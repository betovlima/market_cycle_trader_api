from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.environment import load_project_environment
from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
    utc_now,
)
from ..schemas.requests import BacktestExecutionRequest
from .capital_rotation import ROTATION_FEATURES, _build_walk_forward_folds, prepare_rotation_panel
from .market_data import load_market_bars, validate_and_clean_bars
from .temporal_runtime import run_independent_fit_tasks, temporal_fit_worker_count

load_project_environment()


@dataclass(frozen=True)
class _ConstantProbabilityClassifier:
    probability: float

    def predict_proba(self, x: Any) -> np.ndarray:
        value = float(np.clip(self.probability, 1e-6, 1.0 - 1e-6))
        positive = np.full(len(x), value, dtype=float)
        return np.column_stack((1.0 - positive, positive))


@dataclass(frozen=True)
class _PlattCalibrator:
    model: Any | None

    def transform(self, raw_probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.model is None:
            return values
        logits = np.log(values / (1.0 - values)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logits)[:, 1], 1e-6, 1.0 - 1e-6)


@dataclass(frozen=True)
class _BinaryModelBundle:
    model: Any
    calibrator: _PlattCalibrator
    baseline_probability: float
    validation_auc: float | None
    validation_brier_skill: float | None
    validation_samples: int


@dataclass(frozen=True)
class _DrawdownModelBundle:
    model: Any
    baseline_drawdown: float
    validation_mae_skill: float | None
    validation_rank_correlation: float | None
    validation_samples: int


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lightgbm_settings(config: Any) -> dict[str, Any]:
    snapshot = getattr(config, "research_model_settings", {}) or {}
    values = snapshot.get("lightgbm") if isinstance(snapshot, dict) else None
    if not isinstance(values, dict):
        raise ValueError("Temporal Intelligence requires the immutable LightGBM settings saved with the selected Strategy.")
    required = {
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "min_child_weight", "subsample", "subsample_freq",
        "colsample_bytree", "reg_alpha", "reg_lambda", "max_bin", "n_jobs",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError("Temporal Intelligence LightGBM settings are incomplete: " + ", ".join(missing))
    return dict(values)


def _effective_n_jobs(configured: int) -> int:
    override = str(os.getenv("MCT_MODEL_THREADS_OVERRIDE") or "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    # Respect the immutable Strategy model profile when no runtime safety override is active.
    return 1 if int(configured) == 0 else int(configured)


def _model_kwargs(config: Any) -> dict[str, Any]:
    settings = _lightgbm_settings(config)
    n_jobs = _effective_n_jobs(int(settings["n_jobs"]))
    if bool(config.deterministic_execution):
        n_jobs = max(1, int(config.numeric_thread_limit))
    return {
        "boosting_type": "gbdt",
        "n_estimators": int(settings["n_estimators"]),
        "learning_rate": float(settings["learning_rate"]),
        "max_depth": int(settings["max_depth"]),
        "num_leaves": int(settings["num_leaves"]),
        "min_child_samples": int(settings["min_child_samples"]),
        "min_child_weight": float(settings["min_child_weight"]),
        "subsample": float(settings["subsample"]),
        "subsample_freq": int(settings["subsample_freq"]),
        "colsample_bytree": float(settings["colsample_bytree"]),
        "reg_alpha": float(settings["reg_alpha"]),
        "reg_lambda": float(settings["reg_lambda"]),
        "max_bin": int(settings["max_bin"]),
        "random_state": int(config.random_state),
        "n_jobs": n_jobs,
        "deterministic": bool(config.deterministic_execution),
        "force_col_wise": bool(config.deterministic_execution),
        "verbosity": -1,
    }


def _decision_barriers(horizon: int) -> tuple[float, float]:
    scale = math.sqrt(max(1.0, float(horizon)) / 20.0)
    profit_barrier = float(np.clip(0.08 * scale, 0.025, 0.18))
    loss_barrier = float(np.clip(0.05 * scale, 0.015, 0.12))
    return profit_barrier, loss_barrier


def _first_barrier_outcome(
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    *,
    upper_price: float,
    lower_price: float,
) -> float:
    upper_hits = np.flatnonzero(np.asarray(future_highs, dtype=float) >= float(upper_price))
    lower_hits = np.flatnonzero(np.asarray(future_lows, dtype=float) <= float(lower_price))
    upper_index = int(upper_hits[0]) if len(upper_hits) else None
    lower_index = int(lower_hits[0]) if len(lower_hits) else None
    if upper_index is None:
        return 0.0
    if lower_index is None:
        return 1.0
    if upper_index == lower_index:
        return float("nan")
    return 1.0 if upper_index < lower_index else 0.0


def _future_target_matrices(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    horizons: list[int],
) -> dict[int, dict[str, pd.DataFrame]]:
    size = len(common_dates)
    result: dict[int, dict[str, pd.DataFrame]] = {}
    for horizon in horizons:
        profit_barrier, loss_barrier = _decision_barriers(horizon)
        returns = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        drawdowns = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        profit_before_loss = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        bottom = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        top = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        trend_persistence = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        trend_direction = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        max_upside = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        max_downside = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
        for symbol in symbols:
            frame = frames[symbol]
            opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
            closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
            highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float) if "high" in frame else closes.copy()
            lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float) if "low" in frame else closes.copy()
            ema_relation = (
                pd.to_numeric(frame["ema_20_vs_50"], errors="coerce").to_numpy(dtype=float)
                if "ema_20_vs_50" in frame
                else np.zeros(size, dtype=float)
            )
            direction_values = np.where(np.isfinite(ema_relation), np.sign(ema_relation), np.nan)
            trend_direction[symbol] = direction_values
            symbol_return = np.full(size, np.nan, dtype=float)
            symbol_drawdown = np.full(size, np.nan, dtype=float)
            symbol_profit_before_loss = np.full(size, np.nan, dtype=float)
            symbol_bottom = np.full(size, np.nan, dtype=float)
            symbol_top = np.full(size, np.nan, dtype=float)
            symbol_trend_persistence = np.full(size, np.nan, dtype=float)
            symbol_max_upside = np.full(size, np.nan, dtype=float)
            symbol_max_downside = np.full(size, np.nan, dtype=float)
            for index in range(0, max(0, size - horizon)):
                entry = opens[index + 1]
                future_closes = closes[index + 1:index + horizon + 1]
                future_highs = highs[index + 1:index + horizon + 1]
                future_lows = lows[index + 1:index + horizon + 1]
                if (
                    not np.isfinite(entry)
                    or entry <= 0
                    or len(future_closes) != horizon
                    or len(future_highs) != horizon
                    or len(future_lows) != horizon
                    or not np.isfinite(future_closes).all()
                    or not np.isfinite(future_highs).all()
                    or not np.isfinite(future_lows).all()
                ):
                    continue
                exit_price = float(future_closes[-1])
                if exit_price <= 0:
                    continue
                symbol_return[index] = math.log(max(exit_price / entry, 1e-12))
                path = np.concatenate(([entry], future_closes))
                running_peak = np.maximum.accumulate(path)
                dd = 1.0 - np.divide(path, running_peak, out=np.zeros_like(path), where=running_peak > 0)
                symbol_drawdown[index] = max(0.0, float(np.nanmax(dd)))
                upside = max(0.0, float(np.nanmax(future_highs) / entry - 1.0))
                downside = max(0.0, float(1.0 - np.nanmin(future_lows) / entry))
                symbol_max_upside[index] = upside
                symbol_max_downside[index] = downside
                symbol_profit_before_loss[index] = _first_barrier_outcome(
                    future_highs,
                    future_lows,
                    upper_price=entry * (1.0 + profit_barrier),
                    lower_price=entry * (1.0 - loss_barrier),
                )
                symbol_bottom[index] = float(upside >= profit_barrier and downside <= loss_barrier * 0.50)
                symbol_top[index] = float(downside >= loss_barrier and upside <= profit_barrier * 0.50)
                direction = direction_values[index]
                if np.isfinite(direction) and direction != 0:
                    terminal_return = exit_price / entry - 1.0
                    material_move = 0.25 * min(profit_barrier, loss_barrier)
                    symbol_trend_persistence[index] = float(direction * terminal_return >= material_move)
            returns[symbol] = symbol_return
            drawdowns[symbol] = symbol_drawdown
            profit_before_loss[symbol] = symbol_profit_before_loss
            bottom[symbol] = symbol_bottom
            top[symbol] = symbol_top
            trend_persistence[symbol] = symbol_trend_persistence
            max_upside[symbol] = symbol_max_upside
            max_downside[symbol] = symbol_max_downside
        benchmark = returns.mean(axis=1, skipna=True)
        alpha = returns.subtract(benchmark, axis=0)
        result[horizon] = {
            "return": returns,
            "benchmark": benchmark.to_frame("benchmark_return"),
            "alpha": alpha,
            "drawdown": drawdowns,
            "profit_before_loss": profit_before_loss,
            "bottom": bottom,
            "top": top,
            "trend_persistence": trend_persistence,
            "trend_direction": trend_direction,
            "max_upside": max_upside,
            "max_downside": max_downside,
            "profit_barrier": profit_barrier,
            "loss_barrier": loss_barrier,
        }
    return result

def _pooled_dataset(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    targets: dict[str, pd.DataFrame],
    target_name: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    feature_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    target_matrix = targets[target_name]
    for symbol in symbols:
        frame = frames[symbol].reindex(dates)
        feature_frame = frame[ROTATION_FEATURES].replace([np.inf, -np.inf], np.nan)
        target = pd.to_numeric(target_matrix[symbol].reindex(dates), errors="coerce")
        valid = feature_frame.notna().all(axis=1) & target.notna()
        if not bool(valid.any()):
            continue
        current_features = feature_frame.loc[valid].astype(float)
        feature_parts.append(current_features)
        target_parts.append(target.loc[valid].to_numpy(dtype=float))
        metadata_parts.append(pd.DataFrame({
            "timestamp": current_features.index,
            "symbol": symbol,
        }, index=current_features.index))
    if not feature_parts:
        return pd.DataFrame(columns=ROTATION_FEATURES), np.asarray([], dtype=float), pd.DataFrame(columns=["timestamp", "symbol"])
    return (
        pd.concat(feature_parts, axis=0, ignore_index=True),
        np.concatenate(target_parts),
        pd.concat(metadata_parts, axis=0, ignore_index=True),
    )


def _pooled_features(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = frames[symbol].reindex(dates)
        features = frame[ROTATION_FEATURES].replace([np.inf, -np.inf], np.nan)
        valid = features.notna().all(axis=1)
        if not bool(valid.any()):
            continue
        current = features.loc[valid].astype(float)
        feature_parts.append(current)
        metadata_parts.append(pd.DataFrame({"timestamp": current.index, "symbol": symbol}, index=current.index))
    if not feature_parts:
        return pd.DataFrame(columns=ROTATION_FEATURES), pd.DataFrame(columns=["timestamp", "symbol"])
    return pd.concat(feature_parts, ignore_index=True), pd.concat(metadata_parts, ignore_index=True)


def _fit_regressor(x: pd.DataFrame, y: np.ndarray, config: Any, *, objective: str = "regression") -> Any:
    from lightgbm import LGBMRegressor

    if len(x) < 100:
        raise ValueError(f"Temporal Intelligence regression needs at least 100 pooled rows; received {len(x)}.")
    model = LGBMRegressor(objective=objective, **_model_kwargs(config))
    model.fit(x, y)
    return model


def _fit_classifier(x: pd.DataFrame, y: np.ndarray, config: Any) -> Any:
    from lightgbm import LGBMClassifier

    labels = np.asarray(y > 0.0, dtype=int)
    if len(x) < 100 or len(np.unique(labels)) < 2:
        raise ValueError("Temporal Intelligence classification needs at least 100 pooled rows with both alpha classes.")
    model = LGBMClassifier(objective="binary", **_model_kwargs(config))
    model.fit(x, labels)
    return model


def _fit_binary_classifier_relaxed(x: pd.DataFrame, y: np.ndarray, config: Any) -> Any:
    labels = np.asarray(y > 0.0, dtype=int)
    if len(x) < 100:
        raise ValueError(f"Temporal Decision Intelligence classification needs at least 100 pooled rows; received {len(x)}.")
    if len(np.unique(labels)) < 2:
        return _ConstantProbabilityClassifier(float(labels.mean()) if len(labels) else 0.0)
    return _fit_classifier(x, y, config)


def _fit_platt_calibrator(raw_probability: np.ndarray, y_alpha: np.ndarray) -> _PlattCalibrator:
    from sklearn.linear_model import LogisticRegression

    probabilities = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(y_alpha > 0.0, dtype=int)
    if len(probabilities) < 100 or len(np.unique(labels)) < 2:
        return _PlattCalibrator(None)
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = LogisticRegression(random_state=42, solver="lbfgs")
    model.fit(logits, labels)
    return _PlattCalibrator(model)


def _expected_calibration_error(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float | None:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probability) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(probability))
    error = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (probability >= left) & (probability < right if index < bins - 1 else probability <= right)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def _safe_auc(labels: np.ndarray, probability: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return None
    return _finite(roc_auc_score(labels, probability))


def _regression_metrics(realized: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    actual = np.asarray(realized, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(estimate)
    if not bool(valid.any()):
        return {"mae": None, "rmse": None, "rank_correlation": None}
    actual = actual[valid]
    estimate = estimate[valid]
    actual_series = pd.Series(actual, dtype=float)
    predicted_series = pd.Series(estimate, dtype=float)
    rank_correlation = None
    if actual_series.nunique(dropna=True) > 1 and predicted_series.nunique(dropna=True) > 1:
        rank_correlation = _finite(actual_series.corr(predicted_series, method="spearman"))
    return {
        "mae": _finite(mean_absolute_error(actual, estimate)),
        "rmse": _finite(math.sqrt(mean_squared_error(actual, estimate))),
        "rank_correlation": rank_correlation,
    }


def _classification_metrics(realized_alpha: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss

    realized = np.asarray(realized_alpha, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(realized) & np.isfinite(probability)
    if not bool(valid.any()):
        return {"brier": None, "log_loss": None, "auc": None, "calibration_error": None, "positive_rate": None}
    labels = np.asarray(realized[valid] > 0.0, dtype=int)
    probability = np.clip(probability[valid], 1e-6, 1.0 - 1e-6)
    return {
        "brier": _finite(brier_score_loss(labels, probability)),
        "log_loss": _finite(log_loss(labels, probability, labels=[0, 1])),
        "auc": _safe_auc(labels, probability),
        "calibration_error": _expected_calibration_error(probability, labels),
        "positive_rate": _finite(labels.mean()),
        "probability_mean": _finite(probability.mean()),
    }


def _confidence_bins(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    edges = (0.0, 0.50, 0.60, 0.70, 0.80, 1.000001)
    rows: list[dict[str, Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (frame["alpha_probability"] >= left) & (frame["alpha_probability"] < right)
        subset = frame.loc[mask]
        if subset.empty:
            continue
        rows.append({
            "from_probability": left,
            "to_probability": min(1.0, right),
            "samples": int(len(subset)),
            "mean_probability": _finite(subset["alpha_probability"].mean()),
            "realized_positive_rate": _finite((subset["realized_alpha"] > 0).mean()),
            "mean_realized_alpha": _finite(subset["realized_alpha"].mean()),
            "mean_predicted_alpha": _finite(subset["predicted_alpha"].mean()),
            "mean_realized_drawdown": _finite(subset["realized_drawdown"].mean()),
        })
    return rows


def _risk_buckets(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if len(frame) < 3:
        return []
    try:
        bucket = pd.qcut(frame["predicted_drawdown"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    except ValueError:
        return []
    rows: list[dict[str, Any]] = []
    for label in ["low", "medium", "high"]:
        subset = frame.loc[bucket == label]
        if subset.empty:
            continue
        rows.append({
            "bucket": label,
            "samples": int(len(subset)),
            "mean_predicted_drawdown": _finite(subset["predicted_drawdown"].mean()),
            "mean_realized_drawdown": _finite(subset["realized_drawdown"].mean()),
            "p90_realized_drawdown": _finite(subset["realized_drawdown"].quantile(0.90)),
        })
    return rows


def _prediction_frame(
    *,
    metadata: pd.DataFrame,
    realized_alpha: np.ndarray,
    realized_drawdown: np.ndarray,
    predicted_alpha: np.ndarray,
    raw_probability: np.ndarray,
    probability: np.ndarray,
    predicted_drawdown: np.ndarray,
    baseline_alpha: float,
    baseline_probability: float,
    baseline_drawdown: float,
    fold_id: int,
    horizon: int,
) -> pd.DataFrame:
    result = metadata.reset_index(drop=True).copy()
    result["fold_id"] = int(fold_id)
    result["horizon"] = int(horizon)
    result["realized_alpha"] = np.asarray(realized_alpha, dtype=float)
    result["realized_drawdown"] = np.asarray(realized_drawdown, dtype=float)
    result["predicted_alpha"] = np.asarray(predicted_alpha, dtype=float)
    result["baseline_alpha"] = float(baseline_alpha)
    result["raw_alpha_probability"] = np.asarray(raw_probability, dtype=float)
    result["alpha_probability"] = np.asarray(probability, dtype=float)
    result["baseline_alpha_probability"] = float(np.clip(baseline_probability, 1e-6, 1.0 - 1e-6))
    result["predicted_drawdown"] = np.clip(np.asarray(predicted_drawdown, dtype=float), 0.0, 1.0)
    result["baseline_drawdown"] = float(max(0.0, baseline_drawdown))
    return result


def _align_test_targets(
    metadata: pd.DataFrame,
    target_matrix: pd.DataFrame,
) -> np.ndarray:
    values = []
    for row in metadata.itertuples(index=False):
        values.append(target_matrix.at[pd.Timestamp(row.timestamp), str(row.symbol)])
    return np.asarray(values, dtype=float)


def _skill_score(model_error: float | None, baseline_error: float | None) -> float | None:
    if model_error is None or baseline_error is None or baseline_error <= 0:
        return None
    return _finite(1.0 - float(model_error) / float(baseline_error))


def _fold_summary(frame: pd.DataFrame) -> dict[str, Any]:
    alpha = _regression_metrics(frame["realized_alpha"].to_numpy(), frame["predicted_alpha"].to_numpy())
    alpha_baseline = _regression_metrics(frame["realized_alpha"].to_numpy(), frame["baseline_alpha"].to_numpy())
    classification = _classification_metrics(frame["realized_alpha"].to_numpy(), frame["alpha_probability"].to_numpy())
    raw_classification = _classification_metrics(frame["realized_alpha"].to_numpy(), frame["raw_alpha_probability"].to_numpy())
    baseline_classification = _classification_metrics(frame["realized_alpha"].to_numpy(), frame["baseline_alpha_probability"].to_numpy())
    risk = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["predicted_drawdown"].to_numpy())
    risk_baseline = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["baseline_drawdown"].to_numpy())
    high = frame.loc[frame["alpha_probability"] >= 0.70]
    return {
        "samples": int(len(frame)),
        "alpha_mae": alpha["mae"],
        "alpha_rmse": alpha["rmse"],
        "alpha_rank_correlation": alpha["rank_correlation"],
        "alpha_baseline_mae": alpha_baseline["mae"],
        "alpha_mae_skill": _skill_score(alpha["mae"], alpha_baseline["mae"]),
        "brier": classification["brier"],
        "raw_brier": raw_classification["brier"],
        "baseline_brier": baseline_classification["brier"],
        "brier_skill": _skill_score(classification["brier"], baseline_classification["brier"]),
        "calibration_brier_gain": (
            _finite(float(raw_classification["brier"]) - float(classification["brier"]))
            if raw_classification["brier"] is not None and classification["brier"] is not None else None
        ),
        "log_loss": classification["log_loss"],
        "auc": classification["auc"],
        "calibration_error": classification["calibration_error"],
        "positive_rate": classification["positive_rate"],
        "drawdown_mae": risk["mae"],
        "drawdown_rmse": risk["rmse"],
        "drawdown_rank_correlation": risk["rank_correlation"],
        "drawdown_baseline_mae": risk_baseline["mae"],
        "drawdown_mae_skill": _skill_score(risk["mae"], risk_baseline["mae"]),
        "high_confidence_samples": int(len(high)),
        "high_confidence_positive_rate": _finite((high["realized_alpha"] > 0).mean()) if len(high) else None,
        "high_confidence_mean_alpha": _finite(high["realized_alpha"].mean()) if len(high) else None,
    }


def _latest_models_and_forecasts(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    horizon: int,
    targets: dict[str, pd.DataFrame],
    config: Any,
) -> list[dict[str, Any]]:
    purge = max(int(config.rotation_purge_days), int(horizon))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    label_end = len(common_dates) - int(horizon)
    calibration_end = label_end
    calibration_start = calibration_end - calibration_days
    train_end = calibration_start - purge
    if train_end < int(config.rotation_minimum_training_rows):
        return []

    train_dates = common_dates[:train_end]
    calibration_dates = common_dates[calibration_start:calibration_end]
    final_dates = common_dates[:label_end]
    latest_date = pd.DatetimeIndex([common_dates[-1]])

    x_train, y_alpha_train, _ = _pooled_dataset(frames, symbols, train_dates, targets, "alpha")
    classifier_for_calibration = _fit_classifier(x_train, y_alpha_train, config)
    x_cal, y_alpha_cal, _ = _pooled_dataset(frames, symbols, calibration_dates, targets, "alpha")
    raw_cal = classifier_for_calibration.predict_proba(x_cal)[:, 1]
    calibrator = _fit_platt_calibrator(raw_cal, y_alpha_cal)

    x_final_alpha, y_final_alpha, _ = _pooled_dataset(frames, symbols, final_dates, targets, "alpha")
    x_final_dd, y_final_dd, _ = _pooled_dataset(frames, symbols, final_dates, targets, "drawdown")
    alpha_model = _fit_regressor(x_final_alpha, y_final_alpha, config)
    classifier = _fit_classifier(x_final_alpha, y_final_alpha, config)
    drawdown_model = _fit_regressor(x_final_dd, y_final_dd, config, objective="regression_l1")

    x_latest, metadata = _pooled_features(frames, symbols, latest_date)
    if x_latest.empty:
        return []
    predicted_alpha = alpha_model.predict(x_latest)
    probability = calibrator.transform(classifier.predict_proba(x_latest)[:, 1])
    predicted_drawdown = np.clip(drawdown_model.predict(x_latest), 0.0, 1.0)

    rows: list[dict[str, Any]] = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        rows.append({
            "symbol": str(row["symbol"]),
            "as_of": pd.Timestamp(row["timestamp"]),
            "horizon": int(horizon),
            "expected_alpha": _finite(predicted_alpha[index]),
            "probability_positive_alpha": _finite(probability[index]),
            "expected_max_drawdown": _finite(predicted_drawdown[index]),
        })
    rows.sort(
        key=lambda item: (
            -(item.get("probability_positive_alpha") or 0.0),
            -(item.get("expected_alpha") or -1e9),
            item["symbol"],
        )
    )
    return rows



def _prepared_xy(split: dict[str, Any], target_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    x = split["x"]
    y = np.asarray(split["targets"][target_name], dtype=float)
    valid = np.isfinite(y)
    if bool(valid.all()):
        return x, y
    indices = np.flatnonzero(valid)
    return x.iloc[indices], y[indices]


def _fit_calibrated_binary_bundle(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    final_fit_dates: pd.DatetimeIndex,
    targets: dict[str, Any],
    target_name: str,
    config: Any,
    *,
    prepared_datasets: dict[str, dict[str, Any]] | None = None,
) -> _BinaryModelBundle:
    if prepared_datasets is None:
        x_train, y_train, _ = _pooled_dataset(frames, symbols, train_dates, targets, target_name)
    else:
        x_train, y_train = _prepared_xy(prepared_datasets["train"], target_name)
    calibration_model = _fit_binary_classifier_relaxed(x_train, y_train, config)
    if prepared_datasets is None:
        x_calibration, y_calibration, _ = _pooled_dataset(frames, symbols, calibration_dates, targets, target_name)
    else:
        x_calibration, y_calibration = _prepared_xy(prepared_datasets["calibration"], target_name)
    raw_calibration = calibration_model.predict_proba(x_calibration)[:, 1]
    train_baseline_probability = float(np.clip(np.mean(y_train > 0.0), 1e-6, 1.0 - 1e-6))
    baseline_calibration = np.full(len(y_calibration), train_baseline_probability, dtype=float)
    raw_metrics = _classification_metrics(y_calibration, raw_calibration)
    baseline_metrics = _classification_metrics(y_calibration, baseline_calibration)
    validation_brier_skill = _skill_score(raw_metrics.get("brier"), baseline_metrics.get("brier"))
    calibrator = _fit_platt_calibrator(raw_calibration, y_calibration)
    if prepared_datasets is None:
        x_final, y_final, _ = _pooled_dataset(frames, symbols, final_fit_dates, targets, target_name)
    else:
        x_final, y_final = _prepared_xy(prepared_datasets["final_fit"], target_name)
    model = _fit_binary_classifier_relaxed(x_final, y_final, config)
    return _BinaryModelBundle(
        model=model,
        calibrator=calibrator,
        baseline_probability=float(np.mean(y_final > 0.0)),
        validation_auc=raw_metrics.get("auc"),
        validation_brier_skill=validation_brier_skill,
        validation_samples=int(len(y_calibration)),
    )


def _fit_drawdown_bundle(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    final_fit_dates: pd.DatetimeIndex,
    targets: dict[str, Any],
    config: Any,
    *,
    prepared_datasets: dict[str, dict[str, Any]] | None = None,
) -> _DrawdownModelBundle:
    if prepared_datasets is None:
        x_train, y_train, _ = _pooled_dataset(frames, symbols, train_dates, targets, "drawdown")
    else:
        x_train, y_train = _prepared_xy(prepared_datasets["train"], "drawdown")
    validation_model = _fit_regressor(x_train, y_train, config, objective="regression_l1")
    if prepared_datasets is None:
        x_calibration, y_calibration, _ = _pooled_dataset(frames, symbols, calibration_dates, targets, "drawdown")
    else:
        x_calibration, y_calibration = _prepared_xy(prepared_datasets["calibration"], "drawdown")
    predicted_calibration = np.clip(validation_model.predict(x_calibration), 0.0, 1.0)
    training_baseline = float(max(0.0, np.mean(y_train)))
    baseline_calibration = np.full(len(y_calibration), training_baseline, dtype=float)
    validation = _regression_metrics(y_calibration, predicted_calibration)
    validation_baseline = _regression_metrics(y_calibration, baseline_calibration)
    if prepared_datasets is None:
        x_final, y_final, _ = _pooled_dataset(frames, symbols, final_fit_dates, targets, "drawdown")
    else:
        x_final, y_final = _prepared_xy(prepared_datasets["final_fit"], "drawdown")
    model = _fit_regressor(x_final, y_final, config, objective="regression_l1")
    return _DrawdownModelBundle(
        model=model,
        baseline_drawdown=float(max(0.0, np.mean(y_final))),
        validation_mae_skill=_skill_score(validation.get("mae"), validation_baseline.get("mae")),
        validation_rank_correlation=validation.get("rank_correlation"),
        validation_samples=int(len(y_calibration)),
    )


def _binary_quality_from_metrics(auc: float | None, skill: float | None, *, primary: bool = False) -> float:
    auc_value = _finite(auc)
    skill_value = _finite(skill)
    if auc_value is None or skill_value is None:
        return 0.35 if primary else 0.0
    auc_strength = float(np.clip((auc_value - 0.50) / 0.10, 0.0, 1.0))
    skill_strength = float(np.clip(skill_value / 0.04, 0.0, 1.0))
    quality = auc_strength * skill_strength
    return max(0.35, quality) if primary else quality


def _binary_quality_weight(bundle: _BinaryModelBundle, *, primary: bool = False) -> float:
    return _binary_quality_from_metrics(bundle.validation_auc, bundle.validation_brier_skill, primary=primary)


def _drawdown_quality_from_metrics(skill: float | None, rank: float | None) -> float:
    skill_value = _finite(skill)
    rank_value = _finite(rank)
    if skill_value is None or rank_value is None or skill_value <= 0.0 or rank_value <= 0.0:
        return 0.0
    return float(np.clip(skill_value / 0.10, 0.0, 1.0) * np.clip(rank_value / 0.40, 0.0, 1.0))


def _drawdown_quality_weight(bundle: _DrawdownModelBundle) -> float:
    return _drawdown_quality_from_metrics(bundle.validation_mae_skill, bundle.validation_rank_correlation)


def _historical_oos_quality(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    by_signal = {str(item["signal"]): item for item in _decision_signal_metrics(frame)}
    quality = {
        signal: _binary_quality_from_metrics(
            metrics.get("auc"), metrics.get("brier_skill"), primary=signal == "profit_before_loss"
        )
        for signal, metrics in by_signal.items()
    }
    risk = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["predicted_drawdown"].to_numpy())
    risk_baseline = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["baseline_drawdown"].to_numpy())
    quality["drawdown"] = _drawdown_quality_from_metrics(
        _skill_score(risk.get("mae"), risk_baseline.get("mae")),
        risk.get("rank_correlation"),
    )
    return quality


_QUALITY_COLUMNS = {
    "profit_before_loss": "profit_before_loss_quality_weight",
    "bottom": "bottom_quality_weight",
    "top": "top_quality_weight",
    "trend_persistence": "trend_persistence_quality_weight",
    "drawdown": "drawdown_quality_weight",
}


def _quality_values_from_row(row: pd.Series) -> dict[str, float]:
    values: dict[str, float] = {}
    for signal, column in _QUALITY_COLUMNS.items():
        value = _finite(row.get(column))
        values[signal] = float(value) if value is not None else (0.35 if signal == "profit_before_loss" else 0.0)
    return values


def _latest_online_quality(frame: pd.DataFrame) -> tuple[dict[str, float] | None, int]:
    if frame.empty or "timestamp" not in frame.columns:
        return None, 0
    latest_timestamp = pd.Timestamp(frame["timestamp"].max())
    latest = frame.loc[pd.to_datetime(frame["timestamp"]) == latest_timestamp]
    if latest.empty:
        return None, 0
    row = latest.iloc[0]
    return _quality_values_from_row(row), int(row.get("quality_history_samples") or 0)


def _apply_online_matured_quality(
    frame: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    horizon: int,
    *,
    window_sessions: int = 126,
    update_every_sessions: int = 5,
    minimum_history_samples: int = 250,
    full_trust_samples: int = 1500,
    smoothing_alpha: float = 0.35,
) -> pd.DataFrame:
    """Update signal-quality weights using only OOS labels already knowable at each decision close."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    ordered_market_dates = pd.DatetimeIndex(pd.to_datetime(common_dates))
    market_position = {pd.Timestamp(value): index for index, value in enumerate(ordered_market_dates)}
    decision_dates = sorted(pd.Timestamp(value) for value in result["timestamp"].dropna().unique())
    previous_quality: dict[str, float] | None = None
    previous_samples = 0
    previous_cutoff: pd.Timestamp | None = None
    last_update_market_index: int | None = None

    for timestamp in decision_dates:
        timestamp_rows = result.loc[result["timestamp"] == timestamp]
        if timestamp_rows.empty:
            continue
        baseline_quality = _quality_values_from_row(timestamp_rows.iloc[0])
        active_quality = dict(previous_quality or baseline_quality)
        history_samples = previous_samples
        maturity_cutoff = previous_cutoff
        current_market_index = market_position.get(timestamp)
        should_update = (
            current_market_index is not None
            and current_market_index >= int(horizon)
            and (
                last_update_market_index is None
                or current_market_index - last_update_market_index >= max(1, int(update_every_sessions))
            )
        )
        if should_update:
            maturity_index = int(current_market_index) - int(horizon)
            cutoff = pd.Timestamp(ordered_market_dates[maturity_index])
            start_index = max(0, maturity_index - max(1, int(window_sessions)) + 1)
            start = pd.Timestamp(ordered_market_dates[start_index])
            history = result.loc[(result["timestamp"] >= start) & (result["timestamp"] <= cutoff)]
            if len(history) >= max(1, int(minimum_history_samples)):
                observed_quality = _historical_oos_quality(history)
                trust_denominator = max(1, int(full_trust_samples) - int(minimum_history_samples))
                trust = float(np.clip(
                    (len(history) - int(minimum_history_samples)) / trust_denominator, 0.0, 1.0
                ))
                target_quality: dict[str, float] = {}
                for signal in _QUALITY_COLUMNS:
                    observed = float(observed_quality.get(signal, baseline_quality[signal]))
                    target_quality[signal] = (1.0 - trust) * baseline_quality[signal] + trust * observed
                if previous_quality is None:
                    active_quality = target_quality
                else:
                    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
                    active_quality = {
                        signal: (1.0 - alpha) * previous_quality[signal] + alpha * target_quality[signal]
                        for signal in _QUALITY_COLUMNS
                    }
                previous_quality = dict(active_quality)
                previous_samples = int(len(history))
                previous_cutoff = cutoff
                history_samples = previous_samples
                maturity_cutoff = cutoff
            elif previous_quality is None:
                previous_quality = dict(baseline_quality)
                previous_samples = int(timestamp_rows.iloc[0].get("quality_history_samples") or 0)
                active_quality = dict(previous_quality)
                history_samples = previous_samples
            last_update_market_index = int(current_market_index)

        mask = result["timestamp"] == timestamp
        for signal, column in _QUALITY_COLUMNS.items():
            result.loc[mask, column] = float(active_quality[signal])
        result.loc[mask, "quality_source"] = "online_matured_oos"
        result.loc[mask, "quality_history_samples"] = int(history_samples)
        result.loc[mask, "online_quality_window_sessions"] = int(window_sessions)
        result.loc[mask, "online_quality_maturity_cutoff"] = maturity_cutoff
    return result


def _binary_signal_metrics(
    realized: np.ndarray,
    probability: np.ndarray,
    raw_probability: np.ndarray,
    baseline_probability: np.ndarray,
) -> dict[str, Any]:
    calibrated = _classification_metrics(realized, probability)
    raw = _classification_metrics(realized, raw_probability)
    baseline = _classification_metrics(realized, baseline_probability)
    realized_values = np.asarray(realized, dtype=float)
    probability_values = np.asarray(probability, dtype=float)
    valid = np.isfinite(realized_values) & np.isfinite(probability_values)
    high = valid & (probability_values >= 0.70)
    labels = realized_values > 0.0
    high_rate = _finite(labels[high].mean()) if bool(high.any()) else None
    positive_rate = _finite(labels[valid].mean()) if bool(valid.any()) else None
    return {
        "brier": calibrated["brier"],
        "raw_brier": raw["brier"],
        "baseline_brier": baseline["brier"],
        "brier_skill": _skill_score(calibrated["brier"], baseline["brier"]),
        "calibration_brier_gain": (
            _finite(float(raw["brier"]) - float(calibrated["brier"]))
            if raw["brier"] is not None and calibrated["brier"] is not None else None
        ),
        "log_loss": calibrated["log_loss"],
        "auc": calibrated["auc"],
        "calibration_error": calibrated["calibration_error"],
        "positive_rate": positive_rate,
        "high_confidence_samples": int(high.sum()),
        "high_confidence_positive_rate": high_rate,
        "high_confidence_lift": (
            _finite(float(high_rate) - float(positive_rate))
            if high_rate is not None and positive_rate is not None else None
        ),
    }


def _decision_signal_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    definitions = (
        ("profit_before_loss", "realized_profit_before_loss", "profit_before_loss_probability", "raw_profit_before_loss_probability", "baseline_profit_before_loss_probability"),
        ("bottom", "realized_bottom", "bottom_probability", "raw_bottom_probability", "baseline_bottom_probability"),
        ("top", "realized_top", "top_probability", "raw_top_probability", "baseline_top_probability"),
        ("trend_persistence", "realized_trend_persistence", "trend_persistence_probability", "raw_trend_persistence_probability", "baseline_trend_persistence_probability"),
    )
    rows: list[dict[str, Any]] = []
    for signal, realized_column, probability_column, raw_column, baseline_column in definitions:
        rows.append({
            "signal": signal,
            **_binary_signal_metrics(
                frame[realized_column].to_numpy(dtype=float),
                frame[probability_column].to_numpy(dtype=float),
                frame[raw_column].to_numpy(dtype=float),
                frame[baseline_column].to_numpy(dtype=float),
            ),
        })
    return rows


def _group_rank_score(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) <= 1:
        return pd.Series(np.ones(len(numeric), dtype=float), index=numeric.index)
    rank = numeric.rank(method="average", ascending=not higher_is_better)
    return 1.0 - (rank - 1.0) / max(1.0, float(len(numeric) - 1))


def _cross_sectional_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    profit = pd.to_numeric(result["profit_before_loss_probability"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(result["predicted_drawdown"], errors="coerce").fillna(1.0)
    grouped = result.assign(_profit=profit, _risk=risk).groupby("timestamp", sort=False)

    result["profit_percentile"] = grouped["_profit"].transform(lambda values: _group_rank_score(values, higher_is_better=True))
    result["risk_safety_percentile"] = grouped["_risk"].transform(lambda values: _group_rank_score(values, higher_is_better=False))
    result["profit_median"] = grouped["_profit"].transform("median")
    result["profit_q25"] = grouped["_profit"].transform(lambda values: float(values.quantile(0.25)))
    result["profit_q75"] = grouped["_profit"].transform(lambda values: float(values.quantile(0.75)))
    result["profit_iqr"] = (result["profit_q75"] - result["profit_q25"]).clip(lower=0.0)
    result["profit_spread_vs_median"] = profit - result["profit_median"]

    top_values = grouped["_profit"].transform("max")
    second_values = grouped["_profit"].transform(
        lambda values: float(values.nlargest(2).iloc[-1]) if len(values) >= 2 else float(values.iloc[0])
    )
    result["profit_top_probability"] = top_values
    result["profit_second_probability"] = second_values
    result["profit_top_gap"] = (top_values - second_values).clip(lower=0.0)
    result["is_profit_top"] = profit >= top_values - 1e-12

    robust_scale = np.maximum(result["profit_iqr"].to_numpy(dtype=float), 0.02)
    separation = np.clip(result["profit_spread_vs_median"].to_numpy(dtype=float) / (1.5 * robust_scale), 0.0, 1.0)
    result["profit_separation_strength"] = separation
    gap_scale = np.maximum(result["profit_iqr"].to_numpy(dtype=float), 0.015)
    gap_strength = np.clip(result["profit_top_gap"].to_numpy(dtype=float) / gap_scale, 0.0, 1.0)
    result["profit_top_gap_strength"] = np.where(result["is_profit_top"].to_numpy(dtype=bool), gap_strength, 0.0)

    baseline_profit = pd.to_numeric(result["baseline_profit_before_loss_probability"], errors="coerce").fillna(0.5)
    above_baseline = pd.Series((profit.to_numpy(dtype=float) > baseline_profit.to_numpy(dtype=float)).astype(float), index=result.index)
    result["profit_breadth_above_baseline"] = above_baseline.groupby(result["timestamp"], sort=False).transform("mean")
    result["risk_median"] = grouped["_risk"].transform("median")
    result["drawdown_vs_median"] = risk - result["risk_median"]
    return result


def _decision_components(
    frame: pd.DataFrame,
    *,
    profit_barrier: float,
    loss_barrier: float,
    one_side_cost: float,
) -> pd.DataFrame:
    enriched = _cross_sectional_context(frame)
    effective_profit = max(1e-6, float(profit_barrier) - 2.0 * float(one_side_cost))
    effective_loss = float(loss_barrier) + 2.0 * float(one_side_cost)
    breakeven_probability = effective_loss / max(1e-9, effective_profit + effective_loss)

    profit_probability = pd.to_numeric(enriched["profit_before_loss_probability"], errors="coerce").fillna(0.0)
    profit_quality = pd.to_numeric(enriched["profit_before_loss_quality_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    adjusted_profit_probability = breakeven_probability + profit_quality * (profit_probability - breakeven_probability)
    expected_barrier_return = adjusted_profit_probability * effective_profit - (1.0 - adjusted_profit_probability) * effective_loss

    predicted_drawdown = pd.to_numeric(enriched["predicted_drawdown"], errors="coerce").fillna(float(loss_barrier))
    risk_quality = pd.to_numeric(enriched["drawdown_quality_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    risk_safety = pd.to_numeric(enriched["risk_safety_percentile"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    risk_excess = np.maximum(0.0, predicted_drawdown - float(loss_barrier))
    risk_component = -0.45 * risk_quality * risk_excess

    bottom_probability = pd.to_numeric(enriched["bottom_probability"], errors="coerce").fillna(0.0)
    top_probability = pd.to_numeric(enriched["top_probability"], errors="coerce").fillna(0.0)
    baseline_bottom = pd.to_numeric(enriched["baseline_bottom_probability"], errors="coerce").fillna(0.0)
    baseline_top = pd.to_numeric(enriched["baseline_top_probability"], errors="coerce").fillna(0.0)
    bottom_quality = pd.to_numeric(enriched["bottom_quality_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    top_quality = pd.to_numeric(enriched["top_quality_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    trend_direction = pd.to_numeric(enriched["trend_direction"], errors="coerce").fillna(0.0)
    persistence = pd.to_numeric(enriched["trend_persistence_probability"], errors="coerce").fillna(0.5)
    trend_quality = pd.to_numeric(enriched["trend_persistence_quality_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    trend_long_support = pd.Series(np.where(trend_direction >= 0.0, persistence, 1.0 - persistence), index=enriched.index, dtype=float)

    optional_signal = (
        bottom_quality * (bottom_probability - baseline_bottom)
        - top_quality * (top_probability - baseline_top)
        + trend_quality * (trend_long_support - 0.5)
    )
    optional_support = pd.Series(np.clip(0.5 + optional_signal.to_numpy(dtype=float), 0.0, 1.0), index=enriched.index)

    profit_rank_weight = 0.72 + 0.08 * profit_quality
    risk_rank_weight = 0.23 - 0.03 * profit_quality
    optional_rank_weight = 1.0 - profit_rank_weight - risk_rank_weight
    asset_rank_score = (
        profit_rank_weight * pd.to_numeric(enriched["profit_percentile"], errors="coerce").fillna(0.0)
        + risk_rank_weight * risk_safety
        + optional_rank_weight * optional_support
    )

    separation = pd.to_numeric(enriched["profit_separation_strength"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    top_gap_strength = pd.to_numeric(enriched["profit_top_gap_strength"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    breadth = pd.to_numeric(enriched["profit_breadth_above_baseline"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    quality_modifier = 0.75 + 0.25 * profit_quality
    opportunity_gate_score = quality_modifier * (0.50 * separation + 0.15 * top_gap_strength + 0.25 * risk_safety + 0.10 * breadth)

    relative_strength = pd.Series(
        np.clip(0.5 + enriched["profit_spread_vs_median"].to_numpy(dtype=float) / (2.0 * np.maximum(enriched["profit_iqr"].to_numpy(dtype=float), 0.02)), 0.0, 1.0),
        index=enriched.index,
    )
    hold_score = 0.62 * relative_strength + 0.28 * risk_safety + 0.10 * optional_support

    entry_threshold = 0.30 + min(0.05, 10.0 * float(one_side_cost)) + 0.04 * (1.0 - profit_quality)
    exit_threshold = 0.39 + min(0.04, 8.0 * float(one_side_cost))
    rotation_hurdle = 0.07 + min(0.05, 10.0 * float(one_side_cost))

    return pd.DataFrame({
        "breakeven_probability": float(breakeven_probability),
        "adjusted_profit_probability": adjusted_profit_probability,
        "expected_barrier_return": expected_barrier_return,
        "risk_component": risk_component,
        "bottom_component": 0.20 * effective_profit * bottom_quality * (bottom_probability - baseline_bottom),
        "top_component": -0.20 * effective_loss * top_quality * (top_probability - baseline_top),
        "trend_component": 0.15 * effective_profit * trend_quality * (trend_long_support - 0.5),
        "profit_percentile": enriched["profit_percentile"],
        "risk_safety_percentile": risk_safety,
        "profit_median": enriched["profit_median"],
        "profit_iqr": enriched["profit_iqr"],
        "profit_spread_vs_median": enriched["profit_spread_vs_median"],
        "profit_top_probability": enriched["profit_top_probability"],
        "profit_second_probability": enriched["profit_second_probability"],
        "profit_top_gap": enriched["profit_top_gap"],
        "profit_separation_strength": separation,
        "profit_top_gap_strength": top_gap_strength,
        "profit_breadth_above_baseline": breadth,
        "risk_median": enriched["risk_median"],
        "drawdown_vs_median": enriched["drawdown_vs_median"],
        "optional_support": optional_support,
        "asset_rank_score": asset_rank_score,
        "opportunity_gate_score": opportunity_gate_score,
        "entry_score": opportunity_gate_score,
        "hold_score": hold_score,
        "cash_score": entry_threshold,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "rotation_hurdle": rotation_hurdle,
    }, index=frame.index)

def _decision_prediction_frame(
    *,
    metadata: pd.DataFrame,
    realized_profit_before_loss: np.ndarray,
    realized_bottom: np.ndarray,
    realized_top: np.ndarray,
    realized_trend_persistence: np.ndarray,
    trend_direction: np.ndarray,
    realized_drawdown: np.ndarray,
    classifier_predictions: dict[str, tuple[np.ndarray, np.ndarray, _BinaryModelBundle]],
    predicted_drawdown: np.ndarray,
    drawdown_bundle: _DrawdownModelBundle,
    fold_id: int,
    horizon: int,
    profit_barrier: float,
    loss_barrier: float,
    one_side_cost: float,
    quality_overrides: dict[str, float] | None = None,
    quality_history_samples: int = 0,
) -> pd.DataFrame:
    result = metadata.reset_index(drop=True).copy()
    result["fold_id"] = int(fold_id)
    result["horizon"] = int(horizon)
    result["quality_source"] = "prior_oos" if quality_overrides else "pretest_validation"
    result["quality_history_samples"] = int(quality_history_samples)
    result["realized_profit_before_loss"] = np.asarray(realized_profit_before_loss, dtype=float)
    result["realized_bottom"] = np.asarray(realized_bottom, dtype=float)
    result["realized_top"] = np.asarray(realized_top, dtype=float)
    result["realized_trend_persistence"] = np.asarray(realized_trend_persistence, dtype=float)
    result["trend_direction"] = np.asarray(trend_direction, dtype=float)
    result["realized_drawdown"] = np.asarray(realized_drawdown, dtype=float)
    for name, (raw_probability, probability, bundle) in classifier_predictions.items():
        result[f"raw_{name}_probability"] = np.asarray(raw_probability, dtype=float)
        result[f"{name}_probability"] = np.asarray(probability, dtype=float)
        result[f"baseline_{name}_probability"] = float(np.clip(bundle.baseline_probability, 1e-6, 1.0 - 1e-6))
        result[f"{name}_validation_auc"] = bundle.validation_auc
        result[f"{name}_validation_brier_skill"] = bundle.validation_brier_skill
        result[f"{name}_validation_samples"] = bundle.validation_samples
        default_quality = _binary_quality_weight(bundle, primary=name == "profit_before_loss")
        result[f"{name}_quality_weight"] = float((quality_overrides or {}).get(name, default_quality))
    result["trend_reversal_probability"] = 1.0 - result["trend_persistence_probability"]
    result["predicted_drawdown"] = np.clip(np.asarray(predicted_drawdown, dtype=float), 0.0, 1.0)
    result["baseline_drawdown"] = float(max(0.0, drawdown_bundle.baseline_drawdown))
    result["drawdown_validation_mae_skill"] = drawdown_bundle.validation_mae_skill
    result["drawdown_validation_rank_correlation"] = drawdown_bundle.validation_rank_correlation
    result["drawdown_validation_samples"] = drawdown_bundle.validation_samples
    result["drawdown_quality_weight"] = float((quality_overrides or {}).get("drawdown", _drawdown_quality_weight(drawdown_bundle)))
    result["profit_barrier"] = float(profit_barrier)
    result["loss_barrier"] = float(loss_barrier)
    components = _decision_components(
        result,
        profit_barrier=profit_barrier,
        loss_barrier=loss_barrier,
        one_side_cost=one_side_cost,
    )
    for column in components.columns:
        result[column] = components[column]
    result["decision_score"] = result["entry_score"]
    return result


def _decision_fold_summary(frame: pd.DataFrame) -> dict[str, Any]:
    risk = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["predicted_drawdown"].to_numpy())
    risk_baseline = _regression_metrics(frame["realized_drawdown"].to_numpy(), frame["baseline_drawdown"].to_numpy())
    signal_metrics = _decision_signal_metrics(frame)
    by_signal = {str(item["signal"]): item for item in signal_metrics}
    profit = by_signal.get("profit_before_loss", {})
    return {
        "samples": int(len(frame)),
        "signal_metrics": signal_metrics,
        "profit_before_loss_auc": profit.get("auc"),
        "profit_before_loss_brier": profit.get("brier"),
        "profit_before_loss_brier_skill": profit.get("brier_skill"),
        "profit_before_loss_calibration_error": profit.get("calibration_error"),
        "profit_before_loss_high_confidence_lift": profit.get("high_confidence_lift"),
        "bottom_auc": by_signal.get("bottom", {}).get("auc"),
        "top_auc": by_signal.get("top", {}).get("auc"),
        "trend_persistence_auc": by_signal.get("trend_persistence", {}).get("auc"),
        "drawdown_mae": risk["mae"],
        "drawdown_rmse": risk["rmse"],
        "drawdown_rank_correlation": risk["rank_correlation"],
        "drawdown_baseline_mae": risk_baseline["mae"],
        "drawdown_mae_skill": _skill_score(risk["mae"], risk_baseline["mae"]),
    }


def _open_price_matrix(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=common_dates, columns=symbols, dtype=float)
    for symbol in symbols:
        matrix[symbol] = pd.to_numeric(frames[symbol].reindex(common_dates)["open"], errors="coerce")
    return matrix


def _state_duration_metrics(state_history: list[tuple[int, str]]) -> dict[str, Any]:
    position_lengths: list[int] = []
    cash_lengths: list[int] = []
    previous_fold: int | None = None
    current_state: str | None = None
    length = 0
    for fold_id, state in state_history:
        if previous_fold != fold_id or current_state != state:
            if length > 0 and current_state is not None:
                (cash_lengths if current_state == "CASH" else position_lengths).append(length)
            previous_fold = int(fold_id)
            current_state = str(state)
            length = 1
        else:
            length += 1
    if length > 0 and current_state is not None:
        (cash_lengths if current_state == "CASH" else position_lengths).append(length)

    return {
        "position_spell_count": int(len(position_lengths)),
        "average_holding_days": _finite(np.mean(position_lengths)) if position_lengths else None,
        "median_holding_days": _finite(np.median(position_lengths)) if position_lengths else None,
        "short_holding_ratio_2d": _finite(np.mean(np.asarray(position_lengths) <= 2)) if position_lengths else None,
        "cash_spell_count": int(len(cash_lengths)),
        "average_cash_days": _finite(np.mean(cash_lengths)) if cash_lengths else None,
        "median_cash_days": _finite(np.median(cash_lengths)) if cash_lengths else None,
        "max_cash_days": int(max(cash_lengths)) if cash_lengths else 0,
    }


def _trend_capture_rotation_advantage(challenger: pd.Series, incumbent: pd.Series) -> float:
    return float(
        0.50 * (float(challenger["asset_rank_score"]) - float(incumbent["asset_rank_score"]))
        + 0.20 * (float(challenger["short_profit_consensus"]) - float(incumbent["short_profit_consensus"]))
        + 0.15 * (float(challenger["long_profit_confirmation"]) - float(incumbent["long_profit_confirmation"]))
        + 0.15 * (float(challenger["all_horizon_risk_safety"]) - float(incumbent["all_horizon_risk_safety"]))
    )


def _trend_capture_challenger_confirmation(challenger: pd.Series) -> float:
    return float(
        0.45 * float(challenger["short_horizon_agreement"])
        + 0.25 * float(challenger["long_profit_confirmation"])
        + 0.20 * float(challenger["all_horizon_risk_safety"])
        + 0.10 * float(challenger["horizon_agreement"])
    )


def _shadow_capital_study(
    frame: pd.DataFrame,
    open_prices: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    config: Any,
    *,
    include_diagnostics: bool = False,
    include_economic_curve: bool = False,
    decision_policy: str = "standard",
) -> dict[str, Any]:
    if frame.empty:
        return {}
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(common_dates)}
    daily_returns: list[float] = []
    action_counts = {"buy": 0, "hold": 0, "sell": 0, "rotate": 0, "cash": 0}
    diagnostics: list[dict[str, Any]] = []
    economic_rows: list[dict[str, Any]] = []
    exposure_days = 0
    switch_count = 0
    state_history: list[tuple[int, str]] = []
    reentry_count = 0
    next_day_reentry_count = 0
    rotation_before_cash_count = 0
    incumbent_entry_recovery_hold_count = 0
    defensive_rotation_count = 0
    signal_unavailable_rotation_count = 0
    defensive_exit_cash_count = 0
    opportunity_exit_cash_count = 0
    trend_capture = decision_policy in {"trend_capture_hysteresis", "adaptive_trend_capture", "adaptive_rotation_before_cash"}
    adaptive_trend_capture = decision_policy in {"adaptive_trend_capture", "adaptive_rotation_before_cash"}
    rotation_before_cash = decision_policy == "adaptive_rotation_before_cash"
    for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
        current_symbol: str | None = None
        position_age = 0
        cash_recovery = False
        cash_recovery_mode: str | None = None
        cash_age = 0
        dates = sorted(pd.Timestamp(value) for value in fold_frame["timestamp"].dropna().unique())
        for timestamp in dates:
            date_index = date_to_index.get(timestamp)
            if date_index is None or date_index + 2 >= len(common_dates):
                continue
            rows = fold_frame.loc[fold_frame["timestamp"] == timestamp].copy()
            rows = rows.loc[np.isfinite(rows["entry_score"]) & np.isfinite(rows["hold_score"])]
            if rows.empty:
                continue
            ranking_column = "asset_rank_score" if "asset_rank_score" in rows.columns else "entry_score"
            rows = rows.sort_values([ranking_column, "profit_before_loss_probability"], ascending=[False, False])
            best = rows.iloc[0]
            best_symbol = str(best["symbol"])
            best_entry_score = float(best["entry_score"])
            entry_threshold = float(best["entry_threshold"])
            current_row = rows.loc[rows["symbol"] == current_symbol].iloc[0] if current_symbol is not None and bool((rows["symbol"] == current_symbol).any()) else None
            current_hold_score = float(current_row["hold_score"]) if current_row is not None else None
            exit_threshold = float(current_row["exit_threshold"]) if current_row is not None else float(best["exit_threshold"])
            rotation_hurdle = float(current_row["rotation_hurdle"]) if current_row is not None else float(best["rotation_hurdle"])

            target_symbol = current_symbol
            action = "hold" if current_symbol is not None else "cash"
            reason = "hold_relative_opportunity_preserved" if current_symbol is not None else "cash_relative_gate_not_cleared"
            base_entry_threshold = entry_threshold
            effective_entry_threshold = entry_threshold
            active_reentry_margin = 0.0
            incumbent_persistence = _finite(current_row.get("incumbent_persistence_score")) if current_row is not None else None
            rotation_advantage = None
            dynamic_rotation_hurdle = rotation_hurdle
            challenger_confirmation = None
            severe_risk_exit = False
            risk_break_exit = False
            risk_deterioration_exit = False
            should_exit = False
            incumbent_risk_health = None
            challenger_confirmation_threshold = 0.58
            position_age_before = int(position_age)
            cash_age_before = int(cash_age)

            if trend_capture:
                if current_symbol is None and cash_recovery:
                    reentry_margin = float(best.get("reentry_margin") or 0.11)
                    reentry_decay = max(1.0, float(best.get("reentry_decay_sessions") or 5.0))
                    if rotation_before_cash:
                        if cash_recovery_mode == "opportunity":
                            reentry_margin *= 0.35
                            reentry_decay = max(1.0, reentry_decay * 0.50)
                        elif cash_recovery_mode == "signal_unavailable":
                            reentry_margin *= 0.50
                            reentry_decay = max(1.0, reentry_decay * 0.60)
                    active_reentry_margin = reentry_margin * math.exp(-float(cash_age) / reentry_decay)
                    effective_entry_threshold += active_reentry_margin
                best_entry_eligible = best_entry_score > effective_entry_threshold

                if current_symbol is None:
                    if best_entry_eligible:
                        if cash_recovery:
                            reentry_count += 1
                            if cash_age <= 1:
                                next_day_reentry_count += 1
                        target_symbol = best_symbol
                        action = "buy"
                        reason = "trend_capture_entry_or_reentry_confirmed"
                        cash_recovery = False
                        cash_recovery_mode = None
                        cash_age = 0
                        position_age = 1
                    else:
                        target_symbol = None
                        action = "cash"
                        reason = "cash_reentry_hysteresis_not_cleared" if cash_recovery else "cash_risk_adjusted_entry_not_cleared"
                        cash_age += 1
                        position_age = 0
                elif current_row is None or incumbent_persistence is None:
                    best_risk_safety = float(best.get("all_horizon_risk_safety") or 0.0)
                    unavailable_rotation_eligible = (
                        rotation_before_cash
                        and best_symbol != current_symbol
                        and best_entry_score > base_entry_threshold + 0.02
                        and best_risk_safety >= 0.12
                    )
                    if unavailable_rotation_eligible:
                        target_symbol = best_symbol
                        action = "rotate"
                        reason = "incumbent_signal_unavailable_rotates_to_strong_challenger"
                        signal_unavailable_rotation_count += 1
                        rotation_before_cash_count += 1
                        position_age = 1
                    else:
                        target_symbol = None
                        action = "sell"
                        reason = "incumbent_signal_unavailable"
                        cash_recovery = True
                        cash_recovery_mode = "signal_unavailable" if rotation_before_cash else None
                        cash_age = 1
                        position_age = 0
                else:
                    current_risk = float(current_row.get("all_horizon_risk_safety") or 0.0)
                    current_short_risk = float(current_row.get("short_risk_safety") or 0.0)
                    current_long_risk = float(current_row.get("long_risk_safety") or 0.0)
                    current_short_profit = float(current_row.get("short_profit_consensus") or 0.0)
                    current_long_profit = float(current_row.get("long_profit_confirmation") or 0.0)
                    incumbent_risk_health = float(current_row.get("incumbent_risk_health") or 0.0)
                    if adaptive_trend_capture:
                        severe_risk_exit = (
                            current_risk <= 0.020
                            or (current_risk <= 0.045 and min(current_short_risk, current_long_risk) <= 0.10)
                        )
                        risk_break_exit = (
                            current_risk <= 0.10
                            and (current_short_risk <= 0.18 or current_long_risk <= 0.16)
                        )
                        risk_deterioration_exit = (
                            current_risk < 0.18 or current_short_profit < 0.28 or current_long_profit < 0.28
                        )
                        should_exit = (
                            severe_risk_exit
                            or risk_break_exit
                            or (incumbent_persistence <= exit_threshold and risk_deterioration_exit)
                        )
                    else:
                        severe_risk_exit = current_risk <= 0.035 and current_short_risk <= 0.063
                        risk_deterioration_exit = (
                            current_risk < 0.16 or current_short_profit < 0.30 or current_long_profit < 0.30
                        )
                        should_exit = severe_risk_exit or (incumbent_persistence <= exit_threshold and risk_deterioration_exit)
                    if best_symbol != current_symbol:
                        rotation_advantage = _trend_capture_rotation_advantage(best, current_row)
                        challenger_confirmation = _trend_capture_challenger_confirmation(best)
                    if adaptive_trend_capture:
                        risk_pressure = 1.0 - float(np.clip(incumbent_risk_health or 0.0, 0.0, 1.0))
                        dynamic_rotation_hurdle = float(np.clip(
                            rotation_hurdle
                            + 0.07 * incumbent_persistence * (1.0 - risk_pressure)
                            - 0.055 * risk_pressure,
                            0.005,
                            0.12,
                        ))
                        challenger_confirmation_threshold = 0.58 - 0.14 * risk_pressure
                    elif best_symbol != current_symbol:
                        dynamic_rotation_hurdle = max(
                            0.015,
                            rotation_hurdle
                            + 0.08 * incumbent_persistence
                            - 0.04 * max(0.0, 0.20 - current_risk) / 0.20,
                        )

                    defensive_exit = bool(severe_risk_exit or risk_break_exit)
                    opportunity_exit = bool(should_exit and not defensive_exit)
                    best_risk_safety = float(best.get("all_horizon_risk_safety") or 0.0)
                    if should_exit and rotation_before_cash:
                        defensive_rotation_eligible = (
                            defensive_exit
                            and best_symbol != current_symbol
                            and best_entry_score > base_entry_threshold
                            and best_risk_safety >= 0.35
                            and rotation_advantage is not None
                            and rotation_advantage > 0.0
                            and challenger_confirmation is not None
                            and challenger_confirmation > max(0.72, challenger_confirmation_threshold + 0.10)
                        )
                        incumbent_entry_recovery = (
                            opportunity_exit
                            and best_symbol == current_symbol
                            and best_entry_score > base_entry_threshold
                            and incumbent_risk_health >= 0.20
                        )
                        opportunity_rotation_hurdle = min(0.01, max(0.0, 0.50 * dynamic_rotation_hurdle))
                        opportunity_rotation_eligible = (
                            opportunity_exit
                            and best_symbol != current_symbol
                            and best_entry_score > base_entry_threshold
                            and rotation_advantage is not None
                            and rotation_advantage > opportunity_rotation_hurdle
                            and challenger_confirmation is not None
                            and challenger_confirmation > challenger_confirmation_threshold
                        )
                        if defensive_rotation_eligible:
                            target_symbol = best_symbol
                            action = "rotate"
                            reason = "defensive_exit_rotates_to_safe_strong_challenger"
                            defensive_rotation_count += 1
                            rotation_before_cash_count += 1
                            position_age = 1
                        elif incumbent_entry_recovery:
                            target_symbol = current_symbol
                            action = "hold"
                            reason = "incumbent_entry_signal_overrides_soft_exit"
                            incumbent_entry_recovery_hold_count += 1
                            position_age += 1
                        elif opportunity_rotation_eligible:
                            target_symbol = best_symbol
                            action = "rotate"
                            reason = "opportunity_exit_rotates_before_cash"
                            rotation_before_cash_count += 1
                            position_age = 1
                        else:
                            target_symbol = None
                            action = "sell"
                            reason = (
                                "severe_risk_exit" if severe_risk_exit
                                else "risk_break_exit" if risk_break_exit
                                else "incumbent_persistence_below_exit"
                            )
                            cash_recovery = True
                            cash_recovery_mode = "defensive" if defensive_exit else "opportunity"
                            if defensive_exit:
                                defensive_exit_cash_count += 1
                            else:
                                opportunity_exit_cash_count += 1
                            cash_age = 1
                            position_age = 0
                    elif should_exit:
                        if (
                            best_symbol != current_symbol
                            and best_entry_score > base_entry_threshold
                            and rotation_advantage is not None
                            and rotation_advantage > 0.02
                        ):
                            target_symbol = best_symbol
                            action = "rotate"
                            reason = "incumbent_deteriorated_challenger_confirmed"
                            position_age = 1
                        else:
                            target_symbol = None
                            action = "sell"
                            reason = (
                                "severe_risk_exit" if severe_risk_exit
                                else "risk_break_exit" if risk_break_exit
                                else "incumbent_persistence_below_exit"
                            )
                            cash_recovery = True
                            cash_age = 1
                            position_age = 0
                    elif best_symbol != current_symbol and best_entry_score > base_entry_threshold:
                        if (
                            rotation_advantage is not None
                            and rotation_advantage > dynamic_rotation_hurdle
                            and challenger_confirmation is not None
                            and challenger_confirmation > challenger_confirmation_threshold
                        ):
                            target_symbol = best_symbol
                            action = "rotate"
                            reason = "challenger_clears_dynamic_hysteresis"
                            position_age = 1
                        else:
                            target_symbol = current_symbol
                            action = "hold"
                            reason = "incumbent_trend_protected"
                            position_age += 1
                    else:
                        target_symbol = current_symbol
                        action = "hold"
                        reason = "incumbent_trend_persists"
                        position_age += 1
            else:
                if current_symbol is None:
                    if best_entry_score > entry_threshold:
                        target_symbol = best_symbol
                        action = "buy"
                        reason = "relative_opportunity_above_cash"
                elif current_row is None or current_hold_score is None or current_hold_score <= exit_threshold:
                    if best_symbol != current_symbol and best_entry_score > entry_threshold:
                        target_symbol = best_symbol
                        action = "rotate"
                        reason = "current_exit_and_relative_entry"
                    elif best_symbol == current_symbol and best_entry_score > entry_threshold and current_hold_score is not None and current_hold_score > exit_threshold:
                        target_symbol = current_symbol
                        action = "hold"
                        reason = "current_relative_hold_recovered"
                    else:
                        target_symbol = None
                        action = "sell"
                        reason = "current_relative_hold_below_exit"
                elif best_symbol != current_symbol and best_entry_score > entry_threshold and best_entry_score > current_hold_score + rotation_hurdle:
                    target_symbol = best_symbol
                    action = "rotate"
                    reason = "relative_rank_clears_rotation_hurdle"

            cost_sides = 0
            if action == "buy":
                action_counts["buy"] += 1
                cost_sides = 1
                switch_count += 1
            elif action == "sell":
                action_counts["sell"] += 1
                cost_sides = 1
                switch_count += 1
            elif action == "rotate":
                action_counts["rotate"] += 1
                cost_sides = 2
                switch_count += 1
            elif action == "hold":
                action_counts["hold"] += 1
            else:
                action_counts["cash"] += 1

            interval_return = 0.0
            if target_symbol is not None:
                entry_open = _finite(open_prices.at[common_dates[date_index + 1], target_symbol])
                next_open = _finite(open_prices.at[common_dates[date_index + 2], target_symbol])
                if entry_open is not None and next_open is not None and entry_open > 0:
                    interval_return = next_open / entry_open - 1.0
                    exposure_days += 1
                else:
                    target_symbol = None
                    if action in {"buy", "rotate"}:
                        action_counts[action] = max(0, action_counts[action] - 1)
                        action_counts["cash"] += 1
                        switch_count = max(0, switch_count - 1)
                        cost_sides = 0
                        action = "cash"
                        reason = "target_open_unavailable"

            if include_diagnostics:
                diagnostics.append({
                    "fold_id": int(fold_id),
                    "timestamp": timestamp,
                    "current_symbol": current_symbol or "CASH",
                    "best_symbol": best_symbol,
                    "target_symbol": target_symbol or "CASH",
                    "action": action.upper(),
                    "reason": reason,
                    "quality_source": str(best.get("quality_source") or ""),
                    "quality_history_samples": int(best.get("quality_history_samples") or 0),
                    "asset_rank_score": _finite(best.get("asset_rank_score")),
                    "opportunity_gate_score": _finite(best.get("opportunity_gate_score")),
                    "entry_score": _finite(best.get("entry_score")),
                    "risk_adjusted_entry_score": _finite(best.get("risk_adjusted_entry_score")),
                    "entry_risk_multiplier": _finite(best.get("entry_risk_multiplier")),
                    "risk_entry_threshold_penalty": _finite(best.get("risk_entry_threshold_penalty")),
                    "current_hold_score": _finite(current_hold_score),
                    "incumbent_persistence_score": _finite(incumbent_persistence),
                    "incumbent_risk_health": _finite(incumbent_risk_health),
                    "cash_score": _finite(best.get("cash_score")),
                    "base_entry_threshold": _finite(base_entry_threshold),
                    "entry_threshold": _finite(effective_entry_threshold),
                    "active_reentry_margin": _finite(active_reentry_margin),
                    "exit_threshold": _finite(exit_threshold),
                    "rotation_hurdle": _finite(rotation_hurdle),
                    "dynamic_rotation_hurdle": _finite(dynamic_rotation_hurdle),
                    "rotation_advantage": _finite(rotation_advantage),
                    "challenger_confirmation": _finite(challenger_confirmation),
                    "challenger_confirmation_threshold": _finite(challenger_confirmation_threshold),
                    "severe_risk_exit": bool(severe_risk_exit),
                    "risk_break_exit": bool(risk_break_exit),
                    "risk_deterioration_exit": bool(risk_deterioration_exit),
                    "position_age_before": int(position_age_before),
                    "cash_age_before": int(cash_age_before),
                    "cash_recovery_mode": cash_recovery_mode,
                    "defensive_exit": bool(severe_risk_exit or risk_break_exit),
                    "opportunity_exit": bool(should_exit and not (severe_risk_exit or risk_break_exit)),
                    "probability_profit_before_loss": _finite(best.get("profit_before_loss_probability")),
                    "breakeven_probability": _finite(best.get("breakeven_probability")),
                    "adjusted_profit_probability": _finite(best.get("adjusted_profit_probability")),
                    "expected_barrier_return": _finite(best.get("expected_barrier_return")),
                    "profit_percentile": _finite(best.get("profit_percentile")),
                    "profit_median": _finite(best.get("profit_median")),
                    "profit_iqr": _finite(best.get("profit_iqr")),
                    "profit_spread_vs_median": _finite(best.get("profit_spread_vs_median")),
                    "profit_top_gap": _finite(best.get("profit_top_gap")),
                    "profit_separation_strength": _finite(best.get("profit_separation_strength")),
                    "profit_top_gap_strength": _finite(best.get("profit_top_gap_strength")),
                    "profit_breadth_above_baseline": _finite(best.get("profit_breadth_above_baseline")),
                    "risk_safety_percentile": _finite(best.get("risk_safety_percentile")),
                    "risk_median": _finite(best.get("risk_median")),
                    "drawdown_vs_median": _finite(best.get("drawdown_vs_median")),
                    "expected_max_drawdown": _finite(best.get("predicted_drawdown")),
                    "risk_component": _finite(best.get("risk_component")),
                    "bottom_component": _finite(best.get("bottom_component")),
                    "top_component": _finite(best.get("top_component")),
                    "trend_component": _finite(best.get("trend_component")),
                    "profit_quality_weight": _finite(best.get("profit_before_loss_quality_weight")),
                    "drawdown_quality_weight": _finite(best.get("drawdown_quality_weight")),
                    "bottom_quality_weight": _finite(best.get("bottom_quality_weight")),
                    "top_quality_weight": _finite(best.get("top_quality_weight")),
                    "trend_quality_weight": _finite(best.get("trend_persistence_quality_weight")),
                    "entry_rank_score": _finite(best.get("entry_rank_score")),
                    "entry_rank_percentile": _finite(best.get("entry_rank_percentile")),
                    "entry_separation_strength": _finite(best.get("entry_separation_strength")),
                    "entry_top_gap_strength": _finite(best.get("entry_top_gap_strength")),
                    "short_profit_consensus": _finite(best.get("short_profit_consensus")),
                    "short_risk_safety": _finite(best.get("short_risk_safety")),
                    "short_bottom_support": _finite(best.get("short_bottom_support")),
                    "short_horizon_agreement": _finite(best.get("short_horizon_agreement")),
                    "long_profit_confirmation": _finite(best.get("long_profit_confirmation")),
                    "long_risk_safety": _finite(best.get("long_risk_safety")),
                    "long_trend_support": _finite(best.get("long_trend_support")),
                    "long_horizon_agreement": _finite(best.get("long_horizon_agreement")),
                    "cross_horizon_agreement": _finite(best.get("cross_horizon_agreement")),
                    "horizon_agreement": _finite(best.get("horizon_agreement")),
                    "all_horizon_risk_safety": _finite(best.get("all_horizon_risk_safety")),
                })

            state_history.append((int(fold_id), target_symbol or "CASH"))
            factor = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + interval_return)
            daily_returns.append(factor - 1.0)
            if include_economic_curve:
                economic_rows.append({
                    "fold_id": int(fold_id),
                    "decision_timestamp": timestamp,
                    "execution_date": pd.Timestamp(common_dates[date_index + 1]),
                    "next_execution_date": pd.Timestamp(common_dates[date_index + 2]),
                    "current_symbol": current_symbol or "CASH",
                    "target_symbol": target_symbol or "CASH",
                    "action": action.upper(),
                    "reason": reason,
                    "gross_interval_return": _finite(interval_return),
                    "cost_sides": int(cost_sides),
                    "one_side_cost_rate": float(one_side_cost),
                    "net_interval_return": _finite(factor - 1.0),
                })
            current_symbol = target_symbol
        if current_symbol is not None and daily_returns:
            daily_returns[-1] = max(1e-9, (1.0 + daily_returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
    if not daily_returns:
        return {}
    returns = np.asarray(daily_returns, dtype=float)
    equity = float(config.initial_capital) * np.cumprod(1.0 + returns)
    ending_capital = float(equity[-1])
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    cagr = (ending_capital / float(config.initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    running_peak = np.maximum.accumulate(equity)
    drawdown_curve = equity / running_peak - 1.0 if len(equity) else np.asarray([], dtype=float)
    max_drawdown = float(np.min(drawdown_curve)) if len(equity) else 0.0
    if include_economic_curve:
        for index, row in enumerate(economic_rows):
            if index >= len(equity):
                break
            row["strategy_equity"] = _finite(equity[index])
            row["strategy_drawdown"] = _finite(drawdown_curve[index])
            row["cumulative_return"] = _finite(equity[index] / float(config.initial_capital) - 1.0)
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
    result = {
        "initial_capital": float(config.initial_capital),
        "ending_capital": ending_capital,
        "total_return": ending_capital / float(config.initial_capital) - 1.0,
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(max_drawdown),
        "exposure": _finite(exposure_days / max(1, len(returns))),
        "decision_days": int(len(returns)),
        "switch_count": int(switch_count),
        "action_counts": action_counts,
        "one_side_cost_rate": float(one_side_cost),
        "decision_policy": str(decision_policy),
        "reentry_count": int(reentry_count),
        "next_day_reentry_count": int(next_day_reentry_count),
        "rotation_before_cash_count": int(rotation_before_cash_count),
        "incumbent_entry_recovery_hold_count": int(incumbent_entry_recovery_hold_count),
        "defensive_rotation_count": int(defensive_rotation_count),
        "signal_unavailable_rotation_count": int(signal_unavailable_rotation_count),
        "defensive_exit_cash_count": int(defensive_exit_cash_count),
        "opportunity_exit_cash_count": int(opportunity_exit_cash_count),
        **_state_duration_metrics(state_history),
    }
    if include_diagnostics:
        result["decision_diagnostics"] = diagnostics
    if include_economic_curve:
        result["economic_curve"] = economic_rows
    return result


def _latest_decision_forecasts(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    horizon: int,
    targets: dict[str, Any],
    config: Any,
    quality_overrides: dict[str, float] | None = None,
    quality_history_samples: int = 0,
    quality_source: str | None = None,
) -> list[dict[str, Any]]:
    purge = max(int(config.rotation_purge_days), int(horizon))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    label_end = len(common_dates) - int(horizon)
    calibration_end = label_end
    calibration_start = calibration_end - calibration_days
    train_end = calibration_start - purge
    if train_end < int(config.rotation_minimum_training_rows):
        return []
    train_dates = common_dates[:train_end]
    calibration_dates = common_dates[calibration_start:calibration_end]
    final_dates = common_dates[:label_end]
    latest_date = pd.DatetimeIndex([common_dates[-1]])
    bundles: dict[str, _BinaryModelBundle] = {}
    for target_name in ("profit_before_loss", "bottom", "top", "trend_persistence"):
        bundles[target_name] = _fit_calibrated_binary_bundle(
            frames,
            symbols,
            train_dates,
            calibration_dates,
            final_dates,
            targets,
            target_name,
            config,
        )
    drawdown_bundle = _fit_drawdown_bundle(
        frames, symbols, train_dates, calibration_dates, final_dates, targets, config
    )
    x_latest, metadata = _pooled_features(frames, symbols, latest_date)
    if x_latest.empty:
        return []
    latest = metadata.reset_index(drop=True).copy()
    for target_name, bundle in bundles.items():
        latest[f"{target_name}_probability"] = bundle.calibrator.transform(bundle.model.predict_proba(x_latest)[:, 1])
        latest[f"baseline_{target_name}_probability"] = bundle.baseline_probability
        latest[f"{target_name}_validation_auc"] = bundle.validation_auc
        latest[f"{target_name}_validation_brier_skill"] = bundle.validation_brier_skill
        default_quality = _binary_quality_weight(bundle, primary=target_name == "profit_before_loss")
        latest[f"{target_name}_quality_weight"] = float((quality_overrides or {}).get(target_name, default_quality))
    latest["trend_reversal_probability"] = 1.0 - latest["trend_persistence_probability"]
    latest["predicted_drawdown"] = np.clip(drawdown_bundle.model.predict(x_latest), 0.0, 1.0)
    latest["drawdown_validation_mae_skill"] = drawdown_bundle.validation_mae_skill
    latest["drawdown_validation_rank_correlation"] = drawdown_bundle.validation_rank_correlation
    latest["drawdown_quality_weight"] = float((quality_overrides or {}).get("drawdown", _drawdown_quality_weight(drawdown_bundle)))
    trend_matrix = targets["trend_direction"]
    latest["trend_direction"] = [
        trend_matrix.at[pd.Timestamp(row.timestamp), str(row.symbol)]
        for row in latest.itertuples(index=False)
    ]
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))
    components = _decision_components(
        latest,
        profit_barrier=float(targets["profit_barrier"]),
        loss_barrier=float(targets["loss_barrier"]),
        one_side_cost=one_side_cost,
    )
    for column in components.columns:
        latest[column] = components[column]
    latest["decision_score"] = latest["entry_score"]
    latest = latest.sort_values(["asset_rank_score", "profit_before_loss_probability"], ascending=[False, False]).reset_index(drop=True)
    has_target = bool(len(latest) and float(latest.iloc[0]["entry_score"]) > float(latest.iloc[0]["entry_threshold"]))
    rows: list[dict[str, Any]] = []
    for index, row in latest.iterrows():
        direction = _finite(row.get("trend_direction")) or 0.0
        rows.append({
            "symbol": str(row["symbol"]),
            "as_of": pd.Timestamp(row["timestamp"]),
            "horizon": int(horizon),
            "profit_barrier": float(targets["profit_barrier"]),
            "loss_barrier": float(targets["loss_barrier"]),
            "probability_profit_before_loss": _finite(row.get("profit_before_loss_probability")),
            "baseline_profit_before_loss_probability": _finite(row.get("baseline_profit_before_loss_probability")),
            "probability_bottom": _finite(row.get("bottom_probability")),
            "baseline_bottom_probability": _finite(row.get("baseline_bottom_probability")),
            "probability_top": _finite(row.get("top_probability")),
            "baseline_top_probability": _finite(row.get("baseline_top_probability")),
            "probability_trend_persistence": _finite(row.get("trend_persistence_probability")),
            "baseline_trend_persistence_probability": _finite(row.get("baseline_trend_persistence_probability")),
            "probability_trend_reversal": _finite(row.get("trend_reversal_probability")),
            "expected_max_drawdown": _finite(row.get("predicted_drawdown")),
            "trend_state": "up" if direction > 0 else "down" if direction < 0 else "flat",
            "quality_source": quality_source or ("complete_oos" if quality_overrides else "pretest_validation"),
            "quality_history_samples": int(quality_history_samples),
            "decision_score": _finite(row.get("decision_score")),
            "asset_rank_score": _finite(row.get("asset_rank_score")),
            "opportunity_gate_score": _finite(row.get("opportunity_gate_score")),
            "entry_score": _finite(row.get("entry_score")),
            "hold_score": _finite(row.get("hold_score")),
            "entry_threshold": _finite(row.get("entry_threshold")),
            "exit_threshold": _finite(row.get("exit_threshold")),
            "breakeven_probability": _finite(row.get("breakeven_probability")),
            "adjusted_profit_probability": _finite(row.get("adjusted_profit_probability")),
            "expected_barrier_return": _finite(row.get("expected_barrier_return")),
            "profit_percentile": _finite(row.get("profit_percentile")),
            "profit_median": _finite(row.get("profit_median")),
            "profit_iqr": _finite(row.get("profit_iqr")),
            "profit_spread_vs_median": _finite(row.get("profit_spread_vs_median")),
            "profit_top_gap": _finite(row.get("profit_top_gap")),
            "profit_separation_strength": _finite(row.get("profit_separation_strength")),
            "profit_top_gap_strength": _finite(row.get("profit_top_gap_strength")),
            "profit_breadth_above_baseline": _finite(row.get("profit_breadth_above_baseline")),
            "risk_safety_percentile": _finite(row.get("risk_safety_percentile")),
            "risk_median": _finite(row.get("risk_median")),
            "drawdown_vs_median": _finite(row.get("drawdown_vs_median")),
            "risk_component": _finite(row.get("risk_component")),
            "bottom_component": _finite(row.get("bottom_component")),
            "top_component": _finite(row.get("top_component")),
            "trend_component": _finite(row.get("trend_component")),
            "profit_quality_weight": _finite(row.get("profit_before_loss_quality_weight")),
            "drawdown_quality_weight": _finite(row.get("drawdown_quality_weight")),
            "bottom_quality_weight": _finite(row.get("bottom_quality_weight")),
            "top_quality_weight": _finite(row.get("top_quality_weight")),
            "trend_quality_weight": _finite(row.get("trend_persistence_quality_weight")),
            "shadow_target": bool(has_target and index == 0),
            "shadow_state": "ENTRY" if has_target and index == 0 else "CASH",
        })
    return rows


def _multi_horizon_roles(horizons: list[int]) -> dict[str, list[int]]:
    ordered = sorted({int(value) for value in horizons})
    if not ordered:
        return {"entry": [], "hold": [], "risk": []}
    entry = ordered[: min(2, len(ordered))]
    hold = ordered[2:] if len(ordered) > 2 else list(entry)
    return {"entry": entry, "hold": hold, "risk": list(ordered)}


def _weighted_average_columns(
    frame: pd.DataFrame,
    value_columns: list[str],
    weight_columns: list[str] | None = None,
    *,
    neutral_when_unweighted: float | None = None,
) -> pd.Series:
    if not value_columns:
        return pd.Series(np.full(len(frame), 0.5 if neutral_when_unweighted is None else neutral_when_unweighted), index=frame.index, dtype=float)
    values = np.column_stack([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in value_columns
    ])
    finite_values = np.isfinite(values)
    if weight_columns:
        weights = np.column_stack([
            np.clip(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
            for column in weight_columns
        ])
        weights = np.where(finite_values, weights, 0.0)
    else:
        weights = finite_values.astype(float)
    denominator = weights.sum(axis=1)
    numerator = np.where(finite_values, values, 0.0) * weights
    averaged = np.divide(numerator.sum(axis=1), denominator, out=np.full(len(frame), np.nan, dtype=float), where=denominator > 1e-12)
    if neutral_when_unweighted is not None:
        averaged = np.where(denominator > 1e-12, averaged, float(neutral_when_unweighted))
    else:
        fallback_denominator = finite_values.sum(axis=1)
        fallback = np.divide(
            np.where(finite_values, values, 0.0).sum(axis=1),
            fallback_denominator,
            out=np.full(len(frame), 0.5, dtype=float),
            where=fallback_denominator > 0,
        )
        averaged = np.where(denominator > 1e-12, averaged, fallback)
    return pd.Series(averaged, index=frame.index, dtype=float)


def _agreement_score(frame: pd.DataFrame, columns: list[str], *, neutral: float = 0.5) -> pd.Series:
    if not columns:
        return pd.Series(np.full(len(frame), neutral), index=frame.index, dtype=float)
    values = np.column_stack([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in columns
    ])
    result = np.full(len(frame), neutral, dtype=float)
    for row_index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if len(finite) == 1:
            result[row_index] = 1.0
        elif len(finite) >= 2:
            result[row_index] = float(np.clip(1.0 - (float(np.max(finite)) - float(np.min(finite))), 0.0, 1.0))
    return pd.Series(result, index=frame.index, dtype=float)


def _quality_filtered_agreement(
    frame: pd.DataFrame,
    value_columns: list[str],
    quality_columns: list[str],
    *,
    neutral: float = 0.5,
) -> pd.Series:
    if not value_columns:
        return pd.Series(np.full(len(frame), neutral), index=frame.index, dtype=float)
    values = np.column_stack([pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) for column in value_columns])
    qualities = np.column_stack([
        np.clip(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
        for column in quality_columns
    ])
    result = np.full(len(frame), neutral, dtype=float)
    for row_index, (row_values, row_qualities) in enumerate(zip(values, qualities)):
        active = np.isfinite(row_values) & (row_qualities > 1e-6)
        finite = row_values[active]
        if len(finite) == 1:
            result[row_index] = 1.0
        elif len(finite) >= 2:
            result[row_index] = float(np.clip(1.0 - (float(np.max(finite)) - float(np.min(finite))), 0.0, 1.0))
    return pd.Series(result, index=frame.index, dtype=float)


def _multi_horizon_merge(
    frames_by_horizon: dict[int, pd.DataFrame],
    horizons: list[int],
) -> pd.DataFrame:
    keys = ["timestamp", "symbol", "fold_id"]
    merged: pd.DataFrame | None = None
    selected_columns = (
        "profit_before_loss_probability",
        "baseline_profit_before_loss_probability",
        "profit_percentile",
        "profit_before_loss_quality_weight",
        "predicted_drawdown",
        "risk_safety_percentile",
        "drawdown_quality_weight",
        "bottom_probability",
        "baseline_bottom_probability",
        "bottom_quality_weight",
        "top_probability",
        "baseline_top_probability",
        "top_quality_weight",
        "trend_direction",
        "trend_persistence_probability",
        "baseline_trend_persistence_probability",
        "trend_persistence_quality_weight",
        "realized_profit_before_loss",
        "realized_bottom",
        "realized_top",
        "realized_trend_persistence",
        "realized_drawdown",
        "quality_history_samples",
    )
    for horizon in horizons:
        source = frames_by_horizon.get(int(horizon))
        if source is None or source.empty:
            continue
        available = [column for column in selected_columns if column in source.columns]
        current = source[keys + available].copy()
        current = current.rename(columns={column: f"{column}_h{int(horizon)}" for column in available})
        merged = current if merged is None else merged.merge(current, on=keys, how="inner", validate="one_to_one")
    return merged if merged is not None else pd.DataFrame(columns=keys)


def _multi_horizon_relative_context(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    result = frame.copy()
    score = pd.to_numeric(result[score_column], errors="coerce").fillna(0.0)
    grouped = result.assign(_score=score).groupby("timestamp", sort=False)
    result["entry_rank_percentile"] = grouped["_score"].transform(lambda values: _group_rank_score(values, higher_is_better=True))
    result["entry_rank_median"] = grouped["_score"].transform("median")
    result["entry_rank_q25"] = grouped["_score"].transform(lambda values: float(values.quantile(0.25)))
    result["entry_rank_q75"] = grouped["_score"].transform(lambda values: float(values.quantile(0.75)))
    result["entry_rank_iqr"] = (result["entry_rank_q75"] - result["entry_rank_q25"]).clip(lower=0.0)
    result["entry_rank_spread_vs_median"] = score - result["entry_rank_median"]
    top_values = grouped["_score"].transform("max")
    second_values = grouped["_score"].transform(
        lambda values: float(values.nlargest(2).iloc[-1]) if len(values) >= 2 else float(values.iloc[0])
    )
    result["entry_rank_top_gap"] = (top_values - second_values).clip(lower=0.0)
    result["is_entry_top"] = score >= top_values - 1e-12
    separation_scale = np.maximum(result["entry_rank_iqr"].to_numpy(dtype=float), 0.04)
    result["entry_separation_strength"] = np.clip(
        result["entry_rank_spread_vs_median"].to_numpy(dtype=float) / (1.5 * separation_scale), 0.0, 1.0
    )
    gap_scale = np.maximum(result["entry_rank_iqr"].to_numpy(dtype=float), 0.03)
    gap_strength = np.clip(result["entry_rank_top_gap"].to_numpy(dtype=float) / gap_scale, 0.0, 1.0)
    result["entry_top_gap_strength"] = np.where(result["is_entry_top"].to_numpy(dtype=bool), gap_strength, 0.0)
    return result


def _multi_horizon_components(
    merged: pd.DataFrame,
    horizons: list[int],
    *,
    one_side_cost: float,
) -> pd.DataFrame:
    if merged.empty:
        return merged.copy()
    roles = _multi_horizon_roles(horizons)
    entry_horizons = roles["entry"]
    hold_horizons = roles["hold"]
    risk_horizons = roles["risk"]
    result = merged.copy()

    entry_profit_columns = [f"profit_percentile_h{h}" for h in entry_horizons]
    entry_profit_quality_columns = [f"profit_before_loss_quality_weight_h{h}" for h in entry_horizons]
    entry_risk_columns = [f"risk_safety_percentile_h{h}" for h in entry_horizons]
    entry_risk_quality_columns = [f"drawdown_quality_weight_h{h}" for h in entry_horizons]
    hold_profit_columns = [f"profit_percentile_h{h}" for h in hold_horizons]
    hold_profit_quality_columns: list[str] = []
    for horizon in hold_horizons:
        source_quality = pd.to_numeric(result[f"profit_before_loss_quality_weight_h{horizon}"], errors="coerce").fillna(0.0)
        strict_column = f"_strict_profit_quality_h{horizon}"
        result[strict_column] = np.clip((source_quality - 0.35) / 0.65, 0.0, 1.0)
        hold_profit_quality_columns.append(strict_column)
    hold_risk_columns = [f"risk_safety_percentile_h{h}" for h in hold_horizons]
    hold_risk_quality_columns = [f"drawdown_quality_weight_h{h}" for h in hold_horizons]
    all_risk_columns = [f"risk_safety_percentile_h{h}" for h in risk_horizons]
    all_risk_quality_columns = [f"drawdown_quality_weight_h{h}" for h in risk_horizons]

    result["short_profit_consensus"] = _weighted_average_columns(result, entry_profit_columns, entry_profit_quality_columns)
    result["short_risk_safety"] = _weighted_average_columns(result, entry_risk_columns, entry_risk_quality_columns)
    result["long_profit_confirmation"] = _weighted_average_columns(
        result, hold_profit_columns, hold_profit_quality_columns, neutral_when_unweighted=0.5
    )
    result["long_risk_safety"] = _weighted_average_columns(result, hold_risk_columns, hold_risk_quality_columns)
    result["all_horizon_risk_safety"] = _weighted_average_columns(result, all_risk_columns, all_risk_quality_columns)
    result["short_horizon_agreement"] = _agreement_score(result, entry_profit_columns, neutral=1.0)
    result["long_horizon_agreement"] = _quality_filtered_agreement(
        result, hold_profit_columns, hold_profit_quality_columns, neutral=0.5
    )
    strict_long_quality = _weighted_average_columns(
        result, hold_profit_quality_columns, None, neutral_when_unweighted=0.0
    ).clip(0.0, 1.0)
    raw_cross_agreement = 1.0 - np.abs(result["short_profit_consensus"] - result["long_profit_confirmation"])
    result["cross_horizon_agreement"] = strict_long_quality * raw_cross_agreement + (1.0 - strict_long_quality) * result["short_horizon_agreement"]
    result["horizon_agreement"] = np.clip(
        0.50 * result["short_horizon_agreement"]
        + 0.20 * result["long_horizon_agreement"]
        + 0.30 * result["cross_horizon_agreement"],
        0.0,
        1.0,
    )

    short_quality = _weighted_average_columns(
        result,
        entry_profit_quality_columns,
        None,
        neutral_when_unweighted=0.35,
    ).clip(0.0, 1.0)
    result["short_profit_quality"] = short_quality

    bottom_support_columns: list[str] = []
    bottom_weight_columns: list[str] = []
    for horizon in entry_horizons:
        probability_column = f"bottom_probability_h{horizon}"
        baseline_column = f"baseline_bottom_probability_h{horizon}"
        quality_column = f"bottom_quality_weight_h{horizon}"
        support_column = f"_bottom_support_h{horizon}"
        result[support_column] = np.clip(
            0.5
            + pd.to_numeric(result[probability_column], errors="coerce").fillna(0.0)
            - pd.to_numeric(result[baseline_column], errors="coerce").fillna(0.0),
            0.0,
            1.0,
        )
        bottom_support_columns.append(support_column)
        bottom_weight_columns.append(quality_column)
    result["short_bottom_support"] = _weighted_average_columns(
        result, bottom_support_columns, bottom_weight_columns, neutral_when_unweighted=0.5
    )

    trend_support_columns: list[str] = []
    trend_weight_columns: list[str] = []
    for horizon in hold_horizons:
        direction = pd.to_numeric(result[f"trend_direction_h{horizon}"], errors="coerce").fillna(0.0)
        persistence = pd.to_numeric(result[f"trend_persistence_probability_h{horizon}"], errors="coerce").fillna(0.5)
        support_column = f"_trend_support_h{horizon}"
        result[support_column] = np.where(direction >= 0.0, persistence, 1.0 - persistence)
        trend_support_columns.append(support_column)
        trend_weight_columns.append(f"trend_persistence_quality_weight_h{horizon}")
    result["long_trend_support"] = _weighted_average_columns(
        result, trend_support_columns, trend_weight_columns, neutral_when_unweighted=0.5
    )

    result["entry_rank_score"] = np.clip(
        0.58 * result["short_profit_consensus"]
        + 0.20 * result["short_risk_safety"]
        + 0.10 * result["short_horizon_agreement"]
        + 0.07 * result["long_profit_confirmation"]
        + 0.05 * result["short_bottom_support"],
        0.0,
        1.0,
    )
    result = _multi_horizon_relative_context(result, "entry_rank_score")

    result["opportunity_gate_score"] = np.clip(
        0.40 * result["entry_separation_strength"]
        + 0.15 * result["entry_top_gap_strength"]
        + 0.15 * result["short_horizon_agreement"]
        + 0.12 * result["short_risk_safety"]
        + 0.10 * result["horizon_agreement"]
        + 0.08 * result["long_profit_confirmation"],
        0.0,
        1.0,
    )
    risk_safety = result["all_horizon_risk_safety"].clip(0.0, 1.0)
    result["entry_risk_multiplier"] = np.clip(0.42 + 0.58 * np.sqrt(risk_safety), 0.0, 1.0)
    result["risk_adjusted_entry_score"] = np.clip(
        result["opportunity_gate_score"] * result["entry_risk_multiplier"], 0.0, 1.0
    )
    risk_deterioration = np.maximum(0.0, 0.22 - risk_safety)
    result["incumbent_risk_health"] = np.clip((risk_safety - 0.08) / 0.42, 0.0, 1.0)
    result["incumbent_persistence_raw"] = np.clip(
        0.42 * result["long_profit_confirmation"]
        + 0.20 * risk_safety
        + 0.13 * result["short_profit_consensus"]
        + 0.12 * result["horizon_agreement"]
        + 0.13 * result["long_trend_support"],
        0.0,
        1.0,
    )
    result["incumbent_persistence_score"] = np.clip(
        result["incumbent_persistence_raw"] * (0.60 + 0.40 * result["incumbent_risk_health"])
        - 0.12 * np.clip(risk_deterioration / 0.22, 0.0, 1.0),
        0.0,
        1.0,
    )
    result["hold_score"] = result["incumbent_persistence_score"]
    result["risk_entry_threshold_penalty"] = 0.10 * np.power(
        np.clip((0.35 - risk_safety) / 0.35, 0.0, 1.0), 1.5
    )
    result["entry_threshold"] = (
        0.34
        + min(0.05, 10.0 * float(one_side_cost))
        + 0.04 * (1.0 - short_quality)
        + result["risk_entry_threshold_penalty"]
    )
    result["reentry_margin"] = 0.11
    result["reentry_decay_sessions"] = 5.0
    result["exit_threshold"] = 0.37 + min(0.04, 8.0 * float(one_side_cost))
    result["rotation_hurdle"] = 0.03 + min(0.05, 10.0 * float(one_side_cost))
    result["cash_score"] = result["entry_threshold"]
    result["entry_score"] = result["risk_adjusted_entry_score"]
    result["risk_adjusted_asset_rank_score"] = np.clip(
        0.90 * result["entry_rank_score"] + 0.10 * risk_safety, 0.0, 1.0
    )
    result["asset_rank_score"] = result["risk_adjusted_asset_rank_score"]
    result["decision_score"] = result["entry_score"]
    result["profit_before_loss_probability"] = _weighted_average_columns(
        result,
        [f"profit_before_loss_probability_h{h}" for h in entry_horizons],
        entry_profit_quality_columns,
    )
    result["predicted_drawdown"] = _weighted_average_columns(
        result,
        [f"predicted_drawdown_h{h}" for h in risk_horizons],
        all_risk_quality_columns,
    )
    result["profit_percentile"] = result["short_profit_consensus"]
    result["risk_safety_percentile"] = result["all_horizon_risk_safety"]
    result["profit_median"] = result["entry_rank_median"]
    result["profit_iqr"] = result["entry_rank_iqr"]
    result["profit_spread_vs_median"] = result["entry_rank_spread_vs_median"]
    result["profit_top_gap"] = result["entry_rank_top_gap"]
    result["profit_separation_strength"] = result["entry_separation_strength"]
    result["profit_top_gap_strength"] = result["entry_top_gap_strength"]
    result["profit_breadth_above_baseline"] = result["horizon_agreement"]
    result["risk_median"] = result["predicted_drawdown"].groupby(result["timestamp"], sort=False).transform("median")
    result["drawdown_vs_median"] = result["predicted_drawdown"] - result["risk_median"]
    result["quality_source"] = "multi_horizon_online_matured_oos"
    history_columns = [f"quality_history_samples_h{h}" for h in horizons if f"quality_history_samples_h{h}" in result.columns]
    if history_columns:
        result["quality_history_samples"] = result[history_columns].apply(pd.to_numeric, errors="coerce").fillna(0).min(axis=1).astype(int)
    else:
        result["quality_history_samples"] = 0
    result["breakeven_probability"] = np.nan
    result["adjusted_profit_probability"] = result["profit_before_loss_probability"]
    result["expected_barrier_return"] = np.nan
    result["risk_component"] = -risk_deterioration
    result["bottom_component"] = result["short_bottom_support"] - 0.5
    result["top_component"] = 0.0
    result["trend_component"] = result["long_trend_support"] - 0.5
    result["profit_before_loss_quality_weight"] = short_quality
    result["drawdown_quality_weight"] = _weighted_average_columns(
        result, all_risk_quality_columns, None, neutral_when_unweighted=0.0
    ).clip(0.0, 1.0)
    result["bottom_quality_weight"] = _weighted_average_columns(
        result, bottom_weight_columns, None, neutral_when_unweighted=0.0
    ).clip(0.0, 1.0)
    result["top_quality_weight"] = 0.0
    result["trend_persistence_quality_weight"] = _weighted_average_columns(
        result, trend_weight_columns, None, neutral_when_unweighted=0.0
    ).clip(0.0, 1.0)
    return result


def _multi_horizon_frame(
    frames_by_horizon: dict[int, pd.DataFrame],
    horizons: list[int],
    *,
    one_side_cost: float,
) -> pd.DataFrame:
    merged = _multi_horizon_merge(frames_by_horizon, horizons)
    return _multi_horizon_components(merged, horizons, one_side_cost=one_side_cost)


def _multi_horizon_observation_rows(
    frame: pd.DataFrame,
    horizons: list[int],
    *,
    frames_by_symbol: dict[str, pd.DataFrame] | None = None,
    common_dates: pd.DatetimeIndex | None = None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    base_columns = [
        "timestamp", "fold_id", "symbol", "entry_rank_score", "risk_adjusted_asset_rank_score", "entry_rank_percentile",
        "opportunity_gate_score", "risk_adjusted_entry_score", "entry_risk_multiplier", "risk_entry_threshold_penalty", "entry_threshold",
        "reentry_margin", "reentry_decay_sessions", "hold_score", "incumbent_persistence_raw", "incumbent_persistence_score", "incumbent_risk_health",
        "exit_threshold", "rotation_hurdle",
        "short_profit_consensus", "short_risk_safety", "short_bottom_support", "short_horizon_agreement",
        "long_profit_confirmation", "long_risk_safety", "long_trend_support", "long_horizon_agreement",
        "cross_horizon_agreement", "horizon_agreement", "all_horizon_risk_safety", "predicted_drawdown",
        "entry_separation_strength", "entry_top_gap_strength", "short_profit_quality", "quality_history_samples",
    ]
    horizon_columns: list[str] = []
    for horizon in horizons:
        for prefix in (
            "profit_before_loss_probability", "profit_percentile", "profit_before_loss_quality_weight",
            "predicted_drawdown", "risk_safety_percentile", "drawdown_quality_weight",
            "bottom_probability", "bottom_quality_weight", "top_probability", "top_quality_weight",
            "trend_direction", "trend_persistence_probability", "trend_persistence_quality_weight",
            "realized_profit_before_loss", "realized_bottom", "realized_top",
            "realized_trend_persistence", "realized_drawdown",
        ):
            column = f"{prefix}_h{horizon}"
            if column in frame.columns:
                horizon_columns.append(column)
    columns = [column for column in base_columns + horizon_columns if column in frame.columns]
    rows: list[dict[str, Any]] = []
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(common_dates)} if common_dates is not None else {}
    for item in frame[columns].sort_values(["timestamp", "symbol"]).to_dict(orient="records"):
        normalized = {
            key: (pd.Timestamp(value) if key == "timestamp" else _finite(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in item.items()
        }
        timestamp = pd.Timestamp(normalized.get("timestamp")) if normalized.get("timestamp") is not None else None
        symbol = str(normalized.get("symbol") or "")
        symbol_frame = (frames_by_symbol or {}).get(symbol)
        date_index = date_to_index.get(timestamp) if timestamp is not None else None
        if symbol_frame is not None and date_index is not None and date_index + 2 < len(common_dates):
            execution_date = pd.Timestamp(common_dates[date_index + 1])
            next_execution_date = pd.Timestamp(common_dates[date_index + 2])
            def market_value(date: pd.Timestamp, column: str) -> float | None:
                if column not in symbol_frame.columns or date not in symbol_frame.index:
                    return None
                return _finite(symbol_frame.at[date, column])
            execution_open = market_value(execution_date, "open")
            next_open = market_value(next_execution_date, "open")
            normalized.update({
                "decision_close": market_value(timestamp, "close"),
                "execution_date": execution_date,
                "next_execution_date": next_execution_date,
                "execution_open": execution_open,
                "execution_high": market_value(execution_date, "high"),
                "execution_low": market_value(execution_date, "low"),
                "execution_close": market_value(execution_date, "close"),
                "execution_volume": market_value(execution_date, "volume"),
                "next_open": next_open,
                "open_to_open_return": (next_open / execution_open - 1.0) if execution_open not in {None, 0.0} and next_open is not None else None,
            })
        rows.append(normalized)
    return rows


def _latest_multi_horizon_frame(
    latest_forecasts: list[dict[str, Any]],
    horizons: list[int],
    *,
    one_side_cost: float,
) -> pd.DataFrame:
    frames_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        rows = [item for item in latest_forecasts if int(item.get("horizon") or -1) == int(horizon)]
        if not rows:
            continue
        canonical_rows: list[dict[str, Any]] = []
        for item in rows:
            canonical_rows.append({
                "timestamp": pd.Timestamp(item.get("as_of")),
                "symbol": str(item.get("symbol")),
                "fold_id": 0,
                "profit_before_loss_probability": item.get("probability_profit_before_loss"),
                "baseline_profit_before_loss_probability": item.get("baseline_profit_before_loss_probability"),
                "profit_percentile": item.get("profit_percentile"),
                "profit_before_loss_quality_weight": item.get("profit_quality_weight"),
                "predicted_drawdown": item.get("expected_max_drawdown"),
                "risk_safety_percentile": item.get("risk_safety_percentile"),
                "drawdown_quality_weight": item.get("drawdown_quality_weight"),
                "bottom_probability": item.get("probability_bottom"),
                "baseline_bottom_probability": item.get("baseline_bottom_probability"),
                "bottom_quality_weight": item.get("bottom_quality_weight"),
                "top_probability": item.get("probability_top"),
                "baseline_top_probability": item.get("baseline_top_probability"),
                "top_quality_weight": item.get("top_quality_weight"),
                "trend_direction": 1.0 if item.get("trend_state") == "up" else -1.0 if item.get("trend_state") == "down" else 0.0,
                "trend_persistence_probability": item.get("probability_trend_persistence"),
                "baseline_trend_persistence_probability": item.get("baseline_trend_persistence_probability"),
                "trend_persistence_quality_weight": item.get("trend_quality_weight"),
            })
        frames_by_horizon[int(horizon)] = pd.DataFrame(canonical_rows)
    return _multi_horizon_frame(frames_by_horizon, horizons, one_side_cost=one_side_cost)


def _multi_horizon_latest_rows(frame: pd.DataFrame, horizons: list[int]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ordered = frame.sort_values(["asset_rank_score", "profit_before_loss_probability"], ascending=[False, False]).reset_index(drop=True)
    has_target = bool(len(ordered) and float(ordered.iloc[0]["entry_score"]) > float(ordered.iloc[0]["entry_threshold"]))
    rows: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        item: dict[str, Any] = {
            "symbol": str(row["symbol"]),
            "as_of": pd.Timestamp(row["timestamp"]),
            "asset_rank_score": _finite(row.get("asset_rank_score")),
            "risk_adjusted_asset_rank_score": _finite(row.get("risk_adjusted_asset_rank_score")),
            "entry_rank_percentile": _finite(row.get("entry_rank_percentile")),
            "opportunity_gate_score": _finite(row.get("opportunity_gate_score")),
            "risk_adjusted_entry_score": _finite(row.get("risk_adjusted_entry_score")),
            "entry_risk_multiplier": _finite(row.get("entry_risk_multiplier")),
            "risk_entry_threshold_penalty": _finite(row.get("risk_entry_threshold_penalty")),
            "entry_threshold": _finite(row.get("entry_threshold")),
            "reentry_margin": _finite(row.get("reentry_margin")),
            "hold_score": _finite(row.get("hold_score")),
            "incumbent_persistence_raw": _finite(row.get("incumbent_persistence_raw")),
            "incumbent_persistence_score": _finite(row.get("incumbent_persistence_score")),
            "incumbent_risk_health": _finite(row.get("incumbent_risk_health")),
            "exit_threshold": _finite(row.get("exit_threshold")),
            "short_profit_consensus": _finite(row.get("short_profit_consensus")),
            "short_risk_safety": _finite(row.get("short_risk_safety")),
            "short_bottom_support": _finite(row.get("short_bottom_support")),
            "short_horizon_agreement": _finite(row.get("short_horizon_agreement")),
            "long_profit_confirmation": _finite(row.get("long_profit_confirmation")),
            "long_risk_safety": _finite(row.get("long_risk_safety")),
            "long_trend_support": _finite(row.get("long_trend_support")),
            "horizon_agreement": _finite(row.get("horizon_agreement")),
            "all_horizon_risk_safety": _finite(row.get("all_horizon_risk_safety")),
            "expected_max_drawdown": _finite(row.get("predicted_drawdown")),
            "shadow_target": bool(has_target and index == 0),
            "shadow_state": "ENTRY" if has_target and index == 0 else "CASH",
        }
        for horizon in horizons:
            item[f"profit_probability_{horizon}d"] = _finite(row.get(f"profit_before_loss_probability_h{horizon}"))
            item[f"profit_percentile_{horizon}d"] = _finite(row.get(f"profit_percentile_h{horizon}"))
            item[f"risk_safety_{horizon}d"] = _finite(row.get(f"risk_safety_percentile_h{horizon}"))
        rows.append(item)
    return rows



_TEMPORAL_COST_STRESS_BPS = (0.0, 1.0, 2.0, 5.0, 10.0)


def _cost_stress_metrics(
    gross_returns: list[float],
    cost_sides: list[int],
    fold_ids: list[int],
    fold_close_cost: list[bool],
    initial_capital: float,
) -> list[dict[str, Any]]:
    if not gross_returns:
        return []
    gross = np.asarray(gross_returns, dtype=float)
    sides = np.asarray(cost_sides, dtype=int)
    close_flags = np.asarray(fold_close_cost, dtype=bool)
    rows: list[dict[str, Any]] = []
    for cost_bps in _TEMPORAL_COST_STRESS_BPS:
        one_side_cost = float(cost_bps) / 10_000.0
        factors = np.maximum(1e-9, 1.0 - sides.astype(float) * one_side_cost) * np.maximum(1e-9, 1.0 + gross)
        if one_side_cost > 0.0 and close_flags.any():
            factors = factors.copy()
            factors[close_flags] *= max(1e-9, 1.0 - one_side_cost)
        returns = factors - 1.0
        equity = float(initial_capital) * np.cumprod(factors)
        running_peak = np.maximum.accumulate(equity)
        drawdown = equity / running_peak - 1.0
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
        rows.append({
            "one_side_cost_bps": float(cost_bps),
            "ending_capital": _finite(float(equity[-1])),
            "total_return": _finite(float(equity[-1]) / float(initial_capital) - 1.0),
            "sharpe": _finite(sharpe),
            "max_drawdown": _finite(float(np.min(drawdown))),
            "switch_cost_events": int(np.count_nonzero(sides)),
            "fold_close_cost_events": int(np.count_nonzero(close_flags)),
        })
    return rows


_WINNER_TIMING_BASE_WEAK_THRESHOLD = 0.50
_WINNER_TIMING_CHALLENGER_MINIMUM = 0.60
_WINNER_TIMING_MINIMUM_ADVANTAGE = 0.25
_WINNER_TIMING_MAXIMUM_ADVANTAGE = 0.65


def _reference_asset(value: Any) -> str:
    symbol = str(value or "CASH").strip().upper()
    return symbol or "CASH"


def _reference_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _reference_cost_sides(previous: str | None, target: str | None) -> int:
    previous_asset = _reference_asset(previous)
    target_asset = _reference_asset(target)
    if previous_asset == target_asset:
        return 0
    if previous_asset == "CASH" or target_asset == "CASH":
        return 1
    return 2


def _reference_counts_cash_transitions_as_rotations(reference_analytics: dict[str, Any]) -> bool:
    processing_kind = str(reference_analytics.get("processing_kind") or "").strip().lower()
    return processing_kind in {"strategy_research_temporal", "strategy_research_stateful", "strategy_research_decision_optimization"}


def _reference_rotation_increment(
    previous: str | None,
    target: str | None,
    *,
    count_cash_transitions: bool,
) -> int:
    previous_asset = _reference_asset(previous)
    target_asset = _reference_asset(target)
    if previous_asset == target_asset:
        return 0
    if not count_cash_transitions and (previous_asset == "CASH" or target_asset == "CASH"):
        return 0
    return 1


def _strategy_research_reference_study(
    frame: pd.DataFrame,
    winner_daily_rows: list[dict[str, Any]],
    reference_analytics: dict[str, Any],
    open_prices: pd.DataFrame,
    config: Any,
    *,
    include_diagnostics: bool = False,
    include_economic_curve: bool = False,
    enable_timing_override: bool = True,
) -> dict[str, Any]:
    if frame.empty or not winner_daily_rows or not isinstance(reference_analytics, dict):
        return {}
    equity_rows = [
        dict(row) for row in (reference_analytics.get("equity") or [])
        if isinstance(row, dict) and _reference_timestamp(row.get("timestamp")) is not None
        and _finite(row.get("simulation_equity")) is not None
    ]
    equity_rows.sort(key=lambda row: _reference_timestamp(row.get("timestamp")))
    if len(equity_rows) < 2:
        return {}

    winner_by_execution: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in winner_daily_rows:
        if not isinstance(row, dict):
            continue
        stamp = _reference_timestamp(row.get("timestamp"))
        if stamp is not None:
            winner_by_execution[stamp] = row

    frame_by_timestamp: dict[pd.Timestamp, pd.DataFrame] = {}
    for raw_timestamp, rows in frame.groupby("timestamp", sort=False):
        stamp = _reference_timestamp(raw_timestamp)
        if stamp is not None:
            frame_by_timestamp[stamp] = rows.copy()

    aligned: list[tuple[dict[str, Any], dict[str, Any], pd.DataFrame]] = []
    for session in equity_rows:
        execution_stamp = _reference_timestamp(session.get("timestamp"))
        winner_row = winner_by_execution.get(execution_stamp) if execution_stamp is not None else None
        decision_stamp = _reference_timestamp((winner_row or {}).get("decision_date"))
        rows = frame_by_timestamp.get(decision_stamp) if decision_stamp is not None else None
        if winner_row is not None:
            aligned.append((session, winner_row, rows.copy() if rows is not None else pd.DataFrame()))
    if len(aligned) < 2:
        return {}

    first_equity = _finite(aligned[0][0].get("simulation_equity"))
    source_metrics = reference_analytics.get("metrics") if isinstance(reference_analytics.get("metrics"), dict) else {}
    initial_capital = _finite(source_metrics.get("initial_capital")) or float(config.initial_capital)
    if first_equity is None or initial_capital <= 0:
        return {}

    rotations = [row for row in (reference_analytics.get("rotations") or []) if isinstance(row, dict)]
    count_cash_transitions_as_rotations = _reference_counts_cash_transitions_as_rotations(reference_analytics)
    first_execution = _reference_timestamp(aligned[0][0].get("timestamp"))
    first_rotation = next(
        (row for row in rotations if _reference_timestamp(row.get("executed_at")) == first_execution),
        None,
    )
    current_symbol = _reference_asset(
        aligned[0][1].get("strategy_research_control_previous_asset")
        or aligned[0][1].get("previous_asset")
        or (first_rotation or {}).get("from_asset")
    )
    candidate_capital = float(first_equity)
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))

    equity_curve: list[float] = []
    gross_return_history: list[float] = []
    cost_sides_history: list[int] = []
    fold_id_history: list[int] = []
    fold_close_cost_history: list[bool] = []
    economic_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state_history: list[tuple[int, str]] = []
    action_counts = {"buy": 0, "hold": 0, "sell": 0, "rotate": 0, "cash": 0}
    exposure_days = 0
    cash_days = 0
    switch_count = 0
    timing_override_count = 0
    anchor_days = 0
    anchor_top1_days = 0

    for index, (session, winner_row, rows) in enumerate(aligned):
        execution_stamp = _reference_timestamp(session.get("timestamp"))
        decision_stamp = _reference_timestamp(winner_row.get("decision_date"))
        fold_id = int(winner_row.get("walk_forward_fold") or winner_row.get("decision_fold_id") or winner_row.get("fold_id") or 0)
        rows_by_symbol = rows.set_index("symbol", drop=False) if not rows.empty and "symbol" in rows.columns else pd.DataFrame()
        control_previous = _reference_asset(
            winner_row.get("strategy_research_control_previous_asset")
            or winner_row.get("previous_asset")
            or (
                aligned[index - 1][1].get("strategy_research_control_asset")
                if index > 0 else (first_rotation or {}).get("from_asset")
            )
        )
        base_asset = _reference_asset(
            winner_row.get("strategy_research_control_asset")
            or winner_row.get("selected_asset")
            or session.get("selected_asset")
        )
        base_symbol = None if base_asset == "CASH" else base_asset
        top1_value = winner_row.get("top_1_asset") or winner_row.get("raw_best_asset") or winner_row.get("best_asset")
        top2_value = winner_row.get("top_2_asset") or winner_row.get("second_asset")
        top1_asset = _reference_asset(top1_value)
        challenger_asset = _reference_asset(top2_value)
        top1_symbol = None if top1_asset == "CASH" else top1_asset
        challenger_symbol = None if challenger_asset == "CASH" else challenger_asset
        anchor_days += 1
        if base_symbol is not None and base_symbol == top1_symbol:
            anchor_top1_days += 1

        base_short = None
        challenger_short = None
        temporal_advantage = None
        timing_candidate = False
        override = False
        if base_symbol is not None and base_symbol in rows_by_symbol.index:
            base_short = _finite(rows_by_symbol.loc[base_symbol].get("short_profit_consensus"))
        if challenger_symbol is not None and challenger_symbol in rows_by_symbol.index:
            challenger_short = _finite(rows_by_symbol.loc[challenger_symbol].get("short_profit_consensus"))
        if base_short is not None and challenger_short is not None:
            temporal_advantage = float(challenger_short) - float(base_short)
            timing_candidate = bool(
                base_symbol == top1_symbol
                and challenger_symbol != base_symbol
                and float(base_short) < _WINNER_TIMING_BASE_WEAK_THRESHOLD
                and float(challenger_short) >= _WINNER_TIMING_CHALLENGER_MINIMUM
                and temporal_advantage >= _WINNER_TIMING_MINIMUM_ADVANTAGE
                and temporal_advantage <= _WINNER_TIMING_MAXIMUM_ADVANTAGE
            )
            override = bool(enable_timing_override and timing_candidate)

        target_symbol = challenger_symbol if override else base_symbol
        reason = "strategy_research_reference"
        if override:
            timing_override_count += 1
            reason = "temporal_short_timing_overrides_strategy_research"

        def interval_return_for(symbol: str | None) -> float | None:
            if index >= len(aligned) - 1:
                return None
            if symbol is None:
                return 0.0
            if symbol not in open_prices.columns:
                return None
            current_execution = execution_stamp
            next_execution = _reference_timestamp(aligned[index + 1][0].get("timestamp"))
            if current_execution is None or next_execution is None:
                return None
            entry = _finite(open_prices.at[current_execution, symbol]) if current_execution in open_prices.index else None
            nxt = _finite(open_prices.at[next_execution, symbol]) if next_execution in open_prices.index else None
            if entry is None or nxt is None or entry <= 0:
                return None
            return float(nxt / entry - 1.0)

        if target_symbol is not None and interval_return_for(target_symbol) is None and index < len(aligned) - 1:
            if override and base_symbol is not None and interval_return_for(base_symbol) is not None:
                target_symbol = base_symbol
                override = False
                timing_override_count = max(0, timing_override_count - 1)
                reason = "strategy_research_reference_after_temporal_candidate_open_unavailable"
            else:
                target_symbol = base_symbol
                reason = "strategy_research_reference_open_unavailable"

        if current_symbol == "CASH" and target_symbol is not None:
            action = "buy"
        elif current_symbol != "CASH" and target_symbol is None:
            action = "sell"
        elif current_symbol != "CASH" and target_symbol is not None and current_symbol != target_symbol:
            action = "rotate"
        elif current_symbol == "CASH" and target_symbol is None:
            action = "cash"
        else:
            action = "hold"
        action_counts[action] += 1
        sides = _reference_cost_sides(current_symbol, target_symbol)
        switch_count += _reference_rotation_increment(
            current_symbol,
            target_symbol,
            count_cash_transitions=count_cash_transitions_as_rotations,
        )

        target_asset = _reference_asset(target_symbol)
        if target_asset == "CASH":
            cash_days += 1
        else:
            exposure_days += 1
        state_history.append((fold_id, target_asset))

        equity_curve.append(float(candidate_capital))
        if include_economic_curve:
            economic_rows.append({
                "fold_id": fold_id,
                "decision_timestamp": decision_stamp,
                "execution_date": execution_stamp,
                "current_symbol": current_symbol,
                "target_symbol": target_asset,
                "action": action.upper(),
                "reason": reason,
                "winner_anchor_symbol": base_asset,
                "winner_top2_symbol": challenger_asset,
                "temporal_timing_candidate": bool(timing_candidate),
                "temporal_timing_override": bool(override),
                "strategy_equity": _finite(candidate_capital),
            })

        if include_diagnostics:
            base_row = rows_by_symbol.loc[base_symbol] if base_symbol is not None and base_symbol in rows_by_symbol.index else None
            challenger_row = rows_by_symbol.loc[challenger_symbol] if challenger_symbol is not None and challenger_symbol in rows_by_symbol.index else None
            diagnostics.append({
                "fold_id": fold_id,
                "timestamp": decision_stamp,
                "current_symbol": current_symbol,
                "best_symbol": top1_asset,
                "target_symbol": target_asset,
                "action": action.upper(),
                "reason": reason,
                "winner_anchor_symbol": base_asset,
                "winner_top1_symbol": top1_asset,
                "winner_top2_symbol": challenger_asset,
                "winner_anchor_score": _finite(winner_row.get("decision_score")),
                "temporal_timing_candidate": bool(timing_candidate),
                "temporal_timing_override": bool(override),
                "winner_anchor_short_profit_consensus": base_short,
                "winner_top2_short_profit_consensus": challenger_short,
                "temporal_short_profit_advantage": _finite(temporal_advantage),
                "winner_anchor_risk_safety": _finite(base_row.get("all_horizon_risk_safety")) if base_row is not None else None,
                "winner_top2_risk_safety": _finite(challenger_row.get("all_horizon_risk_safety")) if challenger_row is not None else None,
                "timing_base_weak_threshold": _WINNER_TIMING_BASE_WEAK_THRESHOLD,
                "timing_challenger_minimum": _WINNER_TIMING_CHALLENGER_MINIMUM,
                "timing_minimum_advantage": _WINNER_TIMING_MINIMUM_ADVANTAGE,
                "timing_maximum_advantage": _WINNER_TIMING_MAXIMUM_ADVANTAGE,
            })

        if index < len(aligned) - 1:
            current_reference = _finite(session.get("simulation_equity"))
            next_reference = _finite(aligned[index + 1][0].get("simulation_equity"))
            if current_reference in {None, 0.0} or next_reference is None:
                return {}
            baseline_factor = float(next_reference / current_reference)
            control_return = interval_return_for(base_symbol)
            candidate_return = interval_return_for(target_symbol)
            control_sides = _reference_cost_sides(control_previous, base_symbol)
            if target_asset == base_asset and current_symbol == control_previous:
                candidate_factor = baseline_factor
                gross_return = baseline_factor / max(1e-9, 1.0 - sides * one_side_cost) - 1.0
            else:
                if candidate_return is None:
                    target_symbol = base_symbol
                    target_asset = base_asset
                    candidate_return = control_return
                    sides = _reference_cost_sides(current_symbol, target_symbol)
                    override = False
                if control_return is not None:
                    expected_control = max(1e-9, 1.0 - control_sides * one_side_cost) * max(1e-9, 1.0 + float(control_return))
                    residual = baseline_factor / expected_control if expected_control > 0 else 1.0
                else:
                    residual = baseline_factor / max(1e-9, 1.0 - control_sides * one_side_cost)
                candidate_factor = residual * max(1e-9, 1.0 - sides * one_side_cost) * max(1e-9, 1.0 + float(candidate_return or 0.0))
                gross_return = residual * max(1e-9, 1.0 + float(candidate_return or 0.0)) - 1.0
            gross_return_history.append(float(gross_return))
            cost_sides_history.append(int(sides))
            fold_id_history.append(fold_id)
            fold_close_cost_history.append(False)
            candidate_capital *= max(1e-9, float(candidate_factor))
            if include_economic_curve and economic_rows:
                economic_rows[-1]["reference_interval_factor"] = _finite(baseline_factor)
                economic_rows[-1]["net_interval_return"] = _finite(candidate_factor - 1.0)
                economic_rows[-1]["gross_interval_return"] = _finite(gross_return)
                economic_rows[-1]["cost_sides"] = int(sides)
                economic_rows[-1]["one_side_cost_rate"] = float(one_side_cost)
        current_symbol = target_asset

    if not equity_curve:
        return {}
    values = np.asarray(equity_curve, dtype=float)
    daily_returns = [float(values[0] / initial_capital - 1.0)]
    daily_returns.extend(float(values[index] / values[index - 1] - 1.0) if values[index - 1] > 0 else 0.0 for index in range(1, len(values)))
    returns = np.asarray(daily_returns, dtype=float)
    ending_capital = float(values[-1])
    years = max(len(values) / 252.0, 1.0 / 252.0)
    cagr = (ending_capital / initial_capital) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    running_peak = np.maximum.accumulate(values)
    drawdown_curve = values / running_peak - 1.0
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
    if include_economic_curve:
        for index, row in enumerate(economic_rows):
            if index >= len(values):
                break
            row["strategy_equity"] = _finite(values[index])
            row["strategy_drawdown"] = _finite(drawdown_curve[index])
            row["cumulative_return"] = _finite(values[index] / initial_capital - 1.0)

    result = {
        "initial_capital": float(initial_capital),
        "ending_capital": ending_capital,
        "total_return": ending_capital / float(initial_capital) - 1.0,
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(float(np.min(drawdown_curve))),
        "exposure": _finite(exposure_days / max(1, len(values))),
        "cash_days": int(cash_days),
        "decision_days": int(len(values)),
        "switch_count": int(switch_count),
        "action_counts": action_counts,
        "one_side_cost_rate": float(one_side_cost),
        "decision_policy": "strategy_research_temporal_timing" if enable_timing_override else "strategy_research_reference_replay",
        "winner_anchor_days": int(anchor_days),
        "winner_anchor_top1_days": int(anchor_top1_days),
        "timing_override_count": int(timing_override_count),
        "timing_base_weak_threshold": _WINNER_TIMING_BASE_WEAK_THRESHOLD,
        "timing_challenger_minimum": _WINNER_TIMING_CHALLENGER_MINIMUM,
        "timing_minimum_advantage": _WINNER_TIMING_MINIMUM_ADVANTAGE,
        "timing_maximum_advantage": _WINNER_TIMING_MAXIMUM_ADVANTAGE,
        "reference_coverage_start": bson_value(_reference_timestamp(aligned[0][0].get("timestamp"))),
        "reference_coverage_end": bson_value(_reference_timestamp(aligned[-1][0].get("timestamp"))),
        "reference_equity_sessions": int(len(aligned)),
        "cost_stress": _cost_stress_metrics(
            gross_return_history,
            cost_sides_history,
            fold_id_history,
            fold_close_cost_history,
            float(first_equity),
        ) if gross_return_history else [],
        **_state_duration_metrics(state_history),
    }
    if include_diagnostics:
        result["decision_diagnostics"] = diagnostics
    if include_economic_curve:
        result["economic_curve"] = economic_rows
    return result


def _winner_anchored_temporal_study(
    frame: pd.DataFrame,
    winner_daily_rows: list[dict[str, Any]],
    open_prices: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    config: Any,
    *,
    include_diagnostics: bool = False,
    include_economic_curve: bool = False,
    enable_timing_override: bool = True,
) -> dict[str, Any]:
    """Use the immutable Winner as allocator and Temporal short-horizon evidence only as a timing overlay."""
    if frame.empty or not winner_daily_rows:
        return {}

    common_index = pd.DatetimeIndex(pd.to_datetime(common_dates, utc=True))
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(common_index)}
    winner_by_decision: dict[pd.Timestamp, dict[str, Any]] = {}
    for item in winner_daily_rows:
        if not isinstance(item, dict) or item.get("decision_date") is None:
            continue
        timestamp = pd.Timestamp(item["decision_date"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        winner_by_decision[timestamp] = item

    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))
    daily_returns: list[float] = []
    gross_return_history: list[float] = []
    cost_sides_history: list[int] = []
    fold_id_history: list[int] = []
    fold_close_cost_history: list[bool] = []
    economic_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state_history: list[tuple[int, str]] = []
    action_counts = {"buy": 0, "hold": 0, "sell": 0, "rotate": 0, "cash": 0}
    exposure_days = 0
    switch_count = 0
    timing_override_count = 0
    anchor_days = 0
    anchor_top1_days = 0

    for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
        current_symbol: str | None = None
        dates = sorted(pd.Timestamp(value) for value in fold_frame["timestamp"].dropna().unique())
        for raw_timestamp in dates:
            timestamp = raw_timestamp.tz_localize("UTC") if raw_timestamp.tzinfo is None else raw_timestamp.tz_convert("UTC")
            date_index = date_to_index.get(timestamp)
            if date_index is None or date_index + 2 >= len(common_index):
                continue
            winner_row = winner_by_decision.get(timestamp)
            if winner_row is None:
                continue

            rows = fold_frame.loc[fold_frame["timestamp"] == raw_timestamp].copy()
            if rows.empty:
                continue
            rows_by_symbol = rows.set_index("symbol", drop=False)
            top1_value = winner_row.get("top_1_asset") or winner_row.get("raw_best_asset") or winner_row.get("best_asset")
            top2_value = winner_row.get("top_2_asset") or winner_row.get("second_asset")
            base_symbol_value = winner_row.get("selected_asset") or winner_row.get("final_action_asset") or top1_value
            base_symbol = str(base_symbol_value) if base_symbol_value not in {None, "", "CASH"} else None
            top1_symbol = str(top1_value) if top1_value not in {None, "", "CASH"} else None
            challenger_symbol = str(top2_value) if top2_value not in {None, "", "CASH"} else None
            anchor_days += 1
            if base_symbol is not None and base_symbol == top1_symbol:
                anchor_top1_days += 1

            base_short = None
            challenger_short = None
            temporal_advantage = None
            timing_candidate = False
            override = False
            if base_symbol is not None and base_symbol in rows_by_symbol.index:
                base_short = _finite(rows_by_symbol.loc[base_symbol].get("short_profit_consensus"))
            if challenger_symbol is not None and challenger_symbol in rows_by_symbol.index:
                challenger_short = _finite(rows_by_symbol.loc[challenger_symbol].get("short_profit_consensus"))
            if base_short is not None and challenger_short is not None:
                temporal_advantage = float(challenger_short) - float(base_short)
                timing_candidate = bool(
                    base_symbol == top1_symbol
                    and challenger_symbol != base_symbol
                    and float(base_short) < _WINNER_TIMING_BASE_WEAK_THRESHOLD
                    and float(challenger_short) >= _WINNER_TIMING_CHALLENGER_MINIMUM
                    and temporal_advantage >= _WINNER_TIMING_MINIMUM_ADVANTAGE
                    and temporal_advantage <= _WINNER_TIMING_MAXIMUM_ADVANTAGE
                )
                override = bool(enable_timing_override and timing_candidate)

            target_symbol = challenger_symbol if override else base_symbol
            reason = "winner_anchor"
            if override:
                timing_override_count += 1
                reason = "temporal_short_timing_overrides_winner_top1_with_top2"

            def has_open_path(symbol: str | None) -> bool:
                if symbol is None or symbol not in open_prices.columns:
                    return symbol is None
                entry = _finite(open_prices.at[common_index[date_index + 1], symbol])
                nxt = _finite(open_prices.at[common_index[date_index + 2], symbol])
                return entry is not None and nxt is not None and entry > 0

            def interval_return_for(symbol: str | None) -> float | None:
                if symbol is None or symbol not in open_prices.columns:
                    return 0.0 if symbol is None else None
                entry = _finite(open_prices.at[common_index[date_index + 1], symbol])
                nxt = _finite(open_prices.at[common_index[date_index + 2], symbol])
                if entry is None or nxt is None or entry <= 0:
                    return None
                return float(nxt / entry - 1.0)

            anchor_interval_return = interval_return_for(base_symbol)
            challenger_interval_return = interval_return_for(challenger_symbol)
            timing_alpha_return = None
            timing_alpha_log_return = None
            if anchor_interval_return is not None and challenger_interval_return is not None:
                timing_alpha_return = float(challenger_interval_return) - float(anchor_interval_return)
                if anchor_interval_return > -1.0 and challenger_interval_return > -1.0:
                    timing_alpha_log_return = float(math.log1p(challenger_interval_return) - math.log1p(anchor_interval_return))

            if target_symbol is not None and not has_open_path(target_symbol):
                if override and base_symbol is not None and has_open_path(base_symbol):
                    target_symbol = base_symbol
                    override = False
                    timing_override_count = max(0, timing_override_count - 1)
                    reason = "winner_anchor_after_temporal_candidate_open_unavailable"
                else:
                    target_symbol = None
                    reason = "winner_anchor_open_unavailable"

            if current_symbol is None and target_symbol is not None:
                action = "buy"
                cost_sides = 1
            elif current_symbol is not None and target_symbol is None:
                action = "sell"
                cost_sides = 1
            elif current_symbol is not None and target_symbol is not None and current_symbol != target_symbol:
                action = "rotate"
                cost_sides = 2
            elif current_symbol is None and target_symbol is None:
                action = "cash"
                cost_sides = 0
            else:
                action = "hold"
                cost_sides = 0

            action_counts[action] += 1
            if action in {"buy", "sell", "rotate"}:
                switch_count += 1

            interval_return = interval_return_for(target_symbol)
            if interval_return is None:
                interval_return = 0.0
            elif target_symbol is not None:
                exposure_days += 1

            factor = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + float(interval_return))
            daily_returns.append(factor - 1.0)
            gross_return_history.append(float(interval_return))
            cost_sides_history.append(int(cost_sides))
            fold_id_history.append(int(fold_id))
            fold_close_cost_history.append(False)
            state_history.append((int(fold_id), target_symbol or "CASH"))

            if include_diagnostics:
                base_row = rows_by_symbol.loc[base_symbol] if base_symbol is not None and base_symbol in rows_by_symbol.index else None
                challenger_row = rows_by_symbol.loc[challenger_symbol] if challenger_symbol is not None and challenger_symbol in rows_by_symbol.index else None
                diagnostics.append({
                    "fold_id": int(fold_id),
                    "timestamp": timestamp,
                    "current_symbol": current_symbol or "CASH",
                    "best_symbol": top1_symbol or "CASH",
                    "target_symbol": target_symbol or "CASH",
                    "action": action.upper(),
                    "reason": reason,
                    "winner_anchor_symbol": base_symbol or "CASH",
                    "winner_top1_symbol": top1_symbol or "CASH",
                    "winner_top2_symbol": challenger_symbol or "CASH",
                    "winner_top1_score": _finite(winner_row.get("top_1_score")) if _finite(winner_row.get("top_1_score")) is not None else _finite(winner_row.get("raw_best_score") or winner_row.get("best_score")),
                    "winner_top2_score": _finite(winner_row.get("top_2_score")) if _finite(winner_row.get("top_2_score")) is not None else _finite(winner_row.get("second_score")),
                    "winner_anchor_score": _finite(winner_row.get("decision_score")),
                    "temporal_timing_candidate": bool(timing_candidate),
                    "temporal_timing_override": bool(override),
                    "winner_anchor_short_profit_consensus": base_short,
                    "winner_top2_short_profit_consensus": challenger_short,
                    "temporal_short_profit_advantage": _finite(temporal_advantage),
                    "winner_anchor_interval_return": _finite(anchor_interval_return),
                    "winner_top2_interval_return": _finite(challenger_interval_return),
                    "temporal_timing_alpha_return": _finite(timing_alpha_return),
                    "temporal_timing_alpha_log_return": _finite(timing_alpha_log_return),
                    "winner_anchor_risk_safety": _finite(base_row.get("all_horizon_risk_safety")) if base_row is not None else None,
                    "winner_top2_risk_safety": _finite(challenger_row.get("all_horizon_risk_safety")) if challenger_row is not None else None,
                    "winner_anchor_predicted_drawdown": _finite(base_row.get("predicted_drawdown")) if base_row is not None else None,
                    "winner_top2_predicted_drawdown": _finite(challenger_row.get("predicted_drawdown")) if challenger_row is not None else None,
                    "timing_base_weak_threshold": _WINNER_TIMING_BASE_WEAK_THRESHOLD,
                    "timing_challenger_minimum": _WINNER_TIMING_CHALLENGER_MINIMUM,
                    "timing_minimum_advantage": _WINNER_TIMING_MINIMUM_ADVANTAGE,
                    "timing_maximum_advantage": _WINNER_TIMING_MAXIMUM_ADVANTAGE,
                })

            if include_economic_curve:
                economic_rows.append({
                    "fold_id": int(fold_id),
                    "decision_timestamp": timestamp,
                    "execution_date": pd.Timestamp(common_index[date_index + 1]),
                    "next_execution_date": pd.Timestamp(common_index[date_index + 2]),
                    "current_symbol": current_symbol or "CASH",
                    "target_symbol": target_symbol or "CASH",
                    "action": action.upper(),
                    "reason": reason,
                    "winner_anchor_symbol": base_symbol or "CASH",
                    "winner_top2_symbol": challenger_symbol or "CASH",
                    "temporal_timing_candidate": bool(timing_candidate),
                    "temporal_timing_override": bool(override),
                    "winner_anchor_interval_return": _finite(anchor_interval_return),
                    "winner_top2_interval_return": _finite(challenger_interval_return),
                    "temporal_timing_alpha_return": _finite(timing_alpha_return),
                    "temporal_timing_alpha_log_return": _finite(timing_alpha_log_return),
                    "gross_interval_return": _finite(interval_return),
                    "cost_sides": int(cost_sides),
                    "one_side_cost_rate": float(one_side_cost),
                    "net_interval_return": _finite(factor - 1.0),
                })
            current_symbol = target_symbol

        if current_symbol is not None and daily_returns:
            daily_returns[-1] = max(1e-9, (1.0 + daily_returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
            fold_close_cost_history[-1] = True

    if not daily_returns:
        return {}
    returns = np.asarray(daily_returns, dtype=float)
    equity = float(config.initial_capital) * np.cumprod(1.0 + returns)
    ending_capital = float(equity[-1])
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    cagr = (ending_capital / float(config.initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    running_peak = np.maximum.accumulate(equity)
    drawdown_curve = equity / running_peak - 1.0
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
    if include_economic_curve:
        for index, row in enumerate(economic_rows):
            if index >= len(equity):
                break
            row["strategy_equity"] = _finite(equity[index])
            row["strategy_drawdown"] = _finite(drawdown_curve[index])
            row["cumulative_return"] = _finite(equity[index] / float(config.initial_capital) - 1.0)

    result = {
        "initial_capital": float(config.initial_capital),
        "ending_capital": ending_capital,
        "total_return": ending_capital / float(config.initial_capital) - 1.0,
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(float(np.min(drawdown_curve))),
        "exposure": _finite(exposure_days / max(1, len(returns))),
        "decision_days": int(len(returns)),
        "switch_count": int(switch_count),
        "action_counts": action_counts,
        "one_side_cost_rate": float(one_side_cost),
        "decision_policy": "winner_anchored_temporal_timing" if enable_timing_override else "winner_anchor_replay",
        "winner_anchor_days": int(anchor_days),
        "winner_anchor_top1_days": int(anchor_top1_days),
        "timing_override_count": int(timing_override_count),
        "timing_base_weak_threshold": _WINNER_TIMING_BASE_WEAK_THRESHOLD,
        "timing_challenger_minimum": _WINNER_TIMING_CHALLENGER_MINIMUM,
        "timing_minimum_advantage": _WINNER_TIMING_MINIMUM_ADVANTAGE,
        "timing_maximum_advantage": _WINNER_TIMING_MAXIMUM_ADVANTAGE,
        "cost_stress": _cost_stress_metrics(
            gross_return_history,
            cost_sides_history,
            fold_id_history,
            fold_close_cost_history,
            float(config.initial_capital),
        ),
        **_state_duration_metrics(state_history),
    }
    if include_diagnostics:
        result["decision_diagnostics"] = diagnostics
    if include_economic_curve:
        result["economic_curve"] = economic_rows
    return result

def _winner_reference_replay(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    from .capital_rotation import run_rotation_models
    from .compound_rotation_backtest import apply_slippage, calculate_reference_fees

    def report(percent: float, stage: str, _completed_runs: int = 0) -> None:
        if progress_callback:
            progress_callback(94.0 + 4.5 * max(0.0, min(100.0, float(percent))) / 100.0, f"Winner reference replay · {stage}")

    results = run_rotation_models(
        bars_by_symbol,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=report,
    )
    if not results:
        raise ValueError("Winner reference replay produced no result.")
    replay = results[0]
    metrics = dict(replay.metrics or {})
    winner_daily_rows: list[dict[str, Any]] = []
    if isinstance(replay.predictions, pd.DataFrame) and not replay.predictions.empty:
        daily_frame = replay.predictions.reset_index()
        winner_daily_rows = [bson_value(dict(item)) for item in daily_frame.to_dict(orient="records")]
    winner_trade_rows: list[dict[str, Any]] = []
    if isinstance(replay.trades, pd.DataFrame) and not replay.trades.empty:
        winner_trade_rows = [bson_value(dict(item)) for item in replay.trades.to_dict(orient="records")]
    return {
        "reference_type": "immutable_winner_replay",
        "same_frozen_market_snapshot": True,
        "model_family": metrics.get("model_family") or str(config.research_model_family),
        "strategy_mode": metrics.get("strategy_mode") or str(config.strategy_mode),
        "initial_capital": _finite(metrics.get("initial_capital")) or float(config.initial_capital),
        "ending_capital": _finite(metrics.get("strategy_ending_capital")),
        "total_return": _finite(metrics.get("strategy_return")),
        "cagr": _finite(metrics.get("strategy_cagr")),
        "sharpe": _finite(metrics.get("strategy_sharpe")),
        "max_drawdown": _finite(metrics.get("strategy_maximum_drawdown")),
        "exposure": _finite(metrics.get("market_exposure")),
        "switch_count": int(metrics.get("capital_rotations") or 0),
        "benchmark_ending_capital": _finite(metrics.get("buy_hold_ending_capital")),
        "benchmark_return": _finite(metrics.get("buy_hold_return")),
        "benchmark_cagr": _finite(metrics.get("buy_hold_cagr")),
        "benchmark_sharpe": _finite(metrics.get("buy_hold_sharpe")),
        "benchmark_max_drawdown": _finite(metrics.get("buy_hold_maximum_drawdown")),
        "oos_start": metrics.get("champion_oos_start") or metrics.get("test_start"),
        "oos_end": metrics.get("test_end"),
        "folds": [dict(item) for item in (metrics.get("walk_forward_folds") or []) if isinstance(item, dict)],
        "replay_count": int(len(results)),
        "_daily_rows": winner_daily_rows,
        "_trade_rows": winner_trade_rows,
    }


def _stateful_strategy_reference_rows(
    winner_daily_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    stateful_reference_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not winner_daily_rows:
        return [], [], 0
    if not isinstance(stateful_reference_bundle, dict):
        return deepcopy(winner_daily_rows), [], 0

    from ..services.temporal_policy_tuning import observations_from_rows
    from ..services.temporal_winner_transition_stateful import (
        _dynamic_transition_features,
        _policy_target,
        _serialized_risk_score,
        _winner_history_by_decision,
    )

    def ts(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def key(value: Any) -> str | None:
        stamp = ts(value)
        return stamp.isoformat() if stamp is not None else None

    def asset(value: Any) -> str:
        symbol = str(value or "CASH").strip().upper()
        return symbol or "CASH"

    observations = observations_from_rows(observation_rows)
    histories = _winner_history_by_decision(winner_daily_rows)
    risk_models = {int(year): dict(payload) for year, payload in (stateful_reference_bundle.get("risk_models") or {}).items() if isinstance(payload, dict)}
    confidence_by_year = {int(year): dict(payload) for year, payload in (stateful_reference_bundle.get("confidence_by_year") or {}).items() if isinstance(payload, dict)}
    settings = dict(stateful_reference_bundle.get("policy_settings") or {})
    mode = str(stateful_reference_bundle.get("mode") or "conservative_one_session")

    required_settings = ("timing_base_weak_threshold", "timing_challenger_minimum", "timing_minimum_advantage")
    if any(name not in settings for name in required_settings):
        raise ValueError("Selected Stateful Strategy Research policy is missing timing settings.")

    ordered = sorted(
        (deepcopy(row) for row in winner_daily_rows if isinstance(row, dict)),
        key=lambda row: (int(row.get("fold_id") or 0), ts(row.get("decision_date")) or pd.Timestamp.min.tz_localize("UTC")),
    )
    output: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    current_by_fold: dict[int, str] = {}
    cooldown_by_fold: dict[int, bool] = {}
    interventions = 0

    for row in ordered:
        decision_key = key(row.get("decision_date"))
        if decision_key is None:
            output.append(row)
            continue
        fold_id = int(row.get("fold_id") or 0)
        payload = observations.get(decision_key) or {}
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        base_value = row.get("selected_asset") or row.get("final_action_asset") or row.get("top_1_asset") or row.get("raw_best_asset") or row.get("best_asset")
        control_target = asset(base_value)
        if fold_id in current_by_fold:
            previous = current_by_fold[fold_id]
        else:
            previous = asset(row.get("previous_asset") or row.get("current_asset") or control_target)
        target = control_target
        intervention = False
        risk_score = None
        risk_threshold = None
        confidence_margin = None
        confidence_threshold = None
        reason = "strategy_research_stateful_control"

        sample = next((item for item in rows_by_symbol.values() if isinstance(item, dict)), None)
        execution_stamp = ts((sample or {}).get("execution_date")) or ts(row.get("decision_date"))
        year = int(execution_stamp.year) if execution_stamp is not None else 0
        confidence = confidence_by_year.get(year) or {"active": False, "margin_threshold": None}
        model_payload = risk_models.get(year)
        cooldown = bool(cooldown_by_fold.get(fold_id, False))

        gate_allowed = bool(
            previous != "CASH"
            and control_target != "CASH"
            and previous != control_target
            and bool(confidence.get("active"))
            and rows_by_symbol
            and rows_by_symbol.get(previous) is not None
            and rows_by_symbol.get(control_target) is not None
            and _finite((rows_by_symbol.get(previous) or {}).get("open_to_open_return")) is not None
            and _finite((rows_by_symbol.get(control_target) or {}).get("open_to_open_return")) is not None
        )
        if gate_allowed:
            proposed = _policy_target(rows_by_symbol, row, settings)
            gate_allowed = bool(
                not proposed.get("timing_override")
                and control_target == asset(proposed.get("base_symbol"))
                and control_target == asset(proposed.get("top1_symbol"))
            )
        if mode == "conservative_one_session" and cooldown:
            gate_allowed = False

        if gate_allowed and model_payload:
            feature_row = _dynamic_transition_features(
                history_rows=histories.get(decision_key) or [row],
                observations=observations,
                target_symbol=control_target,
                incumbent_symbol=previous,
            )
            risk_score = _serialized_risk_score(model_payload, feature_row)
            risk_threshold = _finite(model_payload.get("risk_threshold"))
            confidence_threshold = _finite(confidence.get("margin_threshold"))
            if risk_score is not None and risk_threshold is not None:
                confidence_margin = float(risk_score - risk_threshold)
                intervention = bool(confidence_threshold is not None and confidence_margin >= confidence_threshold)

        if intervention:
            target = previous
            interventions += 1
            reason = "strategy_research_stateful_confidence_defer"
            if mode == "conservative_one_session":
                cooldown_by_fold[fold_id] = True
        elif mode == "conservative_one_session" and cooldown:
            cooldown_by_fold[fold_id] = False

        row["strategy_research_control_asset"] = control_target
        row["strategy_research_control_previous_asset"] = asset(row.get("previous_asset") or row.get("current_asset") or previous)
        row["previous_asset"] = previous
        if "current_asset" in row:
            row["current_asset"] = previous
        row["selected_asset"] = target
        row["final_action_asset"] = target
        row["stateful_intervention"] = bool(intervention)
        row["stateful_reason"] = reason
        row["stateful_risk_score"] = _finite(risk_score)
        row["stateful_risk_threshold"] = _finite(risk_threshold)
        row["stateful_confidence_margin"] = _finite(confidence_margin)
        row["stateful_confidence_threshold"] = _finite(confidence_threshold)
        output.append(row)

        if target != previous:
            transitions.append({
                "decision_date": row.get("decision_date"),
                "executed_at": row.get("timestamp") or (sample or {}).get("execution_date"),
                "fold_id": fold_id,
                "from_asset": previous,
                "to_asset": target,
                "reason": reason,
                "stateful_intervention": bool(intervention),
                "risk_score": _finite(risk_score),
                "risk_threshold": _finite(risk_threshold),
                "confidence_margin": _finite(confidence_margin),
                "confidence_threshold": _finite(confidence_threshold),
            })
        current_by_fold[fold_id] = target

    return output, transitions, interventions


def _bind_strategy_research_reference_analytics(
    *,
    winner_reference: dict[str, Any],
    winner_daily_rows: list[dict[str, Any]],
    reference_analytics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = reference_analytics.get("metrics") if isinstance(reference_analytics.get("metrics"), dict) else {}
    equity_rows = [
        dict(row) for row in (reference_analytics.get("equity") or [])
        if isinstance(row, dict) and _reference_timestamp(row.get("timestamp")) is not None
        and _finite(row.get("simulation_equity")) is not None
    ]
    equity_rows.sort(key=lambda row: _reference_timestamp(row.get("timestamp")))
    if len(equity_rows) < 2:
        raise ValueError("Selected Strategy Research reference replay does not contain enough equity sessions.")

    rotations = [
        dict(row) for row in (reference_analytics.get("rotations") or [])
        if isinstance(row, dict) and _reference_timestamp(row.get("executed_at")) is not None
    ]
    rotations.sort(key=lambda row: _reference_timestamp(row.get("executed_at")))
    winner_by_execution = {
        _reference_timestamp(row.get("timestamp")): dict(row)
        for row in winner_daily_rows
        if isinstance(row, dict) and _reference_timestamp(row.get("timestamp")) is not None
    }

    first_stamp = _reference_timestamp(equity_rows[0].get("timestamp"))
    first_rotation = next(
        (row for row in rotations if _reference_timestamp(row.get("executed_at")) == first_stamp),
        None,
    )
    current_asset = _reference_asset((first_rotation or {}).get("from_asset"))
    rotation_index = 0
    synthetic_context_sessions = 0
    bound_rows: list[dict[str, Any]] = []
    last_fold_id = 0

    for reference in equity_rows:
        execution_stamp = _reference_timestamp(reference.get("timestamp"))
        if execution_stamp is None:
            continue
        previous_asset = current_asset
        while rotation_index < len(rotations):
            rotation_stamp = _reference_timestamp(rotations[rotation_index].get("executed_at"))
            if rotation_stamp is None or rotation_stamp > execution_stamp:
                break
            current_asset = _reference_asset(rotations[rotation_index].get("to_asset"))
            rotation_index += 1

        source = deepcopy(winner_by_execution.get(execution_stamp) or {})
        if not source:
            synthetic_context_sessions += 1
            source = {
                "timestamp": execution_stamp,
                "decision_date": None,
                "walk_forward_fold": last_fold_id,
                "decision_fold_id": last_fold_id,
            }
        else:
            last_fold_id = int(
                source.get("walk_forward_fold")
                or source.get("decision_fold_id")
                or source.get("fold_id")
                or last_fold_id
                or 0
            )

        source["strategy_research_control_asset"] = current_asset
        source["strategy_research_control_previous_asset"] = previous_asset
        source["previous_asset"] = previous_asset
        if "current_asset" in source:
            source["current_asset"] = previous_asset
        source["selected_asset"] = current_asset
        source["final_action_asset"] = current_asset
        source["strategy_equity"] = _finite(reference.get("simulation_equity"))
        source["strategy_research_reference_source"] = "exact_processing_analytics"
        bound_rows.append(source)

    if len(bound_rows) != len(equity_rows):
        raise ValueError(
            "Selected Strategy Research reference replay could not bind every reference equity session "
            f"({len(bound_rows)}/{len(equity_rows)} sessions bound)."
        )
    if synthetic_context_sessions:
        raise ValueError(
            "Strategy Research market context is incomplete: "
            f"{synthetic_context_sessions} of {len(equity_rows)} reference sessions have no complete market context. "
            "This usually indicates discontinuous history in one or more Strategy assets; the research run was stopped before downstream analysis."
        )

    fold_equity: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    reference_by_execution = {
        _reference_timestamp(row.get("timestamp")): row for row in equity_rows
        if _reference_timestamp(row.get("timestamp")) is not None
    }
    for row in bound_rows:
        stamp = _reference_timestamp(row.get("timestamp"))
        reference = reference_by_execution.get(stamp)
        if reference is None:
            continue
        fold_id = int(row.get("walk_forward_fold") or row.get("decision_fold_id") or row.get("fold_id") or 0)
        fold_equity.setdefault(fold_id, []).append((row, reference))
    exact_folds: list[dict[str, Any]] = []
    count_cash_transitions_as_rotations = _reference_counts_cash_transitions_as_rotations(reference_analytics)
    for fold_id, items in sorted(fold_equity.items()):
        values = np.asarray([float(item[1].get("simulation_equity") or 0.0) for item in items], dtype=float)
        if len(values) < 2 or values[0] <= 0:
            continue
        returns = np.asarray([float(values[index] / values[index - 1] - 1.0) for index in range(1, len(values))], dtype=float)
        peaks = np.maximum.accumulate(values)
        drawdown = values / peaks - 1.0
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
        years = max((len(values) - 1) / 252.0, 1.0 / 252.0)
        fold_return = float(values[-1] / values[0] - 1.0)
        fold_cagr = float((values[-1] / values[0]) ** (1.0 / years) - 1.0) if values[-1] > 0 else -1.0
        assets = [_reference_asset(item[0].get("selected_asset")) for item in items]
        cash_days = sum(asset == "CASH" for asset in assets)
        switches = sum(
            _reference_rotation_increment(
                assets[index - 1],
                assets[index],
                count_cash_transitions=count_cash_transitions_as_rotations,
            )
            for index in range(1, len(assets))
        )
        exact_folds.append({
            "fold_id": int(fold_id),
            "test_start": items[0][0].get("fold_test_start") or items[0][1].get("timestamp"),
            "test_end": items[-1][0].get("fold_test_end") or items[-1][1].get("timestamp"),
            "strategy_return": fold_return,
            "strategy_ending_capital": _finite(values[-1]),
            "strategy_cagr": _finite(fold_cagr),
            "strategy_sharpe": _finite(sharpe),
            "strategy_maximum_drawdown": _finite(float(np.min(drawdown))),
            "market_exposure": _finite((len(assets) - cash_days) / max(1, len(assets))),
            "capital_rotations": int(switches),
        })

    updated = deepcopy(winner_reference)
    initial_capital = _finite(metrics.get("initial_capital"))
    if initial_capital is None:
        initial_capital = _finite(metrics.get("starting_capital"))
    ending_capital = _finite(metrics.get("ending_capital"))
    strategy_return = _finite(metrics.get("strategy_return"))
    if strategy_return is None:
        strategy_return = _finite(metrics.get("simulation_return"))
    switch_count_value = metrics.get("capital_rotations")
    if switch_count_value is None:
        switch_count_value = metrics.get("position_changes")
    switch_count = int(switch_count_value if switch_count_value is not None else len(rotations))
    coverage_start = _reference_timestamp(equity_rows[0].get("timestamp"))
    coverage_end = _reference_timestamp(equity_rows[-1].get("timestamp"))
    updated.update({
        "reference_type": "selected_strategy_research_exact_processing_replay",
        "same_frozen_market_snapshot": True,
        "initial_capital": initial_capital or updated.get("initial_capital"),
        "ending_capital": ending_capital,
        "total_return": strategy_return,
        "cagr": _finite(metrics.get("cagr")),
        "sharpe": _finite(metrics.get("sharpe")),
        "max_drawdown": _finite(metrics.get("maximum_drawdown")),
        "exposure": _finite(metrics.get("market_exposure")),
        "cash_days": int(metrics.get("cash_days") or 0),
        "switch_count": switch_count,
        "folds": exact_folds,
        "oos_start": bson_value(coverage_start),
        "oos_end": bson_value(coverage_end),
        "reference_equity_sessions": int(len(equity_rows)),
        "reference_context_sessions": int(len(winner_by_execution)),
        "reference_uncovered_market_sessions": int(synthetic_context_sessions),
        "reference_coverage_start": bson_value(coverage_start),
        "reference_coverage_end": bson_value(coverage_end),
        "reference_processing_id": reference_analytics.get("processing_id") or reference_analytics.get("job_id"),
        "source_stateful_replay_id": reference_analytics.get("source_stateful_replay_id"),
        "strategy_profile_id": reference_analytics.get("strategy_profile_id"),
        "strategy_profile_revision": reference_analytics.get("strategy_profile_revision"),
        "strategy_configuration_hash": reference_analytics.get("strategy_configuration_hash"),
    })
    if ending_capital is None:
        raise ValueError("Selected Strategy Research reference replay does not contain ending capital.")
    return updated, bound_rows, rotations


def _strategy_research_reference_parity(
    reference: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    reference_capital = _finite(reference.get("ending_capital"))
    replay_capital = _finite(replay.get("ending_capital"))
    reference_exposure = _finite(reference.get("exposure"))
    replay_exposure = _finite(replay.get("exposure"))
    reference_cash = int(reference.get("cash_days") or 0)
    replay_cash = int(replay.get("cash_days") or 0)
    reference_switches = int(reference.get("switch_count") or 0)
    replay_switches = int(replay.get("switch_count") or 0)
    reference_sessions = int(reference.get("reference_equity_sessions") or 0)
    replay_sessions = int(replay.get("reference_equity_sessions") or replay.get("decision_days") or 0)
    capital_delta = (
        float(replay_capital / reference_capital - 1.0)
        if replay_capital is not None and reference_capital not in {None, 0.0}
        else None
    )
    exposure_delta = (
        float(replay_exposure - reference_exposure)
        if replay_exposure is not None and reference_exposure is not None
        else None
    )
    checks = {
        "ending_capital": capital_delta is not None and abs(capital_delta) <= 1e-10,
        "cash_days": replay_cash == reference_cash,
        "market_exposure": exposure_delta is not None and abs(exposure_delta) <= 1e-12,
        "switches": replay_switches == reference_switches,
        "equity_sessions": replay_sessions == reference_sessions,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "ending_capital_delta_rate": _finite(capital_delta),
        "market_exposure_delta": _finite(exposure_delta),
        "reference": {
            "ending_capital": reference_capital,
            "cash_days": reference_cash,
            "market_exposure": reference_exposure,
            "switches": reference_switches,
            "equity_sessions": reference_sessions,
        },
        "replay": {
            "ending_capital": replay_capital,
            "cash_days": replay_cash,
            "market_exposure": replay_exposure,
            "switches": replay_switches,
            "equity_sessions": replay_sessions,
        },
    }


def _replace_reference_with_stateful_strategy(
    *,
    winner_reference: dict[str, Any],
    winner_daily_rows: list[dict[str, Any]],
    multi_horizon_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    open_prices: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    config: BacktestExecutionRequest,
    stateful_reference_bundle: dict[str, Any],
    observation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stateful_rows, stateful_trades, intervention_count = _stateful_strategy_reference_rows(
        winner_daily_rows, observation_rows, stateful_reference_bundle
    )
    if not stateful_rows or multi_horizon_frame.empty:
        raise ValueError("Selected Stateful Strategy Research could not be replayed on the frozen Temporal snapshot.")
    replay = _winner_anchored_temporal_study(
        multi_horizon_frame, stateful_rows, open_prices, common_dates, config, enable_timing_override=False
    )
    if not replay:
        raise ValueError("Selected Stateful Strategy Research replay produced no reference metrics.")

    updated = deepcopy(winner_reference)
    updated.update({
        "reference_type": "selected_strategy_research_stateful_replay",
        "ending_capital": _finite(replay.get("ending_capital")),
        "total_return": _finite(replay.get("total_return")),
        "cagr": _finite(replay.get("cagr")),
        "sharpe": _finite(replay.get("sharpe")),
        "max_drawdown": _finite(replay.get("max_drawdown")),
        "exposure": _finite(replay.get("exposure")),
        "switch_count": int(replay.get("switch_count") or 0),
        "stateful_intervention_count": int(intervention_count),
        "stateful_policy_mode": stateful_reference_bundle.get("mode"),
        "stateful_source_replay_id": stateful_reference_bundle.get("source_replay_id"),
        "stateful_source_run_id": stateful_reference_bundle.get("source_run_id"),
    })

    old_folds = {int(item.get("fold_id") or 0): dict(item) for item in (updated.get("folds") or []) if isinstance(item, dict)}
    new_folds: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold.get("fold_id") or 0)
        fold_frame = multi_horizon_frame.loc[multi_horizon_frame["fold_id"] == fold_id].copy()
        fold_replay = _winner_anchored_temporal_study(
            fold_frame, stateful_rows, open_prices, common_dates, config, enable_timing_override=False
        ) if not fold_frame.empty else {}
        item = old_folds.get(fold_id, {"fold_id": fold_id, "test_start": fold.get("test_start"), "test_end": fold.get("test_end")})
        if fold_replay:
            item["strategy_return"] = _finite(fold_replay.get("total_return"))
            item["strategy_ending_capital"] = _finite(fold_replay.get("ending_capital"))
            item["strategy_cagr"] = _finite(fold_replay.get("cagr"))
            item["strategy_sharpe"] = _finite(fold_replay.get("sharpe"))
            item["strategy_maximum_drawdown"] = _finite(fold_replay.get("max_drawdown"))
            item["market_exposure"] = _finite(fold_replay.get("exposure"))
            item["capital_rotations"] = int(fold_replay.get("switch_count") or 0)
        new_folds.append(item)
    updated["folds"] = new_folds
    return updated, stateful_rows, stateful_trades


def _attach_winner_reference(
    horizon_summaries: list[dict[str, Any]],
    fold_summaries: list[dict[str, Any]],
    winner_reference: dict[str, Any],
) -> None:
    winner_ending = _finite(winner_reference.get("ending_capital"))
    winner_return = _finite(winner_reference.get("total_return"))
    for item in horizon_summaries:
        capital = item.get("shadow_capital") if isinstance(item.get("shadow_capital"), dict) else {}
        shadow_ending = _finite(capital.get("ending_capital"))
        shadow_return = _finite(capital.get("total_return"))
        item["winner_reference_ending_capital"] = winner_ending
        item["winner_reference_return"] = winner_return
        item["capital_vs_winner"] = (shadow_ending / winner_ending - 1.0) if shadow_ending is not None and winner_ending not in {None, 0.0} else None
        item["return_gap_vs_winner"] = (shadow_return - winner_return) if shadow_return is not None and winner_return is not None else None
    winner_folds = {int(row.get("fold_id")): row for row in (winner_reference.get("folds") or []) if isinstance(row, dict) and row.get("fold_id") is not None}
    for item in fold_summaries:
        reference = winner_folds.get(int(item.get("fold_id") or 0))
        if reference is None:
            continue
        capital = item.get("shadow_capital") if isinstance(item.get("shadow_capital"), dict) else {}
        shadow_return = _finite(capital.get("total_return"))
        reference_return = _finite(reference.get("strategy_return"))
        item["winner_reference_return"] = reference_return
        item["winner_reference_benchmark_return"] = _finite(reference.get("benchmark_return"))
        item["return_gap_vs_winner"] = (shadow_return - reference_return) if shadow_return is not None and reference_return is not None else None

def run_temporal_intelligence(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
    winner_reference_override: dict[str, Any] | None = None,
    candidate_evaluation_only: bool = False,
    prepared_context: dict[str, Any] | None = None,
    stateful_reference_bundle: dict[str, Any] | None = None,
    strategy_research_reference_analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if str(config.research_model_family) != "lightgbm_utility":
        raise ValueError("Temporal Decision Intelligence is restricted to a LightGBM Strategy snapshot.")

    def ensure_not_cancelled() -> None:
        if cancel_callback is not None:
            cancel_callback()

    ensure_not_cancelled()
    started = time.perf_counter()
    if prepared_context is None:
        if progress_callback:
            progress_callback(18.0, "Building temporal decision feature panel")
        frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
        ensure_not_cancelled()
        symbols = sorted(frames)
        horizons = sorted({int(value) for value in config.rotation_target_horizons})
        folds = _build_walk_forward_folds(common_dates, config)
        targets_by_horizon = _future_target_matrices(frames, common_dates, symbols, horizons)
        open_prices = _open_price_matrix(frames, common_dates, symbols)
    else:
        frames = prepared_context["frames"]
        common_dates = prepared_context["common_dates"]
        symbols = list(prepared_context["symbols"])
        horizons = list(prepared_context["horizons"])
        folds = list(prepared_context["folds"])
        targets_by_horizon = prepared_context["targets_by_horizon"]
        open_prices = prepared_context["open_prices"]
        requested_horizons = sorted({int(value) for value in config.rotation_target_horizons})
        if requested_horizons != horizons:
            raise ValueError("Prepared Temporal training context does not match the candidate horizons.")
        if progress_callback:
            progress_callback(18.0, "Reusing campaign temporal training context")
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))

    all_predictions: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in horizons}
    fold_summaries: list[dict[str, Any]] = []
    total_steps = max(1, len(folds) * len(horizons))
    completed_steps = 0
    total_fit_bundles = max(1, len(folds) * len(horizons) * 5)
    completed_fit_bundles = 0

    for fold_position, fold in enumerate(folds, start=1):
        ensure_not_cancelled()
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[int(fold["calibration_start_index"]): int(fold["calibration_end_index"])]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        test_dates = common_dates[int(fold["test_start_index"]): int(fold["test_end_index"])]

        for horizon in horizons:
            ensure_not_cancelled()
            targets = targets_by_horizon[horizon]
            prepared_datasets = None
            prepared_test = None
            prepared_realized = None
            if prepared_context is not None:
                fold_context = prepared_context["fold_contexts"][int(fold["fold_id"])]
                prepared_datasets = {
                    split_name: {
                        "x": fold_context["splits"][split_name]["x"],
                        "targets": fold_context["targets"][horizon][split_name],
                    }
                    for split_name in ("train", "calibration", "final_fit")
                }
                prepared_test = fold_context["splits"]["test"]
                prepared_realized = fold_context["targets"][horizon]["test"]

            binary_targets = ("profit_before_loss", "bottom", "top", "trend_persistence")
            fit_tasks: dict[str, Callable[[], Any]] = {
                target_name: (
                    lambda name=target_name: _fit_calibrated_binary_bundle(
                        frames,
                        symbols,
                        train_dates,
                        calibration_dates,
                        final_fit_dates,
                        targets,
                        name,
                        config,
                        prepared_datasets=prepared_datasets,
                    )
                )
                for target_name in binary_targets
            }
            fit_tasks["drawdown"] = lambda: _fit_drawdown_bundle(
                frames,
                symbols,
                train_dates,
                calibration_dates,
                final_fit_dates,
                targets,
                config,
                prepared_datasets=prepared_datasets,
            )
            worker_count = temporal_fit_worker_count(len(fit_tasks))
            if progress_callback:
                progress_callback(
                    22.0 + 54.0 * (completed_fit_bundles / total_fit_bundles),
                    f"Fold {fold_position}/{len(folds)} · {horizon} sessions · training {len(fit_tasks)} model bundles on {worker_count} worker(s)",
                )

            def fit_completed(target_name: str, _position: int, _total: int) -> None:
                nonlocal completed_fit_bundles
                completed_fit_bundles += 1
                if progress_callback:
                    progress_callback(
                        22.0 + 54.0 * (completed_fit_bundles / total_fit_bundles),
                        f"Fold {fold_position}/{len(folds)} · {horizon} sessions · completed {target_name}",
                    )

            fitted = run_independent_fit_tasks(
                fit_tasks,
                cancel_check=ensure_not_cancelled,
                completed_callback=fit_completed,
            )
            bundles = {name: fitted[name] for name in binary_targets}
            drawdown_bundle = fitted["drawdown"]
            ensure_not_cancelled()

            if prepared_test is None:
                x_test, metadata = _pooled_features(frames, symbols, test_dates)
            else:
                x_test, metadata = prepared_test["x"], prepared_test["metadata"]
            if x_test.empty:
                completed_steps += 1
                continue
            if prepared_realized is None:
                realized = {
                    name: _align_test_targets(metadata, targets[name])
                    for name in ("profit_before_loss", "bottom", "top", "trend_persistence", "trend_direction", "drawdown")
                }
            else:
                realized = prepared_realized

            classifier_predictions: dict[str, tuple[np.ndarray, np.ndarray, _BinaryModelBundle]] = {}
            for target_name, bundle in bundles.items():
                raw_probability = bundle.model.predict_proba(x_test)[:, 1]
                classifier_predictions[target_name] = (
                    raw_probability,
                    bundle.calibrator.transform(raw_probability),
                    bundle,
                )
            predicted_dd = drawdown_bundle.model.predict(x_test)
            prior_oos = pd.concat(all_predictions[horizon], ignore_index=True) if all_predictions[horizon] else pd.DataFrame()
            quality_overrides = _historical_oos_quality(prior_oos) if not prior_oos.empty else None
            prediction = _decision_prediction_frame(
                metadata=metadata,
                realized_profit_before_loss=realized["profit_before_loss"],
                realized_bottom=realized["bottom"],
                realized_top=realized["top"],
                realized_trend_persistence=realized["trend_persistence"],
                trend_direction=realized["trend_direction"],
                realized_drawdown=realized["drawdown"],
                classifier_predictions=classifier_predictions,
                predicted_drawdown=predicted_dd,
                drawdown_bundle=drawdown_bundle,
                fold_id=int(fold["fold_id"]),
                horizon=horizon,
                profit_barrier=float(targets["profit_barrier"]),
                loss_barrier=float(targets["loss_barrier"]),
                one_side_cost=one_side_cost,
                quality_overrides=quality_overrides,
                quality_history_samples=int(len(prior_oos)),
            )
            all_predictions[horizon].append(prediction)
            fold_summary = _decision_fold_summary(prediction)
            fold_summary["shadow_capital"] = _shadow_capital_study(prediction, open_prices, common_dates, config)
            fold_summaries.append({
                "fold_id": int(fold["fold_id"]),
                "horizon": int(horizon),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                **fold_summary,
            })
            completed_steps += 1

    ensure_not_cancelled()
    combined_predictions_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        frames_for_horizon = all_predictions[horizon]
        combined_raw = pd.concat(frames_for_horizon, ignore_index=True) if frames_for_horizon else pd.DataFrame()
        combined_predictions_by_horizon[horizon] = _apply_online_matured_quality(
            combined_raw, common_dates, horizon
        ) if not combined_raw.empty else combined_raw

    horizon_summaries: list[dict[str, Any]] = []
    latest_forecasts: list[dict[str, Any]] = []
    for horizon in horizons:
        combined = combined_predictions_by_horizon[horizon]
        if combined.empty:
            continue
        summary = _decision_fold_summary(combined)
        summary.update({
            "horizon": int(horizon),
            "profit_barrier": float(targets_by_horizon[horizon]["profit_barrier"]),
            "loss_barrier": float(targets_by_horizon[horizon]["loss_barrier"]),
            "risk_buckets": _risk_buckets(combined),
            "shadow_capital": _shadow_capital_study(combined, open_prices, common_dates, config, include_diagnostics=True),
        })
        horizon_summaries.append(summary)
        if not candidate_evaluation_only:
            if progress_callback:
                progress_callback(
                    78.0 + 14.0 * (horizons.index(horizon) / max(1, len(horizons))),
                    f"Training latest {horizon}-session relative decision forecast",
                )
            latest_quality, latest_quality_samples = _latest_online_quality(combined)
            latest_forecasts.extend(
                _latest_decision_forecasts(
                    frames,
                    common_dates,
                    symbols,
                    horizon,
                    targets_by_horizon[horizon],
                    config,
                    quality_overrides=latest_quality,
                    quality_history_samples=latest_quality_samples,
                    quality_source="online_matured_oos",
                )
            )

    latest_forecasts.sort(
        key=lambda row: (int(row["horizon"]), -(row.get("asset_rank_score") or -1e9), row["symbol"])
    )

    ensure_not_cancelled()
    if progress_callback:
        progress_callback(92.0, "Combining horizons into one temporal decision policy")
    multi_horizon_frame = _multi_horizon_frame(
        combined_predictions_by_horizon,
        horizons,
        one_side_cost=one_side_cost,
    )
    multi_horizon_roles = _multi_horizon_roles(horizons)
    standalone_temporal_shadow = _shadow_capital_study(
        multi_horizon_frame, open_prices, common_dates, config,
        decision_policy="adaptive_rotation_before_cash",
    ) if not multi_horizon_frame.empty else {}

    latest_multi_frame = _latest_multi_horizon_frame(
        latest_forecasts, horizons, one_side_cost=one_side_cost
    )
    multi_horizon_latest_forecasts = _multi_horizon_latest_rows(latest_multi_frame, horizons)

    ensure_not_cancelled()
    if progress_callback:
        progress_callback(95.0, "Replaying selected Strategy Research and applying Temporal timing overlay")
    if winner_reference_override:
        winner_reference = deepcopy(winner_reference_override.get("summary") or {})
        winner_reference_daily_rows = deepcopy(winner_reference_override.get("daily_rows") or [])
        winner_reference_trade_rows = deepcopy(winner_reference_override.get("trade_rows") or [])
    else:
        winner_reference = _winner_reference_replay(bars_by_symbol, config, progress_callback=progress_callback)
        winner_reference_daily_rows = winner_reference.pop("_daily_rows", [])
        winner_reference_trade_rows = winner_reference.pop("_trade_rows", [])

    observation_rows = _multi_horizon_observation_rows(
        multi_horizon_frame, horizons, frames_by_symbol=frames, common_dates=common_dates
    )
    if strategy_research_reference_analytics:
        winner_reference, winner_reference_daily_rows, winner_reference_trade_rows = _bind_strategy_research_reference_analytics(
            winner_reference=winner_reference,
            winner_daily_rows=winner_reference_daily_rows,
            reference_analytics=strategy_research_reference_analytics,
        )
    elif stateful_reference_bundle:
        winner_reference, winner_reference_daily_rows, winner_reference_trade_rows = _replace_reference_with_stateful_strategy(
            winner_reference=winner_reference,
            winner_daily_rows=winner_reference_daily_rows,
            multi_horizon_frame=multi_horizon_frame,
            folds=folds,
            open_prices=open_prices,
            common_dates=common_dates,
            config=config,
            stateful_reference_bundle=stateful_reference_bundle,
            observation_rows=observation_rows,
        )
    _attach_winner_reference(horizon_summaries, fold_summaries, winner_reference)

    if strategy_research_reference_analytics:
        winner_anchor_replay = _strategy_research_reference_study(
            multi_horizon_frame, winner_reference_daily_rows, strategy_research_reference_analytics,
            open_prices, config, enable_timing_override=False,
        ) if not multi_horizon_frame.empty else {}
        multi_horizon_shadow = _strategy_research_reference_study(
            multi_horizon_frame, winner_reference_daily_rows, strategy_research_reference_analytics,
            open_prices, config, include_diagnostics=True, include_economic_curve=True,
            enable_timing_override=True,
        ) if not multi_horizon_frame.empty else {}
    else:
        winner_anchor_replay = _winner_anchored_temporal_study(
            multi_horizon_frame, winner_reference_daily_rows, open_prices, common_dates, config,
            enable_timing_override=False,
        ) if not multi_horizon_frame.empty else {}
        multi_horizon_shadow = _winner_anchored_temporal_study(
            multi_horizon_frame, winner_reference_daily_rows, open_prices, common_dates, config,
            include_diagnostics=True, include_economic_curve=True, enable_timing_override=True,
        ) if not multi_horizon_frame.empty else {}

    reference_parity = None
    if strategy_research_reference_analytics:
        reference_parity = _strategy_research_reference_parity(winner_reference, winner_anchor_replay)
        if str(reference_parity.get("status") or "") != "passed":
            raise ValueError(
                "Selected Strategy Research reference replay failed exact parity before Temporal timing. "
                + json.dumps({
                    "checks": reference_parity.get("checks") or {},
                    "reference": reference_parity.get("reference") or {},
                    "replay": reference_parity.get("replay") or {},
                }, sort_keys=True)
            )

    multi_horizon_fold_metrics: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        fold_frame = multi_horizon_frame.loc[multi_horizon_frame["fold_id"] == fold_id].copy() if not multi_horizon_frame.empty else pd.DataFrame()
        if strategy_research_reference_analytics:
            fold_capital = _strategy_research_reference_study(
                fold_frame, winner_reference_daily_rows, strategy_research_reference_analytics,
                open_prices, config, enable_timing_override=True
            ) if not fold_frame.empty else {}
            fold_anchor = _strategy_research_reference_study(
                fold_frame, winner_reference_daily_rows, strategy_research_reference_analytics,
                open_prices, config, enable_timing_override=False
            ) if not fold_frame.empty else {}
        else:
            fold_capital = _winner_anchored_temporal_study(
                fold_frame, winner_reference_daily_rows, open_prices, common_dates, config, enable_timing_override=True
            ) if not fold_frame.empty else {}
            fold_anchor = _winner_anchored_temporal_study(
                fold_frame, winner_reference_daily_rows, open_prices, common_dates, config, enable_timing_override=False
            ) if not fold_frame.empty else {}
        fold_anchor_capital = _finite(fold_anchor.get("ending_capital")) if fold_anchor else None
        fold_hybrid_capital = _finite(fold_capital.get("ending_capital")) if fold_capital else None
        multi_horizon_fold_metrics.append({
            "fold_id": fold_id,
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "samples": int(len(fold_frame)),
            "winner_anchor_replay_ending_capital": fold_anchor_capital,
            "strategy_research_reference_ending_capital": fold_anchor_capital,
            "capital_lift_vs_winner_anchor_replay": (
                fold_hybrid_capital / fold_anchor_capital - 1.0
                if fold_hybrid_capital is not None and fold_anchor_capital not in {None, 0.0}
                else None
            ),
            "capital_lift_vs_strategy_research_reference": (
                fold_hybrid_capital / fold_anchor_capital - 1.0
                if fold_hybrid_capital is not None and fold_anchor_capital not in {None, 0.0}
                else None
            ),
            "shadow_capital": fold_capital,
        })

    anchor_capital = _finite(winner_anchor_replay.get("ending_capital")) if winner_anchor_replay else None
    multi_capital = _finite(multi_horizon_shadow.get("ending_capital")) if multi_horizon_shadow else None
    multi_horizon_metrics: dict[str, Any] = {
        "samples": int(len(multi_horizon_frame)),
        "entry_horizons": multi_horizon_roles["entry"],
        "hold_horizons": multi_horizon_roles["hold"],
        "risk_horizons": multi_horizon_roles["risk"],
        "shadow_capital": multi_horizon_shadow,
        "winner_anchor_replay": winner_anchor_replay,
        "strategy_research_reference_replay": winner_anchor_replay,
        "winner_anchor_replay_ending_capital": anchor_capital,
        "strategy_research_reference_ending_capital": anchor_capital,
        "winner_anchor_replay_sharpe": _finite(winner_anchor_replay.get("sharpe")) if winner_anchor_replay else None,
        "winner_anchor_replay_max_drawdown": _finite(winner_anchor_replay.get("max_drawdown")) if winner_anchor_replay else None,
        "winner_anchor_replay_switch_count": int(winner_anchor_replay.get("switch_count") or 0) if winner_anchor_replay else 0,
        "capital_lift_vs_winner_anchor_replay": (
            multi_capital / anchor_capital - 1.0
            if multi_capital is not None and anchor_capital not in {None, 0.0}
            else None
        ),
        "capital_lift_vs_strategy_research_reference": (
            multi_capital / anchor_capital - 1.0
            if multi_capital is not None and anchor_capital not in {None, 0.0}
            else None
        ),
        "standalone_temporal_reference": {
            key: value for key, value in standalone_temporal_shadow.items()
            if key not in {"decision_diagnostics", "economic_curve"}
        },
        "strategy_research_reference_parity": reference_parity,
    }

    winner_capital = _finite(winner_reference.get("ending_capital"))
    benchmark_capital = _finite(winner_reference.get("benchmark_ending_capital"))
    if multi_capital is not None and winner_capital is not None and winner_capital > 0:
        capital_vs_reference = multi_capital / winner_capital - 1.0
        multi_horizon_metrics["capital_vs_winner"] = capital_vs_reference
        multi_horizon_metrics["capital_vs_strategy_research_reference"] = capital_vs_reference
    if multi_capital is not None and benchmark_capital is not None and benchmark_capital > 0:
        multi_horizon_metrics["capital_vs_benchmark"] = multi_capital / benchmark_capital - 1.0
    winner_folds = {int(item.get("fold_id")): item for item in (winner_reference.get("folds") or []) if isinstance(item, dict)}
    for fold_item in multi_horizon_fold_metrics:
        reference = winner_folds.get(int(fold_item["fold_id"])) or {}
        capital = fold_item.get("shadow_capital") if isinstance(fold_item.get("shadow_capital"), dict) else {}
        shadow_return = _finite(capital.get("total_return"))
        winner_fold_return = _finite(reference.get("strategy_return"))
        benchmark_fold_return = _finite(reference.get("benchmark_return"))
        fold_item["winner_reference_return"] = winner_fold_return
        fold_item["winner_reference_benchmark_return"] = benchmark_fold_return
        if shadow_return is not None and winner_fold_return is not None:
            fold_item["return_gap_vs_winner"] = shadow_return - winner_fold_return
        if shadow_return is not None and benchmark_fold_return is not None:
            fold_item["return_gap_vs_benchmark"] = shadow_return - benchmark_fold_return

    try:
        import lightgbm
        lightgbm_version = str(lightgbm.__version__)
    except Exception:
        lightgbm_version = None

    ensure_not_cancelled()
    if progress_callback:
        progress_callback(99.0, "Finalizing Temporal Decision Intelligence v8 winner-anchored timing metrics")
    return {
        "experiment": "temporal_decision_intelligence_v8_winner_anchored_timing",
        "model_family": "lightgbm_utility",
        "model_label": "LightGBM Temporal Decision Intelligence v8 — Winner-Anchored Temporal Timing",
        "lightgbm_version": lightgbm_version,
        "target_definition": {
            "profit_before_loss": "upper price barrier touched before lower price barrier from next-session open; unresolved paths are non-opportunities and same-bar ambiguous touches are excluded",
            "bottom": "future upside reaches the profit barrier while adverse excursion stays within half of the loss barrier",
            "top": "future downside reaches the loss barrier while favorable excursion stays within half of the profit barrier",
            "trend_persistence": "current EMA20-vs-EMA50 trend direction remains economically material at the forecast horizon",
            "expected_max_drawdown": "maximum close-path drawdown from next-session open through the forecast horizon",
            "execution_basis": "features at current close; every target path begins at next-session open",
        },
        "decision_policy": {
            "objective": "maximize compounded OOS capital beyond the immutable Winner by keeping Winner utility as the primary allocator and using causal Temporal evidence as a short-horizon timing overlay",
            "asset_ranker": "the immutable Winner provides the primary Top-1/Top-2 utility candidate set; Temporal 5d/10d profit consensus may temporarily replace Winner Top-1 with Winner Top-2 only when the Top-1 timing signal is weak and Top-2 has materially stronger short-horizon evidence",
            "opportunity_gate": "the timing overlay does not continuously penalize high-return candidates for moderate risk; the Winner remains fully authoritative unless the causal short-horizon timing conditions for a Top-2 override are all satisfied",
            "entry": "Winner Top-1 remains the anchor; Top-2 overrides require Winner Top-1 short-profit consensus below 0.50, Top-2 consensus at least 0.60, and a minimum 0.25 absolute consensus advantage",
            "hold_exit": "Winner hysteresis remains the base holding logic; Temporal timing is intentionally narrow and changes only the selected Winner Top-1/Top-2 asset for the next open-to-open interval",
            "cash": "CASH is not introduced by the timing overlay; the overlay is designed to preserve the Winner compound engine rather than suppress return through broad defensive abstention",
            "hysteresis": "Winner selection and switching remain authoritative; Temporal creates only a one-interval Top-2 timing override when all three short-horizon timing conditions are satisfied, with no rigid minimum holding period added by the overlay",
            "horizon_agreement": "agreement measures whether short timing horizons and medium/long confirmation horizons identify a coherent leader; isolated long-horizon anomalies receive limited authority",
            "signal_quality": "signal quality is updated online from a rolling 126-session window of OOS labels only after each label horizon has fully matured; updates occur every five sessions and are smoothed, so no decision can use its own future outcome",
            "winner_reference": "the immutable Winner is replayed on the same frozen market snapshot and fold protocol for apples-to-apples capital comparison",
            "execution": "decision at current close, target position at next-session open, reevaluated every session",
            "costs": "Winner slippage_bps and commission_rate are applied to entries, exits, and rotations",
            "operational": False,
        },
        "horizons": horizons,
        "barriers": {
            str(horizon): {
                "profit": float(targets_by_horizon[horizon]["profit_barrier"]),
                "loss": float(targets_by_horizon[horizon]["loss_barrier"]),
            }
            for horizon in horizons
        },
        "feature_count": len(ROTATION_FEATURES),
        "features": list(ROTATION_FEATURES),
        "assets": symbols,
        "asset_count": len(symbols),
        "walk_forward_fold_count": len(folds),
        "purge_sessions": max(int(config.rotation_purge_days), max(horizons)),
        "oos_start": folds[0]["test_start"],
        "oos_end": folds[-1]["test_end"],
        "latest_as_of": common_dates[-1],
        "horizon_metrics": horizon_summaries,
        "fold_metrics": fold_summaries,
        "latest_forecasts": latest_forecasts,
        "multi_horizon_metrics": multi_horizon_metrics,
        "multi_horizon_fold_metrics": multi_horizon_fold_metrics,
        "multi_horizon_latest_forecasts": multi_horizon_latest_forecasts,
        "_multi_horizon_observations": observation_rows,
        "_winner_reference_daily_rows": winner_reference_daily_rows,
        "_winner_reference_trade_rows": winner_reference_trade_rows,
        "winner_reference": winner_reference,
        "duration_seconds": time.perf_counter() - started,
        "shadow_only": True,
        "affects_strategy_decisions": False,
        "affects_winner": False,
        "affects_paper_trading": False,
    }

def _load_run(db: Any, run_id: str) -> tuple[dict[str, Any], BacktestExecutionRequest]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 0})
    if run is None:
        raise ValueError(f"Temporal Intelligence run not found: {run_id}")
    request = BacktestExecutionRequest.model_validate(run.get("request") or {})
    return run, request


def _emit_progress(db: Any, run_id: str, percent: float, stage: str) -> None:
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {"$set": {"progress": max(0.0, min(100.0, float(percent))), "stage": str(stage), "updated_at": utc_now()}},
    )
    print(f"TEMPORAL_PROGRESS|{float(percent):.1f}|{str(stage).replace('|', '/')}", flush=True)


def _compressed_artifact_documents(
    run_id: str,
    kind: str,
    rows: list[dict[str, Any]],
    *,
    chunk_size: int = 250,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    normalized_rows = [bson_value(dict(row)) for row in rows if isinstance(row, dict)]
    for sequence, start in enumerate(range(0, len(normalized_rows), max(1, int(chunk_size)))):
        chunk = normalized_rows[start:start + max(1, int(chunk_size))]
        encoded = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        documents.append({
            "run_id": str(run_id),
            "kind": str(kind),
            "sequence": int(sequence),
            "encoding": "zlib-json-v1",
            "row_count": int(len(chunk)),
            "payload": zlib.compress(encoded, level=9),
            "created_at": utc_now(),
        })
    return documents


def _externalize_result_diagnostics(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    artifact_rows: list[dict[str, Any]] = []
    counts = {
        "horizon_decision_diagnostics": 0,
        "multi_horizon_decision_diagnostics": 0,
        "multi_horizon_equity_curve": 0,
    }
    for horizon in result.get("horizon_metrics") or []:
        if not isinstance(horizon, dict):
            continue
        capital = horizon.get("shadow_capital")
        if not isinstance(capital, dict):
            continue
        diagnostics = capital.pop("decision_diagnostics", []) or []
        horizon_value = horizon.get("horizon")
        for item in diagnostics:
            if isinstance(item, dict):
                artifact_rows.append({
                    "artifact_kind": "horizon_decision_diagnostics",
                    "horizon": horizon_value,
                    **dict(item),
                })
                counts["horizon_decision_diagnostics"] += 1

    multi_horizon = result.get("multi_horizon_metrics")
    if isinstance(multi_horizon, dict):
        capital = multi_horizon.get("shadow_capital")
        if isinstance(capital, dict):
            diagnostics = capital.pop("decision_diagnostics", []) or []
            for item in diagnostics:
                if isinstance(item, dict):
                    artifact_rows.append({
                        "artifact_kind": "multi_horizon_decision_diagnostics",
                        **dict(item),
                    })
                    counts["multi_horizon_decision_diagnostics"] += 1
            economic_curve = capital.pop("economic_curve", []) or []
            for item in economic_curve:
                if isinstance(item, dict):
                    artifact_rows.append({
                        "artifact_kind": "multi_horizon_equity_curve",
                        **dict(item),
                    })
                    counts["multi_horizon_equity_curve"] += 1
    return artifact_rows, counts


def execute_temporal_run(run_id: str, db: Any) -> dict[str, Any]:
    run, config = _load_run(db, run_id)
    if str(config.research_model_family) != "lightgbm_utility":
        raise ValueError("Temporal Intelligence requires the selected Strategy to use LightGBM.")

    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {"$set": {"status": "running", "stage": "Loading frozen market data", "progress": 2.0, "started_at": utc_now(), "updated_at": utc_now()}},
    )
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(config.assets, start=1):
        _emit_progress(db, run_id, 3.0 + 12.0 * ((position - 1) / max(1, len(config.assets))), f"Loading market data {position}/{len(config.assets)}")
        asset_config = config if symbol in set(config.calendar_anchor_assets) else config.model_copy(update={"market_data_require_complete_history": False})
        raw = load_market_bars(symbol, asset_config)
        bars_by_symbol[symbol] = validate_and_clean_bars(raw, asset_config)

    strategy_research_reference_analytics = None
    processing_id = str(run.get("research_processing_id") or "").strip()
    processing_kind = str(run.get("research_processing_kind") or "").strip()
    strategy_kind = str(run.get("strategy_kind") or "standard").strip() or "standard"
    if strategy_kind == "standard" and not processing_id:
        raise ValueError(
            "Selected standard Strategy Research requires its completed Reference Replay before Temporal Intelligence."
        )
    if processing_id and processing_kind in {"backtest", "strategy_research_stateful", "strategy_research_temporal", "strategy_research_decision_optimization"}:
        from ..services.analytics import processing_analytics
        strategy_research_reference_analytics = processing_analytics(db, processing_id)
        expected_strategy_id = str(run.get("strategy_profile_id") or "").strip()
        expected_revision = int(run.get("strategy_profile_revision") or 1)
        expected_hash = str(run.get("strategy_configuration_hash") or "").strip()

        if processing_kind == "backtest":
            source = db[JOBS_COLLECTION].find_one(
                {"id": processing_id},
                {
                    "_id": 0,
                    "strategy_profile_id": 1,
                    "strategy_profile_revision": 1,
                    "strategy_configuration_hash": 1,
                },
            ) or {}
            actual_strategy_id = str(source.get("strategy_profile_id") or "").strip()
            actual_revision = int(source.get("strategy_profile_revision") or 1)
            actual_hash = str(source.get("strategy_configuration_hash") or "").strip()
        else:
            actual_strategy_id = str(strategy_research_reference_analytics.get("strategy_profile_id") or "").strip()
            actual_revision = int(strategy_research_reference_analytics.get("strategy_profile_revision") or 1)
            actual_hash = str(strategy_research_reference_analytics.get("strategy_configuration_hash") or "").strip()

        if (
            actual_strategy_id != expected_strategy_id
            or actual_revision != expected_revision
            or (expected_hash and actual_hash != expected_hash)
        ):
            raise ValueError("Selected Strategy Research reference does not match the frozen Strategy snapshot.")
        strategy_research_reference_analytics["strategy_profile_id"] = expected_strategy_id
        strategy_research_reference_analytics["strategy_profile_revision"] = expected_revision
        strategy_research_reference_analytics["strategy_configuration_hash"] = expected_hash

    result = run_temporal_intelligence(
        bars_by_symbol,
        config,
        progress_callback=lambda percent, stage: _emit_progress(db, run_id, percent, stage),
        stateful_reference_bundle=(run.get("stateful_reference_bundle") if isinstance(run.get("stateful_reference_bundle"), dict) else None),
        strategy_research_reference_analytics=strategy_research_reference_analytics,
    )
    observation_rows = result.pop("_multi_horizon_observations", [])
    winner_reference_daily_rows = result.pop("_winner_reference_daily_rows", [])
    winner_reference_trade_rows = result.pop("_winner_reference_trade_rows", [])
    diagnostic_artifact_rows, diagnostic_counts = _externalize_result_diagnostics(result)

    observations_collection = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION]
    observations_collection.delete_many({"run_id": run_id})
    if observation_rows:
        grouped_observations: dict[str, dict[str, Any]] = {}
        for row in observation_rows:
            timestamp = pd.Timestamp(row.get("timestamp"))
            key = timestamp.isoformat()
            document = grouped_observations.setdefault(key, {"run_id": run_id, "timestamp": timestamp, "rows": []})
            payload = dict(row)
            payload.pop("timestamp", None)
            document["rows"].append(bson_value(payload))
        documents: list[dict[str, Any]] = []
        for _, document in sorted(grouped_observations.items()):
            rows_payload = bson_value(document.get("rows") or [])
            encoded = json.dumps(rows_payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
            documents.append({
                "run_id": run_id,
                "timestamp": bson_value(document.get("timestamp")),
                "encoding": "zlib-json-v1",
                "row_count": int(len(rows_payload)),
                "payload": zlib.compress(encoded, level=9),
            })
        for start in range(0, len(documents), 200):
            observations_collection.insert_many(documents[start:start + 200], ordered=False)

    artifacts_collection = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION]
    artifacts_collection.delete_many({"run_id": run_id})
    artifact_documents = _compressed_artifact_documents(
        run_id,
        "decision_diagnostics",
        diagnostic_artifact_rows,
        chunk_size=200,
    )
    artifact_documents.extend(_compressed_artifact_documents(
        run_id, "winner_reference_daily", winner_reference_daily_rows, chunk_size=250
    ))
    artifact_documents.extend(_compressed_artifact_documents(
        run_id, "winner_reference_trades", winner_reference_trade_rows, chunk_size=250
    ))
    for start in range(0, len(artifact_documents), 100):
        artifacts_collection.insert_many(artifact_documents[start:start + 100], ordered=False)

    result["artifact_storage"] = {
        "schema_version": 2,
        "decision_diagnostics_external": True,
        "horizon_decision_diagnostics_rows": int(diagnostic_counts["horizon_decision_diagnostics"]),
        "multi_horizon_decision_diagnostics_rows": int(diagnostic_counts["multi_horizon_decision_diagnostics"]),
        "multi_horizon_equity_curve_rows": int(diagnostic_counts["multi_horizon_equity_curve"]),
        "multi_horizon_daily_asset_rows": int(len(observation_rows)),
        "winner_reference_daily_rows": int(len(winner_reference_daily_rows)),
        "winner_reference_trade_rows": int(len(winner_reference_trade_rows)),
    }

    now = utc_now()
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {"$set": {
            "status": "completed",
            "stage": "Completed",
            "progress": 100.0,
            "result": bson_value(result),
            "finished_at": now,
            "updated_at": now,
        }},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        print("Temporal Decision Intelligence shadow research only. No trading decision will be changed.", flush=True)
        execute_temporal_run(str(args.run_id), db)
    except Exception as exc:
        try:
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
                {"id": str(args.run_id)},
                {"$set": {
                    "status": "failed",
                    "stage": "Temporal Intelligence failed",
                    "failure_message": "Temporal Intelligence execution failed. Check protected server logs.",
                    "technical_error": str(exc)[:2000],
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                }},
            )
        except Exception:
            pass
        print(f"ERROR Temporal Intelligence: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
