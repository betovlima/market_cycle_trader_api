from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.environment import load_project_environment
from ..infrastructure.persistence.mongo_repository import (
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
        "n_jobs": _effective_n_jobs(int(settings["n_jobs"])),
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

    if len(realized) == 0:
        return {"mae": None, "rmse": None, "rank_correlation": None}
    actual_series = pd.Series(realized, dtype=float)
    predicted_series = pd.Series(predicted, dtype=float)
    rank_correlation = None
    if actual_series.nunique(dropna=True) > 1 and predicted_series.nunique(dropna=True) > 1:
        rank_correlation = _finite(actual_series.corr(predicted_series, method="spearman"))
    return {
        "mae": _finite(mean_absolute_error(realized, predicted)),
        "rmse": _finite(math.sqrt(mean_squared_error(realized, predicted))),
        "rank_correlation": rank_correlation,
    }


def _classification_metrics(realized_alpha: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss

    labels = np.asarray(realized_alpha > 0.0, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(labels) == 0:
        return {"brier": None, "log_loss": None, "auc": None, "calibration_error": None, "positive_rate": None}
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



def _fit_calibrated_binary_bundle(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    final_fit_dates: pd.DatetimeIndex,
    targets: dict[str, Any],
    target_name: str,
    config: Any,
) -> _BinaryModelBundle:
    x_train, y_train, _ = _pooled_dataset(frames, symbols, train_dates, targets, target_name)
    calibration_model = _fit_binary_classifier_relaxed(x_train, y_train, config)
    x_calibration, y_calibration, _ = _pooled_dataset(frames, symbols, calibration_dates, targets, target_name)
    raw_calibration = calibration_model.predict_proba(x_calibration)[:, 1]
    train_baseline_probability = float(np.clip(np.mean(y_train > 0.0), 1e-6, 1.0 - 1e-6))
    baseline_calibration = np.full(len(y_calibration), train_baseline_probability, dtype=float)
    raw_metrics = _classification_metrics(y_calibration, raw_calibration)
    baseline_metrics = _classification_metrics(y_calibration, baseline_calibration)
    validation_brier_skill = _skill_score(raw_metrics.get("brier"), baseline_metrics.get("brier"))
    calibrator = _fit_platt_calibrator(raw_calibration, y_calibration)
    x_final, y_final, _ = _pooled_dataset(frames, symbols, final_fit_dates, targets, target_name)
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
) -> _DrawdownModelBundle:
    x_train, y_train, _ = _pooled_dataset(frames, symbols, train_dates, targets, "drawdown")
    validation_model = _fit_regressor(x_train, y_train, config, objective="regression_l1")
    x_calibration, y_calibration, _ = _pooled_dataset(frames, symbols, calibration_dates, targets, "drawdown")
    predicted_calibration = np.clip(validation_model.predict(x_calibration), 0.0, 1.0)
    training_baseline = float(max(0.0, np.mean(y_train)))
    baseline_calibration = np.full(len(y_calibration), training_baseline, dtype=float)
    validation = _regression_metrics(y_calibration, predicted_calibration)
    validation_baseline = _regression_metrics(y_calibration, baseline_calibration)
    x_final, y_final, _ = _pooled_dataset(frames, symbols, final_fit_dates, targets, "drawdown")
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


def _binary_signal_metrics(
    realized: np.ndarray,
    probability: np.ndarray,
    raw_probability: np.ndarray,
    baseline_probability: np.ndarray,
) -> dict[str, Any]:
    calibrated = _classification_metrics(realized, probability)
    raw = _classification_metrics(realized, raw_probability)
    baseline = _classification_metrics(realized, baseline_probability)
    high = np.asarray(probability, dtype=float) >= 0.70
    labels = np.asarray(realized, dtype=float) > 0.0
    high_rate = _finite(labels[high].mean()) if bool(high.any()) else None
    positive_rate = _finite(labels.mean()) if len(labels) else None
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


def _shadow_capital_study(
    frame: pd.DataFrame,
    open_prices: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    config: Any,
    *,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    if frame.empty:
        return {}
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(common_dates)}
    daily_returns: list[float] = []
    action_counts = {"buy": 0, "hold": 0, "sell": 0, "rotate": 0, "cash": 0}
    diagnostics: list[dict[str, Any]] = []
    exposure_days = 0
    switch_count = 0
    for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
        current_symbol: str | None = None
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
                    "current_hold_score": _finite(current_hold_score),
                    "cash_score": _finite(best.get("cash_score")),
                    "entry_threshold": _finite(entry_threshold),
                    "exit_threshold": _finite(exit_threshold),
                    "rotation_hurdle": _finite(rotation_hurdle),
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

            factor = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + interval_return)
            daily_returns.append(factor - 1.0)
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
    max_drawdown = float(np.min(equity / running_peak - 1.0)) if len(equity) else 0.0
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
    }
    if include_diagnostics:
        result["decision_diagnostics"] = diagnostics
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
            "quality_source": "complete_oos" if quality_overrides else "pretest_validation",
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
        "realized_drawdown",
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
    risk_deterioration = np.maximum(0.0, 0.35 - result["all_horizon_risk_safety"])
    result["hold_score"] = np.clip(
        0.34 * result["long_profit_confirmation"]
        + 0.28 * result["all_horizon_risk_safety"]
        + 0.20 * result["short_profit_consensus"]
        + 0.10 * result["horizon_agreement"]
        + 0.08 * result["long_trend_support"]
        - 0.20 * risk_deterioration,
        0.0,
        1.0,
    )
    result["entry_threshold"] = 0.34 + min(0.05, 10.0 * float(one_side_cost)) + 0.04 * (1.0 - short_quality)
    result["exit_threshold"] = 0.40 + min(0.04, 8.0 * float(one_side_cost))
    result["rotation_hurdle"] = 0.08 + min(0.05, 10.0 * float(one_side_cost))
    result["cash_score"] = result["entry_threshold"]
    result["entry_score"] = result["opportunity_gate_score"]
    result["asset_rank_score"] = result["entry_rank_score"]
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
    result["quality_source"] = "multi_horizon_prior_oos"
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


def _multi_horizon_observation_rows(frame: pd.DataFrame, horizons: list[int]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    base_columns = [
        "timestamp", "fold_id", "symbol", "entry_rank_score", "entry_rank_percentile",
        "opportunity_gate_score", "entry_threshold", "hold_score", "exit_threshold", "rotation_hurdle",
        "short_profit_consensus", "short_risk_safety", "short_bottom_support", "short_horizon_agreement",
        "long_profit_confirmation", "long_risk_safety", "long_trend_support", "long_horizon_agreement",
        "cross_horizon_agreement", "horizon_agreement", "all_horizon_risk_safety", "predicted_drawdown",
        "entry_separation_strength", "entry_top_gap_strength", "short_profit_quality",
    ]
    horizon_columns: list[str] = []
    for horizon in horizons:
        for prefix in (
            "profit_before_loss_probability", "profit_percentile", "profit_before_loss_quality_weight",
            "predicted_drawdown", "risk_safety_percentile", "drawdown_quality_weight",
            "bottom_probability", "bottom_quality_weight", "top_probability", "top_quality_weight",
            "trend_direction", "trend_persistence_probability", "trend_persistence_quality_weight",
        ):
            column = f"{prefix}_h{horizon}"
            if column in frame.columns:
                horizon_columns.append(column)
    columns = [column for column in base_columns + horizon_columns if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for item in frame[columns].sort_values(["timestamp", "symbol"]).to_dict(orient="records"):
        rows.append({key: (pd.Timestamp(value) if key == "timestamp" else _finite(value) if isinstance(value, (float, np.floating)) else value) for key, value in item.items()})
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
            "entry_rank_percentile": _finite(row.get("entry_rank_percentile")),
            "opportunity_gate_score": _finite(row.get("opportunity_gate_score")),
            "entry_threshold": _finite(row.get("entry_threshold")),
            "hold_score": _finite(row.get("hold_score")),
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
    }


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
) -> dict[str, Any]:
    if str(config.research_model_family) != "lightgbm_utility":
        raise ValueError("Temporal Decision Intelligence is restricted to a LightGBM Strategy snapshot.")

    started = time.perf_counter()
    if progress_callback:
        progress_callback(18.0, "Building temporal decision feature panel")
    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    horizons = sorted({int(value) for value in config.rotation_target_horizons})
    folds = _build_walk_forward_folds(common_dates, config)
    targets_by_horizon = _future_target_matrices(frames, common_dates, symbols, horizons)
    open_prices = _open_price_matrix(frames, common_dates, symbols)
    one_side_cost = max(0.0, float(config.slippage_bps) / 10_000.0) + max(0.0, float(config.commission_rate))

    all_predictions: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in horizons}
    fold_summaries: list[dict[str, Any]] = []
    total_steps = max(1, len(folds) * len(horizons))
    completed_steps = 0

    for fold_position, fold in enumerate(folds, start=1):
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[int(fold["calibration_start_index"]): int(fold["calibration_end_index"])]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        test_dates = common_dates[int(fold["test_start_index"]): int(fold["test_end_index"])]

        for horizon in horizons:
            targets = targets_by_horizon[horizon]
            if progress_callback:
                progress_callback(
                    22.0 + 54.0 * (completed_steps / total_steps),
                    f"Temporal Decision Intelligence v4 fold {fold_position}/{len(folds)} · {horizon} sessions",
                )

            bundles: dict[str, _BinaryModelBundle] = {}
            for target_name in ("profit_before_loss", "bottom", "top", "trend_persistence"):
                bundles[target_name] = _fit_calibrated_binary_bundle(
                    frames,
                    symbols,
                    train_dates,
                    calibration_dates,
                    final_fit_dates,
                    targets,
                    target_name,
                    config,
                )
            drawdown_bundle = _fit_drawdown_bundle(
                frames, symbols, train_dates, calibration_dates, final_fit_dates, targets, config
            )

            x_test, metadata = _pooled_features(frames, symbols, test_dates)
            if x_test.empty:
                completed_steps += 1
                continue
            realized = {
                name: _align_test_targets(metadata, targets[name])
                for name in ("profit_before_loss", "bottom", "top", "trend_persistence", "trend_direction", "drawdown")
            }
            valid = np.ones(len(metadata), dtype=bool)
            for values in realized.values():
                valid &= np.isfinite(values)
            if not bool(valid.any()):
                completed_steps += 1
                continue
            x_test = x_test.loc[valid].reset_index(drop=True)
            metadata = metadata.loc[valid].reset_index(drop=True)
            realized = {name: values[valid] for name, values in realized.items()}

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

    horizon_summaries: list[dict[str, Any]] = []
    latest_forecasts: list[dict[str, Any]] = []
    for horizon in horizons:
        frames_for_horizon = all_predictions[horizon]
        combined = pd.concat(frames_for_horizon, ignore_index=True) if frames_for_horizon else pd.DataFrame()
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
        if progress_callback:
            progress_callback(
                78.0 + 14.0 * (horizons.index(horizon) / max(1, len(horizons))),
                f"Training latest {horizon}-session relative decision forecast",
            )
        latest_forecasts.extend(
            _latest_decision_forecasts(
                frames,
                common_dates,
                symbols,
                horizon,
                targets_by_horizon[horizon],
                config,
                quality_overrides=_historical_oos_quality(combined),
                quality_history_samples=int(len(combined)),
            )
        )

    latest_forecasts.sort(
        key=lambda row: (int(row["horizon"]), -(row.get("asset_rank_score") or -1e9), row["symbol"])
    )

    combined_predictions_by_horizon = {
        horizon: pd.concat(all_predictions[horizon], ignore_index=True) if all_predictions[horizon] else pd.DataFrame()
        for horizon in horizons
    }
    if progress_callback:
        progress_callback(92.0, "Combining horizons into one temporal decision policy")
    multi_horizon_frame = _multi_horizon_frame(
        combined_predictions_by_horizon,
        horizons,
        one_side_cost=one_side_cost,
    )
    multi_horizon_roles = _multi_horizon_roles(horizons)
    multi_horizon_shadow = _shadow_capital_study(
        multi_horizon_frame, open_prices, common_dates, config, include_diagnostics=True
    ) if not multi_horizon_frame.empty else {}
    multi_horizon_fold_metrics: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        fold_frame = multi_horizon_frame.loc[multi_horizon_frame["fold_id"] == fold_id].copy() if not multi_horizon_frame.empty else pd.DataFrame()
        fold_capital = _shadow_capital_study(fold_frame, open_prices, common_dates, config) if not fold_frame.empty else {}
        multi_horizon_fold_metrics.append({
            "fold_id": fold_id,
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "samples": int(len(fold_frame)),
            "shadow_capital": fold_capital,
        })

    latest_multi_frame = _latest_multi_horizon_frame(
        latest_forecasts, horizons, one_side_cost=one_side_cost
    )
    multi_horizon_latest_forecasts = _multi_horizon_latest_rows(latest_multi_frame, horizons)
    multi_horizon_metrics: dict[str, Any] = {
        "samples": int(len(multi_horizon_frame)),
        "entry_horizons": multi_horizon_roles["entry"],
        "hold_horizons": multi_horizon_roles["hold"],
        "risk_horizons": multi_horizon_roles["risk"],
        "shadow_capital": multi_horizon_shadow,
    }

    if progress_callback:
        progress_callback(95.0, "Replaying immutable Winner on the same frozen market snapshot")
    winner_reference = _winner_reference_replay(bars_by_symbol, config, progress_callback=progress_callback)
    _attach_winner_reference(horizon_summaries, fold_summaries, winner_reference)
    winner_capital = _finite(winner_reference.get("ending_capital"))
    benchmark_capital = _finite(winner_reference.get("benchmark_ending_capital"))
    multi_capital = _finite(multi_horizon_shadow.get("ending_capital")) if multi_horizon_shadow else None
    if multi_capital is not None and winner_capital is not None and winner_capital > 0:
        multi_horizon_metrics["capital_vs_winner"] = multi_capital / winner_capital - 1.0
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

    if progress_callback:
        progress_callback(99.0, "Finalizing Multi-Horizon Temporal Decision Intelligence v4 metrics")
    return {
        "experiment": "temporal_decision_intelligence_v4_multi_horizon",
        "model_family": "lightgbm_utility",
        "model_label": "LightGBM Multi-Horizon Temporal Decision Intelligence v4",
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
            "objective": "maximize shadow compounded OOS capital with one multi-horizon long-only Top-1 or CASH policy",
            "asset_ranker": "the shortest two horizons drive ENTRY timing through cross-sectional P(profit-before-loss) ranks, short-horizon risk safety, and skilled bottom support; longer horizons provide only modest confirmation",
            "opportunity_gate": "INVEST versus CASH is decided after multi-horizon Top-1 selection using cross-sectional separation, Top-1 gap, short-horizon agreement, horizon agreement, and validated risk safety",
            "entry": "ENTRY uses the shortest two configured horizons as timing horizons and never requires absolute P(profit) to exceed a fixed barrier breakeven probability",
            "hold_exit": "the remaining medium/long horizons primarily govern HOLD persistence while risk across every horizon can force deterioration; rotations still require a switching hurdle",
            "cash": "CASH is selected when the multi-horizon leader lacks sufficient cross-sectional separation/agreement or carries weak validated risk, rather than because any single horizon is below an absolute probability threshold",
            "horizon_agreement": "agreement measures whether short timing horizons and medium/long confirmation horizons identify a coherent leader; isolated long-horizon anomalies receive limited authority",
            "signal_quality": "fold 1 uses historical pre-test validation; later folds use only completed prior OOS folds, so current OOS outcomes never tune their own decisions",
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
        "_multi_horizon_observations": _multi_horizon_observation_rows(multi_horizon_frame, horizons),
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

    result = run_temporal_intelligence(
        bars_by_symbol,
        config,
        progress_callback=lambda percent, stage: _emit_progress(db, run_id, percent, stage),
    )
    observation_rows = result.pop("_multi_horizon_observations", [])
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
