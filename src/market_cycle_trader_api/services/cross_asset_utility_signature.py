from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES
from market_cycle_trader_api.engine.research_challengers import (
    _effective_n_jobs,
    _lightgbm_settings,
)

TARGET_COLUMN = "forward_risk_adjusted_utility"
PREDICTION_COLUMN = "common_scale_predicted_utility"


@dataclass
class CrossAssetUtilityModel:
    model: Any
    feature_names: list[str]
    training_rows: int
    training_symbols: int
    training_dates: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.feature_names]
        values = np.asarray(self.model.predict(matrix), dtype=float)
        if values.ndim != 1 or len(values) != len(frame):
            raise RuntimeError("Cross-asset utility model returned an invalid prediction shape.")
        return values

    def diagnostics(self) -> dict[str, Any]:
        return {
            "training_rows": int(self.training_rows),
            "training_symbols": int(self.training_symbols),
            "training_dates": int(self.training_dates),
            "feature_count": int(len(self.feature_names)),
        }


def _date_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("timestamp")["symbol"].transform("count").astype(float)
    raw = 1.0 / counts.clip(lower=1.0)
    values = raw.to_numpy(dtype=float, copy=True)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Cross-asset utility sample weights are invalid.")
    return values * (float(len(values)) / total)


def build_pooled_utility_training_frame(
    frames: dict[str, pd.DataFrame],
    symbols: Iterable[str],
    training_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dates = pd.DatetimeIndex(training_dates).sort_values()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        available = dates.intersection(pd.DatetimeIndex(frame.index))
        if available.empty:
            continue
        part = frame.loc[available, [*ROTATION_FEATURES, TARGET_COLUMN]].copy()
        part = part.replace([np.inf, -np.inf], np.nan)
        part = part.dropna(subset=[*ROTATION_FEATURES, TARGET_COLUMN])
        if part.empty:
            continue
        part.insert(0, "timestamp", pd.DatetimeIndex(part.index))
        part.insert(1, "symbol", symbol)
        rows.append(part.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No finite baseline rows are available for pooled utility training.")
    pooled = pd.concat(rows, ignore_index=True)
    pooled["timestamp"] = pd.to_datetime(
        pooled["timestamp"], utc=True, format="mixed", errors="raise"
    )
    pooled = pooled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if pooled.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("Duplicate timestamp/symbol rows in pooled utility training.")
    return pooled


def fit_cross_asset_utility_model(
    frames: dict[str, pd.DataFrame],
    baseline_symbols: Iterable[str],
    training_dates: pd.DatetimeIndex,
    config: Any,
) -> CrossAssetUtilityModel:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Cross-asset utility research requires lightgbm.") from exc

    symbols = list(dict.fromkeys(str(x).strip().upper() for x in baseline_symbols if str(x).strip()))
    pooled = build_pooled_utility_training_frame(frames, symbols, training_dates)
    settings = _lightgbm_settings(config)
    weights = _date_balanced_weights(pooled)

    model = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(settings["n_estimators"]),
        learning_rate=float(settings["learning_rate"]),
        max_depth=int(settings["max_depth"]),
        num_leaves=int(settings["num_leaves"]),
        min_child_samples=int(settings["min_child_samples"]),
        min_child_weight=float(settings["min_child_weight"]),
        subsample=float(settings["subsample"]),
        subsample_freq=int(settings["subsample_freq"]),
        colsample_bytree=float(settings["colsample_bytree"]),
        reg_alpha=float(settings["reg_alpha"]),
        reg_lambda=float(settings["reg_lambda"]),
        max_bin=int(settings["max_bin"]),
        random_state=int(config.random_state),
        n_jobs=_effective_n_jobs(int(settings["n_jobs"])),
        deterministic=bool(config.deterministic_execution),
        force_col_wise=bool(config.deterministic_execution),
        verbosity=-1,
    )
    model.fit(
        pooled[ROTATION_FEATURES],
        pooled[TARGET_COLUMN].to_numpy(dtype=float),
        sample_weight=weights,
    )
    return CrossAssetUtilityModel(
        model=model,
        feature_names=list(ROTATION_FEATURES),
        training_rows=int(len(pooled)),
        training_symbols=int(pooled["symbol"].nunique()),
        training_dates=int(pooled["timestamp"].nunique()),
    )


def score_common_scale_utility(
    model: CrossAssetUtilityModel,
    frames: dict[str, pd.DataFrame],
    symbols: Iterable[str],
    decision_dates: pd.DatetimeIndex,
    *,
    role: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dates = pd.DatetimeIndex(decision_dates).sort_values()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        available = dates.intersection(pd.DatetimeIndex(frame.index))
        if available.empty:
            continue
        part = frame.loc[available, [*ROTATION_FEATURES, TARGET_COLUMN]].copy()
        part = part.replace([np.inf, -np.inf], np.nan)
        finite_features = ~part[ROTATION_FEATURES].isna().any(axis=1)
        part = part.loc[finite_features].copy()
        if part.empty:
            continue
        part.insert(0, "timestamp", pd.DatetimeIndex(part.index))
        part.insert(1, "symbol", symbol)
        part.insert(2, "role", str(role))
        part[PREDICTION_COLUMN] = model.predict(part)
        part["realized_forward_risk_adjusted_utility"] = pd.to_numeric(
            part[TARGET_COLUMN], errors="coerce"
        )
        rows.append(
            part[
                [
                    "timestamp",
                    "symbol",
                    "role",
                    PREDICTION_COLUMN,
                    "realized_forward_risk_adjusted_utility",
                ]
            ].reset_index(drop=True)
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "role",
                PREDICTION_COLUMN,
                "realized_forward_risk_adjusted_utility",
            ]
        )
    scored = pd.concat(rows, ignore_index=True)
    scored["timestamp"] = pd.to_datetime(
        scored["timestamp"], utc=True, format="mixed", errors="raise"
    )
    scored[PREDICTION_COLUMN] = pd.to_numeric(
        scored[PREDICTION_COLUMN], errors="coerce"
    )
    if not np.isfinite(scored[PREDICTION_COLUMN].to_numpy(dtype=float)).all():
        raise RuntimeError("Cross-asset utility model produced non-finite predictions.")
    return scored.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def common_scale_signal_metrics(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {
            "scored_rows": 0,
            "completed_rows": 0,
            "global_pearson": None,
            "global_spearman": None,
            "mean_daily_cross_section_spearman": None,
            "positive_daily_cross_section_spearman_rate": None,
            "top1_realized_utility_mean": None,
            "cross_section_realized_utility_mean": None,
            "top1_minus_cross_section_mean": None,
        }

    frame = scored.copy()
    realized = pd.to_numeric(
        frame["realized_forward_risk_adjusted_utility"], errors="coerce"
    )
    predicted = pd.to_numeric(frame[PREDICTION_COLUMN], errors="coerce")
    finite = np.isfinite(predicted.to_numpy(dtype=float)) & np.isfinite(
        realized.to_numpy(dtype=float)
    )
    complete = frame.loc[finite].copy()
    if complete.empty:
        return {
            "scored_rows": int(len(frame)),
            "completed_rows": 0,
            "global_pearson": None,
            "global_spearman": None,
            "mean_daily_cross_section_spearman": None,
            "positive_daily_cross_section_spearman_rate": None,
            "top1_realized_utility_mean": None,
            "cross_section_realized_utility_mean": None,
            "top1_minus_cross_section_mean": None,
        }

    complete[PREDICTION_COLUMN] = pd.to_numeric(
        complete[PREDICTION_COLUMN], errors="raise"
    )
    complete["realized_forward_risk_adjusted_utility"] = pd.to_numeric(
        complete["realized_forward_risk_adjusted_utility"], errors="raise"
    )
    pearson = complete[PREDICTION_COLUMN].corr(
        complete["realized_forward_risk_adjusted_utility"], method="pearson"
    )
    spearman = complete[PREDICTION_COLUMN].corr(
        complete["realized_forward_risk_adjusted_utility"], method="spearman"
    )

    daily_rank: list[float] = []
    top1_realized: list[float] = []
    cross_means: list[float] = []
    for _, group in complete.groupby("timestamp", sort=True):
        if len(group) >= 3:
            value = group[PREDICTION_COLUMN].corr(
                group["realized_forward_risk_adjusted_utility"], method="spearman"
            )
            if pd.notna(value):
                daily_rank.append(float(value))
        ordered = group.sort_values(
            [PREDICTION_COLUMN, "symbol"], ascending=[False, True]
        )
        top1_realized.append(
            float(ordered.iloc[0]["realized_forward_risk_adjusted_utility"])
        )
        cross_means.append(
            float(group["realized_forward_risk_adjusted_utility"].mean())
        )

    top_mean = float(np.mean(top1_realized)) if top1_realized else None
    cross_mean = float(np.mean(cross_means)) if cross_means else None
    return {
        "scored_rows": int(len(frame)),
        "completed_rows": int(len(complete)),
        "global_pearson": float(pearson) if pd.notna(pearson) else None,
        "global_spearman": float(spearman) if pd.notna(spearman) else None,
        "mean_daily_cross_section_spearman": (
            float(np.mean(daily_rank)) if daily_rank else None
        ),
        "positive_daily_cross_section_spearman_rate": (
            float(np.mean(np.asarray(daily_rank) > 0.0)) if daily_rank else None
        ),
        "top1_realized_utility_mean": top_mean,
        "cross_section_realized_utility_mean": cross_mean,
        "top1_minus_cross_section_mean": (
            float(top_mean - cross_mean)
            if top_mean is not None and cross_mean is not None
            else None
        ),
    }


def choose_candidate_overrides_on_common_scale(
    candidate_scored: pd.DataFrame,
    baseline_action_scores: pd.DataFrame,
) -> pd.DataFrame:
    required_candidate = {
        "timestamp",
        "symbol",
        PREDICTION_COLUMN,
        "realized_forward_risk_adjusted_utility",
    }
    required_baseline = {
        "timestamp",
        "baseline_target_asset",
        "baseline_common_scale_predicted_utility",
        "baseline_realized_forward_risk_adjusted_utility",
    }
    missing_candidate = sorted(required_candidate.difference(candidate_scored.columns))
    missing_baseline = sorted(required_baseline.difference(baseline_action_scores.columns))
    if missing_candidate or missing_baseline:
        raise RuntimeError(
            "Common-scale override inputs are incomplete: "
            f"candidate_missing={missing_candidate}, baseline_missing={missing_baseline}"
        )

    candidates = candidate_scored.copy()
    candidates["timestamp"] = pd.to_datetime(
        candidates["timestamp"], utc=True, format="mixed", errors="raise"
    )
    baseline = baseline_action_scores.copy()
    baseline["timestamp"] = pd.to_datetime(
        baseline["timestamp"], utc=True, format="mixed", errors="raise"
    )
    baseline_by_time = baseline.set_index("timestamp", drop=False)

    rows: list[dict[str, Any]] = []
    for timestamp, group in candidates.groupby("timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        if timestamp not in baseline_by_time.index:
            continue
        base = baseline_by_time.loc[timestamp]
        if isinstance(base, pd.DataFrame):
            raise RuntimeError("Duplicate baseline action score for one timestamp.")
        ordered = group.sort_values(
            [PREDICTION_COLUMN, "symbol"], ascending=[False, True]
        )
        best = ordered.iloc[0]
        candidate_pred = float(best[PREDICTION_COLUMN])
        baseline_pred = float(base["baseline_common_scale_predicted_utility"])
        predicted_advantage = float(candidate_pred - baseline_pred)
        candidate_realized = pd.to_numeric(
            pd.Series([best["realized_forward_risk_adjusted_utility"]]),
            errors="coerce",
        ).iloc[0]
        baseline_realized = pd.to_numeric(
            pd.Series([base["baseline_realized_forward_risk_adjusted_utility"]]),
            errors="coerce",
        ).iloc[0]
        realized_advantage = (
            float(candidate_realized - baseline_realized)
            if pd.notna(candidate_realized) and pd.notna(baseline_realized)
            else np.nan
        )
        override = bool(np.isfinite(predicted_advantage) and predicted_advantage > 0.0)
        rows.append(
            {
                "timestamp": timestamp,
                "baseline_target_asset": str(base["baseline_target_asset"]),
                "best_candidate": str(best["symbol"]),
                "baseline_common_scale_predicted_utility": baseline_pred,
                "candidate_common_scale_predicted_utility": candidate_pred,
                "predicted_common_scale_advantage": predicted_advantage,
                "override_baseline": override,
                "baseline_realized_forward_risk_adjusted_utility": (
                    float(baseline_realized) if pd.notna(baseline_realized) else np.nan
                ),
                "candidate_realized_forward_risk_adjusted_utility": (
                    float(candidate_realized) if pd.notna(candidate_realized) else np.nan
                ),
                "realized_common_scale_utility_advantage": realized_advantage,
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def summarize_common_scale_overrides(decisions: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "decision_dates": 0,
            "override_count": 0,
            "override_rate": 0.0,
            "completed_override_count": 0,
            "realized_utility_advantage_sum": 0.0,
            "realized_utility_advantage_mean": None,
            "positive_realized_advantage_rate": None,
            "is_portfolio_return": False,
        }
    chosen = decisions.loc[decisions["override_baseline"].astype(bool)].copy()
    realized = pd.to_numeric(
        chosen["realized_common_scale_utility_advantage"], errors="coerce"
    )
    finite = realized.loc[np.isfinite(realized.to_numpy(dtype=float))]
    return {
        "decision_dates": int(len(decisions)),
        "override_count": int(len(chosen)),
        "override_rate": float(len(chosen) / len(decisions)),
        "completed_override_count": int(len(finite)),
        "realized_utility_advantage_sum": float(finite.sum()),
        "realized_utility_advantage_mean": (
            float(finite.mean()) if not finite.empty else None
        ),
        "positive_realized_advantage_rate": (
            float((finite > 0.0).mean()) if not finite.empty else None
        ),
        "is_portfolio_return": False,
    }
