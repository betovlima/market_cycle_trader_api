from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine.capital_rotation import (
    _model_utilities,
    _proportional_switch_cost,
    _training_transition_log_return,
)
from .optimal_switching_state import (
    ACTION_STATE_FEATURES,
    action_features,
    market_context,
    ranked_positions,
)


@dataclass(frozen=True)
class CounterfactualActionValueSettings:
    enabled: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "CounterfactualActionValueSettings":
        research = getattr(config, "research_model_settings", {}) or {}
        raw = (
            research.get("counterfactual_action_value", {})
            if isinstance(research, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        return cls(enabled=bool(raw.get("enabled", False)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "algorithm": "supervised_counterfactual_action_value",
            "action_space": "CASH + HOLD + current LightGBM Top-1",
            "target": "Strategy-weighted multi-horizon exact net log return",
            "decision_rule": "argmax predicted action value",
            "manual_switch_margin": False,
            "manual_cash_threshold": False,
            "manual_minimum_hold_guard": False,
        }


@dataclass
class FittedCounterfactualActionValueModel:
    model: Any
    settings: CounterfactualActionValueSettings
    sample_count: int
    decision_date_count: int
    calibration_start: str
    calibration_end: str
    label_horizons: list[int]
    label_weights: list[float]
    purged_tail_sessions: int
    target_mean: float
    target_std: float
    fit_mae: float
    fit_rmse: float

    def metadata(self) -> dict[str, Any]:
        return {
            **self.settings.as_dict(),
            "sample_count": int(self.sample_count),
            "decision_date_count": int(self.decision_date_count),
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "label_horizons": [int(value) for value in self.label_horizons],
            "label_weights": [float(value) for value in self.label_weights],
            "purged_tail_sessions": int(self.purged_tail_sessions),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "fit_mae": float(self.fit_mae),
            "fit_rmse": float(self.fit_rmse),
            "feature_count": int(len(ACTION_STATE_FEATURES)),
        }


def _regressor(model_settings: dict[str, Any], config: Any) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Counterfactual Action Value research requires lightgbm. "
            "Install requirements.txt."
        ) from exc

    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=int(model_settings["n_estimators"]),
        learning_rate=float(model_settings["learning_rate"]),
        max_depth=int(model_settings["max_depth"]),
        num_leaves=int(model_settings["num_leaves"]),
        min_child_samples=int(model_settings["min_child_samples"]),
        min_child_weight=float(model_settings["min_child_weight"]),
        subsample=float(model_settings["subsample"]),
        subsample_freq=int(model_settings["subsample_freq"]),
        colsample_bytree=float(model_settings["colsample_bytree"]),
        reg_alpha=float(model_settings["reg_alpha"]),
        reg_lambda=float(model_settings["reg_lambda"]),
        max_bin=int(model_settings["max_bin"]),
        random_state=int(config.random_state),
        n_jobs=int(model_settings["n_jobs"]),
        deterministic=bool(config.deterministic_execution),
        force_col_wise=bool(config.deterministic_execution),
        verbosity=-1,
    )


def _decision_targets(current_position: int, utilities: np.ndarray) -> list[int]:
    """Return CASH, incumbent, and the current LightGBM Top-1 without a Top-K rule."""
    values = [0]
    if current_position > 0:
        values.append(int(current_position))
    ranked = ranked_positions(utilities)
    if ranked:
        values.append(int(ranked[0]))
    return list(dict.fromkeys(values))


def _weighted_future_action_value(
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    decision_index: int,
    current_position: int,
    target_position: int,
    horizons: list[int],
    weights: np.ndarray,
    config: Any,
) -> float:
    max_horizon = max(horizons)
    if decision_index + max_horizon >= len(dates):
        return float("nan")

    horizon_weights = {
        int(horizon): float(weight)
        for horizon, weight in zip(horizons, weights, strict=True)
    }
    cumulative = 0.0
    weighted_value = 0.0
    for step in range(1, max_horizon + 1):
        date_now = pd.Timestamp(dates[decision_index + step - 1])
        date_next = pd.Timestamp(dates[decision_index + step])
        from_position = current_position if step == 1 else target_position
        try:
            reward = _training_transition_log_return(
                frames,
                symbols,
                date_now,
                date_next,
                from_position,
                target_position,
                config,
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return float("nan")
        if not np.isfinite(reward):
            return float("nan")
        cumulative += float(reward)
        weight = horizon_weights.get(step)
        if weight is not None:
            weighted_value += weight * cumulative
    return float(weighted_value)


def fit_counterfactual_action_value(
    *,
    utility_models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    config: Any,
    model_settings: dict[str, Any],
    technical_log_callback: Callable[[str], None] | None = None,
) -> FittedCounterfactualActionValueModel:
    settings = CounterfactualActionValueSettings.from_config(config)
    if not settings.enabled:
        raise ValueError("Counterfactual Action Value research is not enabled.")

    dates = pd.DatetimeIndex(calibration_dates).sort_values()
    horizons = [int(value) for value in config.rotation_target_horizons]
    weights = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    if not horizons or len(horizons) != len(weights):
        raise ValueError("Rotation target horizons and weights must have equal length.")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Rotation target horizon weights must be non-negative and non-zero.")
    weights = weights / float(weights.sum())
    max_horizon = max(horizons)
    usable_decision_count = len(dates) - max_horizon
    if usable_decision_count < 10:
        raise ValueError(
            "Counterfactual Action Value needs at least ten calibration decisions "
            "after purging the maximum target horizon."
        )

    def log(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    started = time.perf_counter()
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    sample_weights: list[float] = []
    reachable: set[int] = {0}
    used_decision_dates = 0

    log(
        "model=counterfactual_action_value event=dataset_start "
        f"sessions={len(dates)} usable_sessions={usable_decision_count} "
        f"purged_tail_sessions={max_horizon}"
    )

    for decision_index in range(usable_decision_count):
        now = pd.Timestamp(dates[decision_index])
        utilities = _model_utilities(utility_models, frames, symbols, now, config)
        context = market_context(frames, symbols, now, utilities)
        ranked = ranked_positions(utilities)
        current_states = sorted(reachable)
        date_rows: list[tuple[dict[str, float], float]] = []

        for current_position in current_states:
            for target_position in _decision_targets(current_position, utilities):
                value = _weighted_future_action_value(
                    frames=frames,
                    symbols=symbols,
                    dates=dates,
                    decision_index=decision_index,
                    current_position=current_position,
                    target_position=target_position,
                    horizons=horizons,
                    weights=weights,
                    config=config,
                )
                if not np.isfinite(value):
                    continue
                date_rows.append(
                    (
                        action_features(
                            frames=frames,
                            symbols=symbols,
                            timestamp=now,
                            utilities=utilities,
                            context=context,
                            current_position=current_position,
                            target_position=target_position,
                            config=config,
                        ),
                        float(value),
                    )
                )

        if date_rows:
            used_decision_dates += 1
            equal_date_weight = 1.0 / float(len(date_rows))
            for features, value in date_rows:
                rows.append(features)
                targets.append(value)
                sample_weights.append(equal_date_weight)

        if ranked:
            reachable.add(int(ranked[0]))

    if not rows:
        raise RuntimeError(
            "Counterfactual Action Value produced no calibration samples."
        )

    x = pd.DataFrame(rows, columns=list(ACTION_STATE_FEATURES)).fillna(0.0)
    y = np.asarray(targets, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    model = _regressor(model_settings, config)
    model.fit(x, y, sample_weight=w)
    predicted = np.asarray(model.predict(x), dtype=float)
    residual = predicted - y
    fit_mae = float(np.average(np.abs(residual), weights=w))
    fit_rmse = float(np.sqrt(np.average(np.square(residual), weights=w)))
    target_mean = float(np.average(y, weights=w))
    target_std = float(
        np.sqrt(np.average(np.square(y - target_mean), weights=w))
    )

    log(
        "model=counterfactual_action_value event=fit_complete "
        f"samples={len(x)} decision_dates={used_decision_dates} "
        f"duration_seconds={time.perf_counter() - started:.3f} "
        f"fit_mae={fit_mae:.8f} fit_rmse={fit_rmse:.8f}"
    )
    return FittedCounterfactualActionValueModel(
        model=model,
        settings=settings,
        sample_count=int(len(x)),
        decision_date_count=int(used_decision_dates),
        calibration_start=pd.Timestamp(dates[0]).date().isoformat(),
        calibration_end=pd.Timestamp(dates[usable_decision_count - 1]).date().isoformat(),
        label_horizons=horizons,
        label_weights=[float(value) for value in weights],
        purged_tail_sessions=int(max_horizon),
        target_mean=target_mean,
        target_std=target_std,
        fit_mae=fit_mae,
        fit_rmse=fit_rmse,
    )


def build_counterfactual_action_value_policy(
    *,
    fitted_model: FittedCounterfactualActionValueModel,
    utility_models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    fold_id: int,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        del holding_days
        key = pd.Timestamp(timestamp)
        utilities = _model_utilities(utility_models, frames, symbols, key, config)
        ranked = ranked_positions(utilities)
        context = market_context(frames, symbols, key, utilities)
        candidates = _decision_targets(current_position, utilities)
        matrix = pd.DataFrame(
            [
                action_features(
                    frames=frames,
                    symbols=symbols,
                    timestamp=key,
                    utilities=utilities,
                    context=context,
                    current_position=current_position,
                    target_position=target,
                    config=config,
                )
                for target in candidates
            ],
            columns=list(ACTION_STATE_FEATURES),
        ).fillna(0.0)
        predicted = np.asarray(fitted_model.model.predict(matrix), dtype=float)
        values_by_target = {
            target: (
                float(predicted[index])
                if np.isfinite(predicted[index])
                else float("-inf")
            )
            for index, target in enumerate(candidates)
        }
        ranked_choices = sorted(
            candidates,
            key=lambda target: (
                -values_by_target[target],
                _proportional_switch_cost(config, current_position, target),
                int(target != current_position),
                target,
            ),
        )
        selected = int(ranked_choices[0]) if ranked_choices else 0
        selected_value = values_by_target.get(selected, float("-inf"))
        selected_score = (
            float(utilities[selected])
            if selected > 0 and np.isfinite(utilities[selected])
            else 0.0
        )
        current_score = (
            float(utilities[current_position])
            if current_position > 0 and np.isfinite(utilities[current_position])
            else 0.0
        )
        best = int(ranked[0]) if ranked else 0
        best_score = (
            float(utilities[best])
            if best > 0 and np.isfinite(utilities[best])
            else 0.0
        )
        hold_value = values_by_target.get(current_position)
        cash_value = values_by_target.get(0)
        best_value = values_by_target.get(best)
        reason = (
            "CAV_CASH"
            if selected == 0
            else "CAV_HOLD"
            if selected == current_position
            else "CAV_ENTER"
            if current_position == 0
            else "CAV_ROTATE"
        )

        diagnostics[key] = {
            "decision_diagnostics_schema_version": 30,
            "decision_fold_id": int(fold_id),
            "decision_reason": reason,
            "counterfactual_action_value_enabled": True,
            "current_asset": (
                symbols[current_position - 1] if current_position > 0 else "CASH"
            ),
            "current_score": current_score,
            "current_asset_rank": (
                context["ranks"].get(current_position)
                if current_position > 0
                else None
            ),
            "best_asset": symbols[best - 1] if best > 0 else "CASH",
            "best_score": best_score,
            "best_vs_current_gap": best_score - current_score,
            "universe_score_mean": float(context["mean"]),
            "universe_score_std": float(context["std"]),
            "final_action_asset": (
                symbols[selected - 1] if selected > 0 else "CASH"
            ),
            "final_action_score": selected_score,
            "decision_is_rotation": bool(
                current_position > 0
                and selected > 0
                and selected != current_position
            ),
            "cav_selected_value": (
                selected_value if np.isfinite(selected_value) else None
            ),
            "cav_hold_value": (
                hold_value
                if hold_value is not None and np.isfinite(hold_value)
                else None
            ),
            "cav_cash_value": (
                cash_value
                if cash_value is not None and np.isfinite(cash_value)
                else None
            ),
            "cav_best_candidate_value": (
                best_value
                if best_value is not None and np.isfinite(best_value)
                else None
            ),
            "cav_selected_edge_vs_hold": (
                selected_value - hold_value
                if hold_value is not None
                and np.isfinite(selected_value)
                and np.isfinite(hold_value)
                else None
            ),
            "cav_selected_edge_vs_cash": (
                selected_value - cash_value
                if cash_value is not None
                and np.isfinite(selected_value)
                and np.isfinite(cash_value)
                else None
            ),
            "cav_action_count": int(len(candidates)),
        }
        for index, position in enumerate(ranked[:3], start=1):
            diagnostics[key][f"top_{index}_asset"] = symbols[position - 1]
            diagnostics[key][f"top_{index}_score"] = float(utilities[position])
        return selected, selected_score

    return policy
