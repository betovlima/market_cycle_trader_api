from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine.capital_rotation import _model_utilities, _training_transition_log_return
from .optimal_switching_state import (
    ACTION_STATE_FEATURES,
    action_features,
    market_context,
    ranked_positions,
)


@dataclass(frozen=True)
class CounterfactualRotationAdvantageV4Settings:
    enabled: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "CounterfactualRotationAdvantageV4Settings":
        research = getattr(config, "research_model_settings", {}) or {}
        raw = (
            research.get("counterfactual_rotation_advantage_v4", {})
            if isinstance(research, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        return cls(enabled=bool(raw.get("enabled", False)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "algorithm": "supervised_counterfactual_rotation_advantage_v4",
            "action_space": "HOLD incumbent reference vs ROTATE current LightGBM Top-1",
            "target": (
                "persistent relative net log-return advantage of ROTATE versus HOLD "
                "over the shortest frozen Strategy target horizon"
            ),
            "label_horizon_source": "min(rotation_target_horizons)",
            "decision_rule": (
                "ROTATE when predicted economic advantage is positive, otherwise HOLD"
            ),
            "cash_policy": (
                "not part of this experiment; initial CASH enters current Top-1"
            ),
            "manual_switch_margin": False,
            "manual_cash_threshold": False,
            "manual_minimum_hold_guard": False,
        }


@dataclass
class FittedCounterfactualRotationAdvantageV4Model:
    model: Any
    settings: CounterfactualRotationAdvantageV4Settings
    sample_count: int
    decision_date_count: int
    calibration_start: str
    calibration_end: str
    label_horizon_sessions: int
    purged_tail_sessions: int
    target_mean: float
    target_std: float
    target_positive_fraction: float
    fit_mae: float
    fit_rmse: float

    def metadata(self) -> dict[str, Any]:
        return {
            **self.settings.as_dict(),
            "sample_count": int(self.sample_count),
            "decision_date_count": int(self.decision_date_count),
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "label_horizon_sessions": int(self.label_horizon_sessions),
            "purged_tail_sessions": int(self.purged_tail_sessions),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "target_positive_fraction": float(self.target_positive_fraction),
            "fit_mae": float(self.fit_mae),
            "fit_rmse": float(self.fit_rmse),
            "feature_count": int(len(ACTION_STATE_FEATURES)),
        }


def _label_horizon(config: Any) -> int:
    horizons = [
        int(value)
        for value in list(getattr(config, "rotation_target_horizons", []) or [])
        if int(value) > 0
    ]
    if not horizons:
        raise ValueError(
            "Counterfactual Rotation Advantage v4 requires rotation_target_horizons."
        )
    return min(horizons)


def _regressor(model_settings: dict[str, Any], config: Any) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Counterfactual Rotation Advantage v4 requires lightgbm. "
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


def _persistent_rotation_advantage(
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    decision_index: int,
    current_position: int,
    target_position: int,
    horizon_sessions: int,
    config: Any,
) -> float:
    """Relative value of switching now versus keeping the incumbent.

    Both counterfactual paths are measured over the same frozen Strategy horizon.
    The ROTATE path pays the switch cost only on the first transition. The HOLD
    path keeps the incumbent throughout the label window. The live policy is
    still re-evaluated every session; this horizon is a persistence target, not
    a minimum-holding rule.
    """
    if (
        current_position <= 0
        or target_position <= 0
        or current_position == target_position
        or decision_index + horizon_sessions >= len(dates)
    ):
        return float("nan")

    hold_total = 0.0
    rotate_total = 0.0
    for step in range(1, horizon_sessions + 1):
        date_now = pd.Timestamp(dates[decision_index + step - 1])
        date_next = pd.Timestamp(dates[decision_index + step])
        try:
            hold_return = _training_transition_log_return(
                frames,
                symbols,
                date_now,
                date_next,
                current_position,
                current_position,
                config,
            )
            rotate_from = current_position if step == 1 else target_position
            rotate_return = _training_transition_log_return(
                frames,
                symbols,
                date_now,
                date_next,
                rotate_from,
                target_position,
                config,
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return float("nan")
        if not (np.isfinite(hold_return) and np.isfinite(rotate_return)):
            return float("nan")
        hold_total += float(hold_return)
        rotate_total += float(rotate_return)
    return float(rotate_total - hold_total)


def fit_counterfactual_rotation_advantage_v4(
    *,
    utility_models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    config: Any,
    model_settings: dict[str, Any],
    technical_log_callback: Callable[[str], None] | None = None,
) -> FittedCounterfactualRotationAdvantageV4Model:
    settings = CounterfactualRotationAdvantageV4Settings.from_config(config)
    if not settings.enabled:
        raise ValueError(
            "Counterfactual Rotation Advantage v4 research is not enabled."
        )

    dates = pd.DatetimeIndex(calibration_dates).sort_values()
    horizon_sessions = _label_horizon(config)
    usable_decision_count = len(dates) - horizon_sessions
    if usable_decision_count < 10:
        raise ValueError(
            "Counterfactual Rotation Advantage v4 needs at least ten calibration "
            "decisions after purging its persistence label horizon."
        )

    def log(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    started = time.perf_counter()
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    sample_weights: list[float] = []
    reachable: set[int] = set()
    used_decision_dates = 0

    log(
        "model=counterfactual_rotation_advantage_v4 event=dataset_start "
        f"sessions={len(dates)} usable_sessions={usable_decision_count} "
        f"label_horizon_sessions={horizon_sessions} "
        f"purged_tail_sessions={horizon_sessions}"
    )

    for decision_index in range(usable_decision_count):
        now = pd.Timestamp(dates[decision_index])
        utilities = _model_utilities(utility_models, frames, symbols, now, config)
        context = market_context(frames, symbols, now, utilities)
        ranked = ranked_positions(utilities)
        if not ranked:
            continue
        best = int(ranked[0])

        date_rows: list[tuple[dict[str, float], float]] = []
        for current_position in sorted(reachable):
            if current_position <= 0 or current_position == best:
                continue
            advantage = _persistent_rotation_advantage(
                frames=frames,
                symbols=symbols,
                dates=dates,
                decision_index=decision_index,
                current_position=current_position,
                target_position=best,
                horizon_sessions=horizon_sessions,
                config=config,
            )
            if not np.isfinite(advantage):
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
                        target_position=best,
                        config=config,
                    ),
                    float(advantage),
                )
            )

        if date_rows:
            used_decision_dates += 1
            equal_date_weight = 1.0 / float(len(date_rows))
            for features, advantage in date_rows:
                rows.append(features)
                targets.append(advantage)
                sample_weights.append(equal_date_weight)

        reachable.add(best)

    if not rows:
        raise RuntimeError(
            "Counterfactual Rotation Advantage v4 produced no calibration samples."
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
    target_std = float(np.sqrt(np.average(np.square(y - target_mean), weights=w)))
    target_positive_fraction = float(np.average((y > 0.0).astype(float), weights=w))

    log(
        "model=counterfactual_rotation_advantage_v4 event=fit_complete "
        f"samples={len(x)} decision_dates={used_decision_dates} "
        f"label_horizon_sessions={horizon_sessions} "
        f"duration_seconds={time.perf_counter() - started:.3f} "
        f"target_mean={target_mean:.8f} "
        f"positive_fraction={target_positive_fraction:.4f} "
        f"fit_mae={fit_mae:.8f} fit_rmse={fit_rmse:.8f}"
    )

    return FittedCounterfactualRotationAdvantageV4Model(
        model=model,
        settings=settings,
        sample_count=int(len(x)),
        decision_date_count=int(used_decision_dates),
        calibration_start=pd.Timestamp(dates[0]).date().isoformat(),
        calibration_end=pd.Timestamp(
            dates[usable_decision_count - 1]
        ).date().isoformat(),
        label_horizon_sessions=int(horizon_sessions),
        purged_tail_sessions=int(horizon_sessions),
        target_mean=target_mean,
        target_std=target_std,
        target_positive_fraction=target_positive_fraction,
        fit_mae=fit_mae,
        fit_rmse=fit_rmse,
    )


def build_counterfactual_rotation_advantage_v4_policy(
    *,
    fitted_model: FittedCounterfactualRotationAdvantageV4Model,
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
        best = int(ranked[0]) if ranked else 0
        best_score = (
            float(utilities[best])
            if best > 0 and np.isfinite(utilities[best])
            else 0.0
        )
        current_score = (
            float(utilities[current_position])
            if current_position > 0 and np.isfinite(utilities[current_position])
            else 0.0
        )

        rotation_advantage: float | None = None
        if current_position <= 0:
            selected = best
            reason = "CRA4_ENTER_TOP1"
        elif best <= 0 or best == current_position:
            selected = current_position
            reason = "CRA4_HOLD_CURRENT_TOP1"
            rotation_advantage = 0.0
        else:
            matrix = pd.DataFrame(
                [
                    action_features(
                        frames=frames,
                        symbols=symbols,
                        timestamp=key,
                        utilities=utilities,
                        context=context,
                        current_position=current_position,
                        target_position=best,
                        config=config,
                    )
                ],
                columns=list(ACTION_STATE_FEATURES),
            ).fillna(0.0)
            predicted = float(fitted_model.model.predict(matrix)[0])
            rotation_advantage = predicted if np.isfinite(predicted) else None
            if rotation_advantage is not None and rotation_advantage > 0.0:
                selected = best
                reason = "CRA4_ROTATE_POSITIVE_PERSISTENT_ADVANTAGE"
            else:
                selected = current_position
                reason = "CRA4_HOLD_NONPOSITIVE_PERSISTENT_ADVANTAGE"

        selected_score = (
            float(utilities[selected])
            if selected > 0 and np.isfinite(utilities[selected])
            else 0.0
        )
        diagnostics[key] = {
            "decision_diagnostics_schema_version": 33,
            "decision_fold_id": int(fold_id),
            "decision_reason": reason,
            "counterfactual_rotation_advantage_v4_enabled": True,
            "current_asset": (
                symbols[current_position - 1]
                if current_position > 0
                else "CASH"
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
            "cra4_reference_asset": (
                symbols[current_position - 1]
                if current_position > 0
                else "CASH"
            ),
            "cra4_reference_advantage": 0.0,
            "cra4_rotation_candidate_asset": (
                symbols[best - 1] if best > 0 else None
            ),
            "cra4_rotation_advantage": rotation_advantage,
            "cra4_positive_advantage": bool(
                rotation_advantage is not None and rotation_advantage > 0.0
            ),
            "cra4_label_horizon_sessions": int(
                fitted_model.label_horizon_sessions
            ),
            "cra4_cash_action_enabled": False,
        }
        for index, position in enumerate(ranked[:3], start=1):
            diagnostics[key][f"top_{index}_asset"] = symbols[position - 1]
            diagnostics[key][f"top_{index}_score"] = float(utilities[position])
        return selected, selected_score

    return policy
