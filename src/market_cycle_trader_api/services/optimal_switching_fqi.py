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
    _utility_policy,
)
from .optimal_switching_state import (
    ACTION_STATE_FEATURES,
    POSITION_HISTORY_FEATURES,
    action_features,
    candidate_targets,
    market_context,
    next_holding_days,
    position_history_features,
    ranked_positions,
)


@dataclass(frozen=True)
class OptimalSwitchingFQISettings:
    enabled: bool = False
    action_top_k: int = 5
    bellman_iterations: int = 20
    position_aware_state: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "OptimalSwitchingFQISettings":
        research = getattr(config, "research_model_settings", {}) or {}
        raw = (
            research.get("optimal_switching_fqi", {})
            if isinstance(research, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            action_top_k=max(1, int(raw.get("action_top_k", 5))),
            bellman_iterations=max(1, int(raw.get("bellman_iterations", 20))),
            position_aware_state=bool(raw.get("position_aware_state", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "algorithm": "fitted_q_iteration",
            "action_space": "CASH + HOLD + top-K LightGBM challengers",
            "action_top_k": int(self.action_top_k),
            "bellman_iterations": int(self.bellman_iterations),
            "position_aware_state": bool(self.position_aware_state),
            "position_state_features": list(POSITION_HISTORY_FEATURES),
            "training_state_source": (
                "causal baseline-policy trajectories over calibration"
            ),
            "reward": "exact next-session net log return with switching cost",
            "discount": "half-life derived from median Strategy target horizon",
        }


@dataclass
class FittedQSwitchingModel:
    model: Any
    settings: OptimalSwitchingFQISettings
    gamma: float
    half_life_sessions: float
    sample_count: int
    calibration_start: str
    calibration_end: str
    final_target_delta: float
    behavior_trajectory_count: int
    unique_position_state_count: int

    def metadata(self) -> dict[str, Any]:
        return {
            **self.settings.as_dict(),
            "gamma": float(self.gamma),
            "discount_half_life_sessions": float(self.half_life_sessions),
            "sample_count": int(self.sample_count),
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "final_target_delta": float(self.final_target_delta),
            "feature_count": int(len(ACTION_STATE_FEATURES)),
            "behavior_trajectory_count": int(self.behavior_trajectory_count),
            "unique_position_state_count": int(self.unique_position_state_count),
        }


def _regressor(model_settings: dict[str, Any], config: Any) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Optimal Switching FQI requires lightgbm. Install requirements.txt."
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


def _behavior_margins(config: Any) -> tuple[float, ...]:
    values = [float(getattr(config, "rotation_switch_margin", 0.0))]
    values.extend(
        float(value)
        for value in (
            getattr(config, "rotation_switch_margin_candidates", ()) or ()
        )
    )
    return tuple(dict.fromkeys(values))


def fit_optimal_switching_fqi(
    *,
    utility_models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    config: Any,
    model_settings: dict[str, Any],
    technical_log_callback: Callable[[str], None] | None = None,
) -> FittedQSwitchingModel:
    settings = OptimalSwitchingFQISettings.from_config(config)
    if not settings.enabled:
        raise ValueError("Optimal Switching FQI is not enabled.")

    dates = pd.DatetimeIndex(calibration_dates).sort_values()
    if len(dates) < 3:
        raise ValueError(
            "Optimal Switching FQI needs at least three calibration sessions."
        )

    def log(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    started = time.perf_counter()
    utilities_by_date: dict[pd.Timestamp, np.ndarray] = {}
    context_by_date: dict[pd.Timestamp, dict[str, Any]] = {}

    def utilities_for(timestamp: pd.Timestamp) -> np.ndarray:
        key = pd.Timestamp(timestamp)
        cached = utilities_by_date.get(key)
        if cached is None:
            cached = _model_utilities(
                utility_models, frames, symbols, key, config
            )
            utilities_by_date[key] = cached
        return cached

    def context_for(timestamp: pd.Timestamp) -> dict[str, Any]:
        key = pd.Timestamp(timestamp)
        cached = context_by_date.get(key)
        if cached is None:
            cached = market_context(
                frames, symbols, key, utilities_for(key)
            )
            context_by_date[key] = cached
        return cached

    for timestamp in dates:
        utilities_for(pd.Timestamp(timestamp))
        context_for(pd.Timestamp(timestamp))

    rows: list[dict[str, float]] = []
    rewards: list[float] = []
    next_states: list[tuple[pd.Timestamp, int, int] | None] = []
    seen_transitions: set[tuple[pd.Timestamp, int, int, int]] = set()
    unique_position_states: set[tuple[pd.Timestamp, int, int]] = set()

    margins = _behavior_margins(config)
    behavior_policies = [
        _utility_policy(
            utility_models,
            frames,
            symbols,
            config,
            margin,
        )
        for margin in margins
    ]

    for behavior_policy in behavior_policies:
        current_position = 0
        holding_days = 0

        for index in range(len(dates) - 1):
            now = pd.Timestamp(dates[index])
            nxt = pd.Timestamp(dates[index + 1])
            utilities = utilities_for(now)
            context = context_for(now)
            history = position_history_features(
                frames=frames,
                symbols=symbols,
                timestamp=now,
                current_position=current_position,
                holding_days=holding_days,
                utilities_for_timestamp=utilities_for,
            )
            unique_position_states.add(
                (now, int(current_position), int(holding_days))
            )

            for target_position in candidate_targets(
                current_position,
                utilities,
                settings.action_top_k,
            ):
                transition_key = (
                    now,
                    int(current_position),
                    int(holding_days),
                    int(target_position),
                )
                if transition_key in seen_transitions:
                    continue

                reward = _training_transition_log_return(
                    frames,
                    symbols,
                    now,
                    nxt,
                    current_position,
                    target_position,
                    config,
                )
                if not np.isfinite(reward):
                    continue

                rows.append(
                    action_features(
                        frames=frames,
                        symbols=symbols,
                        timestamp=now,
                        utilities=utilities,
                        context=context,
                        current_position=current_position,
                        target_position=target_position,
                        config=config,
                        position_history=history,
                    )
                )
                rewards.append(float(reward))
                next_holding = next_holding_days(
                    current_position,
                    target_position,
                    holding_days,
                )
                next_states.append(
                    None
                    if nxt == pd.Timestamp(dates[-1])
                    else (
                        nxt,
                        int(target_position),
                        int(next_holding),
                    )
                )
                seen_transitions.add(transition_key)

            selected_position, _ = behavior_policy(
                now,
                current_position,
                holding_days,
            )
            holding_days = next_holding_days(
                current_position,
                int(selected_position),
                holding_days,
            )
            current_position = int(selected_position)

    if not rows:
        raise RuntimeError(
            "Optimal Switching FQI produced no calibration transitions."
        )

    x = pd.DataFrame(
        rows,
        columns=list(ACTION_STATE_FEATURES),
    ).fillna(0.0)
    reward_array = np.asarray(rewards, dtype=float)
    half_life = max(
        1.0,
        float(
            np.median(
                [int(value) for value in config.rotation_target_horizons]
            )
        ),
    )
    gamma = float(0.5 ** (1.0 / half_life))
    matrix_cache: dict[
        tuple[pd.Timestamp, int, int], pd.DataFrame
    ] = {}

    def next_matrix(
        state: tuple[pd.Timestamp, int, int]
    ) -> pd.DataFrame:
        if state in matrix_cache:
            return matrix_cache[state]
        timestamp, current_position, holding_days = state
        utilities = utilities_for(timestamp)
        context = context_for(timestamp)
        history = position_history_features(
            frames=frames,
            symbols=symbols,
            timestamp=timestamp,
            current_position=current_position,
            holding_days=holding_days,
            utilities_for_timestamp=utilities_for,
        )
        matrix = pd.DataFrame(
            [
                action_features(
                    frames=frames,
                    symbols=symbols,
                    timestamp=timestamp,
                    utilities=utilities,
                    context=context,
                    current_position=current_position,
                    target_position=target,
                    config=config,
                    position_history=history,
                )
                for target in candidate_targets(
                    current_position,
                    utilities,
                    settings.action_top_k,
                )
            ],
            columns=list(ACTION_STATE_FEATURES),
        ).fillna(0.0)
        matrix_cache[state] = matrix
        return matrix

    targets = reward_array.copy()
    final_delta = float("nan")
    model = None
    log(
        "model=fqi event=fit_start "
        f"samples={len(x)} sessions={len(dates)} "
        f"position_states={len(unique_position_states)} "
        f"behavior_trajectories={len(behavior_policies)} "
        f"action_top_k={settings.action_top_k} gamma={gamma:.6f} "
        f"iterations={settings.bellman_iterations} "
        "state=position_aware"
    )

    for iteration in range(1, settings.bellman_iterations + 1):
        model = _regressor(model_settings, config)
        model.fit(x, targets)

        state_values: dict[
            tuple[pd.Timestamp, int, int], float
        ] = {}
        next_values = np.zeros(len(next_states), dtype=float)
        for row_index, state in enumerate(next_states):
            if state is None:
                continue
            if state not in state_values:
                q_values = np.asarray(
                    model.predict(next_matrix(state)),
                    dtype=float,
                )
                state_values[state] = (
                    float(np.nanmax(q_values))
                    if np.isfinite(q_values).any()
                    else 0.0
                )
            next_values[row_index] = state_values[state]

        updated = reward_array + gamma * next_values
        final_delta = float(
            np.mean(np.abs(updated - targets))
        )
        targets = updated
        if (
            iteration == 1
            or iteration % 5 == 0
            or iteration == settings.bellman_iterations
        ):
            log(
                "model=fqi event=bellman_backup "
                f"iteration={iteration}/{settings.bellman_iterations} "
                f"mean_abs_target_delta={final_delta:.8f}"
            )

    model = _regressor(model_settings, config)
    model.fit(x, targets)
    log(
        "model=fqi event=fit_complete "
        f"samples={len(x)} "
        f"position_states={len(unique_position_states)} "
        f"duration_seconds={time.perf_counter() - started:.3f} "
        f"final_target_delta={final_delta:.8f}"
    )
    return FittedQSwitchingModel(
        model=model,
        settings=settings,
        gamma=gamma,
        half_life_sessions=half_life,
        sample_count=int(len(x)),
        calibration_start=pd.Timestamp(
            dates[0]
        ).date().isoformat(),
        calibration_end=pd.Timestamp(
            dates[-1]
        ).date().isoformat(),
        final_target_delta=float(final_delta),
        behavior_trajectory_count=int(len(behavior_policies)),
        unique_position_state_count=int(len(unique_position_states)),
    )


def build_optimal_switching_policy(
    *,
    fitted_q: FittedQSwitchingModel,
    utility_models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    fold_id: int,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    settings = fitted_q.settings
    utilities_cache: dict[pd.Timestamp, np.ndarray] = {}
    context_cache: dict[pd.Timestamp, dict[str, Any]] = {}

    def utilities_for(timestamp: pd.Timestamp) -> np.ndarray:
        key = pd.Timestamp(timestamp)
        cached = utilities_cache.get(key)
        if cached is None:
            cached = _model_utilities(
                utility_models,
                frames,
                symbols,
                key,
                config,
            )
            utilities_cache[key] = cached
        return cached

    def context_for(timestamp: pd.Timestamp) -> dict[str, Any]:
        key = pd.Timestamp(timestamp)
        cached = context_cache.get(key)
        if cached is None:
            cached = market_context(
                frames,
                symbols,
                key,
                utilities_for(key),
            )
            context_cache[key] = cached
        return cached

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        utilities = utilities_for(key)
        ranked = ranked_positions(utilities)
        context = context_for(key)
        history = position_history_features(
            frames=frames,
            symbols=symbols,
            timestamp=key,
            current_position=current_position,
            holding_days=holding_days,
            utilities_for_timestamp=utilities_for,
        )
        current_score = (
            float(utilities[current_position])
            if (
                current_position > 0
                and np.isfinite(utilities[current_position])
            )
            else 0.0
        )

        if (
            current_position > 0
            and holding_days < int(config.rotation_min_holding_days)
        ):
            selected = current_position
            selected_q = None
            q_by_target: dict[int, float | None] = {
                selected: None
            }
            reason = "FQI_MIN_HOLD_GUARD"
            action = "HOLD"
        else:
            candidates = candidate_targets(
                current_position,
                utilities,
                settings.action_top_k,
            )
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
                        position_history=history,
                    )
                    for target in candidates
                ],
                columns=list(ACTION_STATE_FEATURES),
            ).fillna(0.0)
            q_values = np.asarray(
                fitted_q.model.predict(matrix),
                dtype=float,
            )
            q_by_target = {
                target: (
                    float(q_values[index])
                    if np.isfinite(q_values[index])
                    else None
                )
                for index, target in enumerate(candidates)
            }
            ranked_choices = sorted(
                candidates,
                key=lambda target: (
                    -(
                        q_by_target[target]
                        if q_by_target[target] is not None
                        else float("-inf")
                    ),
                    _proportional_switch_cost(
                        config,
                        current_position,
                        target,
                    ),
                    int(target != current_position),
                    target,
                ),
            )
            selected = (
                int(ranked_choices[0])
                if ranked_choices
                else 0
            )
            selected_q = q_by_target.get(selected)
            if selected == 0:
                reason = "FQI_CASH"
                action = "CASH"
            elif selected == current_position:
                reason = "FQI_HOLD"
                action = "HOLD"
            elif current_position == 0:
                reason = "FQI_ENTER"
                action = "ENTER"
            else:
                reason = "FQI_ROTATE"
                action = "ROTATE"

        selected_score = (
            float(utilities[selected])
            if selected > 0 and np.isfinite(utilities[selected])
            else 0.0
        )
        best = ranked[0] if ranked else 0
        best_score = (
            float(utilities[best])
            if best > 0
            else 0.0
        )
        switch_values = [
            q
            for target, q in q_by_target.items()
            if (
                target not in {0, current_position}
                and q is not None
            )
        ]
        diagnostic = {
            "decision_diagnostics_schema_version": 21,
            "decision_fold_id": int(fold_id),
            "decision_reason": reason,
            "fqi_selected_action": action,
            "optimal_switching_fqi_enabled": True,
            "optimal_switching_position_aware": True,
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
            "best_asset": (
                symbols[best - 1]
                if best > 0
                else "CASH"
            ),
            "best_score": best_score,
            "best_vs_current_gap": best_score - current_score,
            "universe_score_mean": float(context["mean"]),
            "universe_score_std": float(context["std"]),
            "final_action_asset": (
                symbols[selected - 1]
                if selected > 0
                else "CASH"
            ),
            "final_action_score": selected_score,
            "decision_is_rotation": bool(
                current_position > 0
                and selected > 0
                and selected != current_position
            ),
            "fqi_selected_q": selected_q,
            "fqi_hold_q": q_by_target.get(current_position),
            "fqi_cash_q": q_by_target.get(0),
            "fqi_best_switch_q": (
                max(switch_values)
                if switch_values
                else None
            ),
            "fqi_action_count": int(len(q_by_target)),
            "fqi_gamma": float(fitted_q.gamma),
            **{
                f"fqi_state_{name}": float(history[name])
                for name in POSITION_HISTORY_FEATURES
            },
        }
        for index, position in enumerate(
            ranked[:3],
            start=1,
        ):
            diagnostic[f"top_{index}_asset"] = (
                symbols[position - 1]
            )
            diagnostic[f"top_{index}_score"] = float(
                utilities[position]
            )
        diagnostics[key] = diagnostic
        return selected, selected_score

    return policy
