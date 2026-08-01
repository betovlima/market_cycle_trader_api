from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .capital_rotation import (
    _fit_xgb_models,
    _majority_vote_policy,
    _simple_policy_growth,
    _xgb_policy,
    _xgb_utilities,
    prepare_rotation_panel,
    resolve_xgboost_compute_plan,
)


@dataclass(frozen=True)
class LiveXGBoostDecision:
    decision_date: pd.Timestamp
    current_asset: str
    target_asset: str
    raw_best_asset: str
    selected_utility: float
    utilities: dict[str, float]
    effective_switch_margin: float
    calibrated_candidate_margin: float
    calibration_score: float
    training_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    final_fit_end: pd.Timestamp
    effective_compute_device: str
    compute_fallback_reason: str | None
    random_state: int
    ensemble_enabled: bool
    ensemble_seeds: list[int]
    ensemble_agreement: float | None


def build_live_xgboost_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
) -> LiveXGBoostDecision:








    if config.timeframe != "1Day":
        raise ValueError("Paper execution requires 1Day market data.")

    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    if len(common_dates) < 2:
        raise ValueError("At least two aligned completed sessions are required.")

    purge = max(int(config.rotation_purge_days), int(config.rotation_horizon_days))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    minimum_training_rows = int(config.rotation_minimum_training_rows)

    live_test_start = len(common_dates)
    calibration_end_index = live_test_start - purge
    calibration_start_index = calibration_end_index - calibration_days
    train_end_index = calibration_start_index - purge
    final_fit_end_index = live_test_start - purge

    if train_end_index < minimum_training_rows:
        raise ValueError(
            "Not enough completed history for the live XGBoost fold: "
            f"training_rows={train_end_index}, required={minimum_training_rows}, "
            f"calibration={calibration_days}, purge={purge}."
        )
    if calibration_start_index < 0 or calibration_end_index <= calibration_start_index:
        raise ValueError("The live calibration window is invalid.")
    if final_fit_end_index <= 0:
        raise ValueError("The live final-fit window is invalid.")

    train_dates = common_dates[:train_end_index]
    calibration_dates = common_dates[calibration_start_index:calibration_end_index]
    final_fit_dates = common_dates[:final_fit_end_index]
    decision_date = pd.Timestamp(common_dates[-1])

    current_label = str(current_asset or "CASH").strip().upper()
    labels = ["CASH", *symbols]
    if current_label not in labels:
        raise ValueError(
            f"Managed asset {current_label!r} is not present in the locked asset universe."
        )
    current_position = labels.index(current_label)

    compute_plan = resolve_xgboost_compute_plan(config)
    effective_device = compute_plan.selected
    fallback_reasons: list[str] = []
    policies = []
    seed_utilities: list[np.ndarray] = []
    seed_targets: list[int] = []
    seed_selected_scores: list[float] = []
    candidate_margins: list[float] = []
    calibration_scores: list[float] = []
    effective_margins: list[float] = []
    seeds = list(config.ensemble_seeds)

    for seed in seeds:
        rep_config = config.model_copy(update={"random_state": int(seed)})
        calibration_models, effective_device, fallback_reason = _fit_xgb_models(
            frames,
            symbols,
            train_dates,
            rep_config,
            effective_device,
        )
        if fallback_reason:
            fallback_reasons.append(fallback_reason)

        candidates = tuple(float(value) for value in rep_config.rotation_switch_margin_candidates)
        if not candidates:
            raise ValueError("rotation_switch_margin_candidates cannot be empty.")

        best_candidate = candidates[0]
        best_score = float("-inf")
        for candidate in candidates:
            calibration_policy = _xgb_policy(
                calibration_models,
                frames,
                symbols,
                rep_config,
                candidate,
            )
            score = _simple_policy_growth(
                calibration_policy,
                frames,
                symbols,
                calibration_dates,
                rep_config,
            )
            if score > best_score:
                best_score = float(score)
                best_candidate = float(candidate)

        final_models, effective_device, fallback_reason = _fit_xgb_models(
            frames,
            symbols,
            final_fit_dates,
            rep_config,
            effective_device,
        )
        if fallback_reason:
            fallback_reasons.append(fallback_reason)

        effective_margin = max(
            float(rep_config.rotation_switch_margin),
            float(best_candidate),
        )
        policy = _xgb_policy(
            final_models,
            frames,
            symbols,
            rep_config,
            effective_margin,
        )
        utilities_array = _xgb_utilities(
            final_models,
            frames,
            symbols,
            decision_date,
            rep_config,
        )
        if not np.isfinite(utilities_array).all():
            raise ValueError("The live XGBoost utilities contain non-finite values.")

        target_position, selected_utility = policy(
            decision_date,
            current_position,
            int(holding_sessions),
        )
        policies.append(policy)
        seed_utilities.append(utilities_array.astype(float))
        seed_targets.append(int(target_position))
        seed_selected_scores.append(float(selected_utility))
        candidate_margins.append(float(best_candidate))
        calibration_scores.append(float(best_score))
        effective_margins.append(float(effective_margin))

    ensemble_enabled = len(policies) > 1
    ensemble_minimum_agreement = (len(policies) // 2 + 1) / len(policies)
    if ensemble_enabled:
        combined_policy = _majority_vote_policy(
            policies,
            minimum_agreement=ensemble_minimum_agreement,
        )
        target_position, selected_utility = combined_policy(
            decision_date,
            current_position,
            int(holding_sessions),
        )
        vote_count = sum(position == int(target_position) for position in seed_targets)
        agreement = vote_count / len(seed_targets)
        utilities_array = np.median(np.vstack(seed_utilities), axis=0)
    else:
        target_position = seed_targets[0]
        selected_utility = seed_selected_scores[0]
        agreement = 1.0
        utilities_array = seed_utilities[0]

    raw_best_position = int(np.nanargmax(utilities_array))
    utilities = {
        label: float(utilities_array[index])
        for index, label in enumerate(labels)
    }

    return LiveXGBoostDecision(
        decision_date=decision_date,
        current_asset=current_label,
        target_asset=labels[int(target_position)],
        raw_best_asset=labels[raw_best_position],
        selected_utility=float(selected_utility),
        utilities=utilities,
        effective_switch_margin=float(np.median(effective_margins)),
        calibrated_candidate_margin=float(np.median(candidate_margins)),
        calibration_score=float(np.median(calibration_scores)),
        training_end=pd.Timestamp(train_dates[-1]),
        calibration_start=pd.Timestamp(calibration_dates[0]),
        calibration_end=pd.Timestamp(calibration_dates[-1]),
        final_fit_end=pd.Timestamp(final_fit_dates[-1]),
        effective_compute_device=str(effective_device),
        compute_fallback_reason=(
            fallback_reasons[-1]
            if fallback_reasons
            else compute_plan.fallback_reason
        ),
        random_state=int(config.random_state),
        ensemble_enabled=ensemble_enabled,
        ensemble_seeds=[int(seed) for seed in seeds],
        ensemble_agreement=float(agreement),
    )
