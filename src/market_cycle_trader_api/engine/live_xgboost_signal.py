from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .capital_rotation import (
    _fit_xgb_models,
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


def build_live_xgboost_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
) -> LiveXGBoostDecision:
    """Train the validated live fold and decide the next-open target.

    This mirrors the backtest's expanding walk-forward convention for a decision
    made after the latest completed daily close:

    * a purge separates training labels from calibration;
    * a second purge separates final fitting from the live decision;
    * the model reads features from the latest completed session;
    * any position change is intended for the next regular-session open.
    """

    if config.strategy_mode != "COMPOUND_ROTATION_SWING_XGBOOST":
        raise ValueError("Paper execution supports only COMPOUND_ROTATION_SWING_XGBOOST.")
    if list(config.rotation_models) != ["xgboost_utility"]:
        raise ValueError("Paper execution requires rotation_models=['xgboost_utility'].")
    if config.timeframe != "1Day":
        raise ValueError("Paper execution requires 1Day market data.")

    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    if len(common_dates) < 2:
        raise ValueError("At least two aligned completed sessions are required.")

    purge = max(
        int(config.rotation_purge_days),
        max(int(item) for item in config.rotation_target_horizons),
    )
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    minimum_training_rows = int(config.rotation_minimum_training_rows)

    # A live decision occurs after common_dates[-1], so the unseen execution
    # session would start at index len(common_dates).
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

    compute_plan = resolve_xgboost_compute_plan(config)
    effective_device = compute_plan.selected
    fallback_reasons: list[str] = []

    calibration_models, effective_device, fallback_reason = _fit_xgb_models(
        frames,
        symbols,
        train_dates,
        config,
        effective_device,
    )
    if fallback_reason:
        fallback_reasons.append(fallback_reason)

    candidates = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    if not candidates:
        raise ValueError("rotation_switch_margin_candidates cannot be empty.")

    best_candidate = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        calibration_policy = _xgb_policy(
            calibration_models,
            frames,
            symbols,
            config,
            candidate,
        )
        score = _simple_policy_growth(
            calibration_policy,
            frames,
            symbols,
            calibration_dates,
            config,
        )
        if score > best_score:
            best_score = float(score)
            best_candidate = float(candidate)

    final_models, effective_device, fallback_reason = _fit_xgb_models(
        frames,
        symbols,
        final_fit_dates,
        config,
        effective_device,
    )
    if fallback_reason:
        fallback_reasons.append(fallback_reason)

    effective_margin = max(
        float(config.rotation_switch_margin),
        float(best_candidate),
    )

    current_label = str(current_asset or "CASH").strip().upper()
    labels = ["CASH", *symbols]
    if current_label not in labels:
        raise ValueError(
            f"Managed asset {current_label!r} is not present in the locked asset universe."
        )
    current_position = labels.index(current_label)

    utilities_array = _xgb_utilities(
        final_models,
        frames,
        symbols,
        decision_date,
        config,
    )
    if not np.isfinite(utilities_array).all():
        raise ValueError("The live XGBoost utilities contain non-finite values.")

    policy = _xgb_policy(
        final_models,
        frames,
        symbols,
        config,
        effective_margin,
    )
    target_position, selected_utility = policy(
        decision_date,
        current_position,
        int(holding_sessions),
    )
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
        effective_switch_margin=float(effective_margin),
        calibrated_candidate_margin=float(best_candidate),
        calibration_score=float(best_score),
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
    )
