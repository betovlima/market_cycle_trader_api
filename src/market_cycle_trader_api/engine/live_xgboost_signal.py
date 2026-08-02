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


def build_live_xgboost_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
) -> LiveXGBoostDecision:
    if config.timeframe != "1Day":
        raise ValueError("Paper execution requires daily market data.")

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
        raise ValueError("Insufficient completed history for the active model.")
    if calibration_start_index < 0 or calibration_end_index <= calibration_start_index:
        raise ValueError("The calibration window is invalid.")
    if final_fit_end_index <= 0:
        raise ValueError("The final-fit window is invalid.")

    train_dates = common_dates[:train_end_index]
    calibration_dates = common_dates[calibration_start_index:calibration_end_index]
    final_fit_dates = common_dates[:final_fit_end_index]
    decision_date = pd.Timestamp(common_dates[-1])

    current_label = str(current_asset or "CASH").strip().upper()
    labels = ["CASH", *symbols]
    if current_label not in labels:
        raise ValueError("The managed position is outside the active universe.")
    current_position = labels.index(current_label)

    compute_plan = resolve_xgboost_compute_plan(config)
    calibration_models, effective_device, _ = _fit_xgb_models(
        frames,
        symbols,
        train_dates,
        config,
        compute_plan.selected,
    )

    candidates = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    if not candidates:
        raise ValueError("No calibration candidates are configured.")

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

    final_models, _, _ = _fit_xgb_models(
        frames,
        symbols,
        final_fit_dates,
        config,
        effective_device,
    )
    effective_margin = max(float(config.rotation_switch_margin), best_candidate)
    policy = _xgb_policy(
        final_models,
        frames,
        symbols,
        config,
        effective_margin,
    )
    utilities = _xgb_utilities(
        final_models,
        frames,
        symbols,
        decision_date,
        config,
    )
    if not np.isfinite(utilities).all():
        raise ValueError("The active model returned invalid values.")

    target_position, _ = policy(
        decision_date,
        current_position,
        int(holding_sessions),
    )
    return LiveXGBoostDecision(
        decision_date=decision_date,
        current_asset=current_label,
        target_asset=labels[int(target_position)],
    )
