from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import RESEARCH_ONLY_SWING_STRATEGY_MODES

from .capital_rotation import (
    SUPPORTED_ROTATION_MODES,
    _fit_xgb_models,
    _risk_off_enabled,
    _simple_policy_growth,
    _xgb_policy,
    _xgb_utilities,
    prepare_rotation_panel,
    resolve_xgboost_compute_plan,
)
from .live_policy import build_live_rotation_policy, live_model_utilities
from .selective_opportunity import evaluate_opportunity, fit_selective_opportunity_gate, selective_opportunity_enabled


@dataclass(frozen=True)
class LiveXGBoostDecision:
    decision_date: pd.Timestamp
    current_asset: str
    target_asset: str
    raw_best_asset: str
    selected_utility: float
    utilities: dict[str, float]
    cash_edges: dict[str, float]
    opportunity_probability: float | None
    opportunity_confidence: float | None
    opportunity_threshold: float | None
    opportunity_accepted: bool | None
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
    










    if config.strategy_mode not in SUPPORTED_ROTATION_MODES:
        raise ValueError(f"Unsupported Paper compound-rotation strategy mode: {config.strategy_mode}.")
    if str(config.strategy_mode) in RESEARCH_ONLY_SWING_STRATEGY_MODES:
        raise ValueError("Opportunity Cash Gate / Absolute Utility Cash Gate / Portfolio Allocation / Compound Risk Overlay v3.12.0 is research/backtest-only until the compatible Paper executor is enabled.")
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
    calibration_cash_edge_models = None
    if _risk_off_enabled(config):
        calibration_cash_edge_models, effective_device, fallback_reason = _fit_xgb_models(
            frames,
            symbols,
            train_dates,
            config,
            effective_device,
            phase="live_calibration_cash_edge",
            target_column="forward_cash_edge",
        )
        if fallback_reason:
            fallback_reasons.append(fallback_reason)
    opportunity_gate = None
    if selective_opportunity_enabled(config):
        opportunity_gate = fit_selective_opportunity_gate(
            calibration_models,
            frames,
            symbols,
            calibration_dates,
            lambda fitted, panel, labels, ts: _xgb_utilities(fitted, panel, labels, ts, config),
            random_state=int(config.random_state),
            label_horizon=max(int(item) for item in config.rotation_target_horizons),
        )

    candidates = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    if not candidates:
        raise ValueError("rotation_switch_margin_candidates cannot be empty.")

    best_candidate = candidates[0]
    best_score = float("-inf")
    margin_config = (
        config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
        if selective_opportunity_enabled(config)
        else config
    )
    for candidate in candidates:
        calibration_policy = _xgb_policy(
            calibration_models,
            frames,
            symbols,
            margin_config,
            candidate,
            cash_edge_models=calibration_cash_edge_models,
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
    final_cash_edge_models = None
    if _risk_off_enabled(config):
        final_cash_edge_models, effective_device, fallback_reason = _fit_xgb_models(
            frames,
            symbols,
            final_fit_dates,
            config,
            effective_device,
            phase="live_final_cash_edge",
            target_column="forward_cash_edge",
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

    utilities_array = live_model_utilities(
        final_models,
        frames,
        symbols,
        decision_date,
    )
    if not np.isfinite(utilities_array[1:]).any():
        raise ValueError("The live XGBoost models produced no finite asset utilities.")
    opportunity_result = (
        evaluate_opportunity(opportunity_gate, utilities_array, frames, symbols, decision_date)
        if opportunity_gate is not None
        else None
    )

    policy = build_live_rotation_policy(
        final_models,
        frames,
        symbols,
        config,
        effective_margin,
        cash_edge_models=final_cash_edge_models,
        opportunity_gate=opportunity_gate,
    )
    target_position, selected_utility = policy(
        decision_date,
        current_position,
        int(holding_sessions),
    )
    if _risk_off_enabled(config):
        finite_asset_positions = [
            position
            for position in range(1, len(utilities_array))
            if np.isfinite(utilities_array[position])
        ]
        raw_best_position = max(
            finite_asset_positions,
            key=lambda position: float(utilities_array[position]),
        )
    else:
        raw_best_position = int(np.nanargmax(utilities_array))

    utilities = {
        label: float(utilities_array[index])
        for index, label in enumerate(labels)
    }
    cash_edges: dict[str, float] = {}
    if _risk_off_enabled(config) and final_cash_edge_models is not None:
        cash_edge_array = live_model_utilities(
            final_cash_edge_models,
            frames,
            symbols,
            decision_date,
        )
        cash_edges = {
            label: float(cash_edge_array[index])
            for index, label in enumerate(labels)
            if np.isfinite(cash_edge_array[index])
        }

    return LiveXGBoostDecision(
        decision_date=decision_date,
        current_asset=current_label,
        target_asset=labels[int(target_position)],
        raw_best_asset=labels[raw_best_position],
        selected_utility=float(selected_utility),
        utilities=utilities,
        cash_edges=cash_edges,
        opportunity_probability=float(opportunity_result.probability) if opportunity_result is not None else None,
        opportunity_confidence=float(opportunity_result.confidence) if opportunity_result is not None else None,
        opportunity_threshold=float(opportunity_gate.threshold) if opportunity_gate is not None else None,
        opportunity_accepted=bool(opportunity_result.accepted) if opportunity_result is not None else None,
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
