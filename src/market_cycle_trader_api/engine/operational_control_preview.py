"""Isolated, no-order Control decision preview based on frozen TCC v1.0.6.

This module accepts already audited/split-normalized completed daily OHLCV.
It has no database, network, order, Strategy selection or promotion calls.
It is intentionally NOT plugged into the protected Trader scheduler.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..tcc_v106_reference.capital_rotation import (
    _simple_policy_growth as scientific_policy_growth,
    _utility_policy as scientific_utility_policy,
)
from ..tcc_v106_reference.config import (
    CONFIG as FROZEN_CONFIG,
    build_control_config,
)
from ..tcc_v106_reference.research_challengers import (
    _lightgbm_fit_models as scientific_fit_models,
)
from .live_policy import build_live_rotation_policy, live_model_utilities
from .operational_control_contract import prepare_operational_control_panel

TCC_CONTROL_SOURCE_COMMIT = "c0d71772092f0c26c9f28f0211b9933e9c396b95"


def _finite_score(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def build_control_shadow_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    current_asset: str | None,
    holding_sessions: int,
) -> dict[str, Any]:
    """Calculate one prospective Control action without preparing any order.

    Input must be sourced and audited separately; no market-data access occurs
    here. The scientific replay checks the future next-open bar and cannot be
    applied directly at the latest completed session, so *only* the already
    parity-tested live decision rule is used for that last-day action.

    This is a shadow decision, never a trade recommendation or order payload.
    """
    config = build_control_config(FROZEN_CONFIG)
    if type(holding_sessions) is not int or holding_sessions < 0:
        raise ValueError("holding_sessions must be a nonnegative integer.")
    extras = sorted(set(bars_by_symbol) - set(config.assets))
    if extras:
        raise ValueError("OHLCV includes assets outside frozen Control universe: " + ", ".join(extras))

    frames, common_dates, audit = prepare_operational_control_panel(
        bars_by_symbol,
        completed_session=completed_session,
        config=config,
    )
    symbols = sorted(frames)
    labels = ["CASH", *symbols]
    current_label = str(current_asset or "CASH").strip().upper()
    if current_label not in labels:
        raise ValueError(f"Current position {current_label!r} is not present in the Control universe.")
    if current_label == "CASH" and holding_sessions:
        raise ValueError("CASH state cannot have positive holding_sessions.")

    purge = max(
        int(config.rotation_purge_days),
        max(int(item) for item in config.rotation_target_horizons),
    )
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    minimum_training_rows = int(config.rotation_minimum_training_rows)
    decision_index = len(common_dates) - 1
    # The next open has not happened yet. Treat this last known session as the
    # prospective OOS decision, without leaking its future label into fit.
    calibration_end_index = decision_index - purge
    calibration_start_index = calibration_end_index - calibration_days
    train_end_index = calibration_start_index - purge
    final_fit_end_index = decision_index - purge
    if (
        train_end_index < minimum_training_rows
        or calibration_start_index < 0
        or calibration_end_index <= calibration_start_index
        or final_fit_end_index <= 0
    ):
        raise ValueError(
            "Insufficient completed history for frozen Control training/calibration "
            f"(train={train_end_index}, required={minimum_training_rows}, "
            f"calibration={calibration_days}, purge={purge})."
        )

    train_dates = common_dates[:train_end_index]
    calibration_dates = common_dates[calibration_start_index:calibration_end_index]
    final_fit_dates = common_dates[:final_fit_end_index]
    decision_date = pd.Timestamp(common_dates[decision_index])

    calibration_models = scientific_fit_models(
        frames, symbols, train_dates, config, phase="control_shadow_calibration",
    )
    if not calibration_models:
        raise ValueError("No calibrated Control LightGBM models available.")
    candidate_margins = tuple(float(v) for v in config.rotation_switch_margin_candidates)
    if not candidate_margins:
        raise ValueError("Control switch-margin calibration has no candidate.")
    best_candidate = candidate_margins[0]
    best_score = float("-inf")
    for candidate in candidate_margins:
        policy = scientific_utility_policy(
            calibration_models, frames, symbols, config, candidate,
        )
        score = float(scientific_policy_growth(
            policy, frames, symbols, calibration_dates, config,
        ))
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_candidate = candidate
    if not math.isfinite(best_score):
        raise ValueError("Control calibration produced no finite score.")

    final_models = scientific_fit_models(
        frames, symbols, final_fit_dates, config, phase="control_shadow_final",
    )
    if not final_models:
        raise ValueError("No final Control LightGBM models available.")
    effective_margin = max(float(config.rotation_switch_margin), best_candidate)
    utilities_array = live_model_utilities(final_models, frames, symbols, decision_date)
    if not np.isfinite(utilities_array[1:]).any():
        raise ValueError("No finite Control utility available at the completed session.")

    policy = build_live_rotation_policy(
        final_models, frames, symbols, config, effective_margin,
    )
    target_position, selected_utility = policy(
        decision_date, labels.index(current_label), holding_sessions,
    )
    if target_position < 0 or target_position >= len(labels):
        raise ValueError("Control policy returned an invalid target position.")
    raw_best_position = int(np.nanargmax(utilities_array))
    return {
        "status": "shadow_only",
        "order_eligible": False,
        "source_policy": "tcc_v1.0.6_control",
        "source_commit": TCC_CONTROL_SOURCE_COMMIT,
        "decision_date": completed_session,
        "current_asset": current_label,
        "target_asset": labels[target_position],
        "raw_best_asset": labels[raw_best_position],
        "holding_sessions": holding_sessions,
        "selected_utility": _finite_score(selected_utility),
        "utilities": {
            label: _finite_score(utilities_array[i])
            for i, label in enumerate(labels)
        },
        "calibrated_candidate_margin": best_candidate,
        "effective_switch_margin": effective_margin,
        "calibration_score": best_score,
        "training_end": train_dates[-1].date().isoformat(),
        "calibration_start": calibration_dates[0].date().isoformat(),
        "calibration_end": calibration_dates[-1].date().isoformat(),
        "final_fit_end": final_fit_dates[-1].date().isoformat(),
        "input_audit": {
            "requested_assets": audit.requested_assets,
            "available_assets": audit.available_assets,
            "calendar_sessions": audit.calendar_sessions,
            "first_session": audit.first_session,
            "last_session": audit.last_session,
            "reference_assets": list(audit.required_reference_assets),
        },
        "order_submission": "never",
        "source_validation": "caller_must_audit_raw_sip_corporate_actions",
    }
