from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    STRATEGY_PROFILES_COLLECTION,
    utc_now,
)
from ..schemas.model_research import LightGBMResearchSettings
from ..schemas.requests import BacktestRequest
from ..tcc_v106_reference.config import (
    CONFIG as TCC_V106_CONFIG,
    build_control_config,
)
from .strategy_lab import (
    TCC_V106_BACKTEST_ENGINE_BINDING,
    create_strategy,
    get_research_strategy_context,
    get_strategy,
    get_strategy_control,
    select_research_strategy_only,
    update_strategy,
    update_strategy_model,
)

REFERENCE_ENGINE_ID = "tcc-v1.0.6-verbatim"
SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
SOURCE_TAG = "v1.0.6"
SOURCE_COMMIT = "c0d71772092f0c26c9f28f0211b9933e9c396b95"


def _tcc_lightgbm_values() -> dict[str, Any]:
    control = build_control_config(TCC_V106_CONFIG)
    raw = dict(
        (control.research_model_settings or {}).get("lightgbm") or {}
    )
    allowed = set(LightGBMResearchSettings.model_fields)
    return {
        key: deepcopy(value)
        for key, value in raw.items()
        if key in allowed
    }


def _mct_configuration_for_tcc(
    source: BacktestRequest,
) -> BacktestRequest:
    """Mirror the frozen TCC Control contract in an MCT Strategy document."""
    tcc = TCC_V106_CONFIG
    lightgbm = _tcc_lightgbm_values()

    payload = source.model_dump(mode="python")
    payload.update(
        {
            "assets": list(tcc.assets),
            # MCT keeps the historical XGBOOST token as the normalized
            # LightGBM compound-rotation strategy-mode identifier.
            "strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST",
            "start_date": str(tcc.start_date),
            # Lock the normal MCT Simulation to the same scientific cutoff.
            "end_date": str(tcc.analysis_end_date),
            "timeframe": str(tcc.timeframe),
            "market_data_provider": str(tcc.market_data_provider),
            "alpaca_historical_feed": str(tcc.alpaca_historical_feed),
            "alpaca_live_feed": str(tcc.alpaca_live_feed),
            "alpaca_adjustment": str(tcc.alpaca_adjustment),
            "market_data_history_backfill_enabled": bool(
                tcc.market_data_history_backfill_enabled
            ),
            "market_data_history_backfill_provider": str(
                tcc.market_data_history_backfill_provider
            ),
            "market_data_history_start_tolerance_days": int(
                tcc.market_data_history_start_tolerance_days
            ),
            "market_data_require_complete_history": bool(
                tcc.market_data_require_complete_history
            ),
            "rotation_models": ["xgboost_utility"],
            "rotation_horizon_days": int(tcc.rotation_horizon_days),
            "rotation_target_horizons": list(tcc.rotation_target_horizons),
            "rotation_target_horizon_weights": list(
                tcc.rotation_target_horizon_weights
            ),
            "rotation_movement_capture_weight": float(
                tcc.rotation_movement_capture_weight
            ),
            "rotation_trend_persistence_weight": float(
                tcc.rotation_trend_persistence_weight
            ),
            "rotation_minimum_training_rows": int(
                tcc.rotation_minimum_training_rows
            ),
            "rotation_walk_forward_enabled": bool(
                tcc.rotation_walk_forward_enabled
            ),
            "rotation_walk_forward_calibration_days": int(
                tcc.rotation_walk_forward_calibration_days
            ),
            "rotation_walk_forward_test_days": int(
                tcc.rotation_walk_forward_test_days
            ),
            "rotation_walk_forward_min_test_days": int(
                tcc.rotation_walk_forward_min_test_days
            ),
            "rotation_purge_days": int(tcc.rotation_purge_days),
            "rotation_downside_penalty": float(
                tcc.rotation_downside_penalty
            ),
            "rotation_drawdown_penalty": float(
                tcc.rotation_drawdown_penalty
            ),
            "rotation_min_holding_days": int(
                tcc.rotation_min_holding_days
            ),
            "rotation_min_expected_edge": float(
                tcc.rotation_min_expected_edge
            ),
            "rotation_cash_threshold": float(
                tcc.rotation_cash_threshold
            ),
            "rotation_switch_margin": float(
                tcc.rotation_switch_margin
            ),
            "rotation_switch_margin_candidates": list(
                tcc.rotation_switch_margin_candidates
            ),
            "opportunity_utility_entry_threshold": float(
                tcc.opportunity_utility_entry_threshold
            ),
            "opportunity_utility_exit_threshold": float(
                tcc.opportunity_utility_exit_threshold
            ),
            "allocation_lookback_days": int(
                tcc.allocation_lookback_days
            ),
            "allocation_max_asset_weight": float(
                tcc.allocation_max_asset_weight
            ),
            "allocation_cvar_confidence": float(
                tcc.allocation_cvar_confidence
            ),
            "allocation_cvar_penalty": float(
                tcc.allocation_cvar_penalty
            ),
            "allocation_turnover_penalty": float(
                tcc.allocation_turnover_penalty
            ),
            "allocation_minimum_utility": float(
                tcc.allocation_minimum_utility
            ),
            "allocation_signal_scale": float(
                tcc.allocation_signal_scale
            ),
            "rotation_xgb_n_estimators": int(
                lightgbm["n_estimators"]
            ),
            "rotation_xgb_learning_rate": float(
                lightgbm["learning_rate"]
            ),
            "rotation_xgb_max_depth": int(lightgbm["max_depth"]),
            "rotation_accelerator": str(tcc.rotation_accelerator),
            "rotation_allow_cpu_fallback": bool(
                tcc.rotation_allow_cpu_fallback
            ),
            "rotation_xgb_repetitions": int(
                lightgbm["repetitions"]
            ),
            "rotation_seed_step": int(lightgbm["seed_step"]),
            "initial_capital": float(tcc.initial_capital),
            "whole_shares": bool(tcc.whole_shares),
            "slippage_bps": float(tcc.slippage_bps),
            "commission_rate": float(tcc.commission_rate),
            "sec_fee_rate": float(tcc.sec_fee_rate),
            "taf_fee_per_share": float(tcc.taf_fee_per_share),
            "taf_fee_cap": float(tcc.taf_fee_cap),
            "cat_fee_per_share": float(tcc.cat_fee_per_share),
            "xgb_min_child_weight": float(
                lightgbm["min_child_weight"]
            ),
            "xgb_subsample": float(lightgbm["subsample"]),
            "xgb_colsample_bytree": float(
                lightgbm["colsample_bytree"]
            ),
            "xgb_reg_alpha": float(lightgbm["reg_alpha"]),
            "xgb_reg_lambda": float(lightgbm["reg_lambda"]),
            "xgb_n_jobs": int(lightgbm["n_jobs"]),
            "deterministic_execution": bool(
                tcc.deterministic_execution
            ),
            # The frozen TCC config keeps this at 1 even though deterministic
            # execution is disabled.
            "numeric_thread_limit": int(tcc.numeric_thread_limit),
            "random_state": int(lightgbm["random_state"]),
        }
    )
    return BacktestRequest.model_validate(payload)


def _existing_reference_profile(db: Any) -> dict[str, Any] | None:
    return db[STRATEGY_PROFILES_COLLECTION].find_one(
        {
            "backtest_engine_binding": TCC_V106_BACKTEST_ENGINE_BINDING,
            "reference_source_commit": SOURCE_COMMIT,
        }
    )


def install_tcc_v106_research_strategy(
    db: Any,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    """Create/reuse the catalog Strategy and select it only for Research."""
    actor = (actor_email or "").strip().lower() or None

    existing = _existing_reference_profile(db)
    if existing is not None:
        control = get_strategy_control(db)
        selected = select_research_strategy_only(
            db,
            str(existing["_id"]),
            expected_control_revision=int(control["revision"]),
            note="Select frozen TCC v1.0.6 reference engine for Research",
            actor_email=actor,
        )
        return {
            "created": False,
            "strategy": get_strategy(db, str(existing["_id"])),
            "control": selected,
        }

    source_config, source_profile = get_research_strategy_context(db)
    created = create_strategy(
        db,
        name="TCC v1.0.6 Reference",
        description=(
            "Frozen TCC v1.0.6 reference engine for controlled MCT research"
        ),
        clone_from_strategy_id=str(source_profile["id"]),
        actor_email=actor,
    )
    strategy_id = str(created["id"])
    revision = int(created["revision"])

    model_updated = update_strategy_model(
        db,
        strategy_id,
        model_family="lightgbm_utility",
        values=_tcc_lightgbm_values(),
        note="Mirror frozen TCC v1.0.6 LightGBM Control parameters",
        expected_strategy_revision=revision,
        actor_email=actor,
    )

    config = _mct_configuration_for_tcc(source_config)
    updated = update_strategy(
        db,
        strategy_id,
        configuration=config,
        name=str(created.get("name") or ""),
        description=(
            "Frozen TCC v1.0.6 reference engine "
            f"({SOURCE_COMMIT[:12]})"
        ),
        note="Mirror frozen TCC v1.0.6 Strategy parameters",
        expected_revision=int(model_updated["revision"]),
        actor_email=actor,
    )

    now = utc_now()
    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": strategy_id, "revision": int(updated["revision"])},
        {
            "$set": {
                "backtest_engine_binding": TCC_V106_BACKTEST_ENGINE_BINDING,
                "reference_engine_id": REFERENCE_ENGINE_ID,
                "description": (
                    "TCC v1.0.6 frozen reference engine "
                    f"({SOURCE_COMMIT[:12]})"
                ),
                "source_git_commit_message": (
                    "TCC v1.0.6 frozen reference engine "
                    f"({SOURCE_COMMIT[:12]})"
                ),
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_tag": SOURCE_TAG,
                "reference_source_commit": SOURCE_COMMIT,
                "reference_control_variant": "CONTROL",
                "reference_comparison_variant": "SOFT_HORIZON_CONSENSUS",
                "reference_strategy_installed_at": now,
                "reference_strategy_installed_by": actor,
                "updated_at": now,
            }
        },
    )

    control = get_strategy_control(db)
    selected = select_research_strategy_only(
        db,
        strategy_id,
        expected_control_revision=int(control["revision"]),
        note="Create and select TCC v1.0.6 reference Strategy for Research",
        actor_email=actor,
    )
    return {
        "created": True,
        "strategy": get_strategy(db, strategy_id),
        "control": selected,
    }
