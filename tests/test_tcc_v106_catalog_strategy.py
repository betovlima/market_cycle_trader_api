from __future__ import annotations

from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.services.strategy_lab import (
    TCC_V106_BACKTEST_ENGINE_BINDING,
    _trader_runtime_compatibility,
)
from market_cycle_trader_api.services.tcc_v106_reference_strategy import (
    SOURCE_COMMIT,
    _mct_configuration_for_tcc,
    _tcc_lightgbm_values,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    ASSETS,
    CONFIG as TCC_CONFIG,
)


def _source_request() -> BacktestRequest:
    lightgbm = _tcc_lightgbm_values()
    return BacktestRequest.model_validate(
        {
            "assets": list(ASSETS),
            "strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST",
            "start_date": "2016-01-01",
            "end_date": None,
            "timeframe": "1Day",
            "market_data_provider": "alpaca",
            "alpaca_historical_feed": "sip",
            "alpaca_live_feed": "iex",
            "alpaca_adjustment": "raw",
            "market_data_history_backfill_enabled": True,
            "market_data_history_backfill_provider": "alpaca",
            "market_data_history_start_tolerance_days": 10,
            "market_data_require_complete_history": True,
            "rotation_models": ["xgboost_utility"],
            "rotation_horizon_days": 40,
            "rotation_target_horizons": [5, 10, 20, 40, 60],
            "rotation_target_horizon_weights": [0.10, 0.15, 0.20, 0.30, 0.25],
            "rotation_movement_capture_weight": 0.35,
            "rotation_trend_persistence_weight": 0.20,
            "rotation_minimum_training_rows": 700,
            "rotation_walk_forward_enabled": True,
            "rotation_walk_forward_calibration_days": 126,
            "rotation_walk_forward_test_days": 504,
            "rotation_walk_forward_min_test_days": 126,
            "rotation_purge_days": 60,
            "rotation_downside_penalty": 0.20,
            "rotation_drawdown_penalty": 0.35,
            "rotation_min_holding_days": 2,
            "rotation_min_expected_edge": 0.001,
            "rotation_cash_threshold": 0.0,
            "rotation_switch_margin": 0.0005,
            "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
            "allocation_lookback_days": 126,
            "allocation_max_asset_weight": 1.0,
            "allocation_cvar_confidence": 0.95,
            "allocation_cvar_penalty": 1.0,
            "allocation_turnover_penalty": 0.0025,
            "allocation_minimum_utility": 0.0,
            "allocation_signal_scale": 1.0,
            "rotation_xgb_n_estimators": lightgbm["n_estimators"],
            "rotation_xgb_learning_rate": lightgbm["learning_rate"],
            "rotation_xgb_max_depth": lightgbm["max_depth"],
            "rotation_accelerator": "cpu",
            "rotation_allow_cpu_fallback": False,
            "rotation_xgb_repetitions": lightgbm["repetitions"],
            "rotation_seed_step": lightgbm["seed_step"],
            "initial_capital": 10_000.0,
            "whole_shares": False,
            "slippage_bps": 0.0,
            "commission_rate": 0.0,
            "sec_fee_rate": 0.0000206,
            "taf_fee_per_share": 0.000195,
            "taf_fee_cap": 9.79,
            "cat_fee_per_share": 0.000003,
            "xgb_min_child_weight": lightgbm["min_child_weight"],
            "xgb_subsample": lightgbm["subsample"],
            "xgb_colsample_bytree": lightgbm["colsample_bytree"],
            "xgb_reg_alpha": lightgbm["reg_alpha"],
            "xgb_reg_lambda": lightgbm["reg_lambda"],
            "xgb_n_jobs": lightgbm["n_jobs"],
            "deterministic_execution": False,
            "numeric_thread_limit": 1,
            "mongo_cache_enabled": True,
            "mongo_refresh_overlap_days": 5,
            "mongo_write_batch_size": 2_000,
            "random_state": 42,
        }
    )


def test_catalog_strategy_mirrors_tcc_v106_contract() -> None:
    config = _mct_configuration_for_tcc(_source_request())

    assert config.assets == list(ASSETS)
    assert config.start_date == TCC_CONFIG.start_date
    assert config.end_date == TCC_CONFIG.analysis_end_date
    assert config.rotation_target_horizons == [5, 10, 20, 40, 60]
    assert config.rotation_switch_margin == 0.0005
    assert config.rotation_switch_margin_candidates == [0.0, 0.0025, 0.005, 0.01]
    assert config.rotation_accelerator == "cpu"
    assert config.deterministic_execution is False
    assert config.xgb_n_jobs == -1


def test_catalog_reference_uses_frozen_lightgbm_values() -> None:
    values = _tcc_lightgbm_values()

    assert values["n_estimators"] == 329
    assert values["learning_rate"] == 0.020731
    assert values["num_leaves"] == 6
    assert values["n_jobs"] == -1
    assert values["random_state"] == 42
    assert SOURCE_COMMIT == "c0d71772092f0c26c9f28f0211b9933e9c396b95"


def test_reference_engine_strategy_is_not_live_trader_eligible() -> None:
    compatibility = _trader_runtime_compatibility(
        {
            "strategy_kind": "standard",
            "backtest_engine_binding": TCC_V106_BACKTEST_ENGINE_BINDING,
            "research_model_snapshot": {"family": "lightgbm_utility"},
        }
    )

    assert compatibility["eligible"] is False
    assert compatibility["code"] == "research_reference_engine_not_live"
