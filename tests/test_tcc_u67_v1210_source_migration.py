from __future__ import annotations

from market_cycle_trader_api.core.config import (
    API_VERSION,
    TCC_U67_CONTROL_OPERATIONAL_MODE,
)
from market_cycle_trader_api.tcc_u67_v1210_reference.configuracao import (
    CONFIG,
    construir_configuracao_controle,
)
from market_cycle_trader_api.tcc_u67_v1210_reference.contract import (
    EXECUTION_SCHEMA,
    EXPECTED_ENDING_CAPITAL,
    REPRODUCTION_VERSION,
    SOURCE_COMMIT,
    U67_REQUESTED_ASSETS,
)


def test_tcc_u67_source_identity_is_pinned() -> None:
    assert API_VERSION == "10.8.85"
    assert TCC_U67_CONTROL_OPERATIONAL_MODE == (
        "COMPOUND_ROTATION_SWING_TCC_U67_V1210"
    )
    assert SOURCE_COMMIT == "4b5f16030afa8b850790747bb0e3e3063d233e79"
    assert REPRODUCTION_VERSION == "1.21.0"
    assert EXECUTION_SCHEMA == "u67-control-reproduction-v1"
    assert EXPECTED_ENDING_CAPITAL == 58_557_157.67496595


def test_tcc_u67_universe_matches_official_main_checkpoint() -> None:
    assert len(U67_REQUESTED_ASSETS) == 67
    assert len(set(U67_REQUESTED_ASSETS)) == 67
    assert tuple(CONFIG.assets) == tuple(U67_REQUESTED_ASSETS[:56])
    assert U67_REQUESTED_ASSETS[-8:] == (
        "THO",
        "WDAY",
        "EXR",
        "XEL",
        "SBFG",
        "PAYX",
        "MUX",
        "SXC",
    )


def test_tcc_u67_control_parameters_remain_frozen() -> None:
    control = construir_configuracao_controle(
        CONFIG,
        assets=U67_REQUESTED_ASSETS,
    )
    assert control.rotation_target_horizons == (5, 10, 20, 40, 60)
    assert control.rotation_target_horizon_weights == (
        0.10,
        0.15,
        0.20,
        0.30,
        0.25,
    )
    assert control.rotation_minimum_training_rows == 700
    assert control.rotation_walk_forward_calibration_days == 126
    assert control.rotation_walk_forward_test_days == 504
    assert control.rotation_purge_days == 60
    assert control.rotation_min_holding_days == 2
    assert control.rotation_min_expected_edge == 0.001
    assert control.rotation_switch_margin == 0.0005
    assert control.rotation_switch_margin_candidates == (
        0.0,
        0.0025,
        0.005,
        0.01,
    )
    lightgbm = control.research_model_settings["lightgbm"]
    assert lightgbm["n_estimators"] == 329
    assert lightgbm["learning_rate"] == 0.020731
    assert lightgbm["random_state"] == 42
    assert control.research_model_settings["soft_horizon_consensus"] == {
        "enabled": False
    }
