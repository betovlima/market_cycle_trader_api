from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from market_cycle_trader_api.core.config import (
    RESEARCH_ONLY_SWING_STRATEGY_MODES,
    TCC_U67_CONTROL_OPERATIONAL_MODE,
)
from market_cycle_trader_api.engine import live_model_signal
from market_cycle_trader_api.engine import tcc_u67_operational_runtime as runtime
from market_cycle_trader_api.services import paper_trading, strategy_lab
from market_cycle_trader_api.services.model_research import (
    execution_settings_from_values,
    model_execution_snapshot,
)


ENGINE_BINDING = "tcc_u67_v1210_operational_backtest"


def _contract_config() -> dict:
    values = runtime.tcc_u67_strategy_updates()
    values["research_model_family"] = "lightgbm_utility"
    return values


def _contract_model_snapshot() -> dict:
    settings = execution_settings_from_values(
        "lightgbm_utility",
        runtime.tcc_u67_model_values(),
        settings_revision=1,
        profile_id="tcc-u67-v1.21.0-control",
    )
    snapshot = model_execution_snapshot("lightgbm_utility", settings)
    snapshot["source"] = "tcc_u67_operational_contract"
    return snapshot


def test_u67_mode_is_live_runtime_eligible() -> None:
    assert (
        TCC_U67_CONTROL_OPERATIONAL_MODE
        not in RESEARCH_ONLY_SWING_STRATEGY_MODES
    )


def test_u67_model_snapshot_matches_protected_lightgbm() -> None:
    snapshot = _contract_model_snapshot()
    assert runtime.tcc_u67_model_snapshot_issues(snapshot) == []

    broken = _contract_model_snapshot()
    broken["settings_snapshot"]["lightgbm"]["n_estimators"] = 1
    issues = runtime.tcc_u67_model_snapshot_issues(broken)
    assert any("lightgbm.n_estimators:" in issue for issue in issues)


def test_strategy_catalog_accepts_only_exact_u67_live_contract() -> None:
    document = {
        "backtest_engine_binding": ENGINE_BINDING,
        "configuration": _contract_config(),
        "research_model_snapshot": _contract_model_snapshot(),
        "strategy_kind": "standard",
    }
    result = strategy_lab._trader_runtime_compatibility(document)
    assert result["eligible"] is True
    assert result["code"] == "tcc_u67_live_runtime_ready"

    wrong_binding = {
        **document,
        "backtest_engine_binding": "wrong-engine",
    }
    result = strategy_lab._trader_runtime_compatibility(wrong_binding)
    assert result["eligible"] is False
    assert result["code"] == "tcc_u67_engine_binding_mismatch"

    wrong_contract = {
        **document,
        "configuration": {
            **document["configuration"],
            "rotation_switch_margin": 0.02,
        },
    }
    result = strategy_lab._trader_runtime_compatibility(wrong_contract)
    assert result["eligible"] is False
    assert result["code"] == "tcc_u67_contract_mismatch"


def test_live_router_uses_u67_runtime_not_generic_lightgbm() -> None:
    sentinel = object()
    config = SimpleNamespace(
        strategy_mode=TCC_U67_CONTROL_OPERATIONAL_MODE
    )
    with (
        patch.object(
            live_model_signal,
            "build_live_tcc_u67_decision",
            return_value=sentinel,
        ) as u67_runner,
        patch.object(
            live_model_signal,
            "build_live_lightgbm_decision",
        ) as generic_runner,
    ):
        result = live_model_signal.build_live_model_decision(
            {},
            config,
            model_family="lightgbm_utility",
            current_asset="XSD",
            holding_sessions=3,
        )
    assert result is sentinel
    u67_runner.assert_called_once()
    generic_runner.assert_not_called()


def test_live_policy_can_hold_existing_xsd_position() -> None:
    config = SimpleNamespace(
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0005,
        rotation_min_holding_days=2,
    )
    symbols = ["NVDA", "XSD", "MSFT"]
    utilities = np.asarray([0.0, 0.20, 0.35, 0.10], dtype=float)
    target, raw_best, score = runtime._apply_live_u67_policy(
        utilities,
        symbols,
        config,
        current_asset="XSD",
        holding_sessions=5,
        calibrated_margin=0.0005,
    )
    assert target == "XSD"
    assert raw_best == "XSD"
    assert score == 0.35


def test_live_policy_rotates_only_when_margin_is_met() -> None:
    config = SimpleNamespace(
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0005,
        rotation_min_holding_days=2,
    )
    symbols = ["NVDA", "XSD"]
    utilities = np.asarray([0.0, 0.401, 0.400], dtype=float)
    target, raw_best, _ = runtime._apply_live_u67_policy(
        utilities,
        symbols,
        config,
        current_asset="XSD",
        holding_sessions=5,
        calibrated_margin=0.0025,
    )
    assert raw_best == "NVDA"
    assert target == "XSD"

    utilities = np.asarray([0.0, 0.404, 0.400], dtype=float)
    target, raw_best, _ = runtime._apply_live_u67_policy(
        utilities,
        symbols,
        config,
        current_asset="XSD",
        holding_sessions=5,
        calibrated_margin=0.0025,
    )
    assert raw_best == "NVDA"
    assert target == "NVDA"


def test_paper_runtime_uses_fresh_u67_snapshot_and_preserves_account_state() -> None:
    source = inspect.getsource(paper_trading.prepare_next_paper_plan)
    assert "download_u67_operational_snapshot(" in source
    assert "append_u67_current_session_intraday(" in source
    assert '"current_session_intraday"' in source
    assert "u67_snapshot_sha256" in source
    assert "_reconcile_state_with_account(" in source
    assert "current_asset=state.managed_symbol" in source
    assert "holding_sessions=state.holding_sessions" in source


def test_winner_promotion_does_not_touch_broker_or_reinitialize_position() -> None:
    source = inspect.getsource(strategy_lab.promote_strategy_to_trader)
    assert '"broker_interaction_performed": False' in source
    assert '"operational_state_preserved": True' in source
    assert '"paper_state_reinitialization_required": False' in source
    assert '"current_position_preserved": True' in source
