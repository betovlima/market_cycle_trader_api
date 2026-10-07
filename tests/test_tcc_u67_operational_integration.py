from __future__ import annotations

from types import SimpleNamespace

from market_cycle_trader_api.api.routers.tcc_u67_operational import router
from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.core.config import (
    RESEARCH_ONLY_SWING_STRATEGY_MODES,
    TCC_U67_CONTROL_OPERATIONAL_MODE,
)
from market_cycle_trader_api.engine.tcc_u67_operational_market_data import (
    _structural_issue,
)
from market_cycle_trader_api.engine.tcc_u67_operational_runtime import (
    U67_OPERATIONAL_CONTRACT,
    tcc_u67_contract_issues,
    tcc_u67_strategy_updates,
)
from market_cycle_trader_api.services.tcc_u67_operational_strategy import (
    BACKTEST_ENGINE_BINDING,
)
from market_cycle_trader_api.tcc_u67_v1210_reference.contract import (
    SOURCE_COMMIT,
    U67_REQUESTED_ASSETS,
)


def test_u67_operational_mode_has_protected_live_runtime() -> None:
    assert (
        TCC_U67_CONTROL_OPERATIONAL_MODE
        not in RESEARCH_ONLY_SWING_STRATEGY_MODES
    )
    assert BACKTEST_ENGINE_BINDING == (
        "tcc_u67_v1210_operational_backtest"
    )
    assert U67_OPERATIONAL_CONTRACT == "tcc_main_u67_v1.21.0_control"


def test_u67_operational_contract_accepts_exact_profile() -> None:
    payload = tcc_u67_strategy_updates()
    profile = SimpleNamespace(**payload)
    assert tcc_u67_contract_issues(profile) == []
    assert len(profile.assets) == 67
    assert tuple(profile.assets) == tuple(U67_REQUESTED_ASSETS)


def test_u67_operational_contract_rejects_asset_drift() -> None:
    payload = tcc_u67_strategy_updates()
    payload["assets"] = list(payload["assets"][:-1])
    issues = tcc_u67_contract_issues(SimpleNamespace(**payload))
    assert any("exact U67" in issue for issue in issues)


def test_u67_operational_contract_rejects_live_feed_drift() -> None:
    payload = tcc_u67_strategy_updates()
    payload["alpaca_live_feed"] = "sip"
    issues = tcc_u67_contract_issues(SimpleNamespace(**payload))
    assert any("alpaca_live_feed" in issue for issue in issues)


def test_clmt_is_removed_by_operational_identity_policy() -> None:
    issue = _structural_issue("CLMT", [])
    assert issue is not None
    assert issue["reason"] == "structural_identity_change"
    assert issue["action_type"] == "name_change"
    assert issue["old_cusip"] == "131476103"
    assert issue["new_cusip"] == "131428104"


def test_u67_research_router_exposes_install_and_backtest_endpoint() -> None:
    paths = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in router.routes
    }
    assert (
        "/api/research/tcc-u67",
        ("GET",),
    ) in paths
    assert (
        "/api/research/tcc-u67/strategy",
        ("POST",),
    ) in paths
    assert (
        "/api/research/tcc-u67/jobs",
        ("POST",),
    ) in paths


def test_u67_source_commit_remains_main_checkpoint() -> None:
    assert SOURCE_COMMIT == "4b5f16030afa8b850790747bb0e3e3063d233e79"



def test_legacy_tcc_v106_profile_still_validates() -> None:
    payload = tcc_u67_strategy_updates()
    payload["strategy_mode"] = "COMPOUND_ROTATION_SWING_TCC_CONTROL_V106"
    request = BacktestRequest.model_validate(payload)
    assert request.strategy_mode == "COMPOUND_ROTATION_SWING_TCC_CONTROL_V106"
