"""Guardrails for frozen U67+10 Challenger integration."""
from __future__ import annotations

import inspect

from market_cycle_trader_api.api.routers.tcc_u77_static_challenger import router
from market_cycle_trader_api.core.config import (
    RESEARCH_ONLY_SWING_STRATEGY_MODES,
    TCC_U67_CONTROL_OPERATIONAL_MODE,
    TCC_U77_FIXED_CHALLENGER_MODE,
)
from market_cycle_trader_api.engine import compound_rotation_backtest
from market_cycle_trader_api.engine.tcc_u77_static_challenger import (
    BACKTEST_ENGINE_BINDING,
    REFERENCE_CAPITAL,
    SOURCE_COMMIT,
    SOURCE_TAG,
    STATIC_ADDITIONS,
    U77_REQUESTED_ASSETS,
    tcc_u77_contract_issues,
    tcc_u77_strategy_updates,
)
from market_cycle_trader_api.engine.tcc_u67_operational_runtime import (
    tcc_u67_contract_issues,
    tcc_u67_strategy_updates,
)
from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.services.tcc_u77_static_challenger import (
    _strategy_configuration,
)
from market_cycle_trader_api.services.strategy_lab import (
    _trader_runtime_compatibility,
)
from market_cycle_trader_api.tcc_u67_v1210_reference.contract import U67_REQUESTED_ASSETS


def test_exact_scientific_universe_and_provenance() -> None:
    assert len(U67_REQUESTED_ASSETS) == 67
    assert len(U77_REQUESTED_ASSETS) == 77
    assert U77_REQUESTED_ASSETS[:67] == U67_REQUESTED_ASSETS
    assert U77_REQUESTED_ASSETS[67:] == STATIC_ADDITIONS
    assert REFERENCE_CAPITAL == 587_165_941.1330212
    assert len(SOURCE_COMMIT) == 40
    assert SOURCE_TAG == "research-u67plus10-challenger-v1.4.1"


def test_u77_contract_and_schema_do_not_modify_u67() -> None:
    original = tcc_u67_strategy_updates()
    challenger = tcc_u77_strategy_updates()
    assert tcc_u67_contract_issues(original) == []
    assert tcc_u77_contract_issues(challenger) == []
    assert original["strategy_mode"] == TCC_U67_CONTROL_OPERATIONAL_MODE
    assert challenger["strategy_mode"] == TCC_U77_FIXED_CHALLENGER_MODE
    assert BacktestRequest.model_validate(
        {**challenger, "rotation_models": ["xgboost_utility"]}
    ).strategy_mode == TCC_U77_FIXED_CHALLENGER_MODE


def test_u77_rejects_contract_drift() -> None:
    data = tcc_u77_strategy_updates()
    data["assets"] = data["assets"][:-1]
    assert any("exact U67+10" in x for x in tcc_u77_contract_issues(data))
    data = tcc_u77_strategy_updates()
    data["rotation_purge_days"] = 0
    assert any("rotation_purge_days" in x for x in tcc_u77_contract_issues(data))


def test_u77_backtest_only_not_trader_eligible() -> None:
    assert TCC_U77_FIXED_CHALLENGER_MODE in RESEARCH_ONLY_SWING_STRATEGY_MODES
    assert BACKTEST_ENGINE_BINDING == "tcc_u77_fixed_challenger_backtest"
    compatibility = _trader_runtime_compatibility(
        {
            "strategy_kind": "standard",
            "backtest_engine_binding": BACKTEST_ENGINE_BINDING,
            "research_model_snapshot": {"family": "lightgbm_utility"},
            "configuration": tcc_u77_strategy_updates(),
        }
    )
    assert compatibility["eligible"] is False


def test_jobs_have_independent_u77_snapshot_and_replay() -> None:
    source = inspect.getsource(compound_rotation_backtest.run_job)
    assert "assert_tcc_u77_contract(" in source
    assert "U77_REQUESTED_ASSETS" in source
    assert "requested_assets=(" in source
    assert "download_u67_operational_snapshot(" in source
    assert "run_tcc_u67_operational_backtest(" in source


def test_install_and_jobs_endpoints_exist() -> None:
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in router.routes}
    assert ("/api/research/tcc-u77-fixed", ("GET",)) in routes
    assert ("/api/research/tcc-u77-fixed/strategy", ("POST",)) in routes
    assert ("/api/research/tcc-u77-fixed/jobs", ("POST",)) in routes



def test_catalog_installer_builds_exact_challenger_configuration() -> None:
    source = BacktestRequest.model_validate({
        **tcc_u67_strategy_updates(),
        "rotation_models": ["xgboost_utility"],
    })
    installed = _strategy_configuration(source)

    assert tuple(installed.assets) == U77_REQUESTED_ASSETS
    assert installed.strategy_mode == TCC_U77_FIXED_CHALLENGER_MODE
    assert tcc_u77_contract_issues(installed) == []
