from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_multi_horizon_engine_is_the_only_configured_engine() -> None:
    config = (SRC / "core" / "config.py").read_text(encoding="utf-8")
    assert 'ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"' in config
    assert 'API_VERSION = "1.13.1"' in config


def test_admin_strategy_routes_are_composed() -> None:
    main = (SRC / "main.py").read_text(encoding="utf-8")
    assert "strategy_configuration" in main
    assert "parameter_bootstrap" in main
    assert "admin_setup" in main
    assert "paper_market" in main


def test_legacy_public_mutation_routers_are_not_packaged() -> None:
    assert not (SRC / "api" / "routers" / "config.py").exists()
    assert not (SRC / "api" / "routers" / "integrations.py").exists()
    assert not (SRC / "engine" / "multi_asset_extrema_backtest.py").exists()
