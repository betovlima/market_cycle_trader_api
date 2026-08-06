from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_multi_horizon_engine_is_the_only_configured_engine() -> None:
    config = (SRC / "core" / "config.py").read_text(encoding="utf-8")
    assert 'ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"' in config
    assert 'API_VERSION = "1.13.23"' in config


def test_admin_strategy_routes_are_composed() -> None:
    main = (SRC / "main.py").read_text(encoding="utf-8")
    assert "strategy_configuration" in main
    assert "strategy_lab" in main
    assert "parameter_bootstrap" in main
    assert "admin_setup" in main
    assert "paper_market" in main
    assert "dashboard" in main
    assert "admin_rotations" in main
    assert "analytics" in main


def test_legacy_public_mutation_routers_are_not_packaged() -> None:
    assert not (SRC / "api" / "routers" / "config.py").exists()
    assert not (SRC / "api" / "routers" / "integrations.py").exists()
    assert not (SRC / "engine" / "multi_asset_extrema_backtest.py").exists()


def test_winner_install_endpoint_and_file_are_packaged() -> None:
    router = (SRC / "api" / "routers" / "strategy_configuration.py").read_text(encoding="utf-8")
    service = (SRC / "services" / "strategy_configuration.py").read_text(encoding="utf-8")
    winner = SRC / "parameterizations" / "winner-v1.13.2.json"

    assert '@router.post("/winner/install")' in router
    assert 'WINNER_PARAMETERIZATION = "winner-v1.13.2.json"' in service
    assert winner.exists()
    assert not (SRC / "parameterizations" / "001_xgboost_multihorizon_champion_cpu.json").exists()


def test_google_identity_access_is_server_verified_and_token_only_login_is_removed() -> None:
    auth_router = (SRC / "api" / "routers" / "auth.py").read_text(encoding="utf-8")
    verifier = (SRC / "auth" / "google_identity.py").read_text(encoding="utf-8")
    access_service = (SRC / "auth" / "access_service.py").read_text(encoding="utf-8")

    assert '@router.post("/access/preview"' in auth_router
    assert "GoogleAccessRequest" in auth_router
    assert "create_google_session" in auth_router
    assert "verify_oauth2_token" in verifier
    assert "claimed_subject" in access_service
    assert "google_identity_mismatch" in access_service
    assert "create_viewer_session" not in access_service


def test_obsolete_direct_strategy_mutation_payloads_are_not_packaged() -> None:
    scripts = ROOT / "script"
    assert not (scripts / "patch_api_admin_strategy-configuration_cpu.json").exists()
    assert not (scripts / "put_api_admin_strategy-configuration_champion.json").exists()
    recovery = (scripts / "post_api_admin_strategy-configuration_winner_install.json").read_text(encoding="utf-8")
    assert "DISASTER RECOVERY ONLY" in recovery


def test_winner_promotion_is_metadata_only_and_binds_next_plan_to_winner() -> None:
    strategy_lab = (SRC / "services" / "strategy_lab.py").read_text(encoding="utf-8")
    paper = (SRC / "services" / "paper_trading.py").read_text(encoding="utf-8")
    schema = (SRC / "schemas" / "strategy_lab.py").read_text(encoding="utf-8")

    assert "Trader must be in cash before another winner can be promoted" not in strategy_lab
    assert "broker_interaction_performed\": False" in strategy_lab
    assert "operational_state_preserved\": True" in strategy_lab
    assert "paper_state_reinitialization_required\": False" in strategy_lab
    assert "confirm_market_closed: Literal[True]" in schema
    assert "confirm_preserve_operational_state: Literal[True]" in schema
    assert "winner_strategy_id=str(winner_profile[\"id\"])" in paper
    assert "winner_assets=list(strategy.assets)" in paper
    assert "The prepared Paper plan belongs to a different Trader Winner" in paper
