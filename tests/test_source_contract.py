from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_multi_horizon_engine_is_the_only_configured_engine() -> None:
    config = (SRC / "core" / "config.py").read_text(encoding="utf-8")
    assert 'ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"' in config
    assert 'API_VERSION = "3.18.0"' in config


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
    assert "temporal_intelligence" in main


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
    assert "confirm_market_closed: Literal[True] | None = None" in schema
    assert "Winner promotion is allowed only while the XNYS regular market is closed" not in strategy_lab
    assert "confirm_preserve_operational_state: Literal[True]" in schema
    assert "winner_strategy_id=str(winner_profile[\"id\"])" in paper
    assert "winner_assets=list(strategy.assets)" in paper
    assert "The prepared Paper plan belongs to a different Trader Winner" in paper


def test_asset_discovery_is_admin_only_and_isolated_from_winner_promotion() -> None:
    root = SRC
    main = (root / "main.py").read_text(encoding="utf-8")
    router = (root / "api" / "routers" / "asset_discovery.py").read_text(encoding="utf-8")
    service = (root / "services" / "asset_discovery.py").read_text(encoding="utf-8")
    worker = (root / "services" / "asset_discovery_worker.py").read_text(encoding="utf-8")

    assert "asset_discovery.router, dependencies=admin_required" in main
    assert 'prefix="/api/admin/asset-discovery"' in router
    assert '"/start"' in router and '"/stop"' in router
    assert "get_research_strategy_context" in worker
    assert "promote" not in (service + worker).lower()
    assert "trader_winner_strategy_id" not in (service + worker)


def test_asset_discovery_prefilters_before_loading_available_history_cache() -> None:
    market = (SRC / "services" / "asset_discovery_market.py").read_text(encoding="utf-8")
    recent_gate = market.index("if not all(checks.values())")
    full_history = market.index("frame = _available_history(symbol, config)")

    assert recent_gate < full_history
    assert "historical_cache_ready" in market
    assert "RECENT_PREFILTER_DAYS" in market
    assert '"market_data_require_complete_history": False' in market
    assert '"limited_history"' in market
    assert '"young_history"' in market
    assert 'status = "candidate" if model_ready else "watchlist"' in market


def test_asset_discovery_batch_counts_only_evaluable_assets_and_skips_expected_no_data() -> None:
    worker = (SRC / "services" / "asset_discovery_worker.py").read_text(encoding="utf-8")

    assert 'return "skipped"' in worker
    assert 'NoRecentMarketData' in worker
    assert 'NoHistoricalMarketData' in worker
    assert 'result in {"candidate", "watchlist", "rejected"}' in worker
    assert 'if processed >= batch_target' in worker
    assert '"attempted_count"' in worker


def test_asset_discovery_uses_completed_sip_safe_sessions_and_does_not_persist_technical_failures() -> None:
    market = (SRC / "services" / "asset_discovery_market.py").read_text(encoding="utf-8")
    worker = (SRC / "services" / "asset_discovery_worker.py").read_text(encoding="utf-8")

    assert 'SIP_DELAY_BUFFER_MINUTES = 20' in market
    assert '/v2/calendar' in market
    assert '_latest_safe_completed_session_end' in market
    assert 'subscription does not permit querying recent sip data' in market
    assert 'class MarketDataAccessBlocked' in market
    assert 'except MarketDataAccessBlocked as exc' in worker
    assert 'Asset Discovery stopped because Alpaca market-data access is unavailable' in worker
    assert 'Technical evaluation failure for {symbol}' in worker
    assert '"status": "failed",\n                    "reason_codes": ["technical_failure"]' not in worker


def test_research_decision_diagnostics_are_admin_export_only_and_public_trades_are_sanitized() -> None:
    exports = (SRC / "api" / "routers" / "exports.py").read_text(encoding="utf-8")
    results = (SRC / "services" / "results.py").read_text(encoding="utf-8")
    main = (SRC / "main.py").read_text(encoding="utf-8")

    assert '"/api/jobs/{job_id}/runs/{symbol}/{backend}/decision-diagnostics.csv"' in exports
    assert '"experiment_manifest.json"' in exports
    assert "exports.router, dependencies=admin_required" in main
    assert "PROTECTED_DECISION_TRADE_FIELDS" in results
    assert 'not key.startswith("q_")' in results
    assert 'not key.startswith("top_")' in results
    assert '"trades": _public_trade_rows(trades)' in results


def test_model_estimator_metadata_does_not_reject_valid_intermediate_integers() -> None:
    service = (SRC / "services" / "model_research.py").read_text(encoding="utf-8")
    assert service.count('"n_estimators": {"label": "Estimators", "step": 1') == 2
    assert '"n_estimators": {"label": "Estimators", "step": 10' not in service


def test_save_test_strategy_change_reason_is_optional() -> None:
    panel = (ROOT.parent / "market_cycle_trader" / "src" / "features" / "StrategySettingsPanel.jsx").read_text(encoding="utf-8")
    config = (ROOT.parent / "market_cycle_trader" / "src" / "features" / "strategySettings" / "strategySettingsConfig.js").read_text(encoding="utf-8")

    assert "const note = changeNote.trim() || null" in panel
    assert "Enter a change reason for the strategy revision." not in panel
    assert 'label={tr("Change reason (optional)")}' in panel
    assert "maxLength={500} placeholder={tr('Optional audit note')}" in panel
    assert "maxLength={500} required" not in panel
    assert "Saving an editable draft revision does not require a note" in config
