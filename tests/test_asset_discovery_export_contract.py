from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_export_endpoint_is_admin_asset_discovery_json_download() -> None:
    source = (ROOT / "src/market_cycle_trader_api/api/routers/asset_discovery.py").read_text(encoding="utf-8")
    assert '@router.get("/export")' in source
    assert "build_asset_discovery_export" in source
    assert "Content-Disposition" in source
    assert "application/json" in source


def test_export_payload_keeps_analytical_states_and_behavior_profile() -> None:
    source = (ROOT / "src/market_cycle_trader_api/services/asset_discovery_export.py").read_text(encoding="utf-8")
    assert 'EXPORT_STATUSES = ("candidate", "watchlist", "rejected")' in source
    assert '"behavior_profile"' in source
    assert '"summary"' in source
    assert '"runs"' in source
    assert '"last_error"' not in source
    assert '"requested_by"' not in source
    assert '"last_message"' not in source


def test_asset_discovery_records_the_api_and_policy_that_evaluated_each_asset() -> None:
    worker = (ROOT / "src/market_cycle_trader_api/services/asset_discovery_worker.py").read_text(encoding="utf-8")
    service = (ROOT / "src/market_cycle_trader_api/services/asset_discovery.py").read_text(encoding="utf-8")
    export = (ROOT / "src/market_cycle_trader_api/services/asset_discovery_export.py").read_text(encoding="utf-8")
    assert '"discovered_api_version": API_VERSION' in worker
    assert '"last_evaluated_api_version": API_VERSION' in worker
    assert '"evaluation_policy_version": ASSET_DISCOVERY_EVALUATION_POLICY_VERSION' in worker
    assert '"api_version": API_VERSION' in service
    assert 'EXPORT_SCHEMA_VERSION = 2' in export
    assert '"discovered_api_version"' in export
    assert '"last_evaluated_api_version"' in export
    assert '"evaluation_policy_version"' in export
