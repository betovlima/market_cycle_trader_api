from __future__ import annotations

import json
from pathlib import Path

from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest


ROOT = Path(__file__).resolve().parents[1]


def test_final_research_configuration_is_standalone_and_frozen() -> None:
    path = ROOT / "research" / "final_research_v10_8_75.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    request = BacktestExecutionRequest.model_validate(
        payload["request"]
    )

    assert payload["api_version"] == "10.8.75"
    assert payload["experiment"] == "final-fresh-alpaca-standalone-v1"
    assert request.mongo_cache_enabled is False
    assert request.market_data_provider == "alpaca"
    assert request.alpaca_historical_feed == "sip"
    assert request.start_date == "2016-01-01"
    assert request.analysis_end_date == "2026-09-17"
    assert len(request.assets) == 56
    assert "DOC" in request.assets
    assert "CLMT" in request.assets

    lightgbm = request.research_model_settings["lightgbm"]
    assert lightgbm["n_estimators"] == 329
    assert lightgbm["learning_rate"] == 0.020731
    assert lightgbm["max_depth"] == 3
    assert lightgbm["num_leaves"] == 6
    assert lightgbm["min_child_weight"] == 5.0
    assert lightgbm["early_stopping_enabled"] is False

    challenger = payload["challenger"]["soft_horizon_consensus"]
    assert challenger["enabled"] is True
    assert challenger["penalty_strength"] == 1.0


def test_final_research_runner_has_no_mongo_dependency() -> None:
    path = ROOT / "scripts" / "research_final_fresh_alpaca_standalone.py"
    source = path.read_text(encoding="utf-8")

    forbidden = (
        "create_client",
        "get_database",
        "mongo_repository",
        "JOBS_COLLECTION",
        "_latest_job",
    )
    for token in forbidden:
        assert token not in source

    assert "download_stock_bars" in source
    assert "CORPORATE_ACTIONS_ENDPOINT" in source
    assert '"mongo_used": False' in source
    assert '"database_market_data_used": False' in source
    assert '"database_configuration_used": False' in source
