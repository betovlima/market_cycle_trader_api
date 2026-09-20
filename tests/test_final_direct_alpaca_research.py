from __future__ import annotations

import json
from pathlib import Path

from market_cycle_trader_api.infrastructure.market_data.alpaca_corporate_actions import (
    corporate_actions_for_symbol,
    corporate_actions_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_final_runner_has_no_mongodb_dependency() -> None:
    source = (
        ROOT
        / "scripts"
        / "research_final_direct_alpaca.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "mongo_repository",
        "create_client(",
        "get_database(",
        "MongoClient",
        "--job-id",
    )
    for token in forbidden:
        assert token not in source

    assert "download_stock_bars(" in source
    assert "download_corporate_actions(" in source
    assert "database_access=NONE" in source


def test_final_request_disables_database_market_cache() -> None:
    payload = json.loads(
        (
            ROOT
            / "config"
            / "final_research_alpaca_direct.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["market_data_provider"] == "alpaca"
    assert payload["alpaca_historical_feed"] == "sip"
    assert payload["alpaca_adjustment"] == "raw"
    assert payload["mongo_cache_enabled"] is False
    assert payload["market_data_history_backfill_enabled"] is False
    assert payload["start_date"] == "2016-01-01"
    assert payload["end_date"] == "2026-09-17"
    assert payload["analysis_end_date"] == "2026-09-17"
    assert len(payload["assets"]) == 56
    assert payload["rotation_target_horizons"] == [
        5,
        10,
        20,
        40,
        60,
    ]
    assert payload["rotation_target_horizon_weights"] == [
        0.1,
        0.15,
        0.2,
        0.3,
        0.25,
    ]

    lightgbm = payload["research_model_settings"]["lightgbm"]
    assert lightgbm["n_estimators"] == 329
    assert lightgbm["learning_rate"] == 0.020731
    assert lightgbm["max_depth"] == 3
    assert lightgbm["num_leaves"] == 6
    assert lightgbm["min_child_weight"] == 5.0
    assert lightgbm["early_stopping_enabled"] is False


def test_corporate_action_symbol_matching_covers_identity_fields() -> None:
    actions = [
        {
            "id": "split-a",
            "action_type": "forward_split",
            "symbol": "AAA",
            "process_date": "2020-01-01",
        },
        {
            "id": "merger-doc",
            "action_type": "stock_merger",
            "acquiree_symbol": "DOC",
            "acquirer_symbol": "PEAK",
            "process_date": "2024-03-01",
        },
        {
            "id": "rename",
            "action_type": "name_change",
            "old_symbol": "OLD",
            "new_symbol": "NEW",
            "process_date": "2025-01-01",
        },
    ]

    doc = corporate_actions_for_symbol(actions, "DOC")
    peak = corporate_actions_for_symbol(actions, "PEAK")
    old = corporate_actions_for_symbol(actions, "OLD")
    new = corporate_actions_for_symbol(actions, "NEW")

    assert [item["id"] for item in doc] == ["merger-doc"]
    assert [item["id"] for item in peak] == ["merger-doc"]
    assert [item["id"] for item in old] == ["rename"]
    assert [item["id"] for item in new] == ["rename"]


def test_corporate_action_hash_is_deterministic_for_canonical_order() -> None:
    first = [
        {
            "id": "a",
            "action_type": "forward_split",
            "symbol": "AAA",
            "process_date": "2020-01-01",
        },
        {
            "id": "b",
            "action_type": "cash_dividend",
            "symbol": "AAA",
            "process_date": "2020-02-01",
        },
    ]
    second = [dict(item) for item in first]

    assert (
        corporate_actions_sha256(first)
        == corporate_actions_sha256(second)
    )
