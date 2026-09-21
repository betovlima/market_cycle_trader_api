from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import research_soft_horizon_7m_direct_alpaca as research


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_7m_config_preserves_10_8_74_request() -> None:
    document, request = research._load_config(
        ROOT / "research" / "soft_horizon_7m_direct_alpaca_v10_8_83.json"
    )

    assert document["schema_version"] == 1
    assert document["lineage"]["base_api_version"] == "10.8.74"
    assert (
        document["lineage"]["base_commit"]
        == "05b765df0496c905a4195948027a9b3b7adf2bce"
    )
    assert document["lineage"]["change_scope"] == "market_data_transport_only_plus_prevalidated_gpu_backend"
    assert request.research_market_data_mode == "database_only"
    assert request.mongo_cache_enabled is False
    assert request.alpaca_adjustment == "all"
    assert request.start_date == "2016-01-01"
    assert request.end_date is None
    assert request.analysis_end_date == "2026-09-17"
    assert request.rotation_accelerator == "cuda"
    assert request.rotation_allow_cpu_fallback is False
    assert request.deterministic_execution is False
    assert request.xgb_n_jobs == 1
    assert request.numeric_thread_limit == 1
    assert len(request.assets) == 56
    assert "DOC" in request.assets
    assert "CLMT" in request.assets
    assert document["methodology"]["soft_horizon_consensus_penalty_strength"] == 1.0
    assert request.research_model_settings["schema_version"] == 3
    assert request.research_model_settings["settings_revision"] == 2
    assert request.research_model_settings["profile_id"] == "strategy"
    lightgbm = request.research_model_settings["lightgbm"]
    assert lightgbm["n_estimators"] == 329
    assert lightgbm["n_jobs"] == -1
    assert lightgbm["repetitions"] == 1
    assert lightgbm["seed_step"] == 1000
    assert lightgbm["random_state"] == 42
    assert "soft_horizon_consensus" not in request.research_model_settings


def test_direct_alpaca_transport_is_raw_sip_and_never_all() -> None:
    document, request = research._load_config(
        ROOT / "research" / "soft_horizon_7m_direct_alpaca_v10_8_81.json"
    )

    assert request.alpaca_historical_feed == "sip"
    assert request.timeframe == "1Day"
    assert request.alpaca_adjustment == "all"
    assert document["data_source"]["bars_adjustment"] == "raw"
    assert document["data_source"]["download_adjustment"] == "raw"
    assert document["data_source"]["bar_snapshot_as_of_end"] == "2026-09-17"

    source = (
        ROOT
        / "scripts"
        / "research_soft_horizon_7m_direct_alpaca.py"
    ).read_text(encoding="utf-8")

    assert "download_stock_bars(" in source
    assert 'adjustment = "raw"' in source
    assert '"adjustment": "all"' not in source
    assert "Adjustment.ALL" not in source


def test_direct_runner_has_no_mongo_database_dependency() -> None:
    source = (
        ROOT
        / "scripts"
        / "research_soft_horizon_7m_direct_alpaca.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "mongo_repository",
        "create_client(",
        "get_database(",
        "_latest_job(",
    )
    for token in forbidden:
        assert token not in source


def test_bar_record_preserves_raw_ohlcv() -> None:
    row = {
        "t": "2026-09-18T04:00:00Z",
        "o": 100.125,
        "h": 102.5,
        "l": 99.75,
        "c": 101.375,
        "v": 123456,
        "n": 987,
        "vw": 101.02,
    }

    converted = research._bar_record(row)

    assert converted["timestamp"].startswith("2026-09-18T04:00:00")
    assert converted["open"] == 100.125
    assert converted["high"] == 102.5
    assert converted["low"] == 99.75
    assert converted["close"] == 101.375
    assert converted["volume"] == 123456.0


def test_local_csv_loader_keeps_split_adjustable_ohlcv_float(tmp_path: Path) -> None:
    target = tmp_path / "TEST.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02T00:00:00Z",
                "2026-01-05T00:00:00Z",
            ],
            "open": [101, 103],
            "high": [102, 104],
            "low": [100, 102],
            "close": [101, 103],
            "volume": [1001, 1201],
            "trade_count": [100, 120],
            "vwap": [101.1, 103.1],
        }
    ).to_csv(target, index=False)

    frame = research._load_raw_bar_file(target)

    for column in ("open", "high", "low", "close", "volume"):
        assert frame[column].dtype.kind == "f"

    normalized, applied = research._split_normalize(
        frame,
        [
            {
                "action_type": "forward_split",
                "ex_date": "2026-01-06",
                "process_date": "2026-01-05",
                "old_rate": 2,
                "new_rate": 3,
            }
        ],
    )

    assert np.isclose(normalized.iloc[0]["close"], 101.0 * (2.0 / 3.0))
    assert np.isclose(normalized.iloc[0]["volume"], 1001.0 * 1.5)
    assert len(applied) == 1


def test_structural_identity_guard_matches_10_8_74_doc_rule() -> None:
    issue = research._structural_identity_issue(
        "DOC",
        [
            {
                "action_type": "stock_merger",
                "acquiree_symbol": "DOC",
                "acquirer_symbol": "PEAK",
                "process_date": "2024-03-01",
                "effective_date": "2024-03-01",
            }
        ],
    )

    assert issue is not None
    assert issue["reason"] == "structural_identity_change"
    assert issue["acquirer_symbol"] == "PEAK"


def test_snapshot_hash_is_order_stable() -> None:
    left = {"b": [2, 1], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 1]}

    assert research._canonical_sha256(left) == research._canonical_sha256(right)
