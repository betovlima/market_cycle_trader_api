from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import research_final_standalone_alpaca as final_research


ROOT = Path(__file__).resolve().parents[1]


def test_final_config_is_database_free_and_frozen(monkeypatch) -> None:
    monkeypatch.setenv("MCT_ROTATION_ACCELERATOR", "cuda")
    monkeypatch.setenv("MCT_ROTATION_ALLOW_CPU_FALLBACK", "false")
    document, request = final_research._load_config(
        ROOT / "research" / "final_research_v10_8_76.json"
    )

    assert document["schema_version"] == 1
    assert request.research_market_data_mode == "standalone_snapshot"
    assert request.mongo_cache_enabled is False
    assert request.alpaca_adjustment == "raw"
    assert request.end_date == "2026-09-18"
    assert request.analysis_end_date == "2026-09-18"
    assert request.deterministic_execution is False
    assert request.rotation_accelerator == "cuda"
    assert request.rotation_allow_cpu_fallback is False
    assert request.xgb_n_jobs == 1
    assert request.numeric_thread_limit == 1
    assert len(request.assets) == 56
    assert (
        request.research_model_settings["soft_horizon_consensus"][
            "penalty_strength"
        ]
        == 1.0
    )


def test_final_config_has_explicit_clmt_structural_exclusion(monkeypatch) -> None:
    monkeypatch.setenv("MCT_ROTATION_ACCELERATOR", "cuda")
    monkeypatch.setenv("MCT_ROTATION_ALLOW_CPU_FALLBACK", "false")
    document, _ = final_research._load_config(
        ROOT / "research" / "final_research_v10_8_76.json"
    )
    exclusions = final_research._explicit_structural_exclusions(document)

    assert "CLMT" in exclusions
    assert (
        exclusions["CLMT"]["reason"]
        == "known_structural_history_identity_discontinuity"
    )
    assert (
        exclusions["CLMT"]["policy"]
        == "exclude_do_not_bridge_or_reconstruct"
    )


def test_final_runner_has_no_mongo_database_dependency() -> None:
    source = (
        ROOT
        / "scripts"
        / "research_final_standalone_alpaca.py"
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

    converted = final_research._bar_record(row)

    assert converted["timestamp"].startswith("2026-09-18T04:00:00")
    assert converted["open"] == 100.125
    assert converted["high"] == 102.5
    assert converted["low"] == 99.75
    assert converted["close"] == 101.375
    assert converted["volume"] == 123456.0
    assert converted["trade_count"] == 987
    assert converted["vwap"] == 101.02


def test_local_split_normalization_changes_only_pre_ex_date_units() -> None:
    index = pd.to_datetime(
        [
            "2026-01-02T00:00:00Z",
            "2026-01-05T00:00:00Z",
            "2026-01-06T00:00:00Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame(
        {
            "open": [100.0, 102.0, 51.0],
            "high": [101.0, 103.0, 52.0],
            "low": [99.0, 101.0, 50.0],
            "close": [100.0, 102.0, 51.0],
            "volume": [1000.0, 1200.0, 2500.0],
        },
        index=index,
    )
    actions = [
        {
            "action_type": "forward_split",
            "ex_date": "2026-01-06",
            "process_date": "2026-01-05",
            "old_rate": 1,
            "new_rate": 2,
        }
    ]

    normalized, applied = final_research._split_normalize(
        raw,
        actions,
    )

    assert np.isclose(normalized.iloc[0]["close"], 50.0)
    assert np.isclose(normalized.iloc[1]["close"], 51.0)
    assert np.isclose(normalized.iloc[2]["close"], 51.0)
    assert np.isclose(normalized.iloc[0]["volume"], 2000.0)
    assert np.isclose(normalized.iloc[2]["volume"], 2500.0)
    assert len(applied) == 1


def test_split_normalization_accepts_integer_ohlcv_and_fractional_adjustments() -> None:
    index = pd.to_datetime(
        [
            "2026-01-02T00:00:00Z",
            "2026-01-05T00:00:00Z",
            "2026-01-06T00:00:00Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame(
        {
            "open": [101, 103, 69],
            "high": [102, 104, 70],
            "low": [100, 102, 68],
            "close": [101, 103, 69],
            "volume": [1001, 1201, 2501],
        },
        index=index,
        dtype="int64",
    )
    actions = [
        {
            "action_type": "forward_split",
            "ex_date": "2026-01-06",
            "process_date": "2026-01-05",
            "old_rate": 2,
            "new_rate": 3,
        }
    ]

    normalized, applied = final_research._split_normalize(raw, actions)

    assert all(normalized[column].dtype.kind == "f" for column in final_research.OHLCV)
    assert np.isclose(normalized.iloc[0]["close"], 101.0 * (2.0 / 3.0))
    assert np.isclose(normalized.iloc[0]["volume"], 1001.0 * 1.5)
    assert np.isclose(normalized.iloc[2]["close"], 69.0)
    assert np.isclose(normalized.iloc[2]["volume"], 2501.0)
    assert len(applied) == 1


def test_structural_identity_guard_excludes_acquiree_ticker_reuse() -> None:
    issue = final_research._structural_identity_issue(
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

    assert (
        final_research._canonical_sha256(left)
        == final_research._canonical_sha256(right)
    )
