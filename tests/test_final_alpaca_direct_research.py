from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_final_alpaca_direct import (
    DEFAULT_CONFIG,
    _actions_for_symbol,
    _canonical_actions_bytes,
    _frame_csv_bytes,
    _load_request,
    _split_normalize,
    _structural_identity_issue,
)


def test_final_request_is_local_and_disables_mongo_cache() -> None:
    request, document = _load_request(Path(DEFAULT_CONFIG))

    assert document["schema_version"] == 1
    assert request.market_data_provider == "alpaca"
    assert request.alpaca_historical_feed == "sip"
    assert request.alpaca_adjustment == "raw"
    assert request.mongo_cache_enabled is False
    assert request.analysis_end_date == "2026-09-17"
    assert request.research_model_settings["lightgbm"]["n_estimators"] == 329
    assert len(request.assets) == 56


def test_raw_csv_snapshot_serialization_is_stable() -> None:
    index = pd.to_datetime(
        ["2026-01-05T05:00:00Z", "2026-01-06T05:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000.0, 1100.0],
        },
        index=index,
    )
    frame.index.name = "timestamp"

    first = _frame_csv_bytes(frame)
    second = _frame_csv_bytes(frame)

    assert first == second
    assert b"timestamp,open,high,low,close,volume" in first


def test_split_normalization_adjusts_only_pre_ex_date_rows() -> None:
    index = pd.to_datetime(
        [
            "2026-01-05T05:00:00Z",
            "2026-01-06T05:00:00Z",
            "2026-01-07T05:00:00Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame(
        {
            "open": [100.0, 102.0, 51.0],
            "high": [101.0, 103.0, 52.0],
            "low": [99.0, 101.0, 50.0],
            "close": [100.0, 102.0, 51.0],
            "volume": [1000.0, 1100.0, 2200.0],
        },
        index=index,
    )
    actions = [
        {
            "action_type": "forward_split",
            "ex_date": "2026-01-07",
            "process_date": "2026-01-06",
            "old_rate": 1.0,
            "new_rate": 2.0,
        }
    ]

    normalized, applied = _split_normalize(raw, actions)

    assert len(applied) == 1
    assert np.isclose(normalized.iloc[0]["close"], 50.0)
    assert np.isclose(normalized.iloc[1]["close"], 51.0)
    assert np.isclose(normalized.iloc[2]["close"], 51.0)
    assert np.isclose(normalized.iloc[0]["volume"], 2000.0)
    assert np.isclose(normalized.iloc[2]["volume"], 2200.0)


def test_structural_identity_guard_detects_acquiree_merger() -> None:
    actions = [
        {
            "action_type": "stock_merger",
            "acquiree_symbol": "DOC",
            "acquirer_symbol": "PEAK",
            "process_date": "2024-03-01",
            "effective_date": "2024-03-01",
        }
    ]

    issue = _structural_identity_issue("DOC", actions)

    assert issue is not None
    assert issue["reason"] == "structural_identity_change"
    assert issue["acquirer_symbol"] == "PEAK"


def test_actions_for_symbol_covers_identity_fields_and_serializes_stably() -> None:
    documents = [
        {
            "id": "b",
            "action_type": "name_change",
            "old_symbol": "PEAK",
            "new_symbol": "DOC",
            "process_date": "2024-03-04",
        },
        {
            "id": "a",
            "action_type": "stock_merger",
            "acquiree_symbol": "DOC",
            "acquirer_symbol": "PEAK",
            "process_date": "2024-03-01",
        },
    ]

    selected = _actions_for_symbol(documents, "DOC")

    assert len(selected) == 2
    assert selected[0]["id"] == "a"
    assert _canonical_actions_bytes(documents) == _canonical_actions_bytes(documents)
