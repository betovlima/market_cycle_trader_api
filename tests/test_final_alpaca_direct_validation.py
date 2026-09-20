from __future__ import annotations

import pandas as pd

from scripts.research_final_alpaca_direct_validation import (
    _bar_frame_from_payload,
    _split_normalize,
    _structural_identity_issue,
)


def test_bar_payload_is_normalized_without_adjustment() -> None:
    frame = _bar_frame_from_payload(
        [
            {
                "t": "2026-09-17T04:00:00Z",
                "o": 100.0,
                "h": 103.0,
                "l": 99.0,
                "c": 102.0,
                "v": 12345,
                "n": 100,
                "vw": 101.5,
            }
        ]
    )

    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    assert float(frame.iloc[0]["open"]) == 100.0
    assert float(frame.iloc[0]["close"]) == 102.0
    assert str(frame.index.tz) == "UTC"


def test_local_split_normalization_changes_only_pre_ex_date_history() -> None:
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
            "id": "split-1",
            "ex_date": "2026-01-06",
            "process_date": "2026-01-05",
            "old_rate": 1.0,
            "new_rate": 2.0,
        }
    ]

    normalized, applied = _split_normalize(
        raw,
        actions,
    )

    assert len(applied) == 1
    assert float(normalized.iloc[0]["close"]) == 50.0
    assert float(normalized.iloc[1]["close"]) == 51.0
    assert float(normalized.iloc[2]["close"]) == 51.0
    assert float(normalized.iloc[0]["volume"]) == 2000.0
    assert float(normalized.iloc[2]["volume"]) == 2500.0


def test_structural_identity_change_is_excluded() -> None:
    issue = _structural_identity_issue(
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
    assert issue["acquiree_symbol"] == "DOC"
    assert issue["acquirer_symbol"] == "PEAK"
