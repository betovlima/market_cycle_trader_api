from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.research_market_data import (
    RAW_TOTAL_CAUSAL_PROTOCOL,
    effective_research_config,
    split_normalize,
    structural_identity_issue,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
from market_cycle_trader_api.services.reproducibility import (
    market_data_manifest,
)


@dataclass(frozen=True)
class _Config:
    alpaca_adjustment: str = "all"
    research_market_data_protocol: str = RAW_TOTAL_CAUSAL_PROTOCOL

    def model_copy(self, *, update: dict[str, object]):
        return replace(self, **update)


class RawResearchDataParityTests(unittest.TestCase):
    def test_research_protocol_defaults_to_raw_total_causal(self) -> None:
        field = BacktestExecutionRequest.model_fields[
            "research_market_data_protocol"
        ]
        self.assertEqual(field.default, RAW_TOTAL_CAUSAL_PROTOCOL)

    def test_effective_research_config_forces_raw_without_mutating_source(self) -> None:
        original = _Config(alpaca_adjustment="all")

        effective = effective_research_config(original)

        self.assertEqual(original.alpaca_adjustment, "all")
        self.assertEqual(effective.alpaca_adjustment, "raw")
        self.assertEqual(
            effective.research_market_data_protocol,
            RAW_TOTAL_CAUSAL_PROTOCOL,
        )

    def test_split_normalization_matches_tcc_rule(self) -> None:
        index = pd.to_datetime(
            [
                "2020-08-28 04:00:00+00:00",
                "2020-08-31 04:00:00+00:00",
                "2020-09-01 04:00:00+00:00",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "open": [400.0, 100.0, 105.0],
                "high": [404.0, 104.0, 109.0],
                "low": [396.0, 96.0, 101.0],
                "close": [402.0, 102.0, 107.0],
                "volume": [1_000.0, 4_000.0, 5_000.0],
            },
            index=index,
        )
        actions = [
            {
                "action_type": "forward_split",
                "ex_date": "2020-08-31",
                "old_rate": 1.0,
                "new_rate": 4.0,
                "process_date": "2020-08-31",
            }
        ]

        normalized, applied = split_normalize(frame, actions)

        self.assertEqual(len(applied), 1)
        self.assertAlmostEqual(normalized.iloc[0]["open"], 100.0)
        self.assertAlmostEqual(normalized.iloc[0]["close"], 100.5)
        self.assertAlmostEqual(normalized.iloc[0]["volume"], 4_000.0)
        self.assertAlmostEqual(normalized.iloc[1]["open"], 100.0)
        self.assertAlmostEqual(normalized.iloc[1]["close"], 102.0)
        self.assertAlmostEqual(normalized.iloc[1]["volume"], 4_000.0)
        self.assertAlmostEqual(normalized.iloc[2]["open"], 105.0)
        self.assertAlmostEqual(normalized.iloc[2]["volume"], 5_000.0)
        self.assertEqual(
            applied[0]["normalization_direction"],
            "pre_ex_date_history",
        )

    def test_dividend_does_not_adjust_model_ohlcv(self) -> None:
        index = pd.to_datetime(
            [
                "2024-01-02 05:00:00+00:00",
                "2024-01-03 05:00:00+00:00",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "open": [100.0, 99.0],
                "high": [101.0, 100.0],
                "low": [98.0, 97.0],
                "close": [100.0, 99.0],
                "volume": [1_000.0, 1_100.0],
            },
            index=index,
        )
        actions = [
            {
                "action_type": "cash_dividend",
                "ex_date": "2024-01-03",
                "cash": 1.0,
            }
        ]

        normalized, applied = split_normalize(frame, actions)

        pd.testing.assert_frame_equal(normalized, frame)
        self.assertEqual(applied, [])

    def test_structural_merger_excludes_acquiree_symbol(self) -> None:
        issue = structural_identity_issue(
            "DOC",
            [
                {
                    "action_type": "stock_merger",
                    "effective_date": "2024-03-01",
                    "process_date": "2024-03-01",
                    "acquiree_symbol": "DOC",
                    "acquirer_symbol": "PEAK",
                }
            ],
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "structural_identity_change")
        self.assertEqual(issue["symbol"], "DOC")
        self.assertEqual(issue["acquirer_symbol"], "PEAK")

    def test_structural_merger_does_not_exclude_acquirer(self) -> None:
        issue = structural_identity_issue(
            "PEAK",
            [
                {
                    "action_type": "stock_merger",
                    "effective_date": "2024-03-01",
                    "acquiree_symbol": "DOC",
                    "acquirer_symbol": "PEAK",
                }
            ],
        )
        self.assertIsNone(issue)

    def test_reproducibility_signature_includes_raw_protocol_and_splits(self) -> None:
        index = pd.date_range(
            "2020-01-02",
            periods=4,
            freq="B",
            tz="UTC",
        )
        frame = pd.DataFrame(
            {
                "open": np.arange(4, dtype=float) + 10.0,
                "high": np.arange(4, dtype=float) + 11.0,
                "low": np.arange(4, dtype=float) + 9.0,
                "close": np.arange(4, dtype=float) + 10.5,
                "volume": np.arange(4, dtype=float) + 1_000.0,
            },
            index=index,
        )
        frame.attrs["market_data_provenance"] = {
            "history_complete": True,
            "provider": "alpaca",
            "effective_provider": "alpaca",
            "historical_feed": "sip",
            "live_feed": "iex",
            "adjustment": "raw",
            "research_market_data_protocol": RAW_TOTAL_CAUSAL_PROTOCOL,
            "source_adjustment": "raw",
            "effective_adjustment": "raw_plus_causal_split_normalization",
            "corporate_action_count": 3,
            "splits_applied": 1,
            "split_normalization_direction": "pre_ex_date_history",
            "split_normalization_uses_future_events": True,
            "dividend_event_count": 2,
            "dividend_adjustment_applied": False,
            "dividend_events_used_by_model": False,
            "structural_identity_verified": True,
        }

        _, manifests = market_data_manifest({"AAA": frame})
        manifest = manifests["AAA"]

        self.assertEqual(
            manifest["research_market_data_protocol"],
            RAW_TOTAL_CAUSAL_PROTOCOL,
        )
        self.assertEqual(
            manifest["effective_adjustment"],
            "raw_plus_causal_split_normalization",
        )
        self.assertEqual(manifest["splits_applied"], 1)
        self.assertEqual(manifest["corporate_action_count"], 3)
        self.assertEqual(
            manifest["split_normalization_direction"],
            "pre_ex_date_history",
        )
        self.assertTrue(
            manifest["split_normalization_uses_future_events"]
        )
        self.assertEqual(manifest["dividend_event_count"], 2)
        self.assertFalse(manifest["dividend_adjustment_applied"])
        self.assertFalse(manifest["dividend_events_used_by_model"])


if __name__ == "__main__":
    unittest.main()
