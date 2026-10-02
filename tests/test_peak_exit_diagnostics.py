from __future__ import annotations

import unittest

import pandas as pd

from market_cycle_trader_api.engine.rotation_diagnostics import _peak_exit_diagnostics
from market_cycle_trader_api.services.analytics import _peak_exit_analysis


class PeakExitDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.to_datetime(
            [
                "2026-01-05T05:00:00Z",
                "2026-01-06T05:00:00Z",
                "2026-01-07T05:00:00Z",
                "2026-01-08T05:00:00Z",
                "2026-01-09T05:00:00Z",
                "2026-01-12T05:00:00Z",
                "2026-01-13T05:00:00Z",
                "2026-01-14T05:00:00Z",
            ],
            utc=True,
        )
        self.frames = {
            "AAA": pd.DataFrame(
                {
                    "open": [10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5],
                    "high": [12.0, 15.0, 18.0, 14.0, 15.0, 16.0, 15.0, 15.5],
                    "low": [9.5, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
                    "close": [11.0, 12.0, 14.0, 13.0, 14.0, 15.0, 14.5, 15.0],
                },
                index=self.index,
            )
        }

    def test_normal_sell_peak_stops_before_exit_session(self) -> None:
        result = _peak_exit_diagnostics(
            self.frames,
            "AAA",
            self.index[0],
            self.index[2],
            10.0,
            12.0,
            "SELL",
        )

        self.assertEqual(result["peak_exit_diagnostics_schema_version"], 1)
        self.assertAlmostEqual(result["peak_price_while_held"], 15.0)
        self.assertEqual(pd.Timestamp(result["peak_timestamp_while_held"]), self.index[1])
        self.assertAlmostEqual(result["max_runup_pct"], 50.0)
        self.assertAlmostEqual(result["exit_distance_from_peak_pct"], 20.0)
        self.assertAlmostEqual(result["peak_capture_pct"], 40.0)
        self.assertEqual(result["days_from_peak_to_exit"], 1)
        self.assertAlmostEqual(result["post_exit_peak_5d_pct"], 50.0)
        self.assertIsNone(result["post_exit_peak_10d_pct"])
        self.assertIsNone(result["post_exit_peak_20d_pct"])

    def test_final_sell_includes_final_session_high(self) -> None:
        result = _peak_exit_diagnostics(
            self.frames,
            "AAA",
            self.index[0],
            self.index[2],
            10.0,
            14.0,
            "FINAL_SELL",
        )

        self.assertAlmostEqual(result["peak_price_while_held"], 18.0)
        self.assertEqual(pd.Timestamp(result["peak_timestamp_while_held"]), self.index[2])
        self.assertAlmostEqual(result["max_runup_pct"], 80.0)
        self.assertAlmostEqual(result["exit_distance_from_peak_pct"], 100.0 * (18.0 - 14.0) / 18.0)
        self.assertAlmostEqual(result["peak_capture_pct"], 50.0)
        self.assertEqual(result["days_from_peak_to_exit"], 0)
        self.assertAlmostEqual(result["post_exit_peak_5d_pct"], 100.0 * (16.0 / 14.0 - 1.0))

    def test_exit_above_previous_peak_is_treated_as_full_capture(self) -> None:
        result = _peak_exit_diagnostics(
            self.frames,
            "AAA",
            self.index[0],
            self.index[1],
            10.0,
            13.0,
            "SELL",
        )

        self.assertAlmostEqual(result["peak_price_while_held"], 12.0)
        self.assertEqual(result["exit_distance_from_peak_pct"], 0.0)
        self.assertEqual(result["peak_capture_pct"], 100.0)

    def test_asset_aggregation_uses_only_frozen_schema_rows(self) -> None:
        rows = [
            {
                "sequence": 2,
                "asset": "AAA",
                "action": "SELL",
                "peak_exit_diagnostics_schema_version": 1,
                "exit_distance_from_peak_pct": 2.0,
                "peak_capture_pct": 70.0,
                "max_runup_pct": 8.0,
                "days_from_peak_to_exit": 1,
                "post_exit_peak_5d_pct": 3.0,
                "post_exit_peak_10d_pct": 5.0,
                "post_exit_peak_20d_pct": 9.0,
            },
            {
                "sequence": 4,
                "asset": "AAA",
                "action": "SELL",
                "peak_exit_diagnostics_schema_version": 1,
                "exit_distance_from_peak_pct": 6.0,
                "peak_capture_pct": 30.0,
                "max_runup_pct": 12.0,
                "days_from_peak_to_exit": 3,
                "post_exit_peak_5d_pct": 7.0,
                "post_exit_peak_10d_pct": 11.0,
                "post_exit_peak_20d_pct": None,
            },
            {
                "sequence": 6,
                "asset": "OLD",
                "action": "SELL",
                "exit_distance_from_peak_pct": 99.0,
            },
        ]

        analysis = _peak_exit_analysis(rows)
        self.assertEqual(analysis["summary"]["closed_positions"], 2)
        self.assertEqual(analysis["summary"]["assets"], 1)
        self.assertAlmostEqual(analysis["summary"]["median_exit_distance_from_peak_pct"], 4.0)
        self.assertAlmostEqual(analysis["summary"]["median_peak_capture_pct"], 50.0)
        self.assertEqual(analysis["by_asset"][0]["asset"], "AAA")
        self.assertEqual(analysis["by_asset"][0]["closed_positions"], 2)
        self.assertEqual(analysis["by_asset"][0]["post_exit_20d_observations"], 1)


if __name__ == "__main__":
    unittest.main()
