from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest import TestCase

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.services.diagnostics.operation_peak import (
    distance_vs_post_exit_scatter_figure,
    enrich_operation_peak_trades,
    figure_png_bytes,
    metric_bar_figures,
    operation_peak_rows,
    operation_peak_summary,
    peak_distance_boxplot_figures,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=25, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 105.0, 142.0] + [150.0 + i for i in range(22)],
            "high": [110.0, 150.0, 160.0, 165.0, 162.0, 170.0, 168.0]
            + [171.0 + i for i in range(18)],
            "low": [95.0, 104.0, 138.0] + [145.0 + i for i in range(22)],
            "close": [105.0, 140.0, 155.0] + [151.0 + i for i in range(22)],
            "volume": [1000.0] * 25,
        },
        index=index,
    )
    return frame


class OperationPeakAnalysisTests(TestCase):
    def test_normal_sell_stops_peak_at_exit_open_and_tracks_post_exit_upside(self):
        bars = _bars()
        trades = pd.DataFrame(
            [
                {
                    "sequence": 1,
                    "timestamp": bars.index[2],
                    "action": "SELL",
                    "asset": "AAPL",
                    "reason": "ROTATE",
                    "execution_price": 142.0,
                    "position_return": 0.42,
                    "holding_bars": 2,
                    "entry_timestamp": bars.index[0],
                    "entry_price": 100.0,
                    "walk_forward_fold": 1,
                }
            ]
        )

        enriched = enrich_operation_peak_trades(trades, {"AAPL": bars})
        row = enriched.iloc[0]

        self.assertAlmostEqual(row["peak_price_while_held"], 150.0)
        self.assertEqual(
            pd.Timestamp(row["peak_timestamp_while_held"]),
            bars.index[1],
        )
        self.assertAlmostEqual(
            row["exit_distance_from_peak_pct"],
            (142.0 / 150.0 - 1.0) * 100.0,
        )
        self.assertAlmostEqual(row["exit_peak_proximity_pct"], 142.0 / 150.0 * 100.0)
        self.assertAlmostEqual(row["peak_capture_pct"], 84.0)
        self.assertAlmostEqual(row["max_runup_pct"], 50.0)
        self.assertAlmostEqual(row["max_drawdown_from_entry_pct"], -5.0)
        self.assertEqual(row["sessions_from_peak_to_exit"], 1)
        self.assertEqual(row["calendar_days_from_peak_to_exit"], 1)

        # Five post-exit sessions start on the exit session itself because the
        # position was sold at that session's open. Their max high is 170.
        self.assertAlmostEqual(
            row["post_exit_peak_5d_pct"],
            (170.0 / 142.0 - 1.0) * 100.0,
        )
        self.assertEqual(row["post_exit_available_sessions_5d"], 5)

    def test_final_sell_includes_final_session_high(self):
        bars = _bars()
        trades = pd.DataFrame(
            [
                {
                    "sequence": 1,
                    "timestamp": bars.index[2],
                    "action": "FINAL_SELL",
                    "asset": "AAPL",
                    "reason": "FINAL_LIQUIDATION",
                    "execution_price": 155.0,
                    "position_return": 0.55,
                    "holding_bars": 3,
                    "entry_timestamp": bars.index[0],
                    "entry_price": 100.0,
                    "walk_forward_fold": 1,
                }
            ]
        )

        enriched = enrich_operation_peak_trades(trades, {"AAPL": bars})
        row = enriched.iloc[0]
        self.assertAlmostEqual(row["peak_price_while_held"], 160.0)
        self.assertEqual(
            pd.Timestamp(row["peak_timestamp_while_held"]),
            bars.index[2],
        )
        self.assertAlmostEqual(row["peak_capture_pct"], (55.0 / 60.0) * 100.0)
        self.assertEqual(row["sessions_from_peak_to_exit"], 0)

        # FINAL_SELL is at the close, so post-exit starts on the next session.
        self.assertEqual(
            pd.Timestamp(row["post_exit_peak_5d_timestamp"]),
            bars.index[5],
        )

    def test_summary_and_graphs_are_generated_per_asset(self):
        bars = _bars()
        trades = pd.DataFrame(
            [
                {
                    "sequence": 1,
                    "timestamp": bars.index[2],
                    "action": "SELL",
                    "asset": "AAPL",
                    "execution_price": 142.0,
                    "position_return": 0.42,
                    "holding_bars": 2,
                    "entry_timestamp": bars.index[0],
                    "entry_price": 100.0,
                },
                {
                    "sequence": 2,
                    "timestamp": bars.index[3],
                    "action": "SELL",
                    "asset": "NVDA",
                    "execution_price": 150.0,
                    "position_return": 0.25,
                    "holding_bars": 2,
                    "entry_timestamp": bars.index[1],
                    "entry_price": 120.0,
                },
            ]
        )
        frames = {"AAPL": bars, "NVDA": bars}
        enriched = enrich_operation_peak_trades(trades, frames)
        rows = operation_peak_rows(enriched)
        summary = operation_peak_summary(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["asset"] for row in summary}, {"AAPL", "NVDA"})
        self.assertTrue(
            all(row["n_operations"] == 1 for row in summary)
        )

        boxplots = peak_distance_boxplot_figures(rows, page_size=20)
        self.assertEqual(len(boxplots), 1)
        self.assertGreater(len(figure_png_bytes(boxplots[0])), 1000)

        bars_fig = metric_bar_figures(
            summary,
            metric="median_peak_capture_pct",
            title="Peak capture",
            xlabel="Percent",
        )
        self.assertEqual(len(bars_fig), 1)
        self.assertGreater(len(figure_png_bytes(bars_fig[0])), 1000)

        scatter = distance_vs_post_exit_scatter_figure(summary, top_n=25)
        self.assertIsNotNone(scatter)
        self.assertGreater(len(figure_png_bytes(scatter)), 1000)


if __name__ == "__main__":
    import unittest

    unittest.main()
