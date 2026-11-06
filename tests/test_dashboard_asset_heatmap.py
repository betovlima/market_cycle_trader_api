"""Winner daily heatmap: no fake quotes, no incomplete candles, no broker orders."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import dashboard_asset_heatmap as subject

PREFIX = "market_cycle_trader_api.services.dashboard_asset_heatmap."


def sample_bars():
    dates = pd.bdate_range(end="2026-11-05", periods=24, tz="UTC")
    records = [
        ("CAPR", day, 100 + i)
        for i, day in enumerate(dates)
    ]
    records.append(("DNN", dates[-2], 45.0))  # Missing latest completed bar.
    records.append(("CAPR", pd.Timestamp("2026-11-06", tz="UTC"), 999.0))  # Future!
    index = pd.MultiIndex.from_tuples(
        [(symbol, date) for symbol, date, _ in records],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame({"close": [price for _, _, price in records]}, index=index)


class DailyStockHeatmapTests(unittest.TestCase):
    def setUp(self):
        subject._CACHE.clear()
        self.db = MagicMock()
        self.strategy = SimpleNamespace(assets=["CAPR", "DNN", "AMZN", "CAPR"])
        self.profile = {"id": "winner-1", "name": "Winner", "revision": 6}
        self.patchers = [
            patch(PREFIX + "get_trader_winner_context",
                  return_value=(self.strategy, self.profile)),
            patch(PREFIX + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-11-05", tz="UTC")),
            patch(PREFIX + "get_alpaca_credentials",
                  return_value={"api_key_id": "mock", "secret_key": "mock"}),
            patch(PREFIX + "_fetch_completed_bars", return_value=sample_bars()),
        ]
        self.mocks = [item.start() for item in self.patchers]
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patchers)])

    def test_winner_universe_and_real_daily_change(self):
        result = subject.dashboard_asset_heatmap(self.db)
        self.assertEqual(result["as_of"], "2026-11-05")
        self.assertEqual(result["source"], "alpaca_sip_completed_daily_raw")
        self.assertEqual(result["total_assets"], 3)
        self.assertEqual(result["available_assets"], 1)
        self.assertEqual(result["timeframes"], [1, 5, 20])
        rows = {item["symbol"]: item for item in result["items"]}
        self.assertEqual(set(rows), {"CAPR", "DNN", "AMZN"})
        self.assertEqual(rows["CAPR"]["price"], 123.0)
        self.assertAlmostEqual(rows["CAPR"]["change_1d"], round(100 * (123 / 122 - 1), 4))
        self.assertAlmostEqual(rows["CAPR"]["change_5d"], round(100 * (123 / 118 - 1), 4))
        self.assertAlmostEqual(rows["CAPR"]["change_20d"], round(100 * (123 / 103 - 1), 4))
        self.assertEqual(rows["DNN"]["date"], "2026-11-04")
        self.assertTrue(rows["DNN"]["stale"])
        self.assertIsNone(rows["DNN"]["change_1d"])
        self.assertIsNone(rows["AMZN"]["price"])
        self.assertIsNone(rows["AMZN"]["change_20d"])
        self.assertEqual(self.mocks[3].call_count, 1)
        self.mocks[3].assert_called_with(
            ["CAPR", "DNN", "AMZN"],
            start=unittest.mock.ANY, end=unittest.mock.ANY,
            credentials={"api_key_id": "mock", "secret_key": "mock"},
        )

    def test_cache_prevents_repeated_provider_requests(self):
        first = subject.dashboard_asset_heatmap(self.db)
        second = subject.dashboard_asset_heatmap(self.db)
        self.assertIs(first, second)
        self.assertEqual(self.mocks[3].call_count, 1)

    def test_missing_or_invalid_symbols_are_not_fabricated(self):
        self.strategy.assets = ["CAPR", "$BAD", "", "CAPR"]
        result = subject.dashboard_asset_heatmap(self.db)
        self.assertEqual(result["total_assets"], 1)
        self.assertEqual(result["items"][0]["symbol"], "CAPR")

    def test_missing_all_symbols_is_rejected(self):
        self.strategy.assets = []
        with self.assertRaisesRegex(ValueError, "registered symbols"):
            subject.dashboard_asset_heatmap(self.db)
        self.mocks[3].assert_not_called()

    def test_endpoint_is_authenticated_and_read_only(self):
        from market_cycle_trader_api.api.routers import dashboard
        route = next(item for item in dashboard.router.routes
                     if item.path == "/api/dashboard/asset-heatmap")
        self.assertIn("GET", route.methods)
        self.assertNotIn("POST", route.methods)
        self.assertTrue(route.dependant.dependencies)


if __name__ == "__main__":
    unittest.main()
