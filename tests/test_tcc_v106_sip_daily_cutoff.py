"""SIP historical entitlement: bound recent daily request end to a safe XNYS close."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from market_cycle_trader_api.engine import market_data


class SIPDailyHistoricalEndTests(unittest.TestCase):
    def test_september_28_recent_sip_ends_at_close_not_future_midnight(self) -> None:
        end, mode = market_data.daily_historical_api_end(
            "2026-09-28", feed="sip",
            now="2026-09-28T20:25:00Z",
        )
        self.assertEqual(end, pd.Timestamp("2026-09-28T20:00:00Z"))
        self.assertEqual(mode, "completed_xnys_session_close_utc")
        self.assertEqual(
            market_data.normalize_end_date("2026-09-28"), "2026-09-28",
        )

    def test_safe_buffer_is_enforced_not_silently_stale(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RecentSIPDailyBarNotYetSafe"):
            market_data.daily_historical_api_end(
                "2026-09-28", feed="sip",
                now="2026-09-28T20:19:59Z",
            )
        end, _ = market_data.daily_historical_api_end(
            "2026-09-28", feed="sip",
            now="2026-09-28T20:20:00Z",
        )
        self.assertEqual(end, pd.Timestamp("2026-09-28T20:00:00Z"))

    def test_older_sip_keeps_original_tcc_single_request_boundary(self) -> None:
        end, mode = market_data.daily_historical_api_end(
            "2026-09-17", feed="sip",
            now="2026-09-28T20:25:00Z",
        )
        self.assertEqual(end, pd.Timestamp("2026-09-18T00:00:00Z"))
        self.assertEqual(mode, "cutoff_plus_one_day_utc")

    def test_iex_is_not_silently_substituted_for_sip(self) -> None:
        end, mode = market_data.daily_historical_api_end(
            "2026-09-28", feed="iex",
            now="2026-09-28T20:25:00Z",
        )
        self.assertEqual(end, pd.Timestamp("2026-09-29T00:00:00Z"))
        self.assertEqual(mode, "cutoff_plus_one_day_utc")

    def test_recent_sip_uses_actual_early_close_and_previous_session(self) -> None:
        end, mode = market_data.daily_historical_api_end(
            "2026-11-27", feed="sip",
            now="2026-11-27T18:25:00Z",
        )
        self.assertEqual(end, pd.Timestamp("2026-11-27T18:00:00Z"))
        self.assertEqual(mode, "completed_xnys_session_close_utc")
        weekend, _ = market_data.daily_historical_api_end(
            "2026-09-27", feed="sip",
            now="2026-09-27T23:50:00Z",
        )
        self.assertEqual(weekend, pd.Timestamp("2026-09-25T20:00:00Z"))

    def test_full_raw_sip_download_uses_bounded_end_and_keeps_last_bar(self) -> None:
        config = SimpleNamespace(
            timeframe="1Day",
            research_market_data_protocol="raw_total_causal_v1",
            research_market_data_refresh_mode="full",
            alpaca_historical_feed="sip",
            alpaca_adjustment="raw",
        )
        last = pd.Timestamp("2026-09-28T04:00:00Z")
        frame = pd.DataFrame(
            {"open": [10.0], "high": [11.0], "low": [9.0],
             "close": [10.5], "volume": [1000.0]},
            index=pd.DatetimeIndex([last]),
        )
        with (
            patch.object(
                market_data, "get_alpaca_credentials",
                return_value={"api_key_id": "test-key", "secret_key": "test-secret"},
            ),
            patch.object(
                market_data, "daily_historical_api_end",
                return_value=(
                    pd.Timestamp("2026-09-28T20:00:00Z"),
                    "completed_xnys_session_close_utc",
                ),
            ) as bounded,
            patch.object(
                market_data, "download_stock_bars", return_value=frame,
            ) as downloader,
        ):
            result = market_data._download_alpaca_bars(
                "NVDA", config, "2016-01-01", "2026-09-28",
            )
        bounded.assert_called_once_with("2026-09-28", feed="sip")
        kwargs = downloader.call_args.kwargs
        self.assertEqual(kwargs["feed"], "sip")
        self.assertEqual(kwargs["adjustment"], "raw")
        self.assertEqual(kwargs["limit"], 10_000)
        self.assertEqual(
            pd.Timestamp(kwargs["end"]), pd.Timestamp("2026-09-28T20:00:00Z"),
        )
        self.assertEqual(result.index.max(), last)
        self.assertEqual(
            result.attrs["research_bar_end_mode"],
            "completed_xnys_session_close_utc",
        )
        self.assertEqual(
            result.attrs["research_bar_api_end_utc"],
            "2026-09-28T20:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
