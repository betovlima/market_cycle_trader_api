"""Regression: Trading and Live score the same bars under the same rules.

Data-feed keys belong to the market-data pipeline; real broker account and
order execution remain isolated. No test sends orders to a broker.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.engine import operational_market_snapshot as engine
from market_cycle_trader_api.services.live_market_bars import load_live_intraday_session
from market_cycle_trader_api.services.live_trader_state import verified_live_state


SESSION = "2026-10-20"
NOW = pd.Timestamp("2026-10-20T15:10:00Z")


def _daily(symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame([{
        "open": 10.0, "high": 11.0, "low": 9.0,
        "close": 10.5, "volume": 1000.0,
    }], index=pd.DatetimeIndex([pd.Timestamp("2026-10-19T04:00:00Z")]))
    frame.attrs["symbol"] = symbol
    return frame


def _minutes() -> pd.DataFrame:
    return pd.DataFrame([
        {"open": 11.0, "high": 12.0, "low": 10.5, "close": 11.3,
         "volume": 300.0, "vwap": 11.2, "trade_count": 5},
        {"open": 11.3, "high": 11.8, "low": 11.1, "close": 11.7,
         "volume": 150.0, "vwap": 11.6, "trade_count": 3},
    ], index=pd.DatetimeIndex([
        pd.Timestamp("2026-10-20T13:30:00Z"),
        pd.Timestamp("2026-10-20T13:31:00Z"),
    ]))


class TradingLiveIntradayParity(unittest.TestCase):
    def setUp(self):
        self.historical = {"MAN": _daily("MAN"), "CAPR": _daily("CAPR")}
        self.config = SimpleNamespace(alpaca_live_feed="iex")
        self.calls = []
        credentials = patch.object(
            engine, "get_alpaca_credentials",
            return_value={"api_key_id": "paper-data-key", "secret_key": "paper-data-secret"},
        )
        credentials.start()
        self.addCleanup(credentials.stop)

    def _compare(self, source):
        self.calls.clear()
        with patch.object(engine, "download_stock_bars", side_effect=source) as fake:
            paper_data, paper_meta = engine.append_current_session_intraday(
                self.historical, self.config, session=SESSION, now=NOW,
                per_asset_pause_seconds=0,
            )
            paper_calls = list(fake.call_args_list)
            fake.reset_mock()
            live_data, live_meta = load_live_intraday_session(
                self.historical, self.config, session=SESSION, now=NOW,
            )
            live_calls = list(fake.call_args_list)
        self.assertEqual(paper_calls, live_calls)
        for key in self.historical:
            pd.testing.assert_frame_equal(paper_data[key], live_data[key])
            self.assertEqual(paper_data[key].attrs, live_data[key].attrs)
        self.assertEqual(live_meta["analysis_mode"], "live_current_session_intraday")
        self.assertEqual(live_meta["data_pipeline"], "trading_intraday_parity")
        for key, value in paper_meta.items():
            if key == "analysis_mode":
                continue
            self.assertEqual(live_meta[key], value)
        self.assertEqual({call.kwargs["api_key_id"] for call in live_calls}, {"paper-data-key"})
        return paper_meta, live_meta

    def test_iex_bars_are_identical(self):
        self._compare(lambda **kwargs: _minutes())

    def test_missing_iex_asset_uses_full_universe_sip_like_trading(self):
        def bars(**kwargs):
            if kwargs["feed"] == "iex" and kwargs["symbol"] == "CAPR":
                return pd.DataFrame()
            return _minutes()
        paper, live = self._compare(bars)
        self.assertEqual(paper["live_feed"], "sip")
        self.assertEqual(live["live_feed"], "sip")
        self.assertEqual(live["intraday_asset_count"], 2)

    def test_delayed_sip_entitlement_fallback_matches_trading(self):
        def bars(**kwargs):
            if kwargs["feed"] == "iex" and kwargs["symbol"] == "CAPR":
                return pd.DataFrame()
            if kwargs["feed"] == "sip" and kwargs["end"] > (NOW - pd.Timedelta(minutes=20)).to_pydatetime():
                raise RuntimeError("subscription does not permit querying recent SIP data")
            return _minutes()
        _, live = self._compare(bars)
        self.assertEqual(live["live_feed"], "sip")
        self.assertEqual(live["intraday_data_delay_minutes"], 20)
        self.assertEqual(live["intraday_market_data_cutoff_utc"], (NOW - pd.Timedelta(minutes=20)).isoformat())

    def test_both_feeds_missing_still_blocks_no_synthetic_bars(self):
        with patch.object(engine, "download_stock_bars", return_value=pd.DataFrame()):
            with self.assertRaisesRegex(RuntimeError, "^LiveIntradayCoverageNotReady:"):
                load_live_intraday_session(self.historical, self.config, session=SESSION, now=NOW)

    def test_no_historical_universe_still_blocks(self):
        with self.assertRaisesRegex(RuntimeError, "no historical universe"):
            load_live_intraday_session({}, self.config, session=SESSION, now=NOW)


class LiveHoldingSessionsMatchTrading(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db["live_trader_strategy_state"].find_one.return_value = {
            "_id": "live-a", "managed_symbol": "CAPR",
            "managed_quantity": 16.890370370, "entry_session": "2026-10-20",
            "revision": 4, "last_execution_session": "2026-10-20",
        }
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(id="live-a")
        self.broker.get_all_positions.return_value = [
            SimpleNamespace(symbol="CAPR", qty="16.890370370", side="long"),
        ]
        self.broker.get_clock.return_value = SimpleNamespace(
            timestamp=pd.Timestamp("2026-10-20T15:15:00Z").to_pydatetime(),
            is_open=True,
        )

    def test_entry_session_is_one_while_market_still_open(self):
        with patch("market_cycle_trader_api.services.live_trader_state.calendar_session_dates") as calendar:
            state = verified_live_state(self.db, self.broker, session="2026-10-20")
        self.assertEqual(state["holding_sessions"], 1)
        calendar.assert_not_called()

    def test_next_open_session_does_not_count_unfinished_day(self):
        self.db["live_trader_strategy_state"].find_one.return_value["entry_session"] = "2026-10-19"
        self.broker.get_clock.return_value.timestamp = pd.Timestamp("2026-10-20T15:15:00Z").to_pydatetime()
        with patch(
            "market_cycle_trader_api.services.live_trader_state.calendar_session_dates",
            return_value=["2026-10-19", "2026-10-20"],
        ) as calendar:
            state = verified_live_state(self.db, self.broker, session="2026-10-20")
        self.assertEqual(state["holding_sessions"], 1)
        self.assertEqual(calendar.call_args.kwargs["end_date"], "2026-10-19")

    def test_next_session_counts_only_previously_completed_day(self):
        self.broker.get_clock.return_value.timestamp = pd.Timestamp("2026-10-21T15:15:00Z").to_pydatetime()
        with patch(
            "market_cycle_trader_api.services.live_trader_state.calendar_session_dates",
            return_value=["2026-10-20"],
        ):
            state = verified_live_state(self.db, self.broker, session="2026-10-21")
        self.assertEqual(state["holding_sessions"], 1)

    def test_session_after_second_completed_day_counts_two(self):
        self.broker.get_clock.return_value.timestamp = pd.Timestamp("2026-10-22T15:15:00Z").to_pydatetime()
        with patch(
            "market_cycle_trader_api.services.live_trader_state.calendar_session_dates",
            return_value=["2026-10-20", "2026-10-21"],
        ) as calendar:
            state = verified_live_state(self.db, self.broker, session="2026-10-22")
        self.assertEqual(state["holding_sessions"], 2)
        self.assertEqual(calendar.call_args.kwargs["end_date"], "2026-10-21")

    def test_quantity_tolerance_matches_trading(self):
        self.broker.get_all_positions.return_value = [
            SimpleNamespace(symbol="CAPR", qty="16.890380370", side="long"),
        ]
        with patch(
            "market_cycle_trader_api.services.live_trader_state.calendar_session_dates",
        ) as calendar:
            calendar.return_value = []
            state = verified_live_state(self.db, self.broker, session="2026-10-20")
        self.assertEqual(state["managed_symbol"], "CAPR")

    def test_different_position_still_blocks(self):
        self.broker.get_all_positions.return_value = [
            SimpleNamespace(symbol="MAN", qty="16.890370370", side="long"),
        ]
        with self.assertRaisesRegex(RuntimeError, "differ"):
            verified_live_state(self.db, self.broker, session="2026-10-20")


if __name__ == "__main__":
    unittest.main()
