"""Live market decisions use Alpaca's time, XNYS calendar, and no artificial caps."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.engine.market_data import latest_safe_completed_xnys_session
from market_cycle_trader_api.services import live_winner_signal
from market_cycle_trader_api.services.live_execution_window import (
    live_execution_window_deadline,
    live_execution_window_seconds,
)
from market_cycle_trader_api.services.live_market_bars import load_live_intraday_session


class LiveExchangeClockTests(TestCase):
    def test_monday_after_close_uses_monday_without_20_minute_wait(self):
        # Real market closes 16:00 EDT = 20:00 UTC on 2026-10-19.
        observed = datetime(2026, 10, 19, 20, 0, 10, tzinfo=timezone.utc)
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(timestamp=observed, is_open=False)
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:2026-10-20", "status": "waiting_for_winner_signal",
            "execution_session": "2026-10-20",
        }
        prefix = "market_cycle_trader_api.services.live_winner_signal."
        with (
            patch(prefix + "create_live_readonly_client", return_value=broker),
            patch(prefix + "latest_safe_completed_xnys_session",
                  wraps=latest_safe_completed_xnys_session) as cutoff,
            patch(prefix + "get_system_settings",
                  return_value={"training": {"enabled": False}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                live_winner_signal.prepare_live_winner_signal(db, "2026-10-20")
        self.assertEqual(cutoff.call_args.kwargs["now"], observed)
        self.assertEqual(cutoff.call_args.kwargs["data_delay_minutes"], 0)

    def test_clock_can_be_monday_even_when_test_process_is_tuesday(self):
        # The broker's timestamp, rather than the container's local date, is
        # always passed to the daily session selector.
        with patch("market_cycle_trader_api.services.live_winner_signal.create_live_readonly_client") as factory:
            clock = SimpleNamespace(
                timestamp=datetime(2026, 10, 19, 20, 1, tzinfo=timezone.utc),
                is_open=False,
            )
            factory.return_value.get_clock.return_value = clock
            db = MagicMock()
            db["live_winner_session_runs"].find_one.return_value = {
                "status": "waiting_for_winner_signal", "execution_session": "2026-10-20",
            }
            with (
                patch("market_cycle_trader_api.services.live_winner_signal.latest_safe_completed_xnys_session",
                      return_value=pd.Timestamp("2026-10-19")) as safe,
                patch("market_cycle_trader_api.services.live_winner_signal.get_system_settings",
                      return_value={"training": {"enabled": False}}),
            ):
                with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                    live_winner_signal.prepare_live_winner_signal(db, "2026-10-20")
            self.assertEqual(safe.call_args.kwargs["now"], clock.timestamp)
            self.assertEqual(safe.call_args.kwargs["data_delay_minutes"], 0)

    def test_open_is_not_artificially_restricted_to_90_minutes(self):
        start = pd.Timestamp("2026-10-20T13:30:00Z")
        close, duration = live_execution_window_deadline(start)
        self.assertEqual(close, pd.Timestamp("2026-10-20T20:00:00Z"))
        self.assertEqual(duration, 23400)
        self.assertGreater(duration, 5400)
        self.assertEqual(live_execution_window_seconds(), 23400)

    def test_early_close_is_resolved_by_exchange_calendar(self):
        close, duration = live_execution_window_deadline(
            pd.Timestamp("2026-11-27T14:30:00Z")
        )
        self.assertEqual(close, pd.Timestamp("2026-11-27T18:00:00Z"))
        self.assertEqual(duration, 12600)

    def test_no_arbitrary_wait_when_first_completed_minute_exists(self):
        bars = pd.DataFrame([
            {"open": 42, "high": 43, "low": 41, "close": 42.5, "volume": 20}
        ], index=pd.DatetimeIndex([pd.Timestamp("2026-10-20T13:30:00Z")]))
        daily = pd.DataFrame([
            {"open": 40, "high": 42, "low": 39, "close": 41, "volume": 30}
        ], index=pd.DatetimeIndex([pd.Timestamp("2026-10-19T04:00:00Z")]))
        with (
            patch("market_cycle_trader_api.services.live_market_bars.live_credentials",
                  return_value=("LIVE", "SECRET")),
            patch("market_cycle_trader_api.services.live_market_bars.download_stock_bars",
                  return_value=bars),
        ):
            merged, audit = load_live_intraday_session(
                {"MAN": daily}, SimpleNamespace(alpaca_live_feed="iex"),
                session="2026-10-20",
                now=datetime(2026, 10, 20, 13, 31, 5, tzinfo=timezone.utc),
            )
        self.assertEqual(audit["intraday_asset_count"], 1)
        self.assertEqual(audit["live_session"], "2026-10-20")
        self.assertEqual(merged["MAN"].iloc[-1]["close"], 42.5)

    def test_post_close_preparing_claim_is_not_stolen_after_hours(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:2026-10-20", "status": "preparing_signal",
            "preparation_started_at": datetime(2026, 10, 16, 20, 1, tzinfo=timezone.utc),
            "execution_session": "2026-10-20",
        }
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 10, 19, 20, 2, tzinfo=timezone.utc),
            is_open=False,
        )
        prefix = "market_cycle_trader_api.services.live_winner_signal."
        with patch(prefix + "create_live_readonly_client", return_value=broker):
            result = live_winner_signal.prepare_live_winner_signal(db, "2026-10-20")
        self.assertEqual(result["status"], "post_close_training_in_progress")
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_live_open_claim_is_not_recovered_based_on_elapsed_time(self):
        from market_cycle_trader_api.services.live_winner_open import refresh_live_open_winner_signal
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:2026-10-20", "status": "open_recalibration",
            "open_calibration_started_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
        }
        with patch("market_cycle_trader_api.services.live_winner_open.create_live_readonly_client") as factory:
            result = refresh_live_open_winner_signal(db, "2026-10-20")
        self.assertEqual(result["status"], "live_open_recalibration_in_progress")
        factory.assert_not_called()
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()
