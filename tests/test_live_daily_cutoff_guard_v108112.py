"""The Live Winner must never prepare a next-day plan on stale daily candles."""
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services.live_winner_signal import prepare_live_cash_winner_signal


class LiveDailyCutoffGuardTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:2026-10-20",
            "execution_session": "2026-10-20",
            "status": "waiting_for_winner_signal",
            "live_account_id": "acct",
            "preparation_attempts": 0,
        }
        self.prefix = "market_cycle_trader_api.services.live_winner_signal."
        self.broker = MagicMock()
        self.broker.get_clock.return_value = type("LiveClock", (), {
            "timestamp": datetime(2026, 10, 19, 20, 1, tzinfo=timezone.utc),
            "is_open": False,
        })()


    def test_waits_after_close_until_daily_bar_is_safe(self):
        # Monday 2026-10-19 is complete but the 20-min buffer is not.
        # latest_safe_completed_xnys_session() still returns Friday 10-16.
        with (
            patch(self.prefix + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-10-16")),
            patch(self.prefix + "get_system_settings") as settings,
            patch(self.prefix + "download_operational_market_snapshot") as download,
            patch(self.prefix + "create_live_readonly_client", return_value=self.broker) as broker,
        ):
            result = prepare_live_cash_winner_signal(self.db, "2026-10-20")
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        self.assertEqual(result["required_day"], "2026-10-19")
        self.assertEqual(result["available_day"], "2026-10-16")
        self.assertFalse(result["order_submission_enabled"])
        settings.assert_not_called()
        download.assert_not_called()
        broker.assert_called_once()
        self.db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_accepts_previous_trading_day_across_weekend(self):
        self.db["live_winner_session_runs"].find_one.return_value["execution_session"] = "2026-10-19"
        with (
            patch(self.prefix + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-10-16")),
            patch(self.prefix + "get_system_settings",
                  return_value={"training": {"enabled": False}}) as settings,
            patch(self.prefix + "download_operational_market_snapshot") as download,
            patch(self.prefix + "create_live_readonly_client", return_value=self.broker),
        ):
            with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                prepare_live_cash_winner_signal(self.db, "2026-10-19")
        settings.assert_called_once()
        download.assert_not_called()

    def test_next_day_is_prepared_only_after_expected_bar_available(self):
        with (
            patch(self.prefix + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-10-19")),
            patch(self.prefix + "get_system_settings",
                  return_value={"training": {"enabled": False}}) as settings,
            patch(self.prefix + "download_operational_market_snapshot") as download,
            patch(self.prefix + "create_live_readonly_client", return_value=self.broker),
        ):
            with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                prepare_live_cash_winner_signal(self.db, "2026-10-20")
        settings.assert_called_once()
        download.assert_not_called()

    def test_waiting_never_consumes_retry_budget(self):
        self.db["live_winner_session_runs"].find_one.return_value["preparation_attempts"] = 1
        with (
            patch(self.prefix + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-10-16")),
            patch(self.prefix + "create_live_readonly_client", return_value=self.broker),
        ):
            result = prepare_live_cash_winner_signal(self.db, "2026-10-20")
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        self.db["live_winner_session_runs"].update_one.assert_not_called()
        self.db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_real_order_rejects_stale_daily_decision(self):
        from market_cycle_trader_api.services.live_order_execution import (
            execute_recalibrated_live_winner_session,
        )
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:2026-10-20", "execution_session": "2026-10-20",
            "status": "recalibrated_no_orders", "winner_signal_prepared": True,
            "post_close_signal_ready": True,
            "snapshot_sha256": "snapshot", "target_asset": "ADEA",
            "action": "buy", "decision_date": "2026-10-16",
            "winner_strategy_id": "winner", "winner_revision": 2,
            "winner_configuration_hash": "hash",
        }
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary",
                  return_value={"id": "winner", "revision": 2, "configuration_hash": "hash"}),
            patch(prefix + "create_live_execution_client") as broker,
        ):
            with self.assertRaisesRegex(RuntimeError, "LivePriorDailySignalRequired"):
                execute_recalibrated_live_winner_session(db, "2026-10-20")
            broker.assert_not_called()
