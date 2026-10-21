"""Legacy Live opening retry no longer requests intraday market data."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import live_winner_open as obsolete
from market_cycle_trader_api.services import live_winner_signal as daily

SESSION = "2026-10-20"
PREFIX = "market_cycle_trader_api.services.live_winner_signal."


class LiveOpenNullClaimRecoveryTests(TestCase):
    def test_obsolete_intraday_entrypoint_disabled_without_access(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayRecalibrationDisabled"):
            obsolete.refresh_live_open_winner_signal(db, SESSION)
        db.__getitem__.assert_not_called()

    def test_legacy_failed_open_migrates_to_daily_recovery_only(self):
        db = MagicMock()
        stale = {
            "_id": "acct:" + SESSION, "execution_session": SESSION,
            "status": "waiting_for_open_recalibration", "live_account_id": "acct",
            "winner_strategy_id": "winner", "winner_revision": 2,
            "winner_configuration_hash": "hash",
        }
        db["live_winner_session_runs"].find_one.return_value = stale
        db["live_winner_session_runs"].find_one_and_update.return_value = {
            **stale, "status": "waiting_for_winner_signal",
        }
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 20, 13, 40, tzinfo=timezone.utc),
        )
        with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
            PREFIX + "latest_safe_completed_xnys_session",
            return_value=pd.Timestamp("2026-10-19"),
        ), patch(PREFIX + "get_system_settings", return_value={
            "training": {"enabled": False},
        }), patch(PREFIX + "download_operational_market_snapshot") as download:
            with self.assertRaisesRegex(RuntimeError, "training is disabled"):
                daily.prepare_live_winner_signal(db, SESSION, allow_open_recovery=True)
        update = db["live_winner_session_runs"].find_one_and_update.call_args
        self.assertEqual(update.args[0]["status"], "waiting_for_open_recalibration")
        self.assertEqual(update.args[1]["$set"]["status"], "waiting_for_winner_signal")
        download.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_completed_live_session_never_recovered(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "acct:" + SESSION, "execution_session": SESSION,
            "status": "completed",
        }
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            daily.prepare_live_winner_signal(db, SESSION, allow_open_recovery=True)
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()
