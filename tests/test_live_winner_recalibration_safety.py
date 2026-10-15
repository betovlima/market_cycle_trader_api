"""Open-session recalibration is safe and cannot submit orders."""
import unittest
from unittest.mock import MagicMock, patch
from market_cycle_trader_api.services.live_winner_recalibration import recalibrate_live_cash_winner_signal


class LiveRecalibrationSafetyTests(unittest.TestCase):
    def test_missing_session_stops(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = None
        with self.assertRaisesRegex(RuntimeError, "No armed Live session"):
            recalibrate_live_cash_winner_signal(db, "2026-10-16")

    def test_already_recalibrated_is_read_only(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "recalibrated_no_orders", "target_asset": "NVDA",
        }
        result = recalibrate_live_cash_winner_signal(db, "2026-10-16")
        self.assertEqual(result["target_asset"], "NVDA")
        self.assertFalse(result["order_submission_enabled"])

    def test_no_post_close_signal_blocks_market_open(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "waiting_for_winner_signal",
        }
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            recalibrate_live_cash_winner_signal(db, "2026-10-16")

    @patch("market_cycle_trader_api.services.live_winner_recalibration.create_live_readonly_client")
    def test_closed_market_must_not_recalibrate(self, factory):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "signal_prepared", "live_account_id": "live-account",
        }
        factory.return_value.get_account.return_value.id = "live-account"
        factory.return_value.get_clock.return_value.is_open = False
        result = recalibrate_live_cash_winner_signal(db, "2026-10-16")
        self.assertEqual(result["status"], "waiting_for_market_open")
        factory.return_value.submit_order.assert_not_called()
