"""Live signal preparation never reads Paper state or submits brokerage orders."""
import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_winner_signal import prepare_live_cash_winner_signal


class LiveSignalSafetyTests(unittest.TestCase):
    def test_session_must_be_armed(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = None
        with self.assertRaisesRegex(RuntimeError, "No Live session"):
            prepare_live_cash_winner_signal(db, "2026-10-16")

    def test_prepared_signal_is_idempotently_read_only(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "signal_prepared", "target_asset": "NVDA",
        }
        result = prepare_live_cash_winner_signal(db, "2026-10-16")
        self.assertEqual(result["target_asset"], "NVDA")
        self.assertFalse(result["order_submission_enabled"])

    def test_review_required_is_not_retried_automatically(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "review_required",
        }
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            prepare_live_cash_winner_signal(db, "2026-10-16")

    @patch("market_cycle_trader_api.services.live_winner_signal.latest_safe_completed_xnys_session",
           return_value=__import__("pandas").Timestamp("2026-10-15"))
    @patch("market_cycle_trader_api.services.live_winner_signal.get_system_settings")
    def test_training_disabled_rejects_before_broker_connect(self, settings, _cutoff):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "waiting_for_winner_signal",
        }
        settings.return_value = {"training": {"enabled": False}}
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            prepare_live_cash_winner_signal(db, "2026-10-16")
