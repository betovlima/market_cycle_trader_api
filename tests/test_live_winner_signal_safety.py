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

    def test_legacy_signal_without_cutoff_is_not_trusted(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "signal_prepared", "target_asset": "NVDA",
        }
        with self.assertRaisesRegex(RuntimeError, "requires explicit review"):
            prepare_live_cash_winner_signal(db, "2026-10-16")
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_review_required_is_not_retried_automatically(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "review_required",
        }
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            prepare_live_cash_winner_signal(db, "2026-10-16")

    @patch("market_cycle_trader_api.services.live_winner_signal.get_system_settings")
    def test_training_disabled_rejects_before_broker_connect(self, settings):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "waiting_for_winner_signal",
        }
        settings.return_value = {"training": {"enabled": False}}
        from datetime import datetime, timezone
        from types import SimpleNamespace
        with (
            patch("market_cycle_trader_api.services.live_winner_signal.latest_safe_completed_xnys_session",
                  return_value=__import__("pandas").Timestamp("2026-10-15")),
            patch("market_cycle_trader_api.services.live_winner_signal.create_live_readonly_client") as factory,
        ):
            factory.return_value.get_clock.return_value = SimpleNamespace(
                timestamp=datetime(2026, 10, 15, 20, 1, tzinfo=timezone.utc)
            )
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                prepare_live_cash_winner_signal(db, "2026-10-16")
