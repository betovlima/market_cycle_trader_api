"""Live next-session arming must be durable, isolated and never place orders."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_winner_session import plan_next_live_session


class LiveSessionTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_winner_session.create_live_readonly_client")
    @patch("market_cycle_trader_api.services.live_winner_session.live_control_status")
    def test_stopped_does_not_touch_broker_or_db(self, control, factory):
        control.return_value = {"control_mode": "stopped"}
        db = MagicMock()
        result = plan_next_live_session(db)
        self.assertEqual(result["status"], "skipped")
        factory.assert_not_called()
        db.__getitem__.assert_not_called()

    @patch("market_cycle_trader_api.services.live_winner_session.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_winner_session.create_live_readonly_client")
    @patch("market_cycle_trader_api.services.live_winner_session.live_control_status")
    def test_active_arms_live_only_and_never_places_order(self, control, factory, winner):
        control.return_value = {"control_mode": "active"}
        broker = factory.return_value
        broker.get_account.return_value = SimpleNamespace(id="live-123")
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 10, 15, 20, tzinfo=timezone.utc),
            next_open=datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc),
        )
        winner.return_value = {
            "id": "winner-1", "revision": 2, "configuration_hash": "abc", "name": "Strategy #30",
        }
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "waiting_for_winner_signal",
            "phase": "session_armed_without_model_decision",
            "winner_name": "Strategy #30",
            "live_account_id": "live-123",
            "winner_strategy_id": "winner-1",
            "winner_revision": 2,
            "winner_configuration_hash": "abc",
        }
        result = plan_next_live_session(db)
        self.assertEqual(result["execution_session"], "2026-10-16")
        self.assertFalse(result["winner_signal_prepared"])
        self.assertFalse(result["order_submission_enabled"])
        db.__getitem__.assert_called_with("live_winner_session_runs")
        broker.submit_order.assert_not_called()
