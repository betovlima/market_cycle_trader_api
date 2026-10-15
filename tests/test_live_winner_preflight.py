"""Live Winner preflight must never touch Paper or issue an order."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_winner_preflight import (
    capture_live_winner_preflight,
)


class LiveWinnerPreflightTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_winner_preflight.live_control_status")
    @patch("market_cycle_trader_api.services.live_winner_preflight.create_live_readonly_client")
    def test_stopped_does_not_query_broker(self, factory, control):
        control.return_value = {"control_mode": "stopped"}
        db = MagicMock()
        result = capture_live_winner_preflight(db)
        self.assertEqual(result["status"], "skipped")
        factory.assert_not_called()
        db.__getitem__.assert_not_called()

    @patch("market_cycle_trader_api.services.live_winner_preflight.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_winner_preflight.create_live_readonly_client")
    @patch("market_cycle_trader_api.services.live_winner_preflight.live_control_status")
    def test_active_creates_live_only_preflight(self, control, factory, winner):
        control.return_value = {"control_mode": "active"}
        broker = factory.return_value
        broker.get_account.return_value = SimpleNamespace(id="live-account")
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=None, is_open=False, next_open=None, next_close=None,
        )
        winner.return_value = {"name": "Winner"}
        db = MagicMock()
        result = capture_live_winner_preflight(db)
        self.assertEqual(result["phase"], "winner_preflight_only")
        self.assertFalse(result["winner_decision_generated"])
        self.assertFalse(result["order_submission_enabled"])
        db.__getitem__.assert_called_once_with("live_winner_preflight_snapshots")
        broker.submit_order.assert_not_called()
