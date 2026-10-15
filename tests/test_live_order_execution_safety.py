import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_order_execution import (
    execute_recalibrated_live_winner_session, _unique_order_id,
)


class LiveOrderSafetyTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_order_execution.order_submission_opted_in", return_value=False)
    @patch("market_cycle_trader_api.services.live_order_execution.create_live_execution_client")
    def test_default_blocks_broker(self, broker_factory, permission):
        result = execute_recalibrated_live_winner_session(MagicMock(), "2026-10-16")
        self.assertEqual(result["status"], "order_submission_disabled")
        broker_factory.assert_not_called()

    @patch("market_cycle_trader_api.services.live_order_execution.order_submission_opted_in", return_value=True)
    @patch("market_cycle_trader_api.services.live_order_execution.live_control_status", return_value={"control_mode": "stopped"})
    @patch("market_cycle_trader_api.services.live_order_execution.create_live_execution_client")
    def test_stopped_does_not_trade(self, broker_factory, control, permission):
        result = execute_recalibrated_live_winner_session(MagicMock(), "2026-10-16")
        self.assertEqual(result["status"], "blocked_by_live_mode")
        broker_factory.assert_not_called()

    def test_deterministic_client_order_identity(self):
        a = _unique_order_id("a", "2026-10-16", "buy", "NVDA")
        self.assertEqual(a, _unique_order_id("a", "2026-10-16", "buy", "NVDA"))
        self.assertNotEqual(a, _unique_order_id("a", "2026-10-17", "buy", "NVDA"))
