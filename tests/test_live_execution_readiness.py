"""Live preflight reflects broker, selected Winner and existing control mode."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from market_cycle_trader_api.services.live_execution_readiness import live_execution_readiness


class LiveReadinessTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_execution_readiness.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_active_mode_with_healthy_broker_and_winner(self, broker, control, winner):
        broker.return_value.get_account.return_value = SimpleNamespace(
            id="acct", trading_blocked=False, account_blocked=False,
        )
        broker.return_value.get_orders.return_value = []
        control.return_value = {"control_mode": "active"}
        winner.return_value = {"id": "winner", "configuration_hash": "hash"}
        result = live_execution_readiness(object())
        self.assertTrue(result["ready_for_real_orders"])
        self.assertTrue(result["order_submission_enabled"])
        broker.return_value.submit_order.assert_not_called()

    @patch("market_cycle_trader_api.services.live_execution_readiness.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_paused_mode_is_not_ready(self, broker, control, winner):
        broker.return_value.get_account.return_value = SimpleNamespace(
            id="acct", trading_blocked=False, account_blocked=False,
        )
        broker.return_value.get_orders.return_value = []
        control.return_value = {"control_mode": "paused"}
        winner.return_value = {"id": "winner", "configuration_hash": "hash"}
        result = live_execution_readiness(object())
        self.assertFalse(result["ready_for_real_orders"])
        self.assertFalse(result["order_submission_enabled"])

    @patch("market_cycle_trader_api.services.live_execution_readiness.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_open_orders_are_reported(self, broker, control, winner):
        broker.return_value.get_account.return_value = SimpleNamespace(
            id="acct", trading_blocked=False, account_blocked=False,
        )
        broker.return_value.get_orders.return_value = [SimpleNamespace(id="pending")]
        control.return_value = {"control_mode": "active"}
        winner.return_value = {"id": "winner", "configuration_hash": "hash"}
        result = live_execution_readiness(object())
        self.assertFalse(result["ready_for_real_orders"])
        self.assertFalse(next(c for c in result["checks"] if c["name"] == "broker_open_orders")["passed"])

    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_broker_unavailable_is_not_ready(self, broker, control):
        broker.side_effect = RuntimeError("unavailable")
        control.return_value = {"control_mode": "stopped"}
        result = live_execution_readiness(object())
        self.assertFalse(result["ready_for_real_orders"])
