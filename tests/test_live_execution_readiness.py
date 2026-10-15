"""Live execution preflight must fail closed and never submit orders."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from market_cycle_trader_api.services.live_execution_readiness import live_execution_readiness


class LiveReadinessTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_healthy_account_still_does_not_enable_real_orders(self, client_factory, control):
        client_factory.return_value.get_account.return_value = SimpleNamespace(
            id="live-account", trading_blocked=False, account_blocked=False,
        )
        control.return_value = {"revision": 3, "control_mode": "active"}
        result = live_execution_readiness(object())
        self.assertFalse(result["ready_for_real_orders"])
        self.assertFalse(result["order_submission_enabled"])
        self.assertFalse(next(x for x in result["checks"] if x["name"] == "reviewed_live_executor")["passed"])
        client_factory.return_value.submit_order.assert_not_called()

    @patch("market_cycle_trader_api.services.live_execution_readiness.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_readiness.create_live_readonly_client")
    def test_credentials_failure_fails_closed(self, client_factory, control):
        client_factory.side_effect = RuntimeError("secret failure")
        control.return_value = {"revision": 0, "control_mode": "stopped"}
        result = live_execution_readiness(object())
        self.assertFalse(result["ready_for_real_orders"])
        self.assertFalse(next(x for x in result["checks"] if x["name"] == "live_account")["passed"])


if __name__ == "__main__":
    unittest.main()
