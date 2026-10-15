"""Live preview has no broker mutation and always reports execution blocked."""
import unittest
from unittest.mock import patch

from market_cycle_trader_api.services.live_execution_dry_run import preview_live_intent


class DryRunTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_execution_dry_run.create_live_readonly_client")
    @patch("market_cycle_trader_api.services.live_execution_dry_run.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_dry_run.live_portfolio_snapshot")
    def test_hypothetical_target_remains_readonly(self, portfolio, status, client_factory):
        client_factory.return_value.get_asset.return_value.symbol = "MSFT"
        client_factory.return_value.get_asset.return_value.tradable = True
        status.return_value = {"control_mode": "active"}
        portfolio.return_value = {
            "portfolio_value": 1000.0,
            "strategy_cash": 750.0,
            "positions": [{"symbol": "AAPL", "market_value": 250.0}],
            "market_clock": {"is_open": True},
            "trading_blocked": False,
        }
        result = preview_live_intent(object(), target="MSFT", max_allocation_fraction=0.1)
        self.assertEqual(result["hypothetical_target_market_value"], 100.0)
        self.assertFalse(result["would_submit_order"])
        self.assertFalse(result["ready_for_real_orders"])
        self.assertFalse(result["checks"]["strategy_signal_verified"])

    @patch("market_cycle_trader_api.services.live_execution_dry_run.live_control_status")
    @patch("market_cycle_trader_api.services.live_execution_dry_run.live_portfolio_snapshot")
    def test_cash_preview_requires_no_new_buy(self, portfolio, status):
        status.return_value = {"control_mode": "stopped"}
        portfolio.return_value = {
            "portfolio_value": 1000.0,
            "strategy_cash": 750.0,
            "positions": [{"symbol": "AAPL", "market_value": 250.0}],
            "market_clock": {"is_open": False},
            "trading_blocked": False,
        }
        result = preview_live_intent(object(), target="CASH", max_allocation_fraction=0.1)
        self.assertEqual(result["hypothetical_target_delta"], 0.0)
        self.assertEqual(result["positions_to_review_for_reduction"], ["AAPL"])
        self.assertFalse(result["would_submit_order"])

    def test_invalid_ticker_rejected_before_broker_access(self):
        with self.assertRaises(ValueError):
            preview_live_intent(object(), target="INVALID-ORDER", max_allocation_fraction=0.1)


if __name__ == "__main__":
    unittest.main()
