"""Read-only Live portfolio displays broker purchasing power without estimating it."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from market_cycle_trader_api.services.live_portfolio import live_portfolio_snapshot


class LiveBuyingPowerTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_portfolio._store_observation", return_value=[])
    @patch("market_cycle_trader_api.services.live_portfolio.create_live_readonly_client")
    def test_buying_power_is_taken_from_broker_and_not_inferred_from_cash(self, factory, store):
        broker = factory.return_value
        broker.get_account.return_value = SimpleNamespace(
            id="live-account",
            cash="108.25", equity="108.25",
            buying_power="80.00", non_marginable_buying_power="75.00",
            status="ACTIVE", trading_blocked=False,
        )
        broker.get_all_positions.return_value = []
        broker.get_orders.return_value = []
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 10, 19, 20, 10, tzinfo=timezone.utc), is_open=False, next_open=None, next_close=None,
        )
        with patch("market_cycle_trader_api.services.live_portfolio.latest_live_winner_decision", return_value=None):
            result = live_portfolio_snapshot(object())
        self.assertEqual(result["strategy_cash"], 108.25)
        self.assertEqual(result["buying_power"], 80)
        self.assertEqual(result["non_marginable_buying_power"], 75)
        broker.submit_order.assert_not_called()

    @patch("market_cycle_trader_api.services.live_portfolio._store_observation", return_value=[])
    @patch("market_cycle_trader_api.services.live_portfolio.create_live_readonly_client")
    def test_missing_broker_buying_power_is_unknown_not_cash(self, factory, store):
        broker = factory.return_value
        broker.get_account.return_value = SimpleNamespace(
            id="live-account", cash="108.25", equity="108.25",
            status="ACTIVE", trading_blocked=False,
        )
        broker.get_all_positions.return_value = []
        broker.get_orders.return_value = []
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 10, 19, 20, 10, tzinfo=timezone.utc), is_open=False, next_open=None, next_close=None,
        )
        with patch("market_cycle_trader_api.services.live_portfolio.latest_live_winner_decision", return_value=None):
            result = live_portfolio_snapshot(object())
        self.assertIsNone(result["buying_power"])
        self.assertIsNone(result["non_marginable_buying_power"])
        broker.submit_order.assert_not_called()
