"""Alpaca account cumulative performance: broker-only and fail-closed for flows."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from market_cycle_trader_api.services.live_account_return import (
    alpaca_account_cumulative_return,
)


class AlpacaAccountReturnTests(unittest.TestCase):
    def broker(self, **overrides):
        history = {
            "timestamp": [1793664000, 1793750400, 1793836800],
            "equity": [100.0, 105.0, 110.0],
            "base_value": 100.0,
            "base_value_asof": "2026-11-02",
            "cashflow": {},
        }
        history.update(overrides)
        broker = Mock()
        broker.get_portfolio_history.return_value = SimpleNamespace(**history)
        return broker

    def test_account_return_uses_live_equity_and_broker_daily_baseline(self):
        broker = self.broker()
        result = alpaca_account_cumulative_return(broker, current_equity=125.0)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["return_pct"], 0.25)
        self.assertEqual(result["profit_loss"], 25.0)
        self.assertEqual(result["start_date"], "2026-11-02")
        self.assertEqual(result["scope"], "alpaca_account")
        self.assertEqual(result["source"], "alpaca_portfolio_history")
        request = broker.get_portfolio_history.call_args.args[0]
        self.assertEqual(request.timeframe, "1D")
        self.assertEqual(request.cashflow_types, "ALL")

    def test_transfer_or_dividend_never_masquerades_as_winner_profit(self):
        for flow in ({"CSD": [0, 200, 200]}, {"CSW": [0, 0, 20]},
                     {"DIV": [0, 0.50, 0.50]}):
            with self.subTest(flow=flow):
                broker = self.broker(cashflow=flow)
                result = alpaca_account_cumulative_return(
                    broker, current_equity=300.0,
                )
                self.assertEqual(result["reason"], "cashflows_present")
                self.assertIsNone(result["return_pct"])

    def test_unknown_or_missing_cashflow_is_not_assumed_zero(self):
        self.assertEqual(
            alpaca_account_cumulative_return(
                self.broker(cashflow=None), current_equity=120,
            )["reason"], "cashflow_unverified",
        )

    def test_empty_insufficient_or_invalid_history_has_no_return(self):
        for override in (
            {"timestamp": [], "equity": []},
            {"base_value": 0},
            {"timestamp": [1, 2], "equity": [None, 100]},
            {"timestamp": [1, 2], "equity": [90, 100], "cashflow": {"CSD": [0]}},
        ):
            with self.subTest(override=override):
                result = alpaca_account_cumulative_return(
                    self.broker(**override), current_equity=120,
                )
                self.assertIsNone(result["return_pct"])

    def test_history_api_failure_does_not_break_portfolio_or_submit_orders(self):
        broker = Mock()
        broker.get_portfolio_history.side_effect = RuntimeError("not authorized")
        result = alpaca_account_cumulative_return(broker, current_equity=120)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["return_pct"])
        broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
