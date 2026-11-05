"""Daily Alpaca account performance, cashflow adjustment and failure safety."""
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

    def test_no_cashflow_uses_last_daily_close_not_unverified_live_equity(self):
        broker = self.broker()
        result = alpaca_account_cumulative_return(broker, current_equity=125.0)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .10)
        self.assertAlmostEqual(result["profit_loss"], 10)
        self.assertEqual(result["start_date"], "2026-11-03")
        self.assertEqual(result["as_of_date"], "2026-11-05")
        self.assertEqual(result["coverage"], "from_first_daily_close")
        self.assertEqual(result["method"], "daily_linked_external_flow_adjusted")
        self.assertFalse(result["cashflow_adjusted"])
        self.assertEqual(result["scope"], "alpaca_account")
        self.assertEqual(result["source"], "alpaca_portfolio_history")
        request = broker.get_portfolio_history.call_args.args[0]
        self.assertEqual(request.timeframe, "1D")
        self.assertEqual(request.cashflow_types, "ALL")
        broker.submit_order.assert_not_called()

    def test_deposit_does_not_count_as_return(self):
        # First day +10% net of a +50 external deposit; next day +10%.
        result = alpaca_account_cumulative_return(self.broker(
            equity=[100, 160, 176], cashflow={"CSD": [0, 50, 50]},
        ), current_equity=300)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .21)
        self.assertAlmostEqual(result["profit_loss"], 26)
        self.assertAlmostEqual(result["external_cashflow_total"], 50)
        self.assertTrue(result["cashflow_adjusted"])

    def test_withdrawal_does_not_look_like_a_loss(self):
        result = alpaca_account_cumulative_return(self.broker(
            equity=[100, 90, 99], cashflow={"CSW": [0, -20, -20]},
        ), current_equity=99)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .21)
        self.assertAlmostEqual(result["profit_loss"], 19)
        self.assertEqual(result["external_cashflow_total"], -20)

    def test_multiple_cumulative_types_only_count_deltas_not_total_every_day(self):
        result = alpaca_account_cumulative_return(self.broker(
            timestamp=[1793664000, 1793750400, 1793836800, 1793923200],
            equity=[100, 150, 140, 155],
            cashflow={"CSD": [0, 50, 50, 50], "CSW": [0, 0, -20, -20]},
        ), current_equity=999)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], (160 / 150) * (155 / 140) - 1)
        self.assertAlmostEqual(result["profit_loss"], 25)
        self.assertAlmostEqual(result["external_cashflow_total"], 30)

    def test_dividend_and_interest_remain_in_return(self):
        result = alpaca_account_cumulative_return(self.broker(
            equity=[100, 105, 110],
            cashflow={"DIV": [0, 5, 5], "INT": [0, 0, 1]},
        ), current_equity=110)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .10)
        self.assertEqual(result["external_cashflow_total"], 0)
        self.assertFalse(result["cashflow_adjusted"])

    def test_first_funded_close_is_baseline_not_the_initial_deposit(self):
        result = alpaca_account_cumulative_return(self.broker(
            equity=[0, 100, 120],
            base_value=0,
            base_value_asof=None,
            cashflow={"CSD": [0, 100, 100]},
        ), current_equity=120)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .20)
        self.assertAlmostEqual(result["profit_loss"], 20)
        self.assertEqual(result["start_date"], "2026-11-04")

    def test_cashflow_enum_keys_are_supported(self):
        class Activity:
            value = "CSD"
        result = alpaca_account_cumulative_return(self.broker(
            equity=[100, 160, 176],
            cashflow={Activity(): [0, 50, 50]},
        ), current_equity=176)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["return_pct"], .21)

    def test_unknown_stock_transfer_or_journal_fails_closed(self):
        for code in ("ACATS", "JNLC", "MYSTERY"):
            with self.subTest(code=code):
                result = alpaca_account_cumulative_return(self.broker(
                    cashflow={code: [0, 1, 1]},
                ), current_equity=120)
                self.assertEqual(result["reason"], "cashflow_type_unverified")
                self.assertIsNone(result["return_pct"])

    def test_unknown_cashflow_with_no_movement_is_not_assumed_a_transfer(self):
        result = alpaca_account_cumulative_return(self.broker(
            cashflow={"MYSTERY": [0, 0, 0]},
        ), current_equity=120)
        self.assertEqual(result["status"], "available")

    def test_missing_malformed_duplicate_or_nan_cashflow_fails_closed(self):
        for flow in (None, {"CSD": [0, 1]}, {"CSD": [0, float("nan"), 1]},
                     {"CSD": [0, None, 0]}):
            with self.subTest(flow=flow):
                result = alpaca_account_cumulative_return(
                    self.broker(cashflow=flow), current_equity=120,
                )
                self.assertEqual(result["reason"], "cashflow_unverified")
                self.assertIsNone(result["return_pct"])

    def test_invalid_gap_or_time_axis_does_not_silently_chain(self):
        for data in (
            {"equity": [100, None, 120]},
            {"equity": [100, 0, 120]},
            {"equity": [100, float("inf"), 120]},
            {"timestamp": [1793664000, 1793664000, 1793836800]},
            {"timestamp": [1793664000, 1793836800, 1793750400]},
            {"equity": [100, 50, 40], "cashflow": {"CSD": [0, 100, 100]}},
        ):
            with self.subTest(data=data):
                result = alpaca_account_cumulative_return(
                    self.broker(**data), current_equity=120,
                )
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["return_pct"])

    def test_insufficient_history_and_broker_error_do_not_submit_orders(self):
        for data in ({"timestamp": [], "equity": []},
                     {"equity": [0, 0, 100]},
                     {"equity": [None, None, 100]}):
            broker = self.broker(**data)
            result = alpaca_account_cumulative_return(broker, current_equity=120)
            self.assertIsNone(result["return_pct"])
            broker.submit_order.assert_not_called()
        broker = Mock()
        broker.get_portfolio_history.side_effect = RuntimeError("not authorized")
        result = alpaca_account_cumulative_return(broker, current_equity=120)
        self.assertEqual(result["reason"], "history_unavailable")
        self.assertIsNone(result["return_pct"])
        broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
