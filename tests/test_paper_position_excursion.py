"""Paper position peak/giveback observations never alter trading policy."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import paper_trading
from market_cycle_trader_api.schemas.paper_trading import PaperTradingState

from market_cycle_trader_api.services.paper_position_excursion import (
    completed_sessions_after_entry,
    most_recent_open_buy,
    position_excursion_audit,
)


class PaperPositionExcursionTests(unittest.TestCase):
    def setUp(self):
        self.orders = [{
            "symbol": "MAN", "side": "buy", "status": "filled",
            "filled_at": "2026-10-07T19:12:03+00:00",
            "filled_quantity": 196.730176625,
            "filled_average_price": 53.22,
        }]
        self.position = {
            "symbol": "MAN", "quantity": 196.730176625,
            "average_entry_price": 53.22,
            "cost_basis": 10469.98, "market_value": 10159.15,
            "current_price": 51.64,
            "unrealized_return": -310.83 / 10469.98,
            "holding_sessions": 1,
            "portfolio_value": 10159.16,
            "observed_at": "2026-10-13T22:38:53+00:00",
        }
        self.history = [
            {"recorded_at": "2026-10-07T19:12:47+00:00",
             "managed_symbol": "MAN", "portfolio_value": 10442.45,
             "market_value": 10442.44},
            {"recorded_at": "2026-10-09T13:12:08+00:00",
             "managed_symbol": "MAN", "portfolio_value": 10733.61,
             "market_value": 10733.60},
            {"recorded_at": "2026-10-13T22:38:47+00:00",
             "managed_symbol": "MAN", "portfolio_value": 10159.16,
             "market_value": 10159.15},
        ]
        self.sessions = [
            "2026-10-07", "2026-10-08", "2026-10-09",
            "2026-10-12", "2026-10-13",
        ]

    def test_man_giveback_uses_only_observed_snapshots(self):
        result = position_excursion_audit(
            self.position, self.orders, self.history,
            completed_sessions=self.sessions,
            decision={"decision_date": "2026-10-13", "holding_sessions_at_decision": 1},
        )
        self.assertEqual(result["symbol"], "MAN")
        self.assertEqual(result["measurement"], "observed_snapshots_only")
        self.assertAlmostEqual(result["observed_peak_return"], 10733.60 / 10469.98 - 1, places=6)
        self.assertAlmostEqual(result["current_return"], -310.83 / 10469.98, places=6)
        self.assertAlmostEqual(result["profit_given_back"], 574.45, places=2)
        self.assertAlmostEqual(result["account_value_given_back"], 574.45, places=2)
        self.assertEqual(result["observed_completed_sessions_since_entry"], 4)
        self.assertEqual(result["observed_sessions_at_decision"], 4)
        self.assertTrue(result["holding_discrepancy"])
        self.assertTrue(result["decision_holding_discrepancy"])
        self.assertEqual(len(result["observations"]), 4)
        self.assertIn("not the intraday high", result["limitations"])

    def test_market_calendar_excludes_entry_day(self):
        self.assertEqual(
            completed_sessions_after_entry(
                "2026-10-07T19:12:03+00:00", self.sessions,
                as_of_session="2026-10-12",
            ), 3,
        )

    def test_partial_fills_and_reentry_are_resolved_from_ledger(self):
        orders = [
            {"symbol": "MAN", "side": "buy", "filled_quantity": 10,
             "filled_at": "2026-10-01T14:00:00Z"},
            {"symbol": "MAN", "side": "sell", "filled_quantity": 10,
             "filled_at": "2026-10-05T14:00:00Z"},
            {"symbol": "MAN", "side": "buy", "filled_quantity": 8,
             "filled_at": "2026-10-07T14:00:00Z"},
            {"symbol": "MAN", "side": "buy", "filled_quantity": 2,
             "filled_at": "2026-10-08T14:00:00Z"},
        ]
        entry = most_recent_open_buy("MAN", orders)
        self.assertTrue(entry["entry_filled_at"].startswith("2026-10-07"))
        self.assertFalse(entry["constant_quantity"])
        self.assertAlmostEqual(entry["filled_quantity_ledger"], 10)

    def test_legacy_snapshots_are_not_inferred_if_quantity_changed(self):
        position = dict(self.position)
        history = list(self.history)
        self.orders.append({
            "symbol": "MAN", "side": "buy", "filled_quantity": 10,
            "filled_at": "2026-10-08T14:30:00+00:00",
        })
        result = position_excursion_audit(position, self.orders, history)
        self.assertFalse(result["quantity_unchanged_in_order_ledger"])
        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["source"], "current_broker_position")

    def test_missing_order_does_not_invent_a_peak(self):
        result = position_excursion_audit(self.position, [], self.history)
        self.assertEqual(result["status"], "entry_fill_unavailable")
        self.assertEqual(result["observations"], [])


class PaperHoldingReconciliationTests(unittest.TestCase):
    """Real MAN fill, interrupted hold plans and market calendar."""

    def setUp(self):
        self.state = PaperTradingState(
            initial_capital=10000,
            strategy_cash=0,
            managed_symbol="MAN",
            managed_quantity=196.730176625,
            average_entry_price=53.22,
            holding_sessions=1,
            realized_pnl=0,
            last_decision_date="2026-10-06",
            last_execution_session="2026-10-07",
        )
        self.fills = [
            {
                "symbol": "MAN",
                "side": "buy",
                "status": "filled",
                "filled_at": "2026-10-07T19:12:03.483+00:00",
                "filled_quantity": 196.730176625,
            },
        ]
        self.order_collection = MagicMock()
        self.order_collection.find.return_value = self.fills
        self.state_collection = MagicMock()
        self.state_collection.update_one.return_value.matched_count = 1
        self.db = {
            paper_trading.PAPER_TRADE_ORDERS_COLLECTION: self.order_collection,
            paper_trading.PAPER_TRADING_STATE_COLLECTION: self.state_collection,
        }
        self.sessions = [
            "2026-10-07", "2026-10-08", "2026-10-09",
            "2026-10-12", "2026-10-13", "2026-10-14",
        ]

    def _reconcile(self, *, is_open, safe_cutoff="2026-10-13"):
        with (
            patch.object(
                paper_trading, "latest_safe_completed_xnys_session",
                return_value=pd.Timestamp(safe_cutoff),
            ),
            patch.object(
                paper_trading, "calendar_session_dates",
                return_value=self.sessions,
            ) as calendar,
        ):
            result = paper_trading._reconcile_paper_holding_sessions(
                self.db,
                MagicMock(),
                self.state,
                clock={
                    "is_open": is_open,
                    "timestamp": "2026-10-14T15:25:46Z",
                },
            )
            return result, calendar

    def test_stale_man_counter_recovered_from_real_fill(self):
        actual, calendar = self._reconcile(is_open=True)
        # The purchase starts holding session 1. Four full trading
        # sessions after the fill make the effective age 5, not 1.
        self.assertEqual(actual.holding_sessions, 5)
        self.assertEqual(self.state.holding_sessions, 1)
        self.assertEqual(calendar.call_args.kwargs["end_date"].isoformat(), "2026-10-13")
        filter_doc, update_doc = self.state_collection.update_one.call_args.args
        self.assertEqual(filter_doc["managed_symbol"], "MAN")
        self.assertEqual(filter_doc["holding_sessions"], 1)
        self.assertEqual(filter_doc["last_execution_session"], "2026-10-07")
        self.assertEqual(update_doc["$set"]["holding_sessions"], 5)

    def test_incomplete_intraday_candle_never_increments_holding(self):
        # Deliberately supply an over-generous historical cutoff.
        actual, calendar = self._reconcile(is_open=True, safe_cutoff="2026-10-14")
        self.assertEqual(actual.holding_sessions, 5)
        self.assertEqual(calendar.call_args.kwargs["end_date"].isoformat(), "2026-10-13")

    def test_persisted_counter_matches_purchase_session(self):
        state = self.state.model_copy(update={"holding_sessions": 1})
        with (
            patch.object(paper_trading, "latest_safe_completed_xnys_session",
                         return_value=pd.Timestamp("2026-10-07")),
            patch.object(paper_trading, "calendar_session_dates",
                         return_value=["2026-10-07"]),
        ):
            result = paper_trading._reconcile_paper_holding_sessions(
                self.db, MagicMock(), state,
                clock={"is_open": False, "timestamp": "2026-10-07T22:00:00Z"},
            )
        self.assertEqual(result.holding_sessions, 1)
        self.state_collection.update_one.assert_not_called()

    def test_cancelled_hold_plans_do_not_change_fill_based_age(self):
        # No execution of cancelled or prepared hold plans occurred.
        self.order_collection.find.return_value = list(self.fills)
        actual, _ = self._reconcile(is_open=True)
        self.assertEqual(actual.holding_sessions, 5)

    def test_no_confirmed_fill_does_not_unlock_rotation(self):
        self.order_collection.find.return_value = [
            {"symbol": "MAN", "side": "buy",
             "created_at": "2026-10-07T19:12:00Z",
             "filled_quantity": 196.730176625}
        ]
        with self.assertRaisesRegex(RuntimeError, "no open entry"):
            self._reconcile(is_open=True)
        self.state_collection.update_one.assert_not_called()

    def test_ledger_quantity_conflict_blocks_reconciliation(self):
        self.fills[0]["filled_quantity"] = 150.0
        with self.assertRaisesRegex(RuntimeError, "quantity"):
            self._reconcile(is_open=True)
        self.state_collection.update_one.assert_not_called()

    def test_concurrent_state_change_is_not_overwritten(self):
        self.state_collection.update_one.return_value.matched_count = 0
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            self._reconcile(is_open=True)

    def test_cash_position_requires_no_calendar_or_ledger(self):
        state = self.state.model_copy(
            update={
                "managed_symbol": None, "managed_quantity": 0.0,
                "average_entry_price": None, "holding_sessions": 0,
            }
        )
        self.assertIs(
            paper_trading._reconcile_paper_holding_sessions(
                self.db, MagicMock(), state, clock={},
            ),
            state,
        )
        self.order_collection.find.assert_not_called()

    def test_sell_then_repurchase_resets_age(self):
        self.order_collection.find.return_value = [
            {"symbol": "MAN", "side": "buy", "filled_quantity": 15,
             "filled_at": "2026-09-22T14:00:00Z"},
            {"symbol": "MAN", "side": "sell", "filled_quantity": 15,
             "filled_at": "2026-10-06T15:00:00Z"},
            *self.fills,
        ]
        actual, _ = self._reconcile(is_open=True)
        self.assertEqual(actual.holding_sessions, 5)


if __name__ == "__main__":
    unittest.main()
