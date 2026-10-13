"""Paper position peak/giveback observations never alter trading policy."""
from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
