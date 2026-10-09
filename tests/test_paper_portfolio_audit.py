from __future__ import annotations

import unittest

from market_cycle_trader_api.services.paper_portfolio_audit import (
    decision_audit,
    enrich_portfolio_history,
    operation_rows,
    pnl_by_asset,
)


class PaperPortfolioAuditTests(unittest.TestCase):
    def test_minimum_holding_reason_is_explicit(self) -> None:
        plan = {
            "plan_id": "plan-1",
            "current_asset": "MAN",
            "target_asset": "MAN",
            "raw_best_asset": "TSLA",
            "effective_switch_margin": 0.0005,
            "holding_sessions_at_decision": 1,
            "minimum_holding_sessions": 2,
            "utilities": {
                "MAN": 0.309053,
                "TSLA": 0.376528,
                "WDAY": 0.365783,
            },
            "cash_edges": {},
        }

        audit = decision_audit(plan, candidate_limit=None)

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["selection_reason"], "minimum_holding_not_reached")
        self.assertTrue(audit["switch_margin_passed"])
        self.assertFalse(audit["holding_rule_satisfied"])
        self.assertEqual(audit["candidate_count"], 3)
        self.assertEqual(audit["candidates"][0]["symbol"], "TSLA")
        self.assertAlmostEqual(
            audit["raw_best_vs_current_utility"],
            0.067475,
            places=6,
        )

    def test_partial_fill_counts_as_economic_execution_even_if_cancelled(self) -> None:
        orders = [
            {
                "created_at": "2026-01-01T14:30:00+00:00",
                "symbol": "MSFT",
                "side": "buy",
                "status": "filled",
                "filled_quantity": 20,
                "filled_average_price": 490.0,
                "plan_id": "buy-plan",
            },
            {
                "created_at": "2026-01-02T14:30:00+00:00",
                "symbol": "MSFT",
                "side": "sell",
                "status": "canceled",
                "filled_quantity": 14,
                "filled_average_price": 492.0,
                "plan_id": "sell-plan",
            },
        ]

        rows = operation_rows(orders, {}, limit=10)
        pnl = pnl_by_asset(orders)

        self.assertTrue(rows[0]["economic_fill"])
        self.assertEqual(rows[0]["status"], "canceled")
        self.assertAlmostEqual(rows[0]["filled_value"], 6888.0, places=6)
        self.assertEqual(pnl[0]["symbol"], "MSFT")
        self.assertAlmostEqual(pnl[0]["realized_pnl"], 28.0, places=6)
        self.assertAlmostEqual(pnl[0]["open_quantity"], 6.0, places=6)

    def test_history_reports_peak_distance_and_maximum_drawdown(self) -> None:
        history = [
            {"recorded_at": "a", "portfolio_value": 10000.0},
            {"recorded_at": "b", "portfolio_value": 11000.0},
            {"recorded_at": "c", "portfolio_value": 8800.0},
            {"recorded_at": "d", "portfolio_value": 9900.0},
        ]

        enriched, summary = enrich_portfolio_history(history)

        self.assertEqual(enriched[-1]["peak_portfolio_value"], 11000.0)
        self.assertAlmostEqual(summary["maximum_drawdown"], -0.2, places=12)
        self.assertAlmostEqual(summary["distance_to_peak"], -0.1, places=12)


if __name__ == "__main__":
    unittest.main()
