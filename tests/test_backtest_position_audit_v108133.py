"""Position audit must never fabricate states or trade/decision explanations."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from market_cycle_trader_api.services.backtest_position_audit import build_position_audit


def stamp(day: str) -> datetime:
    return datetime.fromisoformat(day + "T04:00:00+00:00")


def equity(*days: str) -> list[dict]:
    return [{"timestamp": stamp(day), "simulation_equity": 100.0} for day in days]


def prediction(day: str, asset: str | None, **kwargs) -> dict:
    return {"timestamp": stamp(day), "selected_asset": asset, **kwargs}


class PersistedCashPositionAuditTests(unittest.TestCase):
    def test_two_isolated_cash_sessions_have_complete_coverage_and_grounded_neighbors(self):
        days = ["2022-06-16", "2022-06-17", "2022-06-21", "2024-04-18", "2024-04-19", "2024-04-22"]
        rows = [
            prediction("2022-06-16", "AVGO", assets_held=1, cash_weight=0.0),
            prediction("2022-06-17", "CASH", assets_held=0, cash_weight=1.0),
            prediction("2022-06-21", "NVDA", assets_held=1, cash_weight=0.01),
            prediction("2024-04-18", "MSFT", assets_held=1, cash_weight=0.0),
            prediction("2024-04-19", "CASH", assets_held=0, cash_weight=1.0),
            prediction("2024-04-22", "AAPL", assets_held=1, cash_weight=0.0),
        ]
        diagnostics = [{
            "execution_date": stamp("2022-06-17"),
            "decision_date": stamp("2022-06-16"),
            "decision_reason": "CASH_THRESHOLD",
            "best_asset": "NVDA",
            "best_score": -0.05,
            "cash_threshold_guard_applied": True,
        }]
        rotations = [
            {"executed_at": stamp("2022-06-17"), "from_asset": "AVGO", "to_asset": "CASH"},
            {"executed_at": stamp("2022-06-21"), "from_asset": "CASH", "to_asset": "NVDA"},
        ]
        result = build_position_audit(equity(*days), rows, diagnostics, rotations)
        coverage = result["position_coverage"]
        self.assertEqual(coverage["total_sessions"], 6)
        self.assertEqual(coverage["recorded_sessions"], 6)
        self.assertEqual(coverage["cash_sessions"], 2)
        self.assertTrue(coverage["complete"])
        self.assertEqual([r["sessions"] for r in coverage["cash_episodes"]], [1, 1])
        first, second = result["cash_investigation"]
        self.assertEqual((first["previous_asset"], first["next_asset"]), ("AVGO", "NVDA"))
        self.assertEqual(first["entry_rotation"]["to_asset"], "CASH")
        self.assertEqual(first["exit_rotation"]["from_asset"], "CASH")
        self.assertEqual(first["decision_reason"], "CASH_THRESHOLD")
        self.assertIsNone(second["decision_reason"])
        self.assertIsNone(second["entry_rotation"])

    def test_missing_record_is_not_forward_filled_even_when_rotations_exist(self):
        days = ["2025-03-03", "2025-03-04", "2025-03-05"]
        rows = [prediction("2025-03-03", "AAPL", assets_held=1),
                prediction("2025-03-05", "AAPL", assets_held=1)]
        events = [{"executed_at": stamp("2025-03-04"), "from_asset": "AAPL", "to_asset": "CASH"}]
        result = build_position_audit(equity(*days), rows, [], events)
        self.assertEqual([r["asset"] for r in result["position_history"]], ["AAPL", None, "AAPL"])
        self.assertFalse(result["position_coverage"]["complete"])
        self.assertEqual(result["position_coverage"]["missing_sessions"], 1)
        self.assertEqual(result["position_coverage"]["cash_sessions"], 0)

    def test_final_liquidation_overrides_stale_selected_asset_but_marks_terminal_event(self):
        result = build_position_audit(
            equity("2026-10-28", "2026-10-29"),
            [prediction("2026-10-28", "NVDA", assets_held=1, cash_weight=0.0),
             prediction("2026-10-29", "NVDA", assets_held=0, cash_weight=1.0, trade_action="FINAL_SELL")],
            [], [],
        )
        self.assertEqual([r["asset"] for r in result["position_history"]], ["NVDA", "CASH"])
        self.assertTrue(result["position_history"][-1]["terminal_liquidation"])
        self.assertTrue(result["cash_investigation"][0]["terminal_liquidation"])

    def test_partial_cash_is_not_a_cash_session(self):
        result = build_position_audit(
            equity("2025-01-02", "2025-01-03"),
            [prediction("2025-01-02", "MSFT", assets_held=1, cash_weight=0.35),
             prediction("2025-01-03", None, assets_held=2, cash_weight=0.12)],
            [], [],
        )
        self.assertEqual([x["asset"] for x in result["position_history"]], ["MSFT", None])
        self.assertEqual(result["position_coverage"]["cash_sessions"], 0)
        self.assertEqual(result["position_coverage"]["missing_sessions"], 1)

    def test_consecutive_cash_sessions_form_one_episode(self):
        result = build_position_audit(
            equity("2025-01-02", "2025-01-03", "2025-01-06"),
            [prediction("2025-01-02", "CASH", assets_held=0, cash_weight=1),
             prediction("2025-01-03", "CASH", assets_held=0, cash_weight=1),
             prediction("2025-01-06", "NVDA", assets_held=1, cash_weight=0)],
            [], [],
        )
        self.assertEqual(result["position_coverage"]["cash_episodes"][0]["sessions"], 2)
        self.assertEqual(len(result["cash_investigation"]), 1)


if __name__ == "__main__":
    unittest.main()
