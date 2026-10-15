"""Regression tests for exact Live Winner candidate ranking and account isolation."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_winner_candidates import (
    latest_live_winner_decision, winner_candidate_fields,
)


class LiveWinnerCandidateTests(unittest.TestCase):
    def test_complete_ranking_preserves_actual_utilities_and_raw_leader(self):
        decision = SimpleNamespace(
            utilities={"CASH": 0.1, "NVDA": 1.2, "AAPL": 0.9},
            cash_edges={}, current_asset="CASH", target_asset="AAPL",
            raw_best_asset="NVDA", selected_utility=0.9,
            effective_switch_margin=0.2, calibrated_candidate_margin=0.15,
            calibration_score=0.5,
        )
        fields = winner_candidate_fields(decision)
        self.assertEqual(fields["candidate_count"], 2)
        self.assertEqual(fields["utilities"]["NVDA"], 1.2)
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "_id": "live-acct:2026-10-16", "execution_session": "2026-10-16",
            "winner_signal_prepared": True, "status": "signal_prepared",
            "current_asset": "CASH", "target_asset": "AAPL",
            **fields,
        }
        audit = latest_live_winner_decision(db, account_id="live-acct")
        self.assertEqual([x["symbol"] for x in audit["candidates"]], ["NVDA", "AAPL"])
        self.assertTrue(audit["candidates"][0]["is_raw_best"])
        self.assertTrue(audit["candidates"][1]["is_target"])
        self.assertEqual(audit["candidates"][1]["rank"], 2)
        self.assertFalse(audit["candidates"][1]["passes_switch_margin"])
        query = db["live_winner_session_runs"].find_one.call_args.args[0]
        self.assertEqual(query["live_account_id"], "live-acct")

    def test_absent_signal_returns_no_ranking(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = None
        self.assertIsNone(latest_live_winner_decision(db, account_id="live-acct"))

    def test_invalid_utility_map_is_rejected(self):
        decision = SimpleNamespace(
            utilities={"CASH": 0, "NVDA": float("nan")},
            cash_edges={}, current_asset="CASH", target_asset="NVDA",
            raw_best_asset="NVDA", selected_utility=0,
            effective_switch_margin=0, calibrated_candidate_margin=0,
            calibration_score=0,
        )
        with self.assertRaisesRegex(RuntimeError, "omits a decision asset"):
            winner_candidate_fields(decision)
