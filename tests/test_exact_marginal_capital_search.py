from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.services.exact_marginal_capital_search import (  # noqa: E402
    annotate_preselector_ranks,
    build_preselector_recall,
    economic_outcome,
    summarize_exact_search,
)


class ExactMarginalCapitalSearchTests(unittest.TestCase):
    def test_preselector_rank_never_drops_unrankable_candidates(self) -> None:
        rows = [
            {"symbol": "AAA", "preselector_raw_score": 0.2},
            {"symbol": "BBB", "preselector_raw_score": None},
            {"symbol": "CCC", "preselector_raw_score": 0.9},
        ]
        ranked = annotate_preselector_ranks(rows)
        by_symbol = {row["symbol"]: row for row in ranked}
        self.assertEqual(len(ranked), 3)
        self.assertEqual(by_symbol["CCC"]["preselector_rank"], 1)
        self.assertEqual(by_symbol["AAA"]["preselector_rank"], 2)
        self.assertIsNone(by_symbol["BBB"]["preselector_rank"])

    def test_recall_measures_exact_positive_recovery_not_score_quality(self) -> None:
        rows = annotate_preselector_ranks(
            [
                {
                    "symbol": "A",
                    "preselector_raw_score": 0.9,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": -0.10,
                },
                {
                    "symbol": "B",
                    "preselector_raw_score": 0.8,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": 0.50,
                },
                {
                    "symbol": "C",
                    "preselector_raw_score": 0.7,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": 0.05,
                },
                {
                    "symbol": "D",
                    "preselector_raw_score": 0.6,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": -0.02,
                },
            ]
        )
        recall = build_preselector_recall(rows, cutoffs=(1, 2, 3))
        by_cutoff = {row["requested_cutoff"]: row for row in recall}
        self.assertEqual(by_cutoff[1]["positive_candidates_in_top_k"], 0)
        self.assertEqual(by_cutoff[2]["positive_candidates_in_top_k"], 1)
        self.assertAlmostEqual(by_cutoff[2]["positive_recall"], 0.5)
        self.assertEqual(by_cutoff[3]["positive_candidates_in_top_k"], 2)
        self.assertAlmostEqual(by_cutoff[3]["positive_recall"], 1.0)

    def test_unranked_positive_is_counted_as_preselector_miss(self) -> None:
        rows = annotate_preselector_ranks(
            [
                {
                    "symbol": "A",
                    "preselector_raw_score": 0.9,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": 0.10,
                },
                {
                    "symbol": "B",
                    "preselector_raw_score": None,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": 0.20,
                },
            ]
        )
        recall = build_preselector_recall(rows, cutoffs=(1,))
        self.assertEqual(recall[0]["total_exact_positive_candidates"], 2)
        self.assertEqual(recall[0]["unranked_exact_positive_candidates"], 1)
        self.assertAlmostEqual(recall[0]["positive_recall"], 0.5)

    def test_economic_outcome_uses_only_exact_capital_delta(self) -> None:
        self.assertEqual(
            economic_outcome(
                {"evaluation_status": "completed", "ending_capital_delta_rate": 0.001}
            ),
            "positive",
        )
        self.assertEqual(
            economic_outcome(
                {"evaluation_status": "completed", "ending_capital_delta_rate": -0.001}
            ),
            "negative",
        )
        self.assertEqual(
            economic_outcome(
                {"evaluation_status": "history_rejected", "ending_capital_delta_rate": 10.0}
            ),
            "history_rejected",
        )

    def test_summary_keeps_positive_negative_and_failed_outcomes(self) -> None:
        summary = summarize_exact_search(
            [
                {
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": 0.20,
                    "exact_replay_seconds": 10.0,
                },
                {
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": -0.10,
                    "exact_replay_seconds": 20.0,
                },
                {"evaluation_status": "failed", "exact_replay_seconds": 5.0},
            ]
        )
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["completed_count"], 2)
        self.assertEqual(summary["positive_count"], 1)
        self.assertAlmostEqual(summary["positive_rate"], 0.5)
        self.assertEqual(summary["outcome_counts"]["positive"], 1)
        self.assertEqual(summary["outcome_counts"]["negative"], 1)
        self.assertEqual(summary["outcome_counts"]["failed"], 1)
        self.assertAlmostEqual(summary["total_exact_replay_seconds"], 35.0)


if __name__ == "__main__":
    unittest.main()
