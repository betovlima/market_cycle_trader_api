from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_exact_bidirectional_universe_search_v1 as search  # noqa: E402


class ExactBidirectionalUniverseSearchV1Tests(unittest.TestCase):
    def test_normalize_addition_row(self) -> None:
        row = search.normalize_move_row(
            {
                "evaluation_status": "completed",
                "ending_capital_delta_rate": 0.05,
                "candidate_ending_capital": 105.0,
            },
            "add",
            "aaa",
        )
        self.assertEqual(row["move_key"], "add:AAA")
        self.assertEqual(row["move_delta_rate"], 0.05)
        self.assertEqual(row["resulting_ending_capital"], 105.0)

    def test_normalize_removal_row(self) -> None:
        row = search.normalize_move_row(
            {
                "evaluation_status": "completed",
                "removal_capital_delta_rate": 0.07,
                "reduced_ending_capital": 107.0,
            },
            "remove",
            "bbb",
        )
        self.assertEqual(row["move_key"], "remove:BBB")
        self.assertEqual(row["move_delta_rate"], 0.07)
        self.assertEqual(row["resulting_ending_capital"], 107.0)

    def test_choose_best_positive_move_compares_add_and_remove(self) -> None:
        rows = [
            {
                "move_key": "add:AAA",
                "move_type": "add",
                "move_symbol": "AAA",
                "evaluation_status": "completed",
                "move_delta_rate": 0.04,
            },
            {
                "move_key": "remove:BBB",
                "move_type": "remove",
                "move_symbol": "BBB",
                "evaluation_status": "completed",
                "move_delta_rate": 0.09,
            },
        ]
        chosen = search.choose_best_positive_move(rows)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["move_key"], "remove:BBB")

    def test_choose_best_positive_move_stops_at_zero(self) -> None:
        rows = [
            {
                "move_key": "add:AAA",
                "evaluation_status": "completed",
                "move_delta_rate": 0.0,
            },
            {
                "move_key": "remove:BBB",
                "evaluation_status": "completed",
                "move_delta_rate": -0.01,
            },
        ]
        self.assertIsNone(search.choose_best_positive_move(rows))

    def test_round_completeness_requires_every_move_completed(self) -> None:
        expected = {"add:AAA", "remove:BBB"}
        rows = [
            {"move_key": "add:AAA", "evaluation_status": "completed"},
            {"move_key": "remove:BBB", "evaluation_status": "failed"},
        ]
        issues = search.round_completeness_issues(rows, expected)
        self.assertTrue(any("remove:BBB=failed" in issue for issue in issues))

    def test_round_completeness_detects_missing_move(self) -> None:
        expected = {"add:AAA", "remove:BBB"}
        rows = [{"move_key": "add:AAA", "evaluation_status": "completed"}]
        issues = search.round_completeness_issues(rows, expected)
        self.assertTrue(any("missing:remove:BBB" in issue for issue in issues))

    def test_round_completeness_accepts_full_completed_round(self) -> None:
        expected = {"add:AAA", "remove:BBB"}
        rows = [
            {"move_key": "add:AAA", "evaluation_status": "completed"},
            {"move_key": "remove:BBB", "evaluation_status": "completed"},
        ]
        self.assertEqual(search.round_completeness_issues(rows, expected), [])


if __name__ == "__main__":
    unittest.main()
