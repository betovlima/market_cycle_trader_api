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

import research_sequential_exact_marginal_search as research  # noqa: E402


class SequentialExactMarginalSearchTests(unittest.TestCase):
    def test_choose_best_positive_uses_exact_delta_only(self) -> None:
        rows = [
            {"symbol": "AAA", "evaluation_status": "completed", "ending_capital_delta_rate": 0.05},
            {"symbol": "BBB", "evaluation_status": "completed", "ending_capital_delta_rate": 0.12},
            {"symbol": "CCC", "evaluation_status": "completed", "ending_capital_delta_rate": -0.40},
        ]
        chosen = research.choose_best_positive(rows)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["symbol"], "BBB")

    def test_choose_best_positive_has_zero_as_only_boundary(self) -> None:
        rows = [
            {"symbol": "AAA", "evaluation_status": "completed", "ending_capital_delta_rate": 1e-12},
            {"symbol": "BBB", "evaluation_status": "completed", "ending_capital_delta_rate": 0.0},
        ]
        chosen = research.choose_best_positive(rows)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["symbol"], "AAA")

    def test_no_positive_candidate_stops_round(self) -> None:
        rows = [
            {"symbol": "AAA", "evaluation_status": "completed", "ending_capital_delta_rate": 0.0},
            {"symbol": "BBB", "evaluation_status": "completed", "ending_capital_delta_rate": -0.01},
            {"symbol": "CCC", "evaluation_status": "context_rejected", "ending_capital_delta_rate": 9.0},
        ]
        self.assertIsNone(research.choose_best_positive(rows))

    def test_symbol_normalization_is_stable_and_deduplicated(self) -> None:
        self.assertEqual(
            research.normalize_symbols([" nvda ", "AAPL", "NVDA", "", "aapl", "msft"]),
            ["NVDA", "AAPL", "MSFT"],
        )

    def test_original_25_is_unique(self) -> None:
        self.assertEqual(len(research.ORIGINAL_25), 25)
        self.assertEqual(len(set(research.ORIGINAL_25)), 25)


if __name__ == "__main__":
    unittest.main()
