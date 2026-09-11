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

import research_exact_universe_pruning_v1 as pruning  # noqa: E402


class ExactUniversePruningV1Tests(unittest.TestCase):
    def test_choose_best_positive_removal(self) -> None:
        rows = [
            {"removed_symbol": "AAA", "evaluation_status": "completed", "removal_capital_delta_rate": -0.01},
            {"removed_symbol": "BBB", "evaluation_status": "completed", "removal_capital_delta_rate": 0.03},
            {"removed_symbol": "CCC", "evaluation_status": "completed", "removal_capital_delta_rate": 0.07},
        ]
        chosen = pruning.choose_best_positive_removal(rows)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["removed_symbol"], "CCC")

    def test_choose_best_positive_removal_stops_at_zero(self) -> None:
        rows = [
            {"removed_symbol": "AAA", "evaluation_status": "completed", "removal_capital_delta_rate": 0.0},
            {"removed_symbol": "BBB", "evaluation_status": "completed", "removal_capital_delta_rate": -0.01},
        ]
        self.assertIsNone(pruning.choose_best_positive_removal(rows))

    def test_choose_best_positive_removal_ignores_failed_rows(self) -> None:
        rows = [
            {"removed_symbol": "AAA", "evaluation_status": "failed", "removal_capital_delta_rate": 1.0},
            {"removed_symbol": "BBB", "evaluation_status": "completed", "removal_capital_delta_rate": 0.02},
        ]
        chosen = pruning.choose_best_positive_removal(rows)
        self.assertEqual(chosen["removed_symbol"], "BBB")


if __name__ == "__main__":
    unittest.main()
