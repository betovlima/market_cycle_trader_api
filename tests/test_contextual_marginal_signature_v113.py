from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v113 as research  # noqa: E402


class ContextualMarginalSignatureV113Tests(unittest.TestCase):
    def test_level2_feature_count_for_four_channels_is_ten(self) -> None:
        self.assertEqual(len(research._feature_columns()), 10)

    def test_level2_logsignature_recovers_signed_area(self) -> None:
        path = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        result = research._level2_logsignature(path, channels=("x", "y"))
        self.assertAlmostEqual(result["logsig1__x"], 1.0, places=12)
        self.assertAlmostEqual(result["logsig1__y"], 1.0, places=12)
        self.assertAlmostEqual(result["logsig2__x__y"], 0.5, places=12)

    def test_order_changes_level2_term_with_same_end_point(self) -> None:
        x_then_y = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y_then_x = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        left = research._level2_logsignature(x_then_y, channels=("x", "y"))
        right = research._level2_logsignature(y_then_x, channels=("x", "y"))
        self.assertAlmostEqual(left["logsig1__x"], right["logsig1__x"], places=12)
        self.assertAlmostEqual(left["logsig1__y"], right["logsig1__y"], places=12)
        self.assertAlmostEqual(left["logsig2__x__y"], 0.5, places=12)
        self.assertAlmostEqual(right["logsig2__x__y"], -0.5, places=12)


if __name__ == "__main__":
    unittest.main()
