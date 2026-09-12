from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v110 as research  # noqa: E402


class ContextualMarginalSignatureV110Tests(unittest.TestCase):
    def _frame(self, feature_shift: float = 0.0, output_shift: float = 0.0) -> pd.DataFrame:
        rows = []
        for candidate, base in [("A", 0.01), ("B", -0.02)]:
            row = {
                "universe_name": "U1",
                "decision_date": "2026-01-02",
                "candidate": candidate,
                "evaluation_status": "completed",
                "baseline_ending_capital": 100.0,
                "candidate_ending_capital": 100.0 * (1.0 + base + output_shift),
                "ending_capital_delta_rate": base + output_shift,
                "delta_log_capital": base + output_shift,
            }
            for feature in research.FEATURE_COLUMNS:
                row[feature] = 0.5 + feature_shift
            rows.append(row)
        return pd.DataFrame(rows)

    def test_exact_copy_is_reproducible(self) -> None:
        frame = self._frame()
        summary, differences = research.compare(frame, frame.copy(), research.TOLERANCE)
        self.assertTrue(summary["exactly_reproducible"])
        self.assertEqual(summary["feature_rows_changed"], 0)
        self.assertEqual(summary["delta_log_rows_changed"], 0)
        self.assertTrue(differences.empty)

    def test_feature_drift_is_separated_from_output_drift(self) -> None:
        reference = self._frame()
        feature_drift = self._frame(feature_shift=0.01)
        summary, _ = research.compare(reference, feature_drift, research.TOLERANCE)
        self.assertGreater(summary["feature_rows_changed"], 0)
        self.assertEqual(summary["delta_log_rows_changed"], 0)
        self.assertFalse(summary["exactly_reproducible"])

    def test_output_drift_with_matching_features_is_detected(self) -> None:
        reference = self._frame()
        output_drift = self._frame(output_shift=0.03)
        summary, _ = research.compare(reference, output_drift, research.TOLERANCE)
        self.assertEqual(summary["feature_rows_changed"], 0)
        self.assertGreater(summary["candidate_capital_rows_changed"], 0)
        self.assertGreater(summary["delta_log_rows_changed"], 0)
        self.assertGreater(summary["effect_sign_mismatches"], 0)
        self.assertFalse(summary["exactly_reproducible"])

    def test_hash_is_order_invariant(self) -> None:
        frame = self._frame()
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        self.assertEqual(
            research._stable_hash(frame, research.FEATURE_COLUMNS),
            research._stable_hash(shuffled, research.FEATURE_COLUMNS),
        )


if __name__ == "__main__":
    unittest.main()
