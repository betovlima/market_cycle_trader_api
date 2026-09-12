from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v112_analysis as analysis  # noqa: E402


class ContextualMarginalSignatureV112AnalysisTests(unittest.TestCase):
    def test_exact_log_decomposition(self) -> None:
        native = pd.DataFrame(
            [
                {
                    "decision_date": "2026-01-02",
                    "universe_name": "U",
                    "candidate": "APD",
                    "baseline_ending_capital": 100.0,
                    "candidate_ending_capital": 110.0,
                    "delta_log_capital": __import__("math").log(1.10),
                }
            ]
        )
        frozen = pd.DataFrame(
            [
                {
                    "decision_date": "2026-01-02",
                    "universe_name": "U",
                    "candidate": "APD",
                    "baseline_ending_capital": 100.0,
                    "candidate_ending_capital": 105.0,
                    "delta_log_capital": __import__("math").log(1.05),
                }
            ]
        )
        rows = analysis._decompose(native, frozen)
        row = rows.iloc[0]
        self.assertAlmostEqual(float(row["frozen_marginal_log"]), __import__("math").log(1.05), places=12)
        self.assertAlmostEqual(
            float(row["switch_margin_recalibration_component"]),
            __import__("math").log(1.10) - __import__("math").log(1.05),
            places=12,
        )
        self.assertAlmostEqual(float(row["reconstruction_error"]), 0.0, places=12)

    def test_baseline_change_is_rejected(self) -> None:
        native = pd.DataFrame(
            [{
                "decision_date": "2026-01-02", "universe_name": "U", "candidate": "APD",
                "baseline_ending_capital": 100.0, "candidate_ending_capital": 110.0,
                "delta_log_capital": __import__("math").log(1.10),
            }]
        )
        frozen = pd.DataFrame(
            [{
                "decision_date": "2026-01-02", "universe_name": "U", "candidate": "APD",
                "baseline_ending_capital": 99.0, "candidate_ending_capital": 110.0,
                "delta_log_capital": __import__("math").log(110.0 / 99.0),
            }]
        )
        with self.assertRaisesRegex(RuntimeError, "Baseline capital changed"):
            analysis._decompose(native, frozen)


if __name__ == "__main__":
    unittest.main()
