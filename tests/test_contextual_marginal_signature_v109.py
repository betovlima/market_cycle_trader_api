from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v109 as research  # noqa: E402


class ContextualMarginalSignatureV109Tests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        values = {
            "Original25": 0.04,
            "Original24_MinusADM": -0.01,
            "Original24_MinusADI": 0.03,
            "Original23_MinusADM_ADI": -0.03,
            "Original20_DropLast5": -0.05,
        }
        rows = []
        for universe, y in values.items():
            rows.append({
                "universe_name": universe,
                "decision_date": "2026-07-01",
                "candidate": "MKSI",
                "evaluation_status": "completed",
                "delta_log_capital": y,
            })
        return pd.DataFrame(rows)

    def test_decomposition_reconstructs_total_exactly(self) -> None:
        result = research.build_decomposition(self._frame())
        row = result.iloc[0]
        self.assertAlmostEqual(float(row["adm_main"]), -0.05)
        self.assertAlmostEqual(float(row["adi_main"]), -0.01)
        self.assertAlmostEqual(float(row["adm_adi_interaction"]), -0.01)
        self.assertAlmostEqual(float(row["residual_hd_adc_adea"]), -0.02)
        self.assertAlmostEqual(float(row["total_drop_last5"]), -0.09)
        self.assertAlmostEqual(float(row["reconstruction_error"]), 0.0)

    def test_adm_and_drop_last5_strict_flips_are_detected(self) -> None:
        result = research.build_decomposition(self._frame())
        row = result.iloc[0]
        self.assertTrue(bool(row["adm_sign_changed"]))
        self.assertTrue(bool(row["adm_strict_flip"]))
        self.assertTrue(bool(row["drop_last5_sign_changed"]))
        self.assertTrue(bool(row["drop_last5_strict_flip"]))
        self.assertFalse(bool(row["adi_sign_changed"]))

    def test_component_summary_tracks_signed_and_absolute_effects(self) -> None:
        result = research.build_decomposition(self._frame())
        summary = research.component_summary(result).set_index("component")
        self.assertAlmostEqual(float(summary.loc["adm_main", "mean"]), -0.05)
        self.assertAlmostEqual(float(summary.loc["total_drop_last5", "mean_abs"]), 0.09)
        self.assertEqual(int(summary.loc["adi_main", "negative_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
