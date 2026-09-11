from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_contextual_marginal_signature_v1 as research  # noqa: E402


class ContextualMarginalSignatureV1Tests(unittest.TestCase):
    def test_select_decision_sessions_is_monthly_deterministic_and_horizon_safe(self) -> None:
        sessions = pd.date_range("2020-01-02", periods=260, freq="B", tz="UTC")
        selected = research.select_decision_sessions(
            sessions,
            pd.Timestamp("2020-02-01", tz="UTC"),
            pd.Timestamp("2020-10-31", tz="UTC"),
            5,
            20,
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(len({(item.year, item.month) for item in selected}), 5)
        for decision in selected:
            position = int(sessions.searchsorted(decision))
            self.assertLess(position + 19, len(sessions))

    def test_panel_completeness_detects_missing_and_failed_cells(self) -> None:
        dates = [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")]
        candidates = ["AAA", "BBB"]
        frame = pd.DataFrame(
            [
                {"decision_date": "2024-01-02", "candidate": "AAA", "evaluation_status": "completed"},
                {"decision_date": "2024-01-02", "candidate": "BBB", "evaluation_status": "failed"},
                {"decision_date": "2024-02-01", "candidate": "AAA", "evaluation_status": "completed"},
            ]
        )
        issues = research.panel_completeness_issues(frame, dates, candidates)
        joined = " | ".join(issues)
        self.assertIn("missing:2024-02-01:BBB", joined)
        self.assertIn("non_completed:2024-01-02:BBB=failed", joined)

    def test_temporal_ols_validation_recovers_forward_linear_structure(self) -> None:
        rows = []
        dates = pd.date_range("2020-01-02", periods=10, freq="90D", tz="UTC")
        for date_index, decision in enumerate(dates):
            for candidate_index, candidate in enumerate(["A", "B", "C", "D"]):
                x1 = float(candidate_index - 1.5)
                x2 = float(date_index) / 10.0
                values = {feature: 0.0 for feature in research.MODEL_FEATURES}
                values["relative__return_20"] = x1
                values["corr_60_to_universe"] = x2
                target = 0.08 * x1 - 0.03 * x2
                rows.append(
                    {
                        "decision_date": decision.date().isoformat(),
                        "candidate": candidate,
                        "evaluation_status": "completed",
                        "ending_capital_delta_rate": float(np.expm1(target)),
                        "delta_log_capital": target,
                        **values,
                    }
                )
        dataset = pd.DataFrame(rows)
        predictions, coefficients, summary = research.temporal_ols_validation(
            dataset,
            pd.Timestamp(dates[7]),
        )
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(summary["pearson_prediction_vs_actual"], 0.99)
        self.assertGreater(summary["sign_accuracy"], 0.9)
        self.assertFalse(predictions.empty)
        self.assertIn("relative__return_20", set(coefficients["feature"]))

    def test_model_inputs_do_not_include_forward_target_columns(self) -> None:
        self.assertTrue(all("forward_" not in feature for feature in research.MODEL_FEATURES))
        self.assertTrue(all(feature in research.rotation.ROTATION_FEATURES for feature in research.BASE_FEATURES))


if __name__ == "__main__":
    unittest.main()
