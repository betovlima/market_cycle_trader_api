from __future__ import annotations

from pathlib import Path
import inspect
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

import research_contextual_marginal_signature_v102 as research  # noqa: E402


class ContextualMarginalSignatureV102Tests(unittest.TestCase):
    def test_processor_does_not_import_previous_research_processors(self) -> None:
        source = inspect.getsource(research)
        self.assertNotIn("import research_contextual_marginal_signature_v1", source)
        self.assertNotIn("import research_contextual_marginal_signature_v101", source)
        self.assertNotIn("import research_asset_signature_leave_one_out", source)
        self.assertNotIn("import research_sequential_exact_marginal_search", source)

    def test_horizon_end_accepts_naive_calendar(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=80, freq="B")
        decision = pd.Timestamp("2024-01-15", tz="UTC")
        end = research._horizon_end(sessions, decision, 20)
        self.assertEqual(end.tzinfo, pd.Timestamp("2024-01-01", tz="UTC").tzinfo)
        expected_position = int(research._utc_index(sessions).searchsorted(decision, side="left")) + 19
        self.assertEqual(end, research._utc_index(sessions)[expected_position])

    def test_horizon_end_accepts_aware_calendar(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=80, freq="B", tz="UTC")
        decision = pd.Timestamp("2024-01-15", tz="UTC")
        end = research._horizon_end(sessions, decision, 20)
        expected_position = int(sessions.searchsorted(decision, side="left")) + 19
        self.assertEqual(end, sessions[expected_position])

    def test_temporal_ols_recovers_simple_structure(self) -> None:
        rows = []
        dates = pd.date_range("2020-01-02", periods=10, freq="90D", tz="UTC")
        for date_index, decision in enumerate(dates):
            for candidate_index, candidate in enumerate(["A", "B", "C", "D"]):
                values = {feature: 0.0 for feature in research.MODEL_FEATURES}
                x1 = float(candidate_index - 1.5)
                x2 = float(date_index) / 10.0
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
        predictions, coefficients, summary = research.temporal_ols_validation(dataset, pd.Timestamp(dates[7]))
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(summary["pearson_prediction_vs_actual"], 0.99)
        self.assertGreater(summary["sign_accuracy"], 0.9)
        self.assertFalse(predictions.empty)
        self.assertIn("relative__return_20", set(coefficients["feature"]))

    def test_model_inputs_are_backward_looking(self) -> None:
        self.assertTrue(all("forward_" not in feature for feature in research.MODEL_FEATURES))
        self.assertTrue(all(feature in research.rotation.ROTATION_FEATURES for feature in research.BASE_FEATURES))


if __name__ == "__main__":
    unittest.main()
