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

import research_contextual_marginal_signature_v103 as research  # noqa: E402


class FakeConfig:
    rotation_purge_days = 2
    rotation_target_horizons = [2]
    rotation_walk_forward_calibration_days = 4
    rotation_walk_forward_test_days = 5
    rotation_walk_forward_min_test_days = 3
    rotation_minimum_training_rows = 6
    walk_forward_fold_count_override = None


class ContextualMarginalSignatureV103Tests(unittest.TestCase):
    def test_horizon_end_accepts_naive_calendar(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=30, freq="B")
        decision = pd.Timestamp("2024-01-05", tz="UTC")
        result = research._horizon_end(sessions, decision, 5)
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result, pd.Timestamp("2024-01-11", tz="UTC"))

    def test_executable_window_rejects_pre_oos_and_accepts_later_window(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=50, freq="B", tz="UTC")
        config = FakeConfig()
        early = research._window_is_executable(
            sessions,
            sessions[4],
            sessions[20],
            config,
        )
        later = research._window_is_executable(
            sessions,
            sessions[18],
            sessions[30],
            config,
        )
        self.assertFalse(early)
        self.assertTrue(later)

    def test_select_executable_decisions_filters_early_months(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=120, freq="B", tz="UTC")
        config = FakeConfig()
        selected = research.select_executable_decision_sessions(
            sessions,
            sessions,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-05-31"),
            3,
            5,
            config,
        )
        self.assertTrue(selected)
        self.assertEqual(selected, sorted(selected))
        for decision in selected:
            horizon = research._horizon_end(sessions, decision, 5)
            self.assertTrue(research._window_is_executable(sessions, decision, horizon, config))

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
                rows.append({
                    "decision_date": decision.date().isoformat(),
                    "candidate": candidate,
                    "evaluation_status": "completed",
                    "ending_capital_delta_rate": float(np.expm1(target)),
                    "delta_log_capital": target,
                    **values,
                })
        predictions, coefficients, summary = research.temporal_ols_validation(
            pd.DataFrame(rows), pd.Timestamp(dates[7])
        )
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(summary["pearson_prediction_vs_actual"], 0.99)
        self.assertFalse(predictions.empty)
        self.assertIn("relative__return_20", set(coefficients["feature"]))

    def test_model_features_do_not_use_forward_columns(self) -> None:
        self.assertTrue(all("forward_" not in feature for feature in research.MODEL_FEATURES))
        self.assertTrue(all(feature in research.rotation.ROTATION_FEATURES for feature in research.BASE_FEATURES))


if __name__ == "__main__":
    unittest.main()
