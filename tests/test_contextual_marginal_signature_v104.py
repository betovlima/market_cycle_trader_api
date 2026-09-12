from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v104 as research  # noqa: E402


class ContextualMarginalSignatureV104Tests(unittest.TestCase):
    def test_exact_redundancy_is_not_in_model_features(self) -> None:
        self.assertNotIn("relative__momentum_acceleration_5_20", research.MODEL_FEATURES)
        self.assertIn("relative__return_5", research.MODEL_FEATURES)
        self.assertIn("relative__return_20", research.MODEL_FEATURES)

    def test_inner_folds_keep_whole_decision_dates_together(self) -> None:
        rows = []
        dates = pd.date_range("2020-01-02", periods=7, freq="90D", tz="UTC")
        for date in dates:
            for candidate in ["A", "B", "C"]:
                rows.append({"decision_date": date.date().isoformat(), "candidate": candidate})
        frame = pd.DataFrame(rows)
        folds = research._date_forward_folds(frame, min_train_dates=3)
        self.assertTrue(folds)
        for fit_idx, valid_idx in folds:
            fit_dates = set(frame.loc[fit_idx, "decision_date"])
            valid_dates = set(frame.loc[valid_idx, "decision_date"])
            self.assertEqual(len(valid_dates), 1)
            self.assertTrue(fit_dates.isdisjoint(valid_dates))
            validation_date = next(iter(valid_dates))
            self.assertEqual(set(frame.index[frame["decision_date"] == validation_date]), set(valid_idx))

    def test_temporal_split_uses_train_only_statistics(self) -> None:
        rows = []
        dates = ["2023-01-03", "2023-04-03", "2024-01-02"]
        for date_index, date in enumerate(dates):
            for candidate_index, candidate in enumerate(["A", "B"]):
                row = {
                    "decision_date": date,
                    "candidate": candidate,
                    "evaluation_status": "completed",
                    "delta_log_capital": float(candidate_index),
                    "ending_capital_delta_rate": float(candidate_index),
                    "decision_ts": pd.Timestamp(date, tz="UTC"),
                }
                for feature in research.MODEL_FEATURES:
                    row[feature] = float(date_index + candidate_index)
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.loc[frame["decision_date"] == "2024-01-02", research.MODEL_FEATURES[0]] = 10000.0
        train, test, train_z, test_z, *_ = research._prepare_split(frame, pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertTrue((train["decision_date"] < "2024-01-01").all())
        self.assertTrue((test["decision_date"] >= "2024-01-01").all())
        self.assertAlmostEqual(float(train_z[research.MODEL_FEATURES[0]].mean()), 0.0, places=12)
        self.assertGreater(float(test_z[research.MODEL_FEATURES[0]].abs().max()), 100.0)

    def test_zero_regime_report_separates_zero_positive_negative(self) -> None:
        rows = []
        for value in [0.0, 0.02, -0.01]:
            row = {"candidate": "A", "decision_date": "2023-01-03", "delta_log_capital": value}
            for feature in research.MODEL_FEATURES:
                row[feature] = value
            rows.append(row)
        counts, _ = research._zero_regime_report(pd.DataFrame(rows))
        observed = dict(zip(counts["effect_regime"], counts["rows"]))
        self.assertEqual(observed, {"negative": 1, "positive": 1, "zero": 1})

    def test_alpha_selection_prioritizes_chronological_rank_signal(self) -> None:
        dates = [f"2023-{month:02d}-01" for month in range(1, 7)]
        rows = []
        x_rows = []
        y_values = []
        for date in dates:
            for position, candidate in enumerate(["A", "B", "C"]):
                rows.append({"decision_date": date, "candidate": candidate})
                x_rows.append([float(position - 1)])
                y_values.append(float(position - 1) * 0.01)
        train = pd.DataFrame(rows)
        train_z = pd.DataFrame(x_rows, index=train.index, columns=["x"])
        y_train = pd.Series(y_values, index=train.index)

        class FakeModel:
            def __init__(self, alpha: float):
                self.alpha = float(alpha)
                self.coef_ = np.asarray([1.0 if self.alpha < 1.0 else 0.0])

            def predict(self, values: np.ndarray) -> np.ndarray:
                if self.alpha < 1.0:
                    return values[:, 0] * 100.0
                return np.zeros(len(values), dtype=float)

        def fake_fit(_kind: str, alpha: float, _x: np.ndarray, _y: np.ndarray) -> FakeModel:
            return FakeModel(alpha)

        with patch.object(research, "_fit_regularized", side_effect=fake_fit):
            alpha, report, metric = research._inner_cv_select_alpha(
                kind="ridge",
                alphas=[0.1, 10.0],
                train=train,
                train_z=train_z,
                y_train=y_train,
            )

        self.assertEqual(metric, "mean_within_date_spearman")
        self.assertEqual(alpha, 0.1)
        selected = report.loc[report["alpha"] == 0.1].iloc[0]
        constant = report.loc[report["alpha"] == 10.0].iloc[0]
        self.assertGreater(float(selected["mean_within_date_spearman"]), 0.99)
        self.assertEqual(int(constant["rank_valid_folds"]), 0)

    def test_constant_predictions_do_not_create_artificial_top1(self) -> None:
        test = pd.DataFrame({
            "decision_date": ["2024-01-02"] * 3,
            "candidate": ["A", "B", "C"],
            "ending_capital_delta_rate": [0.0, 0.1, -0.1],
            "delta_log_capital": [0.0, 0.095, -0.105],
        })
        test.index = pd.Index([10, 11, 12])
        y_test = pd.Series(test["delta_log_capital"].to_numpy(), index=test.index)
        y_train = pd.Series([0.0, 0.01, -0.01])
        _, summary, per_date = research._prediction_report(
            model_name="constant",
            prediction=np.zeros(3, dtype=float),
            test=test,
            y_test=y_test,
            y_train=y_train,
        )
        self.assertEqual(summary["ranking_eligible_dates"], 0)
        self.assertIsNone(summary["mean_top1_actual_delta_log"])
        self.assertFalse(bool(per_date.iloc[0]["ranking_eligible"]))
        self.assertTrue(pd.isna(per_date.iloc[0]["chosen_candidate"]))


if __name__ == "__main__":
    unittest.main()
