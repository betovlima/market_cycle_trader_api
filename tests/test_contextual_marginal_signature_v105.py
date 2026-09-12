from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v105 as research  # noqa: E402


class ContextualMarginalSignatureV105Tests(unittest.TestCase):
    def _row(self, date: str, candidate: str, x1: float, x2: float, y: float) -> dict:
        values = {feature: 0.0 for feature in research.MODEL_FEATURES}
        values["relative__return_20"] = x1
        values["relative__vol_20"] = x2
        return {
            "decision_date": date,
            "candidate": candidate,
            "evaluation_status": "completed",
            "ending_capital_delta_rate": float(np.expm1(y)),
            "delta_log_capital": y,
            **values,
        }

    def test_forward_folds_never_split_one_decision_date(self) -> None:
        rows = []
        dates = pd.date_range("2020-01-02", periods=7, freq="90D")
        for date in dates:
            for candidate in ["A", "B", "C"]:
                rows.append(self._row(date.date().isoformat(), candidate, 0.0, 0.0, 0.0))
        frame = pd.DataFrame(rows)
        folds = research._date_forward_folds(frame, min_train_dates=3)
        self.assertTrue(folds)
        for fit_dates, valid_date in folds:
            self.assertNotIn(valid_date, fit_dates)
            self.assertEqual(
                set(frame.loc[frame["decision_date"] == valid_date, "decision_date"]),
                {valid_date},
            )

    def test_depth_selection_uses_chronological_rank(self) -> None:
        rows = []
        dates = pd.date_range("2020-01-02", periods=9, freq="90D")
        candidates = [f"C{i:02d}" for i in range(24)]
        for date_index, date in enumerate(dates):
            for i, candidate in enumerate(candidates):
                x1 = -1.0 if i < 12 else 1.0
                x2 = -1.0 if i % 12 < 6 else 1.0
                y = 0.06 if x1 * x2 > 0 else -0.04
                y += 0.0001 * date_index
                rows.append(self._row(date.date().isoformat(), candidate, x1, x2, y))
        frame = pd.DataFrame(rows)
        cv = research._inner_cv(frame)
        selected = research._select_depth(cv)
        self.assertIn(selected, (2, 3))
        depth1 = float(cv.loc[cv["max_depth"] == 1, "mean_within_date_spearman"].iloc[0])
        selected_score = float(cv.loc[cv["max_depth"] == selected, "mean_within_date_spearman"].iloc[0])
        self.assertGreater(selected_score, depth1)

    def test_tied_predictions_are_not_reported_as_top1(self) -> None:
        rows = [
            self._row("2024-02-01", "A", 0.0, 0.0, 0.03),
            self._row("2024-02-01", "B", 0.0, 0.0, -0.02),
            self._row("2024-02-01", "C", 0.0, 0.0, 0.00),
        ]
        frame = pd.DataFrame(rows)
        per_date, summary = research._ranking_metrics(
            frame, np.zeros(len(frame)), model_name="constant"
        )
        self.assertFalse(bool(per_date.iloc[0]["ranking_eligible"]))
        self.assertEqual(summary["ranking_eligible_dates"], 0)
        self.assertIsNone(summary["top1_positive_rate"])

    def test_exact_redundant_feature_is_not_used(self) -> None:
        self.assertNotIn("relative__momentum_acceleration_5_20", research.MODEL_FEATURES)
        self.assertEqual(research.TREE_DEPTHS, [1, 2, 3])
        self.assertGreaterEqual(research.MIN_SAMPLES_LEAF, 12)


if __name__ == "__main__":
    unittest.main()
