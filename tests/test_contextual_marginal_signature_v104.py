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
        # Make one validation value extreme. It must not influence train means/stds.
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


if __name__ == "__main__":
    unittest.main()
