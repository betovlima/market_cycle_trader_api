from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES
from market_cycle_trader_api.services.cross_asset_utility_signature import (
    CrossAssetUtilityModel,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    build_pooled_utility_training_frame,
    choose_candidate_overrides_on_common_scale,
    common_scale_signal_metrics,
    score_common_scale_utility,
)


class _IdentityReturnModel:
    def predict(self, frame: pd.DataFrame):
        return frame["return_1"].to_numpy(dtype=float)


def _frame(index: pd.DatetimeIndex, return_values: list[float], targets: list[float]):
    data = pd.DataFrame(index=index)
    for feature in ROTATION_FEATURES:
        data[feature] = 0.0
    data["return_1"] = return_values
    data[TARGET_COLUMN] = targets
    return data


class CrossAssetUtilitySignatureTests(unittest.TestCase):
    def test_training_frame_contains_no_symbol_identity_feature(self) -> None:
        dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        frames = {
            "A": _frame(dates, [0.1, 0.2, 0.3], [0.01, 0.02, 0.03]),
            "B": _frame(dates, [0.2, 0.1, 0.4], [0.02, 0.01, 0.04]),
        }
        pooled = build_pooled_utility_training_frame(frames, ["A", "B"], dates)
        self.assertEqual(len(pooled), 6)
        self.assertIn("symbol", pooled.columns)
        self.assertNotIn("symbol", ROTATION_FEATURES)
        self.assertIn(TARGET_COLUMN, pooled.columns)

    def test_scoring_uses_one_common_model_for_different_symbols(self) -> None:
        dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
        frames = {
            "A": _frame(dates, [0.10, 0.20], [0.01, 0.02]),
            "X": _frame(dates, [0.30, 0.40], [0.03, 0.04]),
        }
        model = CrossAssetUtilityModel(
            model=_IdentityReturnModel(),
            feature_names=list(ROTATION_FEATURES),
            training_rows=10,
            training_symbols=2,
            training_dates=5,
        )
        scored = score_common_scale_utility(
            model,
            frames,
            ["A", "X"],
            dates,
            role="test",
        )
        by_symbol = scored.set_index(["timestamp", "symbol"])
        self.assertAlmostEqual(
            float(by_symbol.loc[(dates[0], "A"), PREDICTION_COLUMN]), 0.10
        )
        self.assertAlmostEqual(
            float(by_symbol.loc[(dates[0], "X"), PREDICTION_COLUMN]), 0.30
        )

    def test_override_boundary_is_common_scale_indifference(self) -> None:
        timestamp = pd.Timestamp("2026-01-05", tz="UTC")
        candidates = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "symbol": "X",
                    PREDICTION_COLUMN: 0.12,
                    "realized_forward_risk_adjusted_utility": 0.08,
                },
                {
                    "timestamp": timestamp,
                    "symbol": "Y",
                    PREDICTION_COLUMN: 0.09,
                    "realized_forward_risk_adjusted_utility": 0.15,
                },
            ]
        )
        baseline = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "baseline_target_asset": "A",
                    "baseline_common_scale_predicted_utility": 0.10,
                    "baseline_realized_forward_risk_adjusted_utility": 0.05,
                }
            ]
        )
        decisions = choose_candidate_overrides_on_common_scale(
            candidates, baseline
        )
        self.assertEqual(len(decisions), 1)
        row = decisions.iloc[0]
        self.assertTrue(bool(row["override_baseline"]))
        self.assertEqual(row["best_candidate"], "X")
        self.assertAlmostEqual(float(row["predicted_common_scale_advantage"]), 0.02)
        self.assertAlmostEqual(
            float(row["realized_common_scale_utility_advantage"]), 0.03
        )

    def test_cross_section_metrics_reward_correct_ordering(self) -> None:
        timestamp = pd.Timestamp("2026-01-05", tz="UTC")
        scored = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "symbol": "A",
                    PREDICTION_COLUMN: 0.1,
                    "realized_forward_risk_adjusted_utility": 0.01,
                },
                {
                    "timestamp": timestamp,
                    "symbol": "B",
                    PREDICTION_COLUMN: 0.2,
                    "realized_forward_risk_adjusted_utility": 0.02,
                },
                {
                    "timestamp": timestamp,
                    "symbol": "C",
                    PREDICTION_COLUMN: 0.3,
                    "realized_forward_risk_adjusted_utility": 0.03,
                },
            ]
        )
        metrics = common_scale_signal_metrics(scored)
        self.assertAlmostEqual(float(metrics["global_spearman"]), 1.0)
        self.assertAlmostEqual(
            float(metrics["mean_daily_cross_section_spearman"]), 1.0
        )
        self.assertGreater(
            float(metrics["top1_minus_cross_section_mean"]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
