from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_dense_counterfactual_candidate_advantage as dcca  # noqa: E402
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (  # noqa: E402
    LABEL_AVAILABLE_COLUMN,
    OBSERVED_TARGET_COLUMN,
    TARGET_COLUMN,
)


class DenseCounterfactualCandidateAdvantageV11Tests(unittest.TestCase):
    def test_censor_aware_dense_samples_keeps_open_episode_for_scoring(self) -> None:
        timestamps = pd.to_datetime(["2026-01-05", "2026-01-06"], utc=True)
        daily_features = pd.DataFrame(
            [
                {
                    "fold": 2,
                    "timestamp": timestamps[0],
                    "candidate": "X",
                    "signal": 1.0,
                },
                {
                    "fold": 2,
                    "timestamp": timestamps[1],
                    "candidate": "Y",
                    "signal": 2.0,
                },
            ]
        )
        episodes = pd.DataFrame(
            [
                {
                    "fold": 2,
                    "candidate": "X",
                    "episode_id": 1,
                    "episode_start": timestamps[0],
                    "episode_end": timestamps[1],
                    "episode_observed_through": timestamps[1] + pd.Timedelta(days=1),
                    LABEL_AVAILABLE_COLUMN: timestamps[1] + pd.Timedelta(days=1),
                    "episode_sessions": 2,
                    "right_censored": False,
                    "baseline_asset_at_start": "A",
                    "expanded_asset_at_start": "X",
                    "baseline_holding_days_at_start": 3,
                    "expanded_holding_days_at_start": 1,
                    "policy_min_holding_days": 2,
                    "episode_definition_version": dcca.EPISODE_DEFINITION_VERSION,
                    OBSERVED_TARGET_COLUMN: 0.05,
                    TARGET_COLUMN: 0.05,
                    "positive_episode": True,
                    "changed_position_sessions": 1,
                },
                {
                    "fold": 2,
                    "candidate": "Y",
                    "episode_id": 2,
                    "episode_start": timestamps[1],
                    "episode_end": timestamps[1],
                    "episode_observed_through": timestamps[1] + pd.Timedelta(days=1),
                    LABEL_AVAILABLE_COLUMN: pd.NaT,
                    "episode_sessions": 1,
                    "right_censored": True,
                    "baseline_asset_at_start": "A",
                    "expanded_asset_at_start": "Y",
                    "baseline_holding_days_at_start": 4,
                    "expanded_holding_days_at_start": 1,
                    "policy_min_holding_days": 2,
                    "episode_definition_version": dcca.EPISODE_DEFINITION_VERSION,
                    OBSERVED_TARGET_COLUMN: -0.01,
                    TARGET_COLUMN: np.nan,
                    "positive_episode": None,
                    "changed_position_sessions": 1,
                },
            ]
        )

        original = dcca._ORIGINAL_DENSE_BUILD
        try:
            dcca._ORIGINAL_DENSE_BUILD = lambda *args, **kwargs: (
                episodes.loc[~episodes["right_censored"]].copy(),
                episodes.copy(),
                {},
            )
            samples, _, diagnostics = dcca._censor_aware_dense_samples(
                panel=object(),
                baseline_replay=object(),
                daily_features=daily_features,
                candidate_symbols=["X", "Y"],
                settings={"rotation_min_holding_days": 2},
                fold_id=2,
            )
        finally:
            dcca._ORIGINAL_DENSE_BUILD = original

        self.assertEqual(len(samples), 2)
        open_row = samples.loc[samples["candidate"] == "Y"].iloc[0]
        self.assertTrue(bool(open_row["right_censored"]))
        self.assertTrue(pd.isna(open_row[TARGET_COLUMN]))
        self.assertAlmostEqual(float(open_row[OBSERVED_TARGET_COLUMN]), -0.01)
        self.assertEqual(diagnostics["scoring_episode_rows"], 2)
        self.assertEqual(diagnostics["completed_training_episode_rows"], 1)
        self.assertEqual(diagnostics["right_censored_episode_rows"], 1)

    def test_legacy_numeric_summary_is_restored_to_pending_null(self) -> None:
        start = pd.Timestamp("2026-08-19", tz="UTC")
        decisions = pd.DataFrame(
            [
                {
                    "fold": 4,
                    "episode_start": start,
                    "episode_end": start + pd.Timedelta(days=5),
                    "chosen_candidate": "UTI",
                    "override_baseline": True,
                    "predicted_marginal_advantage": 0.03,
                    "predicted_marginal_std": 0.20,
                    "right_censored": True,
                    "realized_marginal_episode_net_log_return": np.nan,
                    OBSERVED_TARGET_COLUMN: -0.12,
                    "candidate_count_at_start": 1,
                }
            ]
        )
        compatible = dcca._numeric_summary_for_base_runner(decisions)
        self.assertTrue(compatible["_has_pending_override"])
        self.assertEqual(float(compatible["realized_marginal_log_sum"]), 0.0)

        strict, pending = dcca._strictify_summary(compatible)
        self.assertTrue(pending)
        self.assertIsNone(strict["realized_marginal_log_sum"])
        self.assertAlmostEqual(float(strict["observed_marginal_log_sum"]), -0.12)

    def test_fit_completed_dense_labels_ignores_open_target(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "fold": 1,
                    "timestamp": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=index),
                    "candidate": f"C{index}",
                    "feature": float(index),
                    LABEL_AVAILABLE_COLUMN: pd.Timestamp("2026-02-01", tz="UTC"),
                    TARGET_COLUMN: float(index) * 0.01,
                }
            )
        rows.append(
            {
                "fold": 1,
                "timestamp": pd.Timestamp("2026-01-20", tz="UTC"),
                "candidate": "OPEN",
                "feature": 999.0,
                LABEL_AVAILABLE_COLUMN: pd.NaT,
                TARGET_COLUMN: np.nan,
            }
        )
        model = dcca._fit_completed_dense_labels(pd.DataFrame(rows), ["feature"])
        self.assertEqual(model.training_rows, 8)
        self.assertTrue(np.isfinite(float(model.regressor.alpha_)))
        self.assertGreater(float(model.regressor.alpha_), 0.0)


if __name__ == "__main__":
    unittest.main()
