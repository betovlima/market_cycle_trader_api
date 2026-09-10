from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (
    TARGET_COLUMN,
    choose_non_overlapping_episode_overrides,
    extract_candidate_divergence_episodes,
)
from market_cycle_trader_api.services.pooled_candidate_marginal_advantage import (
    fit_pooled_candidate_marginal_model,
)


class PooledCandidateEpisodeAdvantageTests(unittest.TestCase):
    def test_extracts_stateful_episode_until_policy_state_reconverges(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
        baseline = pd.DataFrame(
            {
                "timestamp": timestamps,
                "selected_asset": ["A", "A", "B", "B", "B", "C"],
                "net_log_return": [0.01, 0.01, 0.02, 0.01, 0.00, 0.01],
            }
        )
        expanded = pd.DataFrame(
            {
                "timestamp": timestamps,
                "selected_asset": ["A", "X", "X", "B", "B", "C"],
                "net_log_return": [0.01, 0.03, -0.01, 0.02, 0.00, 0.01],
            }
        )
        episodes = extract_candidate_divergence_episodes(
            baseline_daily=baseline,
            expanded_daily=expanded,
            candidate="X",
            fold_id=1,
        )
        self.assertEqual(len(episodes), 1)
        row = episodes.iloc[0]
        self.assertEqual(pd.Timestamp(row["episode_start"]), timestamps[1])
        self.assertEqual(pd.Timestamp(row["episode_end"]), timestamps[5])
        expected = (
            (0.03 - 0.01)
            + (-0.01 - 0.02)
            + (0.02 - 0.01)
            + (0.00 - 0.00)
            + (0.01 - 0.01)
        )
        self.assertAlmostEqual(float(row[TARGET_COLUMN]), expected)
        self.assertFalse(bool(row["right_censored"]))

    def test_right_censored_episode_is_marked(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        baseline = pd.DataFrame(
            {
                "timestamp": timestamps,
                "selected_asset": ["A", "A", "A"],
                "net_log_return": [0.0, 0.0, 0.0],
            }
        )
        expanded = pd.DataFrame(
            {
                "timestamp": timestamps,
                "selected_asset": ["A", "X", "X"],
                "net_log_return": [0.0, 0.01, 0.01],
            }
        )
        episodes = extract_candidate_divergence_episodes(
            baseline_daily=baseline,
            expanded_daily=expanded,
            candidate="X",
            fold_id=1,
        )
        self.assertEqual(len(episodes), 1)
        self.assertTrue(bool(episodes.iloc[0]["right_censored"]))

    def test_episode_choice_blocks_overlapping_starts(self) -> None:
        timestamps = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03"], utc=True
        )
        scored = pd.DataFrame(
            [
                {
                    "fold": 2,
                    "candidate": "A",
                    "episode_start": timestamps[0],
                    "episode_end": timestamps[2],
                    TARGET_COLUMN: 0.05,
                    "predicted_marginal_advantage": 0.02,
                    "predicted_marginal_std": 0.03,
                },
                {
                    "fold": 2,
                    "candidate": "B",
                    "episode_start": timestamps[0],
                    "episode_end": timestamps[1],
                    TARGET_COLUMN: 0.01,
                    "predicted_marginal_advantage": 0.01,
                    "predicted_marginal_std": 0.02,
                },
                {
                    "fold": 2,
                    "candidate": "C",
                    "episode_start": timestamps[1],
                    "episode_end": timestamps[2],
                    TARGET_COLUMN: 0.50,
                    "predicted_marginal_advantage": 0.50,
                    "predicted_marginal_std": 0.02,
                },
            ]
        )
        decisions = choose_non_overlapping_episode_overrides(scored)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions.iloc[0]["chosen_candidate"], "A")

    def test_pooled_bayesian_weights_keep_positive_precision(self) -> None:
        rng = np.random.default_rng(7)
        feature_names = [f"f_{index}" for index in range(20)]
        timestamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        rows = []
        for timestamp in timestamps:
            for candidate_index in range(10):
                values = rng.normal(size=len(feature_names))
                row = {
                    "fold": 1,
                    "timestamp": timestamp,
                    "candidate": f"C{candidate_index:02d}",
                    "synthetic_target": float(rng.normal(scale=0.02)),
                }
                row.update(
                    {name: float(value) for name, value in zip(feature_names, values, strict=True)}
                )
                rows.append(row)

        model = fit_pooled_candidate_marginal_model(
            pd.DataFrame(rows),
            feature_names,
            target_column="synthetic_target",
        )
        self.assertGreater(float(model.regressor.alpha_), 0.0)
        self.assertGreater(float(model.regressor.lambda_), 0.0)


if __name__ == "__main__":
    unittest.main()
