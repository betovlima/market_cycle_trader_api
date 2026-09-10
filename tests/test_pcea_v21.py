from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (
    LABEL_AVAILABLE_COLUMN, OBSERVED_TARGET_COLUMN, TARGET_COLUMN,
    build_episode_samples, choose_non_overlapping_episode_overrides,
    extract_candidate_divergence_episodes, fit_episode_model, score_episode_samples,
    summarize_episode_decisions, training_episode_samples,
)
import research_pooled_candidate_marginal_advantage as pcma
import research_asset_rotation_leadership as leadership
import research_windows_file_io as file_io


class SignalModel:
    feature_names = ["signal"]

    def predict(self, frame):
        return frame.signal.to_numpy(), np.full(len(frame), 0.01)


def replay(positions, returns=None):
    dates = pd.date_range("2026-01-01", periods=len(positions), tz="UTC")
    frame = pd.DataFrame({
        "timestamp": dates,
        "execution_timestamp": dates + pd.Timedelta(days=1),
        "selected_asset": positions,
        "net_log_return": returns if returns is not None else np.zeros(len(positions)),
    })
    frame.attrs["rotation_min_holding_days"] = 2
    return frame


def examples():
    baseline = replay(["A"] * 7)
    closed = replay(["A", "X", "X", "A", "A", "A", "A"], [0, .03, -.01, 0, 0, 0, 0])
    opened = replay(["A", "Y", "Y", "Y", "Y", "Y", "Y"], [0, -.01, -.02, 0, 0, 0, 0])
    episodes = pd.concat([
        extract_candidate_divergence_episodes(
            baseline_daily=baseline, expanded_daily=expanded, candidate=symbol, fold_id=2,
        ) for expanded, symbol in ((closed, "X"), (opened, "Y"))
    ], ignore_index=True)
    features = pd.DataFrame({
        "fold": [2, 2], "candidate": ["X", "Y"],
        "timestamp": [baseline.timestamp.iloc[1]] * 2, "signal": [.02, .05],
    })
    return episodes, build_episode_samples(daily_features=features, episodes=episodes)


class PCEAV21Tests(unittest.TestCase):
    def test_open_episode_has_observed_prefix_but_no_complete_training_label(self):
        episodes, samples = examples()
        self.assertEqual(len(samples), 2)
        opened = episodes.loc[episodes.candidate == "Y"].iloc[0]
        self.assertTrue(opened.right_censored)
        self.assertAlmostEqual(opened[OBSERVED_TARGET_COLUMN], -.03)
        self.assertTrue(pd.isna(opened[TARGET_COLUMN]))
        self.assertTrue(pd.isna(opened[LABEL_AVAILABLE_COLUMN]))
        closed = episodes.loc[episodes.candidate == "X"].iloc[0]
        self.assertEqual(closed[LABEL_AVAILABLE_COLUMN], closed.episode_end + pd.Timedelta(days=1))

    def test_predictions_do_not_require_or_read_any_future_outcome(self):
        _, samples = examples()
        causal = samples[["fold", "timestamp", "candidate", "episode_start", "signal"]]
        expected = score_episode_samples(SignalModel(), causal)
        altered = samples.copy()
        altered[TARGET_COLUMN] = [np.inf, np.nan]
        altered["episode_end"] = ["unknown", None]
        altered["right_censored"] = [True, False]
        actual = score_episode_samples(SignalModel(), altered)
        self.assertEqual(actual.candidate.tolist(), ["X", "Y"])
        np.testing.assert_array_equal(actual.predicted_marginal_advantage, expected.predicted_marginal_advantage)

    def test_open_candidate_can_win_and_its_future_loss_does_not_change_admission(self):
        _, samples = examples()
        scored = score_episode_samples(SignalModel(), samples)
        decisions = choose_non_overlapping_episode_overrides(scored)
        self.assertEqual(decisions.chosen_candidate.tolist(), ["Y"])
        self.assertEqual(decisions.candidate_count_at_start.tolist(), [2])
        scored[TARGET_COLUMN] = [-100, 100]
        scored[OBSERVED_TARGET_COLUMN] = [100, -100]
        changed = choose_non_overlapping_episode_overrides(scored)
        self.assertEqual(changed.chosen_candidate.tolist(), decisions.chosen_candidate.tolist())
        self.assertTrue(pd.isna(changed.realized_marginal_episode_net_log_return.iloc[0]))

    def test_open_outcome_is_reported_separately_and_never_as_zero(self):
        _, samples = examples()
        summary = summarize_episode_decisions(
            choose_non_overlapping_episode_overrides(score_episode_samples(SignalModel(), samples))
        )
        self.assertEqual(summary["right_censored_override_count"], 1)
        self.assertEqual(summary["completed_override_count"], 0)
        self.assertAlmostEqual(summary["observed_marginal_log_sum"], -.03)
        self.assertIsNone(summary["realized_marginal_log_sum"])
        self.assertIsNone(summary["positive_realized_override_rate"])

    def test_open_choice_blocks_remainder_of_fold_but_not_next_fold(self):
        _, samples = examples()
        scored = score_episode_samples(SignalModel(), samples)
        later = scored.iloc[[0]].copy()
        later["episode_start"] += pd.Timedelta(days=1)
        later["predicted_marginal_advantage"] = 999.0
        next_fold = later.assign(fold=3)
        decisions = choose_non_overlapping_episode_overrides(pd.concat([scored, later, next_fold]))
        self.assertEqual(decisions.chosen_candidate.tolist(), ["Y", "X"])
        self.assertEqual(decisions.fold.tolist(), [2, 3])

    def test_labels_must_be_observed_before_cutoff_even_in_earlier_fold(self):
        _, samples = examples()
        closed = samples.loc[samples.candidate == "X"]
        cutoff = closed[LABEL_AVAILABLE_COLUMN].iloc[0]
        self.assertTrue(training_episode_samples(samples, available_before=cutoff).empty)
        mature = training_episode_samples(samples, available_before=cutoff + pd.Timedelta(days=1))
        self.assertEqual(mature.candidate.tolist(), ["X"])
        with self.assertRaisesRegex(RuntimeError, "label availability"):
            training_episode_samples(samples.drop(columns=LABEL_AVAILABLE_COLUMN), available_before=cutoff)

    def test_training_is_invariant_to_censored_and_immature_target_values(self):
        dates = pd.date_range("2025-01-01", periods=12, tz="UTC")
        train = pd.DataFrame({
            "fold": 1, "timestamp": dates, "episode_start": dates,
            "candidate": "X", "right_censored": [False] * 11 + [True],
            LABEL_AVAILABLE_COLUMN: dates + pd.Timedelta(days=2),
            "signal": np.linspace(-1, 1, 12), TARGET_COLUMN: np.linspace(-.1, .2, 12),
        })
        cutoff = dates[10]
        first = fit_episode_model(train, ["signal"], available_before=cutoff)
        train.loc[8:, TARGET_COLUMN] = 1e9
        second = fit_episode_model(train, ["signal"], available_before=cutoff)
        self.assertEqual(first.training_rows, 8)
        np.testing.assert_array_equal(first.predict(train)[0], second.predict(train)[0])

    def test_string_false_from_csv_is_not_treated_as_true(self):
        _, samples = examples()
        samples["right_censored"] = samples["right_censored"].astype(str)
        mature = training_episode_samples(samples, available_before="2026-02-01")
        self.assertEqual(mature.candidate.tolist(), ["X"])

    def test_infinite_replay_returns_are_rejected(self):
        baseline = replay(["A"] * 3)
        expanded = replay(["A", "X", "X"], [0, np.inf, 0])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            extract_candidate_divergence_episodes(
                baseline_daily=baseline, expanded_daily=expanded, candidate="X", fold_id=1,
            )

    def test_zero_candidate_episodes_is_a_valid_baseline_only_evaluation(self):
        scored = score_episode_samples(SignalModel(), pd.DataFrame())
        decisions = choose_non_overlapping_episode_overrides(scored)
        summary = summarize_episode_decisions(decisions)
        self.assertEqual(summary["override_count"], 0)
        self.assertEqual(summary["realized_marginal_log_sum"], 0)
        self.assertFalse(summary["is_portfolio_capital_return"])

    def test_pcea_feature_builder_never_calls_future_return_calculation(self):
        dates = pd.date_range("2026-01-01", periods=2, tz="UTC")
        # There is deliberately no next-session row and no OHLCV data: inference
        # only needs today's technical features and today's fitted-model scores.
        frames = {s: pd.DataFrame(1.0, index=dates[:1], columns=pcma.ROTATION_FEATURES) for s in ["A", "B", "X"]}
        with (
            patch.object(pcma, "_utility_policy", return_value=lambda *args: (1, .1)),
            patch.object(pcma, "_model_utilities", return_value=np.array([0, .1, .05])),
            patch.object(pcma, "_candidate_score", return_value=.2),
            patch.object(pcma, "_training_transition_log_return", side_effect=AssertionError("Future outcome accessed")),
        ):
            samples = pcma._build_samples(
                fold_id=2, frames=frames, baseline_symbols=["A", "B"], candidate_symbols=["X"],
                baseline_models={"A": object(), "B": object()}, candidate_models={"X": object()},
                candidate_references={"X": (0., 1., np.array([0., .1, .2]))},
                decision_dates=dates, config=SimpleNamespace(), effective_margin=0., calibrated_margin=0.,
                phase_label="unit_inference", include_future_targets=False,
            )
        self.assertEqual(len(samples), 1)
        self.assertNotIn(pcma.TARGET_COLUMN, samples)
        self.assertTrue(set(pcma.MODEL_FEATURES).issubset(samples.columns))

    def test_leadership_csv_and_json_round_trip_in_long_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / ("experiment_" * 12) / ("selection_" * 12)
            csv_path = output / "intrinsic_timing_summary.csv"
            self.assertGreater(len(str(csv_path)), 260)
            frame = pd.DataFrame({"symbol": ["A", "B"], "value": [.1, -.2]})
            leadership._write_csv(csv_path, frame)
            self.assertTrue(file_io.exists(csv_path))
            pd.testing.assert_frame_equal(file_io.read_csv(csv_path), frame)
            leadership._write_json(output / "snapshot.json", {"version": "2.1.0"})
            self.assertEqual(file_io.read_json(output / "snapshot.json"), {"version": "2.1.0"})
            file_io.remove_tree(output.parent)


if __name__ == "__main__":
    unittest.main()
