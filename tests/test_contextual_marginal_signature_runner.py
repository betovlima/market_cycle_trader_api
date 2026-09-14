from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import exchange_calendars as xcals
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature as runner  # noqa: E402
import research_contextual_signature_analysis as analysis  # noqa: E402
import research_contextual_signature_storage as storage  # noqa: E402
import research_contextual_signature_tournament as tournament  # noqa: E402
from research_contextual_signature_runtime import (  # noqa: E402
    compare_replay_outputs,
)

implementation = runner._impl


class ContextualMarginalSignatureRunnerTests(unittest.TestCase):
    def test_stable_entrypoint_and_export_names_have_no_version_suffix(self) -> None:
        self.assertEqual(runner.EXPORT_FOLDER_NAME, "contextual_marginal_signature")
        self.assertEqual(runner.EXPORT_ZIP_NAME, "contextual_marginal_signature.zip")
        self.assertNotIn("v1", runner.EXPORT_FOLDER_NAME)
        self.assertNotIn("v1", runner.EXPORT_ZIP_NAME)
        self.assertEqual(
            runner.SCRIPT_VERSION,
            "contextual-marginal-signature-v1.0.17.5",
        )
        self.assertEqual(
            runner.TOURNAMENT_VERSION,
            "contextual-marginal-signature-v1.0.20.0",
        )
        self.assertEqual(runner.HORIZON_SESSIONS, 40)
        self.assertEqual(len(runner.DECISION_DATES), 23)
        self.assertEqual(len(runner.CANDIDATES), 7)
        self.assertEqual(len(runner.UNIVERSES), 2)

    def test_opportunity_hole_contract(self) -> None:
        self.assertEqual(
            tournament.EXPERIMENT_NAME,
            "contextual_marginal_signature_opportunity_hole_decomposition",
        )
        self.assertEqual(tournament.TARGET, "source_direct_delta_log_capital")
        self.assertEqual(tournament.HOLE_HORIZONS, (5, 10, 20, 40))
        self.assertEqual(tournament.PRIMARY_HOLE_HORIZON, 20)

    def test_context_classifier_separates_covered_policy_miss_and_hole(self) -> None:
        self.assertEqual(
            tournament._classify_context(0.01, -0.20),
            "COVERED",
        )
        self.assertEqual(
            tournament._classify_context(-0.01, 0.05),
            "POLICY_MISS",
        )
        self.assertEqual(
            tournament._classify_context(-0.01, -0.02),
            "UNIVERSE_HOLE",
        )

    def test_clean_candidate_entry_is_not_indirect_perturbation(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, tz="UTC")
        baseline = pd.DataFrame(
            {
                "timestamp": dates,
                "selected_asset": ["A", "A", "A", "A"],
                "strategy_equity": [100.0, 99.0, 98.0, 97.0],
            }
        )
        policy = pd.DataFrame(
            {
                "timestamp": dates,
                "selected_asset": ["A", "A", "C", "C"],
                "strategy_equity": [100.0, 99.0, 101.0, 102.0],
            }
        )
        result = tournament._trace_divergence(baseline, policy, "C")
        self.assertTrue(result["candidate_ever_selected"])
        self.assertTrue(result["divergence_is_candidate_entry"])
        self.assertFalse(result["perturbation_before_candidate_selection"])

    def test_divergence_before_candidate_is_indirect_perturbation(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, tz="UTC")
        baseline = pd.DataFrame(
            {
                "timestamp": dates,
                "selected_asset": ["A", "A", "A", "A"],
                "strategy_equity": [100.0, 99.0, 98.0, 97.0],
            }
        )
        policy = pd.DataFrame(
            {
                "timestamp": dates,
                "selected_asset": ["A", "B", "C", "C"],
                "strategy_equity": [100.0, 99.0, 101.0, 102.0],
            }
        )
        result = tournament._trace_divergence(baseline, policy, "C")
        self.assertFalse(result["divergence_is_candidate_entry"])
        self.assertTrue(result["perturbation_before_candidate_selection"])

    def test_forward_opportunity_label_does_not_use_rows_after_horizon(self) -> None:
        dates = pd.date_range("2025-01-01", periods=8, tz="UTC")
        base = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0, 104.0, 103.0, 102.0, 101.0, 100.0]},
            index=dates,
        )
        changed_future = base.copy()
        changed_future.loc[dates[5]:, "close"] = 10000.0
        before = tournament._forward_log_return(
            base,
            dates[0],
            dates[4],
            0.0,
        )
        after = tournament._forward_log_return(
            changed_future,
            dates[0],
            dates[4],
            0.0,
        )
        self.assertEqual(before, after)

    def test_opportunity_status_requires_clean_coverage(self) -> None:
        summary = {
            "primary_horizon": {
                "universe_holes": 2,
                "holes_with_any_candidate_fill": 2,
                "holes_with_any_clean_coverage": 0,
            }
        }
        self.assertEqual(
            tournament.opportunity_status(summary),
            "COVERAGE_FOUND_BUT_NOT_CLEAN",
        )
        summary["primary_horizon"]["holes_with_any_clean_coverage"] = 1
        self.assertEqual(
            tournament.opportunity_status(summary),
            "CLEAN_OPPORTUNITY_COVERAGE_OBSERVED",
        )

    def test_v1019_trajectory_helper_still_rejects_no_added_value(self) -> None:
        snapshot = {"cumulative_direct_log_gain": 0.20}
        trajectory = {"cumulative_direct_log_gain": 0.19}
        baseline = {"cumulative_direct_log_gain": 0.05}
        holdout = {"cumulative_direct_log_gain": 0.10}
        robust = {"cumulative_direct_log_gain": 0.10}
        self.assertEqual(
            tournament.trajectory_status(
                trajectory,
                snapshot,
                baseline,
                holdout,
                robust,
            ),
            "TRAJECTORY_ADDED_VALUE_NOT_FOUND",
        )

    def test_temporal_states_are_real_xnys_sessions(self) -> None:
        calendar = xcals.get_calendar("XNYS")
        self.assertGreaterEqual(
            pd.Timestamp(runner.DECISION_DATES[0]),
            pd.Timestamp("2020-08-03"),
        )
        for value in runner.DECISION_DATES:
            resolved = calendar.date_to_session(
                pd.Timestamp(value),
                direction="none",
            )
            self.assertEqual(pd.Timestamp(resolved).date(), pd.Timestamp(value).date())

    def test_temporal_states_are_spread_beyond_target_horizon(self) -> None:
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            pd.Timestamp(runner.DECISION_DATES[0]),
            pd.Timestamp(runner.DECISION_DATES[-1]),
        )
        positions = [
            int(sessions.searchsorted(pd.Timestamp(value)))
            for value in runner.DECISION_DATES
        ]
        gaps = [right - left for left, right in zip(positions, positions[1:])]
        self.assertGreaterEqual(min(gaps), runner.HORIZON_SESSIONS)

    def test_console_log_does_not_render_iso_calendar_date(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            analysis.console_log("state 2026-07-01 -> 2026-08-25")
        text = stream.getvalue()
        self.assertNotIn("2026-07-01", text)
        self.assertNotIn("2026-08-25", text)
        self.assertIn("01/07/2026", text)
        self.assertIn("25/08/2026", text)

    def test_mongo_value_normalizes_nat_and_nonfinite_scalars(self) -> None:
        payload = storage.mongo_value(
            {
                "pandas_nat": pd.NaT,
                "numpy_nat": np.datetime64("NaT"),
                "pandas_na": pd.NA,
                "nan": np.nan,
                "positive_infinity": np.inf,
                "nested": [pd.NaT, np.float64("nan")],
            }
        )
        self.assertIsNone(payload["pandas_nat"])
        self.assertIsNone(payload["numpy_nat"])
        self.assertIsNone(payload["pandas_na"])
        self.assertIsNone(payload["nan"])
        self.assertIsNone(payload["positive_infinity"])
        self.assertEqual(payload["nested"], [None, None])

    def test_resume_observation_identity_is_deterministic(self) -> None:
        row = {
            "decision_date": runner.DECISION_DATES[0],
            "universe_name": "Original25",
            "candidate": "XSD",
        }
        expected = (runner.DECISION_DATES[0], "Original25", "XSD")
        self.assertEqual(implementation._observation_key(row), expected)
        self.assertEqual(
            implementation._observation_key(row),
            implementation._observation_key(dict(row)),
        )

    def test_replay_equivalence_guard_accepts_identical_results(self) -> None:
        sessions = pd.date_range("2025-01-01", periods=2, tz="UTC")
        predictions = pd.DataFrame(
            {
                "selected_asset": ["A", "A"],
                "strategy_equity": [100.0, 101.0],
            },
            index=sessions,
        )
        trades = pd.DataFrame({"action": ["BUY"], "price": [10.0]})
        captured = [SimpleNamespace(predictions=predictions, trades=trades)]
        replay = ({"ending_capital": 101.0}, sessions, captured)
        result = compare_replay_outputs(replay, replay)
        self.assertTrue(result["passed"])
        self.assertEqual(result["capital_relative_error"], 0.0)
        self.assertTrue(result["sessions_identical"])
        self.assertTrue(result["predictions_identical"])
        self.assertTrue(result["trades_identical"])

    def test_replay_equivalence_guard_rejects_semantic_divergence(self) -> None:
        sessions = pd.date_range("2025-01-01", periods=2, tz="UTC")
        ref_predictions = pd.DataFrame(
            {
                "selected_asset": ["A", "A"],
                "strategy_equity": [100.0, 101.0],
            },
            index=sessions,
        )
        alt_predictions = pd.DataFrame(
            {
                "selected_asset": ["A", "B"],
                "strategy_equity": [100.0, 100.5],
            },
            index=sessions,
        )
        ref = (
            {"ending_capital": 101.0},
            sessions,
            [SimpleNamespace(predictions=ref_predictions, trades=pd.DataFrame())],
        )
        alt = (
            {"ending_capital": 100.5},
            sessions,
            [SimpleNamespace(predictions=alt_predictions, trades=pd.DataFrame())],
        )
        result = compare_replay_outputs(ref, alt)
        self.assertFalse(result["passed"])
        self.assertFalse(result["predictions_identical"])
        self.assertGreater(result["capital_relative_error"], 0.0)

    def test_readiness_requires_temporal_diversity_and_intervention_abstention(self) -> None:
        rows = []
        for date_index, date in enumerate(runner.DECISION_DATES):
            for universe in ("U1", "U2"):
                for candidate_index, candidate in enumerate(runner.CANDIDATES):
                    value = (
                        -0.01 - candidate_index * 0.001
                        if date_index % 5 == 0
                        else 0.02 - candidate_index * 0.006
                    )
                    rows.append(
                        {
                            "candidate": candidate,
                            "decision_date": date,
                            "universe_name": universe,
                            "action_advantage_log": value,
                        }
                    )
        result = analysis.readiness(pd.DataFrame(rows), len(runner.CANDIDATES), 2)
        self.assertTrue(result["ready"])
        self.assertTrue(result["both_signs"])
        self.assertTrue(result["intervention_and_abstention"])
        self.assertTrue(result["temporal_states"])
        self.assertTrue(result["year_diversity"])

    def test_partial_analysis_identifies_policy_abstention_context(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "decision_date": "2024-01-02",
                    "universe_name": "U1",
                    "candidate": "A",
                    "action_advantage_log": -0.02,
                },
                {
                    "decision_date": "2024-01-02",
                    "universe_name": "U1",
                    "candidate": "B",
                    "action_advantage_log": -0.01,
                },
                {
                    "decision_date": "2024-01-02",
                    "universe_name": "U2",
                    "candidate": "A",
                    "action_advantage_log": 0.03,
                },
                {
                    "decision_date": "2024-01-02",
                    "universe_name": "U2",
                    "candidate": "B",
                    "action_advantage_log": -0.01,
                },
            ]
        )
        result = analysis.partial_analysis(frame, 2)
        self.assertEqual(result["normal_policy_preferred_contexts"], 1)
        self.assertEqual(result["intervention_preferred_contexts"], 1)


if __name__ == "__main__":
    unittest.main()
