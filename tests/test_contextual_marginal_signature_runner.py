from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
import unittest

import exchange_calendars as xcals
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature as runner  # noqa: E402
import research_contextual_signature_analysis as analysis  # noqa: E402


class ContextualMarginalSignatureRunnerTests(unittest.TestCase):
    def test_stable_entrypoint_and_export_names_have_no_version_suffix(self) -> None:
        self.assertEqual(runner.EXPORT_FOLDER_NAME, "contextual_marginal_signature")
        self.assertEqual(runner.EXPORT_ZIP_NAME, "contextual_marginal_signature.zip")
        self.assertNotIn("v1", runner.EXPORT_FOLDER_NAME)
        self.assertNotIn("v1", runner.EXPORT_ZIP_NAME)
        self.assertEqual(runner.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.17.1")
        self.assertEqual(runner.HORIZON_SESSIONS, 40)
        self.assertEqual(len(runner.DECISION_DATES), 23)
        self.assertEqual(len(runner.CANDIDATES), 7)
        self.assertEqual(len(runner.UNIVERSES), 2)
        self.assertGreaterEqual(len({pd.Timestamp(x).year for x in runner.DECISION_DATES}), 7)

    def test_temporal_states_are_real_xnys_sessions_and_after_locked_oos_warmup(self) -> None:
        calendar = xcals.get_calendar("XNYS")
        self.assertGreaterEqual(pd.Timestamp(runner.DECISION_DATES[0]), pd.Timestamp("2020-08-03"))
        for value in runner.DECISION_DATES:
            resolved = calendar.date_to_session(pd.Timestamp(value), direction="none")
            self.assertEqual(pd.Timestamp(resolved).date(), pd.Timestamp(value).date())

    def test_temporal_states_are_spread_beyond_the_40_session_target_horizon(self) -> None:
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            pd.Timestamp(runner.DECISION_DATES[0]),
            pd.Timestamp(runner.DECISION_DATES[-1]),
        )
        positions = [int(sessions.searchsorted(pd.Timestamp(value))) for value in runner.DECISION_DATES]
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

    def test_readiness_requires_temporal_diversity_and_intervention_abstention(self) -> None:
        rows = []
        # Exercise the complete planned campaign because readiness intentionally
        # requires the full seven-year temporal span, not merely the first 20 states.
        for date_index, date in enumerate(runner.DECISION_DATES):
            for universe in ("U1", "U2"):
                for candidate_index, candidate in enumerate(runner.CANDIDATES):
                    if date_index % 5 == 0:
                        value = -0.01 - candidate_index * 0.001
                    else:
                        value = 0.02 - candidate_index * 0.006
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
                {"decision_date": "2024-01-02", "universe_name": "U1", "candidate": "A", "action_advantage_log": -0.02},
                {"decision_date": "2024-01-02", "universe_name": "U1", "candidate": "B", "action_advantage_log": -0.01},
                {"decision_date": "2024-01-02", "universe_name": "U2", "candidate": "A", "action_advantage_log": 0.03},
                {"decision_date": "2024-01-02", "universe_name": "U2", "candidate": "B", "action_advantage_log": -0.01},
            ]
        )
        result = analysis.partial_analysis(frame, 2)
        self.assertEqual(result["normal_policy_preferred_contexts"], 1)
        self.assertEqual(result["intervention_preferred_contexts"], 1)


if __name__ == "__main__":
    unittest.main()
