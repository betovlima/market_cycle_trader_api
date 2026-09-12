from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1112 as research  # noqa: E402


class ContextualMarginalSignatureV1112Tests(unittest.TestCase):
    def test_progress_log_tracks_current_replay_label(self) -> None:
        original_log = research._ORIGINAL_LOG
        messages: list[str] = []
        research._ORIGINAL_LOG = messages.append
        try:
            research._progress_log("  Original25/APD: challenger trace")
            self.assertEqual(research._CURRENT_REPLAY_LABEL, "Original25/APD: challenger trace")
            self.assertEqual(messages[-1], "  Original25/APD: challenger trace")
        finally:
            research._ORIGINAL_LOG = original_log

    def test_run_with_heartbeat_emits_done_summary(self) -> None:
        original_runner = research._ORIGINAL_RUN_WITH_CAPTURE
        original_log = research._ORIGINAL_LOG
        messages: list[str] = []

        def fake_runner(frames, request):
            del frames, request
            time.sleep(0.01)
            return {"ending_capital": 123.45}, pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")]), [object()]

        research._ORIGINAL_RUN_WITH_CAPTURE = fake_runner
        research._ORIGINAL_LOG = messages.append
        research._CURRENT_REPLAY_LABEL = "Original25/APD: challenger trace"
        try:
            metrics, sessions, captured = research._run_with_heartbeat({}, object())
        finally:
            research._ORIGINAL_RUN_WITH_CAPTURE = original_runner
            research._ORIGINAL_LOG = original_log

        self.assertEqual(metrics["ending_capital"], 123.45)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(captured), 1)
        self.assertTrue(any("[done]" in message for message in messages))
        self.assertTrue(any("ending_capital=123.450000" in message for message in messages))

    def test_scientific_protocol_is_unchanged(self) -> None:
        self.assertEqual(research.EXPERIMENT_NAME, research.base.EXPERIMENT_NAME)
        self.assertIs(research._ORIGINAL_RUN_WITH_CAPTURE, research.base._run_with_capture)
        self.assertEqual(research.HEARTBEAT_SECONDS, 30.0)


if __name__ == "__main__":
    unittest.main()
