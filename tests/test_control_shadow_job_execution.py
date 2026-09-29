"""Job orchestration tests: fresh download, persisted logs, no trading path."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from market_cycle_trader_api.services import control_shadow_jobs as jobs


class FreshControlShadowJobTests(TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.job_id = "control-shadow-test123"
        self.session = "2026-09-25"
        self.frames = {"AAPL": MagicMock()}
        self.snapshot = SimpleNamespace(
            directory=Path("dados/control_shadow/snapshots") / self.job_id,
            frames=self.frames,
            manifest={
                "snapshot_sha256": "test-sha",
                "structural_exclusions": [],
            },
        )

    def test_worker_uses_fresh_mct_download_and_writes_shadow_decision_only(self):
        decision = {
            "target_asset": "AAPL",
            "effective_switch_margin": 0.0025,
            "input_audit": {},
            "order_submission": "never",
        }
        with (
            patch.object(jobs, "download_current_control_snapshot",
                         return_value=self.snapshot) as download,
            patch.object(jobs, "build_control_shadow_decision",
                         return_value=decision) as preview,
        ):
            jobs._run_control_shadow_job(
                self.db, self.job_id,
                completed_session=self.session,
                current_asset="CASH",
                holding_sessions=0,
            )
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.kwargs["job_id"], self.job_id)
        self.assertEqual(download.call_args.kwargs["completed_session"], self.session)
        self.assertEqual(preview.call_count, 1)
        self.assertEqual(preview.call_args.kwargs["completed_session"], self.session)
        self.assertEqual(decision["input_audit"]["snapshot_sha256"], "test-sha")
        self.assertEqual(decision["input_audit"]["source_kind"],
                         "fresh_alpaca_raw_sip_local_mct_snapshot")
        self.assertFalse(decision["order_eligible"])
        self.assertEqual(decision["order_submission"], "never")
        changes = [
            call.args[1] for call in
            self.db[jobs.COLLECTION].update_one.call_args_list
        ]
        completed = [ch for ch in changes if ch.get("$set", {}).get("status") == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertIs(completed[0]["$set"]["result"], decision)
        self.assertEqual(completed[0]["$unset"], {"active_key": ""})
        self.assertTrue(any("$push" in ch for ch in changes))

    def test_download_failure_blocks_training_and_releases_active_key(self):
        with (
            patch.object(jobs, "download_current_control_snapshot",
                         side_effect=RuntimeError("Alpaca 403")),
            patch.object(jobs, "build_control_shadow_decision") as preview,
            patch.object(jobs.LOGGER, "exception"),
        ):
            jobs._run_control_shadow_job(
                self.db, self.job_id,
                completed_session=self.session,
                current_asset="CASH",
                holding_sessions=0,
            )
        preview.assert_not_called()
        changes = [
            call.args[1] for call in
            self.db[jobs.COLLECTION].update_one.call_args_list
        ]
        failures = [ch for ch in changes if ch.get("$set", {}).get("status") == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("Alpaca 403", failures[0]["$set"]["error"])
        self.assertEqual(failures[0]["$unset"], {"active_key": ""})


if __name__ == "__main__":
    import unittest
    unittest.main()
