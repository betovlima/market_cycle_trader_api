"""Swagger, auth, input and queue guards for frozen Control shadow."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from fastapi.testclient import TestClient
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from market_cycle_trader_api.api.routers.control_shadow import StartControlShadowRequest
from market_cycle_trader_api.main import create_app
from market_cycle_trader_api.services import control_shadow_jobs as jobs


class ControlShadowApiTests(TestCase):
    def test_routes_are_documented_and_require_admin_session(self):
        app = create_app()
        documented = app.openapi()["paths"]
        path = "/api/admin/control-shadow/jobs"
        self.assertIn(path, documented)
        self.assertIn("post", documented[path])
        self.assertIn("/api/admin/control-shadow/jobs/{job_id}", documented)
        self.assertIn("/api/admin/control-shadow/jobs/{job_id}/logs", documented)
        client = TestClient(app)  # Do not enter lifespan; no real MongoDB.
        response = client.post(path, json={
            "confirm": "RUN_FROZEN_SHADOW_NO_ORDERS",
            "current_asset": "CASH",
            "holding_sessions": 0,
        })
        self.assertEqual(response.status_code, 401)
        self.assertEqual(client.get(path + "/fake-job").status_code, 401)

    def test_body_does_not_accept_server_paths_or_unconfirmed_execution(self):
        with self.assertRaises(ValidationError):
            StartControlShadowRequest.model_validate({
                "confirm": "RUN_FROZEN_SHADOW_NO_ORDERS",
                "frozen_root": "C:/arbitrary/path",
            })
        with self.assertRaises(ValidationError):
            StartControlShadowRequest.model_validate({
                "confirm": "EXECUTE_REAL_ORDER",
            })

    def test_disabled_by_default_even_if_directory_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "manifest.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {
                jobs.ENABLED_ENV: "false",
                jobs.ROOT_ENV: folder,
            }):
                with self.assertRaises(jobs.ControlShadowUnavailable):
                    jobs.start_control_shadow_job(
                        MagicMock(), current_asset="CASH", holding_sessions=0
                    )

    def test_queue_is_unique_and_has_no_order_or_promotion_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "manifest.json").write_text("{}", encoding="utf-8")
            db = MagicMock()
            collection = db[jobs.COLLECTION]
            with (
                patch.dict(os.environ, {
                    jobs.ENABLED_ENV: "true",
                    jobs.ROOT_ENV: folder,
                }),
                patch.object(jobs.threading, "Thread") as thread,
            ):
                queued = jobs.start_control_shadow_job(
                    db, current_asset="CASH", holding_sessions=0
                )
            thread.return_value.start.assert_called_once()
            self.assertEqual(queued["status"], "queued")
            self.assertEqual(queued["source_kind"], "verified_frozen_tcc_snapshot")
            self.assertFalse(queued["order_eligible"])
            self.assertEqual(queued["order_submission"], "never")
            collection.create_index.assert_called_once_with(
                "active_key", unique=True, sparse=True
            )
            inserted = collection.insert_one.call_args.args[0]
            self.assertEqual(inserted["active_key"], jobs.ACTIVE_KEY)
            self.assertNotIn("account_id", inserted)
            self.assertNotIn("winner_strategy_id", inserted)
            jobs._ACTIVE_THREADS.pop(queued["job_id"], None)

    def test_second_job_fails_on_unique_active_key(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "manifest.json").write_text("{}", encoding="utf-8")
            db = MagicMock()
            db[jobs.COLLECTION].insert_one.side_effect = DuplicateKeyError("duplicate")
            with patch.dict(os.environ, {
                jobs.ENABLED_ENV: "true",
                jobs.ROOT_ENV: folder,
            }):
                with self.assertRaises(jobs.ControlShadowConflict):
                    jobs.start_control_shadow_job(
                        db, current_asset="CASH", holding_sessions=0
                    )

    def test_missing_job_is_not_found(self):
        db = MagicMock()
        db[jobs.COLLECTION].find_one.return_value = None
        with self.assertRaises(jobs.ControlShadowNotFound):
            jobs.get_control_shadow_job(db, "unknown")


if __name__ == "__main__":
    import unittest
    unittest.main()
