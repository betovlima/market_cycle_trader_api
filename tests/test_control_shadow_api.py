"""Swagger, auth, input and queue guards for fresh Alpaca Control shadow."""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from market_cycle_trader_api.api.routers.control_shadow import StartControlShadowRequest, router as control_shadow_router
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
        # API route metadata is reflected by the OpenAPI paths above.
        # The security dependency is attached in create_app.include_router,
        # rather than on the child router's route definitions.
        self.assertEqual(len(control_shadow_router.routes), 3)
        registration = inspect.getsource(create_app)
        self.assertIn(
            "admin_required = [Depends(require_admin_session)]", registration
        )
        self.assertIn(
            "application.include_router(control_shadow.router, dependencies=admin_required)",
            registration,
        )

    def test_body_does_not_accept_server_paths_or_unconfirmed_execution(self):
        with self.assertRaises(ValidationError):
            StartControlShadowRequest.model_validate({
                "confirm": "REFRESH_ALPACA_CONTROL_SHADOW_NO_ORDERS",
                "frozen_root": "C:/arbitrary/path",
            })
        with self.assertRaises(ValidationError):
            StartControlShadowRequest.model_validate({
                "confirm": "EXECUTE_REAL_ORDER",
            })

    def test_disabled_by_default_without_requiring_an_existing_dados_directory(self):
        with patch.dict(os.environ, {jobs.ENABLED_ENV: "false"}):
            with self.assertRaises(jobs.ControlShadowUnavailable):
                jobs.start_control_shadow_job(
                    MagicMock(), current_asset="CASH", holding_sessions=0
                )

    def test_queue_is_unique_and_has_no_order_or_promotion_fields(self):
        db = MagicMock()
        collection = db[jobs.COLLECTION]
        with (
            patch.dict(os.environ, {jobs.ENABLED_ENV: "true"}),
            patch.object(jobs.threading, "Thread") as thread,
        ):
            queued = jobs.start_control_shadow_job(
                db, current_asset="CASH", holding_sessions=0
            )
            thread.return_value.start.assert_called_once()
            self.assertEqual(queued["status"], "queued")
            self.assertEqual(queued["source_kind"], "fresh_alpaca_raw_sip_local_mct_snapshot")
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
        db = MagicMock()
        db[jobs.COLLECTION].insert_one.side_effect = DuplicateKeyError("duplicate")
        with patch.dict(os.environ, {jobs.ENABLED_ENV: "true"}):
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
