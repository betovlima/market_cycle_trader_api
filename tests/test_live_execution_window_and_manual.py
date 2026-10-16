"""Live execution deadline parity and operator-triggered guarded execution."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import HTTPException

from market_cycle_trader_api.api.routers.live_market import (
    LiveExecuteWinnerRequest, execute_live_current_plan, router,
)
from market_cycle_trader_api.auth.security import require_admin_session
from market_cycle_trader_api.services.live_execution_window import (
    live_execution_window_deadline, live_execution_window_seconds,
)


class LiveWindowTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(market_open_delay_seconds=60, market_execution_window_seconds=90)

    @patch.dict("os.environ", {"ALPACA_LIVE_EXECUTION_WINDOW_SECONDS": ""})
    def test_live_window_is_not_short_paper_window(self):
        self.assertEqual(live_execution_window_seconds(self.settings), 1800)
        start = pd.Timestamp(datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc))
        deadline, seconds = live_execution_window_deadline(start, self.settings)
        self.assertEqual(seconds, 1800)
        self.assertEqual(deadline, start + pd.Timedelta(minutes=30))

    @patch.dict("os.environ", {"ALPACA_LIVE_EXECUTION_WINDOW_SECONDS": "3600"})
    def test_configurable_live_window(self):
        self.assertEqual(live_execution_window_seconds(self.settings), 3600)

    @patch.dict("os.environ", {"ALPACA_LIVE_EXECUTION_WINDOW_SECONDS": "10"})
    def test_reject_too_short_window(self):
        with self.assertRaisesRegex(RuntimeError, "60..7200"):
            live_execution_window_seconds(self.settings)

    @patch.dict("os.environ", {"ALPACA_LIVE_EXECUTION_WINDOW_SECONDS": "60"})
    def test_window_must_exceed_open_delay(self):
        with self.assertRaisesRegex(RuntimeError, "must exceed"):
            live_execution_window_seconds(self.settings)


class ManualExecutionRouteTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.winner_hash = "a" * 64
        self.run = {
            "status": "recalibrated_no_orders",
            "execution_session": "2026-10-19",
            "target_asset": "AVGO",
            "winner_configuration_hash": self.winner_hash,
        }
        self.db["live_winner_session_runs"].find_one.return_value = self.run
        self.request = LiveExecuteWinnerRequest(
            execution_session="2026-10-19", expected_target_asset="AVGO",
            expected_winner_configuration_hash=self.winner_hash,
            confirm="EXECUTE_WINNER_LIVE_REAL_ORDER",
        )
        self.operator = SimpleNamespace(subject="trader-admin", email=None)

    def test_manual_execution_endpoint_requires_admin(self):
        route = next(r for r in router.routes if r.path == "/api/live-market/winner/execute-current-plan")
        self.assertTrue(any(d.call == require_admin_session for d in route.dependant.dependencies))

    @patch("market_cycle_trader_api.api.routers.live_market.database")
    @patch("market_cycle_trader_api.api.routers.live_market.execute_recalibrated_live_winner_session")
    def test_verified_current_plan_delegates_to_existing_executor(self, executor, db):
        db.return_value = self.db
        executor.return_value = {"status": "completed", "action": "buy", "target_asset": "AVGO"}
        response = execute_live_current_plan(self.request, self.operator)
        self.assertEqual(response["status"], "completed")
        executor.assert_called_once_with(self.db, "2026-10-19")

    @patch("market_cycle_trader_api.api.routers.live_market.database")
    @patch("market_cycle_trader_api.api.routers.live_market.execute_recalibrated_live_winner_session")
    def test_analysis_only_never_becomes_executable(self, executor, db):
        db.return_value = self.db
        self.run["status"] = "analysis_only_completed"
        with self.assertRaises(HTTPException) as ctx:
            execute_live_current_plan(self.request, self.operator)
        self.assertEqual(ctx.exception.status_code, 409)
        executor.assert_not_called()

    @patch("market_cycle_trader_api.api.routers.live_market.database")
    @patch("market_cycle_trader_api.api.routers.live_market.execute_recalibrated_live_winner_session")
    def test_stale_target_or_hash_is_rejected(self, executor, db):
        db.return_value = self.db
        self.run["target_asset"] = "ADEA"
        with self.assertRaises(HTTPException) as ctx:
            execute_live_current_plan(self.request, self.operator)
        self.assertEqual(ctx.exception.status_code, 409)
        executor.assert_not_called()

    @patch("market_cycle_trader_api.api.routers.live_market.database")
    @patch("market_cycle_trader_api.api.routers.live_market.execute_recalibrated_live_winner_session")
    def test_opt_in_disabled_is_reported_as_not_executed(self, executor, db):
        db.return_value = self.db
        executor.return_value = {"status": "order_submission_disabled"}
        with self.assertRaises(HTTPException) as ctx:
            execute_live_current_plan(self.request, self.operator)
        self.assertEqual(ctx.exception.status_code, 409)
