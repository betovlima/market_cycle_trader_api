"""Live execution uses a dedicated route and a bounded current-session deadline."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import HTTPException

from market_cycle_trader_api.api.routers.live_execution import (
    execute_current_winner_live, router,
)
from market_cycle_trader_api.auth.security import require_admin_session
from market_cycle_trader_api.services.live_execution_window import (
    live_execution_window_deadline, live_execution_window_seconds,
)


class LiveWindowTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            market_open_delay_seconds=60, market_execution_window_seconds=90,
        )

    def test_live_window_does_not_expire_before_model_calculates(self):
        self.assertEqual(live_execution_window_seconds(self.settings), 1800)
        start = pd.Timestamp(datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc))
        deadline, seconds = live_execution_window_deadline(start, self.settings)
        self.assertEqual(seconds, 1800)
        self.assertEqual(deadline, start + pd.Timedelta(minutes=30))

    def test_window_respects_existing_larger_operational_setting(self):
        self.settings.market_execution_window_seconds = 3600
        self.assertEqual(live_execution_window_seconds(self.settings), 3600)

    def test_window_must_exceed_open_delay(self):
        self.settings.market_open_delay_seconds = 1800
        with self.assertRaisesRegex(RuntimeError, "must exceed"):
            live_execution_window_seconds(self.settings)


class LiveExecutionRouteTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db["live_winner_session_runs"].find_one.return_value = {
            "status": "recalibrated_no_orders",
        }
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(id="acct")
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 19, 13, 40, tzinfo=timezone.utc)
        )
        self.actor = SimpleNamespace(subject="admin", email=None)

    def test_api_requires_admin(self):
        route = next(r for r in router.routes if r.path == "/api/live-execution/winner/execute")
        self.assertTrue(any(d.call == require_admin_session for d in route.dependant.dependencies))

    @patch("market_cycle_trader_api.api.routers.live_execution.database")
    @patch("market_cycle_trader_api.api.routers.live_execution.live_control_status")
    @patch("market_cycle_trader_api.api.routers.live_execution.create_live_readonly_client")
    @patch("market_cycle_trader_api.api.routers.live_execution.execute_prepared_live_winner_session")
    def test_current_winner_plan_uses_only_live_executor(self, executor, client, mode, db):
        db.return_value = self.db
        mode.return_value = {"control_mode": "active"}
        client.return_value = self.broker
        executor.return_value = {"status": "completed", "target_asset": "AVGO", "action": "buy"}
        result = execute_current_winner_live(self.actor)
        self.assertEqual(result["status"], "completed")
        executor.assert_called_once_with(self.db, "2026-10-19")
        self.assertEqual(self.db["live_winner_session_runs"].find_one.call_args.args[0],
                         {"_id": "acct:2026-10-19"})

    @patch("market_cycle_trader_api.api.routers.live_execution.database")
    @patch("market_cycle_trader_api.api.routers.live_execution.live_control_status")
    @patch("market_cycle_trader_api.api.routers.live_execution.create_live_readonly_client")
    @patch("market_cycle_trader_api.api.routers.live_execution.execute_prepared_live_winner_session")
    def test_late_analysis_is_not_an_executable_plan(self, executor, client, mode, db):
        db.return_value = self.db
        mode.return_value = {"control_mode": "active"}
        client.return_value = self.broker
        self.db["live_winner_session_runs"].find_one.return_value = {"status": "analysis_only_completed"}
        with self.assertRaises(HTTPException) as exc:
            execute_current_winner_live(self.actor)
        self.assertEqual(exc.exception.status_code, 409)
        executor.assert_not_called()

    @patch("market_cycle_trader_api.api.routers.live_execution.database")
    @patch("market_cycle_trader_api.api.routers.live_execution.live_control_status")
    @patch("market_cycle_trader_api.api.routers.live_execution.create_live_readonly_client")
    def test_paused_mode_does_not_open_broker(self, client, mode, db):
        db.return_value = self.db
        mode.return_value = {"control_mode": "paused"}
        with self.assertRaises(HTTPException) as exc:
            execute_current_winner_live(self.actor)
        self.assertEqual(exc.exception.status_code, 409)
        client.assert_not_called()

    @patch("market_cycle_trader_api.api.routers.live_execution.database")
    @patch("market_cycle_trader_api.api.routers.live_execution.live_control_status")
    @patch("market_cycle_trader_api.api.routers.live_execution.create_live_readonly_client")
    def test_closed_market_does_not_execute(self, client, mode, db):
        db.return_value = self.db
        mode.return_value = {"control_mode": "active"}
        client.return_value = self.broker
        self.broker.get_clock.return_value.is_open = False
        with self.assertRaises(HTTPException) as exc:
            execute_current_winner_live(self.actor)
        self.assertEqual(exc.exception.status_code, 409)
