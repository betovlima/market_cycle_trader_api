"""Live execution follows full XNYS sessions and Alpaca order-state constraints."""
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

    def test_full_market_session_replaces_90_minute_limit(self):
        self.assertEqual(live_execution_window_seconds(self.settings), 23400)
        start = pd.Timestamp(datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc))
        deadline, seconds = live_execution_window_deadline(start, self.settings)
        self.assertEqual(seconds, 23400)
        self.assertEqual(deadline, pd.Timestamp("2026-10-19T20:00:00Z"))

    def test_open_delay_and_old_custom_settings_do_not_limit_live(self):
        self.settings.market_execution_window_seconds = 90
        self.settings.market_open_delay_seconds = 5400
        self.assertEqual(live_execution_window_seconds(self.settings), 23400)

    def test_nyse_early_close_and_dst_are_included(self):
        # Black Friday: the official XNYS calendar closes at 13:00 ET.
        open_at = pd.Timestamp("2026-11-27T14:30:00Z")
        deadline, duration = live_execution_window_deadline(open_at)
        self.assertEqual(duration, 12600)
        self.assertEqual(deadline, pd.Timestamp("2026-11-27T18:00:00Z"))

    def test_live_broker_closing_time_overrides_generic_exchange_schedule(self):
        # For unexpected or provider-known special trading sessions, trust
        # Alpaca Live instead of a server-local hardcoded closing hour.
        clock = SimpleNamespace(
            is_open=True,
            next_close=datetime(2026, 10, 20, 19, 30, tzinfo=timezone.utc),
        )
        deadline, duration = live_execution_window_deadline(
            pd.Timestamp("2026-10-20T13:30:00Z"),
            market_clock=clock,
        )
        self.assertEqual(deadline, pd.Timestamp("2026-10-20T19:30:00Z"))
        self.assertEqual(duration, 21600)

    def test_nyse_standard_time_is_dynamic(self):
        # Jan NYSE open: 09:30 EST / 14:30 UTC, closes at 21:00 UTC.
        deadline, seconds = live_execution_window_deadline(pd.Timestamp("2026-01-05T14:30:00Z"))
        self.assertEqual(deadline, pd.Timestamp("2026-01-05T21:00:00Z"))
        self.assertEqual(seconds, 23400)


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
