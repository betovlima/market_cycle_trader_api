"""Regression tests for bounded, account-scoped Live Winner diagnostics."""
import io
import json
import unittest
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.auth.security import require_admin_session
from market_cycle_trader_api.api.routers.live_market import router
from market_cycle_trader_api.services.live_winner_events import (
    append_live_event, list_live_events, export_live_events,
)


class LiveWinnerEventsTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        collections = {}
        self.db.__getitem__.side_effect = lambda name: collections.setdefault(name, MagicMock())
        self.account = SimpleNamespace(id="acct-one")
        self.cursor = MagicMock()
        self.db["live_winner_process_events"].find.return_value = self.cursor
        self.cursor.sort.return_value = self.cursor
        self.cursor.limit.return_value = [{
            "recorded_at": datetime(2026, 10, 16, 17, 10, tzinfo=timezone.utc),
            "execution_session": "2026-10-16",
            "stage": "model_training_started", "level": "info",
            "message": "LightGBM training", "details": {"assets": 76},
        }]

    def test_append_sanitizes_event_details(self):
        append_live_event(
            self.db, account_id="acct-one", session="2026-10-16",
            stage="training", message="Training started",
            details={"assets": 76, "access_token": "never-persist", "password": "secret"},
        )
        doc = self.db["live_winner_process_events"].insert_one.call_args.args[0]
        self.assertEqual(doc["details"], {"assets": 76})
        self.assertEqual(doc["account_id"], "acct-one")
        self.assertNotIn("secret", str(doc))

    def test_database_event_failure_never_breaks_trading(self):
        self.db["live_winner_process_events"].insert_one.side_effect = RuntimeError("mongo offline")
        append_live_event(self.db, account_id="acct-one", session="2026-10-16",
                          stage="training", message="Training started")

    def test_query_is_account_bound_and_bounded(self):
        broker = MagicMock()
        broker.get_account.return_value = self.account
        with patch(
            "market_cycle_trader_api.services.live_winner_events.create_live_readonly_client",
            return_value=broker,
        ):
            payload = list_live_events(self.db, session="2026-10-16", level="error", limit=999)
        query = self.db["live_winner_process_events"].find.call_args.args[0]
        self.assertEqual(query, {
            "account_id": "acct-one", "execution_session": "2026-10-16",
            "level": "error",
        })
        self.assertEqual(self.cursor.limit.call_args.args[0], 200)
        self.assertEqual(payload["events"][0]["stage"], "model_training_started")
        self.assertEqual(payload["events"][0]["recorded_at"], "2026-10-16T17:10:00+00:00")
        self.assertNotIn("account_id", payload["events"][0])

    def test_default_session_uses_only_current_live_account(self):
        broker = MagicMock()
        broker.get_account.return_value = self.account
        self.db["live_winner_session_runs"].find_one.return_value = {
            "execution_session": "2026-10-16",
        }
        with patch(
            "market_cycle_trader_api.services.live_winner_events.create_live_readonly_client",
            return_value=broker,
        ):
            result = list_live_events(self.db)
        self.assertEqual(result["execution_session"], "2026-10-16")
        self.assertEqual(self.db["live_winner_session_runs"].find_one.call_args.args[0],
                         {"live_account_id": "acct-one"})

    def test_zip_contains_only_sanitized_session_logs(self):
        with patch(
            "market_cycle_trader_api.services.live_winner_events.list_live_events",
            return_value={"execution_session": "2026-10-16", "events": [{"stage": "training"}]},
        ):
            packed = export_live_events(self.db, session="2026-10-16")
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            self.assertEqual(json.loads(archive.read("live_winner_events.json"))["events"][0]["stage"],
                             "training")
            self.assertIn("README.txt", archive.namelist())

    def test_diag_routes_require_admin_session(self):
        for path in ("/api/live-market/winner/events",
                     "/api/live-market/winner/events/export.zip"):
            route = next(route for route in router.routes if route.path == path)
            self.assertTrue(any(dependency.call == require_admin_session
                                for dependency in route.dependant.dependencies))
