"""Live missed-window recovery: explicitly requested, analysis-only, and account bound."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_expired_recovery import queue_expired_live_analysis


class ExpiredLiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        collections = {}
        self.db.__getitem__.side_effect = lambda name: collections.setdefault(name, MagicMock())
        self.db["live_winner_session_runs"].find_one.return_value = {
            "_id": "live-acct:2026-10-16",
            "live_account_id": "live-acct",
            "status": "review_required",
            "phase": "execution_window_expired",
            "winner_signal_prepared": True,
        }
        self.db["live_winner_broker_orders"].find_one.return_value = None
        self.db["live_winner_session_runs"].update_one.return_value.matched_count = 1
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(id="live-acct")
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 16, 15, 5, tzinfo=timezone.utc)
        )
        self.broker.get_orders.return_value = []

    def _call(self, *, session="2026-10-16", mode="active"):
        with patch("market_cycle_trader_api.services.live_expired_recovery.live_control_status", return_value={"control_mode": mode}), \
             patch("market_cycle_trader_api.services.live_expired_recovery.create_live_readonly_client", return_value=self.broker):
            return queue_expired_live_analysis(
                self.db, execution_session=session, actor_email="operator@example.com"
            )

    def test_expired_session_is_queued_for_audit_without_orders(self):
        result = self._call()
        self.assertEqual(result["status"], "analysis_only_requested")
        self.assertFalse(result["order_submission_enabled"])
        updated = self.db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]
        self.assertEqual(updated["status"], "analysis_only_requested")
        self.assertFalse(updated["order_submission_enabled"])
        self.broker.submit_order.assert_not_called()

    def test_live_mode_does_not_make_expired_analysis_executable(self):
        self.assertEqual(self._call()["status"], "analysis_only_requested")
        self.broker.submit_order.assert_not_called()

    def test_no_cross_session_replay(self):
        with self.assertRaisesRegex(RuntimeError, "current regular"):
            self._call(session="2026-10-15")
        self.broker.submit_order.assert_not_called()

    def test_only_expired_sessions_can_be_recovered(self):
        self.db["live_winner_session_runs"].find_one.return_value["phase"] = "broker_reconciliation_required"
        with self.assertRaisesRegex(RuntimeError, "No eligible expired"):
            self._call()

    def test_existing_order_intent_blocks_recovery(self):
        self.db["live_winner_broker_orders"].find_one.return_value = {"status": "submitting"}
        with self.assertRaisesRegex(RuntimeError, "order intent"):
            self._call()

    def test_closed_market_recovery_is_blocked(self):
        self.broker.get_clock.return_value.is_open = False
        with self.assertRaisesRegex(RuntimeError, "open regular"):
            self._call()

    def test_paused_mode_recovery_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "must be active"):
            self._call(mode="paused")


class LiveRecoveryRouteIdentityTests(unittest.TestCase):
    def test_password_admin_without_email_uses_authenticated_subject(self):
        from market_cycle_trader_api.api.routers.live_market import (
            LiveExpiredAnalysisRequest,
            recover_expired_live_analysis,
        )
        identity = SimpleNamespace(email=None, subject="trader-admin")
        payload = LiveExpiredAnalysisRequest(
            execution_session="2026-10-16",
            confirm="LIVE_ANALYSIS_ONLY_NO_ORDERS",
        )
        with patch("market_cycle_trader_api.api.routers.live_market.database") as db, \
             patch("market_cycle_trader_api.api.routers.live_market.queue_expired_live_analysis") as recover:
            recover.return_value = {"status": "analysis_only_requested"}
            response = recover_expired_live_analysis(payload, identity)
            self.assertEqual(response["status"], "analysis_only_requested")
            recover.assert_called_once_with(
                db.return_value,
                execution_session="2026-10-16",
                actor_email="trader-admin",
            )

    def test_email_is_used_if_available(self):
        from market_cycle_trader_api.api.routers.live_market import (
            LiveExpiredAnalysisRequest,
            recover_expired_live_analysis,
        )
        identity = SimpleNamespace(email="operator@example.com", subject="google:1234")
        payload = LiveExpiredAnalysisRequest(
            execution_session="2026-10-16",
            confirm="LIVE_ANALYSIS_ONLY_NO_ORDERS",
        )
        with patch("market_cycle_trader_api.api.routers.live_market.database"), \
             patch("market_cycle_trader_api.api.routers.live_market.queue_expired_live_analysis") as recover:
            recover.return_value = {"status": "analysis_only_requested"}
            recover_expired_live_analysis(payload, identity)
            self.assertEqual(recover.call_args.kwargs["actor_email"], "operator@example.com")
