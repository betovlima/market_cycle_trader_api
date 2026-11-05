"""Manual daily Live Winner request never overwrites plans or submits orders."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import live_manual_daily_recalculation as subject

SESSION = "2026-11-06"
PRIOR = "2026-11-05"
NOW = datetime(2026, 11, 5, 21, 40, tzinfo=timezone.utc)
NEXT_OPEN = datetime(2026, 11, 6, 14, 30, tzinfo=timezone.utc)
PREFIX = "market_cycle_trader_api.services.live_manual_daily_recalculation."


def broker(*, open=False, now=NOW):
    obj = MagicMock()
    obj.get_account.return_value = SimpleNamespace(id="account-live")
    obj.get_clock.return_value = SimpleNamespace(
        is_open=open, timestamp=now, next_open=NEXT_OPEN,
    )
    return obj


class LiveDailyRecalculationTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.run = self.db[subject.LIVE_RUNS]
        self.run.find_one.return_value = None
        self.live = broker()

    def test_pending_daily_signal_is_eligible_only_after_daily_close_guard(self):
        with (
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "create_live_readonly_client", return_value=self.live),
        ):
            state = subject.daily_recalculation_status(self.db)
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["can_request"])
        self.assertEqual(state["execution_session"], SESSION)
        self.assertEqual(state["decision_date"], PRIOR)
        self.assertEqual(state["method"], "previous_completed_session_daily")
        self.assertFalse(state["order_submission_enabled"])
        self.run.update_one.assert_not_called()
        self.live.submit_order.assert_not_called()

    def test_market_open_never_eligible_for_an_intraday_retraining(self):
        live = broker(open=True, now=datetime(2026, 11, 5, 15, tzinfo=timezone.utc))
        with (
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "create_live_readonly_client", return_value=live),
        ):
            state = subject.daily_recalculation_status(self.db)
        self.assertEqual(state["status"], "market_open")
        self.assertFalse(state["can_request"])
        self.run.update_one.assert_not_called()
        live.submit_order.assert_not_called()

    def test_pause_blocks_live_daily_reanalysis(self):
        with patch(PREFIX + "live_control_status", return_value={"control_mode": "paused"}), patch(
            PREFIX + "create_live_readonly_client"
        ) as client:
            state = subject.request_daily_recalculation(self.db, actor="admin")
        self.assertEqual(state["status"], "inactive")
        client.assert_not_called()
        self.run.update_one.assert_not_called()

    def test_valid_daily_signal_is_never_retrained_or_replaced(self):
        self.run.find_one.return_value = {
            "status": "recalibrated_no_orders",
            "winner_signal_prepared": True, "post_close_signal_ready": True,
            "decision_date": PRIOR, "target_asset": "CAPR",
        }
        with (
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "create_live_readonly_client", return_value=self.live),
            patch(PREFIX + "plan_next_live_session") as plan,
        ):
            state = subject.request_daily_recalculation(self.db, actor="admin")
        self.assertEqual(state["status"], "already_prepared")
        self.assertFalse(state["can_request"])
        plan.assert_not_called()
        self.run.update_one.assert_not_called()
        self.live.submit_order.assert_not_called()

    def test_completed_or_executing_run_never_reopened(self):
        for old in ("completed", "executing", "review_required", "recalibrated_no_orders"):
            with self.subTest(old=old):
                self.run.find_one.return_value = {"status": old}
                with (
                    patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
                    patch(PREFIX + "create_live_readonly_client", return_value=self.live),
                    patch(PREFIX + "plan_next_live_session") as plan,
                ):
                    state = subject.request_daily_recalculation(self.db, actor="admin")
                self.assertEqual(state["status"], "blocked")
                plan.assert_not_called()
                self.run.update_one.assert_not_called()

    def test_request_is_idempotent_and_account_scoped_with_no_direct_orders(self):
        with (
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "create_live_readonly_client", return_value=self.live),
            patch(PREFIX + "plan_next_live_session",
                  return_value={"execution_session": SESSION,
                                "status": "waiting_for_winner_signal"}) as plan,
        ):
            result = subject.request_daily_recalculation(self.db, actor="admin@example.com")
        self.assertEqual(result["status"], "queued")
        self.assertFalse(result["order_submission_enabled"])
        self.assertEqual(result["execution_session"], SESSION)
        plan.assert_called_once_with(self.db)
        update_filter = self.run.update_one.call_args.args[0]
        self.assertEqual(update_filter["_id"], f"account-live:{SESSION}")
        self.assertEqual(update_filter["status"], "waiting_for_winner_signal")
        self.assertEqual(
            self.run.update_one.call_args.args[1]["$set"]["manual_daily_requested_by"],
            "admin@example.com",
        )
        self.live.submit_order.assert_not_called()
        self.live.cancel_order_by_id.assert_not_called()

    def test_stale_plan_after_request_does_not_write(self):
        with (
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "create_live_readonly_client", return_value=self.live),
            patch(PREFIX + "plan_next_live_session",
                  return_value={"execution_session": SESSION,
                                "status": "recalibrated_no_orders"}),
        ):
            result = subject.request_daily_recalculation(self.db, actor="admin")
        self.assertEqual(result["status"], "already_claimed")
        self.run.update_one.assert_not_called()

    def test_live_routes_demand_admin_session(self):
        from market_cycle_trader_api.api.routers import live_market
        from market_cycle_trader_api.auth.security import require_admin_session
        for suffix in (
            "/api/live-market/winner/daily-recalculation/status",
            "/api/live-market/winner/daily-recalculation/request",
        ):
            route = next(route for route in live_market.router.routes
                         if route.path == suffix)
            self.assertIn(require_admin_session,
                          {dep.call for dep in route.dependant.dependencies})
        route = next(route for route in live_market.router.routes
                     if route.path.endswith("/daily-recalculation/request"))
        self.assertIn("POST", route.methods)

    def test_missing_actor_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "administrator"):
            subject.request_daily_recalculation(self.db, actor="")
        self.run.update_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
