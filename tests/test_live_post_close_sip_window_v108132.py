"""v10.8.132: Live post-close SIP readiness is a wait, not a failed job.

All scheduler effects are mocked. No historical download, training or broker
execution is performed in these tests.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import live_observation_scheduler as scheduler


PREFIX = "market_cycle_trader_api.services.live_observation_scheduler."
SESSION = "2026-10-30"


class PostCloseDailyWindowTests(unittest.TestCase):
    def timing(self, timestamp, session=SESSION):
        return scheduler._post_close_daily_preparation_window(
            session=session, broker_timestamp=timestamp,
        )

    def test_fails_closed_before_close_and_invalid_clocks(self):
        for value in ("2026-10-29T19:59:59Z", None, "invalid", "2026-10-29T20:21:00"):
            self.assertEqual(self.timing(value)[0], "outside_window")
        self.assertEqual(self.timing("2026-10-29T20:21:00Z", session="2026-10-31")[0], "outside_window")

    def test_exact_close_and_sip_readiness_boundaries(self):
        self.assertEqual(self.timing("2026-10-29T20:00:00Z")[0], "waiting_for_daily_bar")
        state, cutoff, retry_after = self.timing("2026-10-29T20:19:59Z")
        self.assertEqual(state, "waiting_for_daily_bar")
        self.assertEqual(cutoff, "2026-10-29")
        self.assertEqual(retry_after, "2026-10-29T20:20:00+00:00")
        self.assertEqual(self.timing("2026-10-29T20:20:00Z")[0], "ready")

    def test_preparation_remains_possible_overnight_but_not_at_open(self):
        self.assertEqual(self.timing("2026-10-30T13:29:59Z")[0], "ready")
        self.assertEqual(self.timing("2026-10-30T13:30:00Z")[0], "outside_window")
        self.assertEqual(self.timing("2026-10-30T14:00:00Z")[0], "outside_window")

    def test_early_close_and_weekend_use_official_exchange_calendar(self):
        session = "2026-11-30"
        self.assertEqual(self.timing("2026-11-27T18:19:59Z", session=session)[0], "waiting_for_daily_bar")
        self.assertEqual(self.timing("2026-11-27T18:20:00Z", session=session)[0], "ready")
        self.assertEqual(self.timing("2026-11-29T15:00:00Z", session=session)[0], "ready")
        self.assertEqual(self.timing("2026-11-30T14:30:00Z", session=session)[0], "outside_window")


class WaitingIsNotAnErrorTests(unittest.TestCase):
    def test_repeated_before_safe_observations_emit_one_wait_without_attempts(self):
        worker = scheduler.LiveObservationScheduler()
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "live_account_id": "account", "status": "waiting_for_winner_signal",
        }
        result = {
            "status": "observed",
            "recorded_at": "2026-10-29T20:01:00Z",
            "market_open": False,
        }
        armed = {"status": "waiting_for_winner_signal", "execution_session": SESSION}
        with (
            patch.object(worker._stop, "is_set", side_effect=[False, False, True]),
            patch.object(worker._stop, "wait"),
            patch(PREFIX + "database", return_value=db),
            patch(PREFIX + "observe_live_cycle", return_value=result),
            patch(PREFIX + "plan_next_live_session", return_value=armed),
            patch(PREFIX + "append_live_event") as event,
            patch(PREFIX + "execute_monitored_live_stage") as processing,
            patch(PREFIX + "create_live_readonly_client") as brokerage,
        ):
            worker._run()
        processing.assert_not_called()
        brokerage.assert_not_called()
        event.assert_called_once()
        self.assertEqual(event.call_args.kwargs["stage"], "signal_waiting_for_daily_bar")
        self.assertEqual(event.call_args.kwargs["details"]["cutoff"], "2026-10-29")
        self.assertEqual(event.call_args.kwargs["details"]["retry_after"], "2026-10-29T20:20:00+00:00")

    def test_wait_then_prepare_uses_only_post_close_calculation(self):
        worker = scheduler.LiveObservationScheduler()
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "live_account_id": "account", "status": "waiting_for_winner_signal",
        }
        before = {"status": "observed", "recorded_at": "2026-10-29T20:19:59Z", "market_open": False}
        after = {"status": "observed", "recorded_at": "2026-10-29T20:20:30Z", "market_open": False}
        armed = {"status": "waiting_for_winner_signal", "execution_session": SESSION}
        with (
            patch.object(worker._stop, "is_set", side_effect=[False, False, True]),
            patch.object(worker._stop, "wait"),
            patch(PREFIX + "database", return_value=db),
            patch(PREFIX + "observe_live_cycle", side_effect=[before, after]),
            patch(PREFIX + "plan_next_live_session", return_value=armed),
            patch(PREFIX + "append_live_event") as event,
            patch(PREFIX + "execute_monitored_live_stage") as processing,
            patch(PREFIX + "create_live_readonly_client") as brokerage,
        ):
            worker._run()
        brokerage.assert_not_called()
        event.assert_called_once()
        processing.assert_called_once()
        self.assertEqual(processing.call_args.kwargs["stage"], "post_close_calculation")
        self.assertFalse(processing.call_args.kwargs.get("brokerage", False))


if __name__ == "__main__":
    unittest.main()
