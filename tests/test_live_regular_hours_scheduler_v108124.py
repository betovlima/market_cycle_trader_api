"""Live's intraday calculations and broker attempts are regular-XNYS-session only.

Post-close non-executable next-session preparation is intentionally unchanged.
All Alpaca interactions below are mocked; no broker orders are submitted.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import live_observation_scheduler as scheduler

PREFIX = "market_cycle_trader_api.services.live_observation_scheduler."
SESSION = "2026-10-20"


class RegularXnysGateTests(unittest.TestCase):
    def is_allowed(self, timestamp, session=SESSION, open=True, next_close=None):
        return scheduler._regular_xnys_session_open(
            session=session,
            broker_timestamp=timestamp,
            broker_is_open=open,
            broker_next_close=next_close,
        )

    def test_regular_market_open_edges_are_exact(self):
        self.assertFalse(self.is_allowed("2026-10-20T13:29:59Z"))
        self.assertTrue(self.is_allowed("2026-10-20T13:30:00Z"))
        self.assertTrue(self.is_allowed("2026-10-20T19:59:59Z"))
        self.assertFalse(self.is_allowed("2026-10-20T20:00:00Z"))
        self.assertFalse(self.is_allowed("2026-10-20T20:00:01Z"))

    def test_alpaca_clock_can_close_market_early(self):
        self.assertFalse(self.is_allowed("2026-10-20T19:30:00Z", open=False))
        self.assertFalse(self.is_allowed(
            "2026-10-20T19:30:00Z", next_close="2026-10-20T19:00:00Z",
        ))
        self.assertTrue(self.is_allowed("2026-10-20T18:59:59Z", next_close="2026-10-20T19:00:00Z"))
        # Broker cannot extend XNYS close.
        self.assertFalse(self.is_allowed("2026-10-20T20:00:01Z", next_close="2026-10-20T21:00:00Z"))

    def test_exchange_early_close_2026_11_27(self):
        self.assertFalse(self.is_allowed("2026-11-27T14:29:59Z", session="2026-11-27"))
        self.assertTrue(self.is_allowed("2026-11-27T14:30:00Z", session="2026-11-27"))
        self.assertTrue(self.is_allowed("2026-11-27T17:59:59Z", session="2026-11-27"))
        self.assertFalse(self.is_allowed("2026-11-27T18:00:00Z", session="2026-11-27"))

    def test_weekends_holidays_stale_and_naive_clocks_block(self):
        self.assertFalse(self.is_allowed("2026-10-24T15:00:00Z", session="2026-10-24"))
        self.assertFalse(self.is_allowed("2026-11-26T15:00:00Z", session="2026-11-26"))
        self.assertFalse(self.is_allowed("2026-10-21T15:00:00Z", session=SESSION))
        self.assertFalse(self.is_allowed(datetime(2026, 10, 20, 19, 0)))
        self.assertFalse(self.is_allowed(None))
        self.assertFalse(self.is_allowed("bad-time"))
        self.assertFalse(self.is_allowed("2026-10-20T15:00:00Z", open=False))

    def test_open_paraguay_local_1613_is_during_ny_open(self):
        # 16:13:43 Paraguay (-03) is 15:13:43 New York (-04) in October.
        self.assertTrue(self.is_allowed("2026-10-20T19:13:43Z"))


class SchedulerStagesTests(unittest.TestCase):
    def _once(self, *, recorded_at, market_open, plan_status,
              calculation_status="recalibrated_no_orders", fresh_clock=None):
        worker = scheduler.LiveObservationScheduler()
        db = MagicMock()
        if fresh_clock is None:
            fresh_clock = SimpleNamespace(
                is_open=True,
                timestamp=datetime(2026, 10, 20, 19, 30, tzinfo=timezone.utc),
                next_close=datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc),
            )
        broker = MagicMock()
        broker.get_clock.return_value = fresh_clock
        result = {"status": "observed", "recorded_at": recorded_at, "market_open": market_open}
        armed = {"status": plan_status, "execution_session": SESSION}
        calls = []

        def run_stage(_db, *, session, stage, operation, brokerage=False):
            calls.append((stage, brokerage))
            return {"status": calculation_status}

        with (
            patch.object(worker._stop, "is_set", side_effect=[False, True]),
            patch.object(worker._stop, "wait"),
            patch(PREFIX + "database", return_value=db),
            patch(PREFIX + "observe_live_cycle", return_value=result),
            patch(PREFIX + "plan_next_live_session", return_value=armed),
            patch(PREFIX + "execute_monitored_live_stage", side_effect=run_stage),
            patch(PREFIX + "create_live_readonly_client", return_value=broker) as client,
        ):
            worker._run()
        return calls, client, broker

    def test_closed_market_does_not_attempt_intraday_or_broker(self):
        calls, client, _ = self._once(
            recorded_at="2026-10-20T20:01:00Z",
            market_open=False, plan_status="waiting_for_open_recalibration",
        )
        self.assertEqual(calls, [])
        client.assert_not_called()

    def test_provider_open_flag_not_enough_outside_xnys_window(self):
        calls, client, _ = self._once(
            recorded_at="2026-10-20T20:01:00Z",
            market_open=True, plan_status="recalibrated_no_orders",
        )
        self.assertEqual(calls, [])
        client.assert_not_called()

    def test_premarket_flag_cannot_trigger_intraday(self):
        calls, _, _ = self._once(
            recorded_at="2026-10-20T13:20:00Z",
            market_open=True, plan_status="waiting_for_open_recalibration",
        )
        self.assertEqual(calls, [])

    def test_post_close_next_session_preparation_remains_possible(self):
        calls, client, _ = self._once(
            recorded_at="2026-10-20T20:01:00Z",
            market_open=False, plan_status="waiting_for_winner_signal",
        )
        self.assertEqual(calls, [("post_close_calculation", False)])
        client.assert_not_called()

    def test_calculation_completes_after_close_skips_broker(self):
        closed_clock = SimpleNamespace(
            is_open=False,
            timestamp=datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc),
            next_close=datetime(2026, 10, 21, 20, 0, tzinfo=timezone.utc),
        )
        calls, _, broker = self._once(
            recorded_at="2026-10-20T19:59:00Z",
            market_open=True, plan_status="recalibrated_no_orders",
            fresh_clock=closed_clock,
        )
        self.assertEqual(calls, [("open_calculation", False)])
        broker.get_clock.assert_called_once()

    def test_ready_signal_must_recheck_clock_before_broker(self):
        calls, _, broker = self._once(
            recorded_at="2026-10-20T19:20:00Z",
            market_open=True, plan_status="recalibrated_no_orders",
        )
        self.assertEqual(calls, [("open_calculation", False), ("broker_execution", True)])
        broker.get_clock.assert_called_once()

    def test_no_exec_when_calculation_needs_retry(self):
        calls, client, _ = self._once(
            recorded_at="2026-10-20T19:20:00Z",
            market_open=True, plan_status="waiting_for_open_recalibration",
            calculation_status="waiting_for_open_retry",
        )
        self.assertEqual(calls, [("open_calculation", False)])
        client.assert_not_called()

    def test_manual_reconciled_session_is_not_replayed(self):
        calls, client, _ = self._once(
            recorded_at="2026-10-20T19:20:00Z",
            market_open=True, plan_status="manual_reconciled",
        )
        self.assertEqual(calls, [])
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
