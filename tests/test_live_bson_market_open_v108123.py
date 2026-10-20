"""Regression for the 2026-10-20 MAN -> CAPR failed automated Live rotation.

MongoDB BSON is UTC but PyMongo decodes dates to tz-naive datetimes by
default. The executor must normalize that UTC timestamp exactly as Trading
does, then validate the XNYS official session open *before* any order.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import exchange_calendars as xcals
import pandas as pd
from bson import BSON

from market_cycle_trader_api.services import live_order_execution as executor
from market_cycle_trader_api.services.live_workflow_monitor import _safe_broker_failure

ACCOUNT_ID = "live-account"
WINNER_HASH = "winner-hash"
PREFIX = "market_cycle_trader_api.services.live_order_execution."


def _date_from_bson(value: datetime) -> datetime:
    """Use the actual PyMongo default BSON codec, not synthetic tz removal."""
    return BSON.encode({"date": value}).decode()["date"]


class LiveExpectedOpenUTCRegression(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.run = {
            "_id": ACCOUNT_ID + ":2026-10-20",
            "live_account_id": ACCOUNT_ID,
            "execution_session": "2026-10-20",
            "status": "recalibrated_no_orders",
            "winner_strategy_id": "winner",
            "winner_revision": 3,
            "winner_configuration_hash": WINNER_HASH,
            "winner_signal_prepared": True,
            "post_close_signal_ready": True,
            "snapshot_sha256": "frozen-daily-sha",
            "market_open_calibration_completed_at": datetime(2026, 10, 20, 15, 11, tzinfo=timezone.utc),
            "decision_date": "2026-10-20",
            "historical_cutoff": "2026-10-19",
            "analysis_mode": "live_current_session_intraday",
            "action": "rotate",
            "current_asset": "MAN",
            "target_asset": "CAPR",
            "live_state_revision": 4,
        }
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(
            id=ACCOUNT_ID, status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False,
        )
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 20, 15, 45, tzinfo=timezone.utc),
            next_close=datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc),
        )
        self.broker.get_orders.return_value = []
        self.db["live_winner_session_runs"].find_one.return_value = self.run
        self.patches = [
            patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}),
            patch(PREFIX + "get_trader_winner_summary",
                  return_value={"id": "winner", "revision": 3, "configuration_hash": WINNER_HASH}),
            patch(PREFIX + "create_live_execution_client", return_value=self.broker),
            # Deliberately halt *after* timestamp and broker-order validation,
            # but before claiming a plan or submitting anything.
            patch(PREFIX + "verified_live_state", side_effect=RuntimeError("Reached verified state check")),
        ]
        self.mocks = [ctx.start() for ctx in self.patches]
        for ctx in self.patches:
            self.addCleanup(ctx.stop)

    def _run(self):
        return executor.execute_prepared_live_winner_session(self.db, self.run["execution_session"])

    def _no_order(self):
        self.broker.submit_order.assert_not_called()
        self.broker.cancel_order_by_id.assert_not_called()
        self.db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_mongo_utc_naive_opens_pass_same_validation_as_trading(self):
        utc_open = datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
        from_bson = _date_from_bson(utc_open)
        self.assertIsNone(from_bson.tzinfo)
        self.assertEqual(from_bson, utc_open.replace(tzinfo=None))
        self.run["expected_market_open"] = from_bson
        with self.assertRaisesRegex(RuntimeError, "Reached verified state check"):
            self._run()
        self.mocks[-1].assert_called_once()
        self._no_order()

    def test_timezone_aware_official_open_also_passes(self):
        self.run["expected_market_open"] = datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
        with self.assertRaisesRegex(RuntimeError, "Reached verified state check"):
            self._run()
        self._no_order()

    def test_naive_local_exchange_time_is_not_misread_as_open(self):
        self.run["expected_market_open"] = datetime(2026, 10, 20, 9, 30)
        with self.assertRaisesRegex(RuntimeError, "differs from the official XNYS"):
            self._run()
        self.mocks[-1].assert_not_called()
        self._no_order()

    def test_wrong_day_is_not_accepted_as_a_market_open(self):
        self.run["expected_market_open"] = datetime(2026, 10, 19, 13, 30)
        with self.assertRaisesRegex(RuntimeError, "differs from the official XNYS"):
            self._run()
        self._no_order()

    def test_non_open_intraday_time_is_rejected(self):
        self.run["expected_market_open"] = datetime(2026, 10, 20, 15, 11)
        with self.assertRaisesRegex(RuntimeError, "differs from the official XNYS"):
            self._run()
        self._no_order()

    def test_missing_timestamp_is_not_reconstructed(self):
        self.run["expected_market_open"] = None
        with self.assertRaisesRegex(RuntimeError, "expected open missing"):
            self._run()
        self._no_order()

    def test_invalid_timestamp_remains_blocked_and_coded(self):
        self.run["expected_market_open"] = "not-an-instant"
        with self.assertRaises(RuntimeError) as raised:
            self._run()
        self.assertEqual(
            _safe_broker_failure(raised.exception)["error_code"],
            "invalid_execution_window",
        )
        self._no_order()

    def test_mismatched_timestamp_has_safe_diagnostic_code(self):
        self.run["expected_market_open"] = datetime(2026, 10, 20, 9, 30)
        with self.assertRaises(RuntimeError) as raised:
            self._run()
        self.assertEqual(
            _safe_broker_failure(raised.exception)["error_code"],
            "invalid_execution_window",
        )
        self._no_order()

    def test_early_close_session_accepts_only_exchange_open(self):
        session = "2026-11-27"
        calendar = xcals.get_calendar("XNYS")
        official = calendar.session_open(pd.Timestamp(session))
        self.run["execution_session"] = session
        self.run["_id"] = ACCOUNT_ID + ":" + session
        self.run["decision_date"] = session
        self.run["historical_cutoff"] = "2026-11-25"
        self.run["expected_market_open"] = _date_from_bson(official.to_pydatetime())
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc),
            next_close=datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc),
        )
        with self.assertRaisesRegex(RuntimeError, "Reached verified state check"):
            self._run()
        self._no_order()

    def test_valid_timestamp_outside_session_still_cannot_order(self):
        self.run["expected_market_open"] = _date_from_bson(
            datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
        )
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 10, 20, 20, 1, tzinfo=timezone.utc),
            next_close=datetime(2026, 10, 21, 20, 0, tzinfo=timezone.utc),
        )
        result = self._run()
        self.assertEqual(result["status"], "execution_window_expired")
        self.assertEqual(
            self.db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]["status"],
            "review_required",
        )
        self.mocks[-1].assert_not_called()
        self._no_order()

    def test_already_reconciled_manual_session_cannot_execute_again(self):
        self.run["status"] = "manual_reconciled"
        self.run["expected_market_open"] = _date_from_bson(
            datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
        )
        result = self._run()
        self.assertEqual(result["status"], "no_recalibrated_live_plan")
        self.broker.get_account.assert_not_called()
        self._no_order()


if __name__ == "__main__":
    unittest.main()
