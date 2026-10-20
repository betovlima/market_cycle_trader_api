"""Regression: a failed Live-open calculation must be eligible again.

All Alpaca, MongoDB, training and order operations are mocked.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE
from market_cycle_trader_api.services import live_winner_open as open_signal


SESSION = "2026-10-20"
PREFIX = "market_cycle_trader_api.services.live_winner_open."


def _run(**overrides):
    return {
        "_id": "acct:" + SESSION,
        "execution_session": SESSION,
        "expected_market_open": datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc),
        "status": "waiting_for_open_recalibration",
        "market_open_calibration_completed_at": None,
        "live_account_id": "acct",
        "winner_strategy_id": "winner",
        "winner_revision": 2,
        "winner_configuration_hash": "hash",
        **overrides,
    }


class LiveOpenNullClaimRecoveryTests(TestCase):
    def test_failed_open_recalibration_can_claim_again_without_orders(self):
        db = MagicMock()
        sessions = MagicMock()
        db.__getitem__.side_effect = lambda name: sessions
        sessions.find_one.return_value = _run()
        sessions.find_one_and_update.return_value = _run(
            status="open_recalibration",
        )
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="acct", equity=100)
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 20, 13, 40, tzinfo=timezone.utc),
        )
        broker.get_orders.return_value = []
        strategy = MagicMock()
        strategy.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
        strategy.assets = ["CAPR", "AAPL"]
        strategy.model_copy.return_value = strategy
        winner = {"id": "winner", "revision": 2, "configuration_hash": "hash"}
        events = MagicMock()
        with ExitStack() as stack:
            def fake(name, value=None, side_effect=None):
                return stack.enter_context(
                    patch(PREFIX + name, return_value=value, side_effect=side_effect)
                )
            fake("live_control_status", {"control_mode": "active"})
            fake("create_live_readonly_client", broker)
            fake("latest_safe_completed_xnys_session", pd.Timestamp("2026-10-19"))
            fake("get_system_settings", {"training": {"enabled": True}})
            fake("get_trader_winner_context", (strategy, winner))
            fake("get_trader_winner_model_snapshot", {
                "family": "lightgbm_utility", "settings_snapshot": {},
            })
            fake("apply_training_runtime_settings", strategy)
            fake("apply_execution_profile", strategy)
            fake("assert_tcc_u77_contract")
            fake("assert_tcc_u67_model_snapshot")
            fake("verified_live_state", {
                "managed_symbol": "CASH", "managed_quantity": 0,
                "holding_sessions": 0, "revision": 1,
            })
            download = fake(
                "download_operational_market_snapshot",
                side_effect=RuntimeError("Live daily history snapshot is incomplete."),
            )
            stack.enter_context(patch(PREFIX + "append_live_event", events))

            first = open_signal.refresh_live_open_winner_signal(db, SESSION)
            second = open_signal.refresh_live_open_winner_signal(db, SESSION)

        self.assertEqual(first["status"], "waiting_for_open_retry")
        self.assertEqual(second["status"], "waiting_for_open_retry")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(sessions.find_one_and_update.call_count, 2)
        for call in sessions.find_one_and_update.call_args_list:
            claim = call.args[0]
            # MongoDB null equality matches missing OR null, but never a
            # successfully completed (non-null timestamp) claim.
            self.assertIsNone(claim["market_open_calibration_completed_at"])
            self.assertEqual(claim["status"]["$in"], open_signal._ELIGIBLE_STATUSES)
        self.assertEqual(sessions.update_one.call_count, 2)
        for call in sessions.update_one.call_args_list:
            self.assertEqual(
                call.args[1]["$set"]["status"],
                "waiting_for_open_recalibration",
            )
            self.assertIsNone(
                call.args[1]["$set"]["market_open_calibration_completed_at"]
            )
            self.assertFalse(call.args[1]["$set"]["order_submission_enabled"])
        self.assertEqual(events.call_count, 2)
        self.assertEqual(events.call_args.kwargs["details"]["stage"], "daily_history")
        self.assertEqual(
            events.call_args.kwargs["details"]["error_code"], "daily_history_failed"
        )
        broker.submit_order.assert_not_called()

    def test_safe_diagnostics_do_not_echo_provider_exception_payload(self):
        exc = RuntimeError(
            "LiveIntradayCoverageNotReady: provider context with account details"
        )
        self.assertEqual(
            open_signal._safe_open_failure_code(exc, "current_session_bars"),
            "LiveIntradayCoverageNotReady",
        )
        self.assertEqual(
            open_signal._safe_open_failure_code(
                RuntimeError("unknown credentials and URL"), "daily_history"
            ),
            "daily_history_failed",
        )

    def test_already_completed_open_session_is_not_claimed(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run(
            status="completed",
            market_open_calibration_completed_at=datetime(
                2026, 10, 20, 13, 45, tzinfo=timezone.utc
            ),
        )
        with patch(PREFIX + "live_control_status", return_value={"control_mode": "active"}):
            result = open_signal.refresh_live_open_winner_signal(db, SESSION)
        self.assertEqual(result["status"], "completed")
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()
