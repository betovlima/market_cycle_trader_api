"""API v10.8.125: day-before Live signal is authoritative at market open.

Tests use mocks only. No broker orders or market-data downloads.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import (
    live_order_execution as execution,
    live_winner_signal as signal,
    live_observation_scheduler as scheduler,
)
from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE

SESSION = "2026-10-21"
PRIOR = "2026-10-20"
PREFIX = "market_cycle_trader_api.services.live_winner_signal."


def ready_run(**changes):
    values = {
        "_id": "acct:" + SESSION,
        "execution_session": SESSION,
        "expected_market_open": datetime(2026, 10, 21, 13, 30, tzinfo=timezone.utc),
        "status": "recalibrated_no_orders",
        "live_account_id": "acct",
        "winner_strategy_id": "winner",
        "winner_revision": 2,
        "winner_configuration_hash": "hash",
        "decision_date": PRIOR,
        "target_asset": "CAPR",
        "current_asset": "CAPR",
        "action": "hold",
        "winner_signal_prepared": True,
        "post_close_signal_ready": True,
        "snapshot_sha256": "prior-daily-hash",
        "live_state_revision": 2,
    }
    values.update(changes)
    return values


class PriorDailyContractTests(TestCase):
    def test_prepared_yesterday_remains_idempotent_even_when_market_is_open(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = ready_run()
        with patch(PREFIX + "download_operational_market_snapshot") as download, patch(
            PREFIX + "create_live_readonly_client"
        ) as broker:
            result = signal.prepare_live_winner_signal(
                db, SESSION, allow_open_recovery=True,
            )
        self.assertEqual(result["status"], "recalibrated_no_orders")
        download.assert_not_called()
        broker.assert_not_called()

    def test_open_recovery_uses_only_yesterday_daily_data(self):
        db = MagicMock()
        pending = ready_run(
            status="waiting_for_winner_signal",
            decision_date=None,
            target_asset=None,
            action=None,
            winner_signal_prepared=False,
            post_close_signal_ready=False,
            snapshot_sha256=None,
        )
        db["live_winner_session_runs"].find_one.return_value = pending
        db["live_winner_session_runs"].find_one_and_update.side_effect = [
            {**pending, "status": "preparing_signal"},
            {**pending, "status": "recalibrated_no_orders", "action": "hold"},
        ]
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="acct", equity=1000.0)
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 10, 21, 13, 40, tzinfo=timezone.utc),
        )
        broker.get_orders.return_value = []
        config = MagicMock()
        config.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
        config.assets = ["CAPR"]
        config.model_copy.return_value = config
        snapshot = SimpleNamespace(
            frames={"CAPR": pd.DataFrame({"close": [1.0]})},
            manifest={"snapshot_sha256": "verified-prior-hash"},
        )
        with ExitStack() as stack:
            def mock(name, value=None):
                return stack.enter_context(patch(PREFIX + name, return_value=value))
            mock("create_live_readonly_client", broker)
            mock("latest_safe_completed_xnys_session", pd.Timestamp(PRIOR))
            mock("get_system_settings", {"training": {"enabled": True}})
            mock("get_trader_winner_context", (
                config, {"id": "winner", "revision": 2, "configuration_hash": "hash"},
            ))
            mock("get_trader_winner_model_snapshot", {
                "family": "lightgbm_utility", "settings_snapshot": {},
            })
            mock("apply_training_runtime_settings", config)
            mock("apply_execution_profile", config)
            mock("assert_tcc_u77_contract")
            mock("assert_tcc_u67_model_snapshot")
            mock("verified_live_state", {
                "managed_symbol": "CAPR", "managed_quantity": 3,
                "holding_sessions": 2, "revision": 2,
            })
            daily = mock("download_operational_market_snapshot", snapshot)
            mock("build_live_model_decision", SimpleNamespace(
                decision_date=pd.Timestamp(PRIOR), target_asset="CAPR",
            ))
            mock("winner_candidate_fields", {})
            mock("append_live_event")
            result = signal.prepare_live_winner_signal(
                db, SESSION, allow_open_recovery=True,
            )
        self.assertEqual(result["status"], "recalibrated_no_orders")
        self.assertEqual(result["decision_date"], PRIOR)
        self.assertEqual(result["action"], "hold")
        self.assertEqual(daily.call_count, 1)
        self.assertEqual(daily.call_args.kwargs["config"].analysis_end_date, PRIOR)
        fields = db["live_winner_session_runs"].find_one_and_update.call_args_list[-1].args[1]["$set"]
        self.assertEqual(fields["analysis_mode"], "previous_completed_session_daily")
        self.assertEqual(fields["historical_cutoff"], PRIOR)
        self.assertEqual(fields["phase"], "daily_signal_recovered_at_open")
        broker.submit_order.assert_not_called()

    def test_normal_daily_preparation_still_refuses_market_open(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = ready_run(
            status="waiting_for_winner_signal", decision_date=None,
        )
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 10, 21, 13, 40, tzinfo=timezone.utc),
        )
        with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
            PREFIX + "latest_safe_completed_xnys_session",
            return_value=pd.Timestamp(PRIOR),
        ), patch(PREFIX + "get_system_settings", return_value={"training": {"enabled": True}}), patch(
            PREFIX + "get_trader_winner_context", return_value=(MagicMock(), {})
        ):
            # Default after-close path does not acquire an open recovery claim.
            with self.assertRaises(RuntimeError):
                signal.prepare_live_winner_signal(db, SESSION)
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_missing_data_fails_closed_without_training_or_orders(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = ready_run(
            status="waiting_for_winner_signal", decision_date=None,
        )
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 21, 13, 35, tzinfo=timezone.utc)
        )
        with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
            PREFIX + "latest_safe_completed_xnys_session",
            return_value=pd.Timestamp("2026-10-19"),
        ), patch(PREFIX + "download_operational_market_snapshot") as daily:
            result = signal.prepare_live_winner_signal(
                db, SESSION, allow_open_recovery=True,
            )
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        daily.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_scheduler_never_calls_obsolete_intraday_recalibration(self):
        source = (Path(scheduler.__file__)).read_text(encoding="utf-8")
        self.assertNotIn("refresh_live_open_winner_signal", source)
        self.assertNotIn("load_live_intraday_session", source)
        self.assertIn('allow_open_recovery=True', source)

    def test_executor_rejects_current_session_and_legacy_intraday_modes(self):
        for changes in (
            {"decision_date": SESSION},
            {"decision_date": "2026-10-19"},
            {"analysis_mode": "live_current_session_intraday"},
            {"historical_cutoff": "2026-10-19"},
        ):
            with self.subTest(changes=changes):
                db = MagicMock()
                db["live_winner_session_runs"].find_one.return_value = ready_run(**changes)
                prefix = "market_cycle_trader_api.services.live_order_execution."
                with patch(prefix + "live_control_status", return_value={"control_mode": "active"}), patch(
                    prefix + "get_trader_winner_summary",
                    return_value={"id": "winner", "revision": 2, "configuration_hash": "hash"},
                ), patch(prefix + "create_live_execution_client") as broker:
                    with self.assertRaisesRegex(RuntimeError, "LivePriorDailySignalRequired"):
                        execution.execute_prepared_live_winner_session(db, SESSION)
                    broker.assert_not_called()

    def test_existing_valid_2026_10_20_signal_passes_daily_guard_without_retraining(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = ready_run()
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id="acct", status="active",
            account_blocked=False, trading_blocked=False,
            trade_suspended_by_user=False,
        )
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 21, 13, 40, tzinfo=timezone.utc)
        )
        broker.get_orders.return_value = []
        db["live_winner_session_runs"].find_one_and_update.return_value = ready_run()
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with ExitStack() as stack:
            stack.enter_context(patch(prefix + "live_control_status", return_value={"control_mode": "active"}))
            stack.enter_context(patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }))
            stack.enter_context(patch(prefix + "create_live_execution_client", return_value=broker))
            stack.enter_context(patch(prefix + "verified_live_state", return_value={
                "account_id": "acct", "managed_symbol": "CAPR",
                "managed_quantity": 3, "revision": 2,
            }))
            leg = stack.enter_context(patch(prefix + "_execute_leg"))
            result = execution.execute_prepared_live_winner_session(db, SESSION)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["action"], "hold")
        leg.assert_not_called()
        broker.submit_order.assert_not_called()
