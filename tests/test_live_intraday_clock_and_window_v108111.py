"""Live post-close-only training, one bounded retry, immutable open-session plan."""
from __future__ import annotations

import ast
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api import services
from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE
from market_cycle_trader_api.services import (
    live_winner_signal as signals,
    live_winner_recalibration as retired,
    live_order_execution as executor,
    live_expired_recovery,
)

SESSION = "2026-10-19"
PREFIX = "market_cycle_trader_api.services.live_winner_signal."


def _run(attempts=0):
    return {
        "_id": "acct:" + SESSION, "execution_session": SESSION,
        "status": "waiting_for_winner_signal",
        "live_account_id": "acct", "winner_strategy_id": "winner",
        "winner_revision": 2, "winner_configuration_hash": "hash",
        "expected_market_open": datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc),
        "preparation_attempts": attempts,
    }


def _prepare_test(*, fail_download=False, attempts=0, market_open=False):
    db = MagicMock()
    run = _run(attempts)
    db["live_winner_session_runs"].find_one.return_value = run
    claimed = {**run, "status": "preparing_signal", "preparation_attempts": attempts + 1}
    saved = {**run, "status": "recalibrated_no_orders", "target_asset": "ADEA", "action": "buy"}
    db["live_winner_session_runs"].find_one_and_update.side_effect = [claimed, saved]
    broker = MagicMock()
    broker.get_account.return_value = SimpleNamespace(id="acct", equity=1000.0)
    broker.get_clock.return_value = SimpleNamespace(is_open=market_open)
    broker.get_orders.return_value = []
    config = MagicMock()
    config.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
    config.assets = ["ADEA"]
    config.model_copy.return_value = config
    profile = {"id": "winner", "revision": 2, "configuration_hash": "hash"}
    model = {"family": "lightgbm_utility", "settings_snapshot": {}}
    snapshot = SimpleNamespace(
        frames={"ADEA": pd.DataFrame({"close": [1.0]})},
        manifest={"snapshot_sha256": "frozen-snapshot-sha"},
    )
    decision = SimpleNamespace(decision_date=pd.Timestamp("2026-10-16"), target_asset="ADEA")
    with ExitStack() as stack:
        def fake(name, result=None, side_effect=None):
            return stack.enter_context(patch(
                PREFIX + name, return_value=result, side_effect=side_effect,
            ))
        fake("get_system_settings", {"training": {"enabled": True}})
        fake("get_trader_winner_context", (config, profile))
        fake("get_trader_winner_model_snapshot", model)
        fake("apply_training_runtime_settings", config)
        fake("apply_execution_profile", config)
        fake("assert_tcc_u77_contract")
        fake("assert_tcc_u67_model_snapshot")
        fake("create_live_readonly_client", broker)
        fake("verified_live_state", {
            "managed_symbol": "CASH", "managed_quantity": 0,
            "holding_sessions": 0, "revision": 0,
        })
        fake("latest_safe_completed_xnys_session", pd.Timestamp("2026-10-16T04:00:00Z"))
        if fail_download:
            snapshot_call = fake("download_u67_operational_snapshot", side_effect=RuntimeError("provider unavailable"))
        else:
            snapshot_call = fake("download_u67_operational_snapshot", snapshot)
        model_call = fake("build_live_model_decision", decision)
        fake("winner_candidate_fields", {})
        events = fake("append_live_event")
        outcome = None
        error = None
        try:
            outcome = signals.prepare_live_cash_winner_signal(db, SESSION)
        except RuntimeError as exc:
            error = exc
    return db, broker, snapshot_call, model_call, outcome, error


class PostCloseSingleCalibrationTests(TestCase):
    def test_success_frozen_once_and_never_retrained(self):
        db, broker, snapshots, models, result, error = _prepare_test()
        self.assertIsNone(error)
        self.assertEqual(result["status"], "recalibrated_no_orders")
        self.assertFalse(result["order_submission_enabled"])
        snapshots.assert_called_once()
        models.assert_called_once()
        updates = db["live_winner_session_runs"].find_one_and_update.call_args_list[-1].args[1]["$set"]
        self.assertEqual(updates["phase"], "post_close_signal_ready_for_execution")
        self.assertTrue(updates["post_close_signal_ready"])
        self.assertEqual(updates["decision_date"], "2026-10-16")
        self.assertEqual(updates["snapshot_sha256"], "frozen-snapshot-sha")
        broker.submit_order.assert_not_called()

    def test_one_retry_after_failure_has_cooldown_and_no_order(self):
        db, broker, snapshots, models, result, error = _prepare_test(fail_download=True)
        self.assertIsNotNone(error)
        updates = db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]
        self.assertEqual(updates["status"], "waiting_for_winner_signal")
        self.assertEqual(updates["phase"], "post_close_signal_retry_wait")
        self.assertIsNotNone(updates["preparation_retry_not_before"])
        self.assertFalse(updates["post_close_signal_ready"])
        models.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_second_failure_exhausts_retry_and_requires_review(self):
        db, broker, snapshots, models, result, error = _prepare_test(fail_download=True, attempts=1)
        self.assertIsNotNone(error)
        updates = db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]
        self.assertEqual(updates["status"], "review_required")
        self.assertEqual(updates["phase"], "post_close_signal_failed")
        self.assertFalse(updates["order_submission_enabled"])
        broker.submit_order.assert_not_called()

    def test_no_preparation_when_regular_market_open(self):
        db, broker, snapshots, models, result, error = _prepare_test(market_open=True)
        self.assertIn("closed regular market", str(error))
        snapshots.assert_not_called()
        models.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_successful_frozen_plan_is_idempotent(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "recalibrated_no_orders", "target_asset": "ADEA",
        }
        result = signals.prepare_live_cash_winner_signal(db, SESSION)
        self.assertEqual(result["target_asset"], "ADEA")
        self.assertFalse(result["order_submission_enabled"])
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_pending_retry_does_not_train_during_wait(self):
        db = MagicMock()
        run = _run()
        run["preparation_retry_not_before"] = datetime(2099, 10, 19, tzinfo=timezone.utc)
        db["live_winner_session_runs"].find_one.return_value = run
        with patch(PREFIX + "create_live_readonly_client") as broker:
            result = signals.prepare_live_cash_winner_signal(db, SESSION)
        self.assertEqual(result["status"], "waiting_for_post_close_retry")
        broker.assert_not_called()

    def test_exhausted_attempts_cannot_start_third_training(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run(attempts=2)
        with patch(PREFIX + "download_u67_operational_snapshot") as download:
            with self.assertRaisesRegex(RuntimeError, "attempts exhausted"):
                signals.prepare_live_cash_winner_signal(db, SESSION)
        download.assert_not_called()

    def test_intraday_recalibration_and_expired_reanalysis_retired(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayRecalibrationDisabled"):
            retired.recalibrate_live_cash_winner_signal(db, SESSION)
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayReanalysisDisabled"):
            live_expired_recovery.queue_expired_live_analysis(
                db, execution_session=SESSION, actor_email="admin@example.com"
            )
        db.__getitem__.assert_not_called()

    def test_scheduler_no_longer_imports_or_calls_intraday_model(self):
        path = Path(services.__file__).parent / "live_observation_scheduler.py"
        source = path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        self.assertFalse(any(
            node.module and node.module.endswith("live_winner_recalibration")
            for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)
        ))
        self.assertNotIn("recalibrate_live_cash_winner_signal(", source)

    def test_execution_requires_verified_after_close_frozen_snapshot(self):
        db = MagicMock()
        run = {
            **_run(), "status": "recalibrated_no_orders",
            "winner_signal_prepared": True, "target_asset": "ADEA",
            "action": "buy", "current_asset": "CASH",
            "live_state_revision": 0,
            "decision_date": "2026-10-16",
            "snapshot_sha256": "frozen-sha",
            # post_close_signal_ready deliberately missing.
        }
        db["live_winner_session_runs"].find_one.return_value = run
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id="acct", status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False
        )
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 19, 13, 40, tzinfo=timezone.utc)
        )
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client", return_value=broker),
            patch(prefix + "_execute_leg") as orders,
        ):
            with self.assertRaisesRegex(RuntimeError, "No verified post-close"):
                executor.execute_recalibrated_live_winner_session(db, SESSION)
            orders.assert_not_called()
            broker.submit_order.assert_not_called()
