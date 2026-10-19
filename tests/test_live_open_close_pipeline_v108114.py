"""Live-only after-close and market-open contract; all Alpaca calls mocked."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE
from market_cycle_trader_api.services import live_winner_signal as close_signal
from market_cycle_trader_api.services import live_winner_open as open_signal
from market_cycle_trader_api.services import live_market_bars
from market_cycle_trader_api.services import live_order_execution as executor

SESSION = "2026-10-20"
PREVIOUS = "2026-10-19"


def _run(**values):
    return {
        "_id": "acct:" + SESSION, "execution_session": SESSION,
        "expected_market_open": datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc),
        "status": "recalibrated_no_orders",
        "phase": "post_close_signal_ready_for_execution",
        "live_account_id": "acct", "winner_strategy_id": "winner",
        "winner_revision": 2, "winner_configuration_hash": "hash",
        "decision_date": "2026-10-16", "target_asset": "CAPR",
        "action": "rotate", "winner_signal_prepared": True,
        "post_close_signal_ready": True, "snapshot_sha256": "friday-hash",
        **values,
    }


class LiveDailyPreparationTests(TestCase):
    def test_friday_signal_waits_for_monday_daily_data(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run()
        with patch.object(close_signal, "latest_safe_completed_xnys_session",
                          return_value=pd.Timestamp("2026-10-16")):
            result = close_signal.prepare_live_cash_winner_signal(db, SESSION)
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        self.assertEqual(result["required_day"], PREVIOUS)
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()

    def test_stale_friday_signal_is_replaced_when_monday_bar_is_safe(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run()
        db["live_winner_session_runs"].find_one_and_update.return_value = _run(
            status="waiting_for_winner_signal", preparation_attempts=0,
        )
        with (
            patch.object(close_signal, "latest_safe_completed_xnys_session",
                         return_value=pd.Timestamp(PREVIOUS)),
            patch.object(close_signal, "get_system_settings",
                         return_value={"training": {"enabled": False}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                close_signal.prepare_live_cash_winner_signal(db, SESSION)
        update = db["live_winner_session_runs"].find_one_and_update.call_args
        self.assertEqual(update.args[1]["$set"]["status"], "waiting_for_winner_signal")
        self.assertEqual(update.args[1]["$set"]["superseded_decision_date"], "2026-10-16")
        self.assertEqual(update.args[1]["$set"]["superseded_target_asset"], "CAPR")


    def test_abandoned_post_close_training_is_rearmed_without_orders(self):
        db = MagicMock()
        started = datetime.now(timezone.utc) - timedelta(minutes=30)
        db["live_winner_session_runs"].find_one.return_value = _run(
            status="preparing_signal", preparation_started_at=started,
        )
        db["live_winner_session_runs"].find_one_and_update.return_value = _run(
            status="waiting_for_winner_signal", preparation_attempts=1,
        )
        with patch.object(close_signal, "latest_safe_completed_xnys_session",
                          return_value=pd.Timestamp("2026-10-16")):
            result = close_signal.prepare_live_cash_winner_signal(db, SESSION)
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        changed = db["live_winner_session_runs"].find_one_and_update.call_args.args[1]["$set"]
        self.assertEqual(changed["phase"], "recovered_abandoned_post_close_calibration")
        self.assertFalse(changed["order_submission_enabled"])


class LiveOpenPipelineTests(TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(id="acct", equity=110.70)
        self.broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 20, 13, 40, tzinfo=timezone.utc),
        )
        self.broker.get_orders.return_value = []

    def test_old_capr_plan_is_not_automatically_executed(self):
        run = _run()
        self.db["live_winner_session_runs"].find_one.return_value = run
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client") as broker_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "LiveOpenCalibrationRequired"):
                executor.execute_recalibrated_live_winner_session(self.db, SESSION)
            broker_factory.assert_not_called()

    def test_open_waits_if_monday_cutoff_not_current(self):
        self.db["live_winner_session_runs"].find_one.return_value = _run()
        prefix = "market_cycle_trader_api.services.live_winner_open."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "create_live_readonly_client", return_value=self.broker),
            patch(prefix + "latest_safe_completed_xnys_session",
                  return_value=pd.Timestamp("2026-10-16")),
            patch(prefix + "download_operational_market_snapshot") as download,
        ):
            result = open_signal.refresh_live_open_winner_signal(self.db, SESSION)
        self.assertEqual(result["status"], "waiting_for_correct_prior_daily_bar")
        download.assert_not_called()

    def test_open_recalibrates_monday_history_and_tuesday_minutes(self):
        existing = _run()
        claimed = _run(status="open_recalibration", open_preparation_attempts=1)
        saved = _run(status="recalibrated_no_orders", decision_date=SESSION,
                     target_asset="MAN", action="hold")
        self.db["live_winner_session_runs"].find_one.return_value = existing
        self.db["live_winner_session_runs"].find_one_and_update.side_effect = [claimed, saved]
        cfg = MagicMock()
        cfg.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
        cfg.assets = ["MAN", "CAPR"]
        cfg.model_copy.return_value = cfg
        profile = {"id": "winner", "revision": 2, "configuration_hash": "hash"}
        snapshot = SimpleNamespace(frames={"MAN": pd.DataFrame({"close": [1]})},
                                   manifest={"snapshot_sha256": "monday-hash"})
        prefix = "market_cycle_trader_api.services.live_winner_open."
        with ExitStack() as stack:
            def fake(name, value=None):
                return stack.enter_context(patch(prefix + name, return_value=value))
            fake("live_control_status", {"control_mode": "active"})
            fake("create_live_readonly_client", self.broker)
            fake("live_credentials", ("LIVE_KEY", "LIVE_SECRET"))
            fake("latest_safe_completed_xnys_session", pd.Timestamp(PREVIOUS))
            fake("get_system_settings", {"training": {"enabled": True}})
            fake("get_trader_winner_context", (cfg, profile))
            fake("get_trader_winner_model_snapshot", {
                "family": "lightgbm_utility", "settings_snapshot": {},
            })
            fake("apply_training_runtime_settings", cfg)
            fake("apply_execution_profile", cfg)
            fake("assert_tcc_u77_contract")
            fake("assert_tcc_u67_model_snapshot")
            fake("verified_live_state", {
                "managed_symbol": "MAN", "managed_quantity": 2.668639053,
                "holding_sessions": 2, "revision": 1,
            })
            download = fake("download_operational_market_snapshot", snapshot)
            live_bars = fake("load_live_intraday_session", (
                snapshot.frames, {
                    "analysis_mode": "live_current_session_intraday",
                    "live_session": SESSION,
                    "intraday_market_data_cutoff_utc": "2026-10-20T13:40:00+00:00",
                },
            ))
            fake("build_live_model_decision", SimpleNamespace(
                decision_date=pd.Timestamp(SESSION), target_asset="MAN"
            ))
            fake("winner_candidate_fields", {})
            fake("append_live_event")
            result = open_signal.refresh_live_open_winner_signal(self.db, SESSION)
        self.assertEqual(result["status"], "recalibrated_no_orders")
        self.assertEqual(result["decision_date"], SESSION)
        self.assertEqual(result["action"], "hold")
        self.assertEqual(live_bars.call_args.kwargs["session"], SESSION)
        self.assertEqual(download.call_args.kwargs["credentials_override"],
                         {"api_key_id": "LIVE_KEY", "secret_key": "LIVE_SECRET"})
        self.assertFalse(result["order_submission_enabled"])
        saved_fields = self.db["live_winner_session_runs"].find_one_and_update.call_args_list[-1].args[1]["$set"]
        self.assertEqual(saved_fields["historical_cutoff"], PREVIOUS)
        self.assertEqual(saved_fields["decision_date"], SESSION)
        self.assertEqual(saved_fields["analysis_mode"], "live_current_session_intraday")
        self.broker.submit_order.assert_not_called()

    def test_abandoned_live_open_claim_recovers_without_orders(self):
        started = datetime.now(timezone.utc) - timedelta(minutes=30)
        # Distinct collections: MagicMock.__getitem__ otherwise returns the
        # same mock for Live sessions and broker intents.
        sessions = MagicMock()
        intents = MagicMock()
        self.db.__getitem__.side_effect = lambda name: (
            sessions if name == "live_winner_session_runs" else intents
        )
        sessions.find_one.return_value = _run(
            status="open_recalibration", open_calibration_started_at=started,
        )
        intents.find_one.return_value = None
        sessions.find_one_and_update.return_value = _run(
            status="waiting_for_open_recalibration",
        )
        prefix = "market_cycle_trader_api.services.live_winner_open."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "create_live_readonly_client") as broker_factory,
        ):
            result = open_signal.refresh_live_open_winner_signal(self.db, SESSION)
        self.assertEqual(result["status"], "recovered_abandoned_live_calibration")
        self.assertFalse(result["order_submission_enabled"])
        broker_factory.assert_not_called()
        self.broker.submit_order.assert_not_called()


    def test_completed_session_is_not_retrained(self):
        self.db["live_winner_session_runs"].find_one.return_value = _run(status="completed")
        prefix = "market_cycle_trader_api.services.live_winner_open."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "create_live_readonly_client") as broker_factory,
        ):
            result = open_signal.refresh_live_open_winner_signal(self.db, SESSION)
        self.assertEqual(result["status"], "completed")
        broker_factory.assert_not_called()


class LiveCurrentSessionBarsTests(TestCase):
    def test_uses_live_credentials_and_aggregates_completed_minutes(self):
        daily = pd.DataFrame([{"open": 100, "high": 101, "low": 99,
                              "close": 100, "volume": 1000}],
                             index=pd.DatetimeIndex([pd.Timestamp("2026-10-19T04:00:00Z")]))
        minutes = pd.DataFrame([
            {"open": 101, "high": 103, "low": 100, "close": 102,
             "volume": 10, "vwap": 102, "trade_count": 1},
            {"open": 102, "high": 104, "low": 101, "close": 103,
             "volume": 20, "vwap": 103, "trade_count": 2},
        ], index=pd.DatetimeIndex([
            pd.Timestamp("2026-10-20T13:35:00Z"),
            pd.Timestamp("2026-10-20T13:36:00Z"),
        ]))
        with (
            patch.object(live_market_bars, "live_credentials",
                         return_value=("LIVE_KEY", "LIVE_SECRET")),
            patch.object(live_market_bars, "download_stock_bars",
                         return_value=minutes) as provider,
        ):
            bars, info = live_market_bars.load_live_intraday_session(
                {"MAN": daily}, SimpleNamespace(alpaca_live_feed="iex"),
                session=SESSION, now=datetime(2026, 10, 20, 13, 40, tzinfo=timezone.utc),
            )
        self.assertEqual(info["intraday_asset_count"], 1)
        self.assertEqual(info["live_feed"], "iex")
        self.assertEqual(info["live_session"], SESSION)
        self.assertEqual(provider.call_args.kwargs["api_key_id"], "LIVE_KEY")
        current = bars["MAN"].iloc[-1]
        self.assertEqual(float(current["close"]), 103)
        self.assertEqual(float(current["volume"]), 30)
        self.assertEqual(len(bars["MAN"]), 2)
