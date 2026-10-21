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
        broker = MagicMock()
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 10, 19, 20, 1, tzinfo=timezone.utc)
        )
        with (
            patch.object(close_signal, "latest_safe_completed_xnys_session",
                         return_value=pd.Timestamp("2026-10-16")),
            patch.object(close_signal, "create_live_readonly_client", return_value=broker),
        ):
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
            patch.object(close_signal, "create_live_readonly_client") as broker_factory,
        ):
            broker_factory.return_value.get_clock.return_value = SimpleNamespace(
                timestamp=datetime(2026, 10, 19, 20, 1, tzinfo=timezone.utc)
            )
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                close_signal.prepare_live_cash_winner_signal(db, SESSION)
        update = db["live_winner_session_runs"].find_one_and_update.call_args
        self.assertEqual(update.args[1]["$set"]["status"], "waiting_for_winner_signal")
        self.assertEqual(update.args[1]["$set"]["superseded_decision_date"], "2026-10-16")
        self.assertEqual(update.args[1]["$set"]["superseded_target_asset"], "CAPR")


    def test_long_running_post_close_training_not_reclaimed_by_elapsed_time(self):
        db = MagicMock()
        started = datetime.now(timezone.utc) - timedelta(minutes=30)
        db["live_winner_session_runs"].find_one.return_value = _run(
            status="preparing_signal", preparation_started_at=started,
        )
        db["live_winner_session_runs"].find_one_and_update.return_value = _run(
            status="waiting_for_winner_signal", preparation_attempts=1,
        )
        with (
            patch.object(close_signal, "latest_safe_completed_xnys_session",
                         return_value=pd.Timestamp("2026-10-16")),
            patch.object(close_signal, "create_live_readonly_client") as broker_factory,
        ):
            broker_factory.return_value.get_clock.return_value = SimpleNamespace(
                timestamp=datetime.now(timezone.utc)
            )
            result = close_signal.prepare_live_cash_winner_signal(db, SESSION)
        self.assertEqual(result["status"], "post_close_training_in_progress")
        db["live_winner_session_runs"].find_one_and_update.assert_not_called()
        self.assertFalse(result["order_submission_enabled"])


class LiveOpenPipelineTests(TestCase):
    """The previous intraday recalibration entry point is retired."""

    def test_legacy_open_recalibration_is_disabled_without_broker_access(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayRecalibrationDisabled"):
            open_signal.refresh_live_open_winner_signal(db, SESSION)
        db.__getitem__.assert_not_called()

    def test_stale_friday_plan_requires_correct_monday_daily_bar(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run()
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client") as broker_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "LivePriorDailySignalRequired"):
                executor.execute_prepared_live_winner_session(db, SESSION)
            broker_factory.assert_not_called()

    def test_obsolete_current_session_decision_is_not_executable(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = _run(
            decision_date=PREVIOUS, historical_cutoff=PREVIOUS,
            analysis_mode="live_current_session_intraday",
        )
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client") as broker_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "LivePriorDailySignalRequired"):
                executor.execute_prepared_live_winner_session(db, SESSION)
            broker_factory.assert_not_called()


class LiveCurrentSessionBarsTests(TestCase):
    def test_uses_trading_data_credentials_and_aggregates_completed_minutes(self):
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
            patch("market_cycle_trader_api.engine.operational_market_snapshot.get_alpaca_credentials",
                  return_value={"api_key_id": "DATA_KEY", "secret_key": "DATA_SECRET"}),
            patch("market_cycle_trader_api.engine.operational_market_snapshot.download_stock_bars",
                  return_value=minutes) as provider,
        ):
            bars, info = live_market_bars.load_live_intraday_session(
                {"MAN": daily}, SimpleNamespace(alpaca_live_feed="iex"),
                session=SESSION, now=datetime(2026, 10, 20, 13, 40, tzinfo=timezone.utc),
            )
        self.assertEqual(info["intraday_asset_count"], 1)
        self.assertEqual(info["live_feed"], "iex")
        self.assertEqual(info["live_session"], SESSION)
        self.assertEqual(provider.call_args.kwargs["api_key_id"], "DATA_KEY")
        current = bars["MAN"].iloc[-1]
        self.assertEqual(float(current["close"]), 103)
        self.assertEqual(float(current["volume"]), 30)
        self.assertEqual(len(bars["MAN"]), 2)
