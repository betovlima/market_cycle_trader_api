"""Contract tests for Live v10.8.111. All Alpaca/DB operations are mocked."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE
from market_cycle_trader_api.services import (
    live_winner_recalibration as recalibration,
    live_order_execution as executor,
)

SESSION = "2026-10-19"
OPEN = datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc)
PREFIX = "market_cycle_trader_api.services.live_winner_recalibration."


def _clock(minute: int, *, second: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        is_open=True,
        timestamp=datetime(2026, 10, 19, 13, minute, second, tzinfo=timezone.utc),
    )


def _run() -> dict:
    return {
        "_id": "acct:2026-10-19",
        "execution_session": SESSION,
        "status": "signal_prepared",
        "expected_market_open": OPEN,
        "live_account_id": "acct",
        "winner_strategy_id": "winner",
        "winner_revision": 2,
        "winner_configuration_hash": "winner-hash",
    }


def test_recalibration_waits_for_a_completed_minute_without_downloading():
    broker = MagicMock()
    broker.get_account.return_value.id = "acct"
    broker.get_clock.return_value = _clock(30, second=46)
    db = MagicMock()
    db["live_winner_session_runs"].find_one.return_value = _run()
    with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
        PREFIX + "download_u67_operational_snapshot"
    ) as download:
        result = recalibration.recalibrate_live_cash_winner_signal(db, SESSION)
    assert result["status"] == "waiting_for_intraday_data"
    assert result["order_submission_enabled"] is False
    download.assert_not_called()
    db["live_winner_session_runs"].find_one_and_update.assert_not_called()


def test_expired_live_session_is_never_recalibrated_or_executed():
    broker = MagicMock()
    broker.get_account.return_value.id = "acct"
    broker.get_clock.return_value = _clock(31) if False else SimpleNamespace(
        is_open=True, timestamp=datetime(2026, 10, 19, 14, 1, tzinfo=timezone.utc)
    )
    db = MagicMock()
    db["live_winner_session_runs"].find_one.return_value = _run()
    with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
        PREFIX + "download_u67_operational_snapshot"
    ) as download, patch(PREFIX + "append_live_event"):
        result = recalibration.recalibrate_live_cash_winner_signal(db, SESSION)
    assert result["status"] == "execution_window_expired"
    assert result["order_submission_enabled"] is False
    download.assert_not_called()
    assert db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]["phase"] == "execution_window_expired"


def _simulate_model_recalibration(final_clock: SimpleNamespace):
    db = MagicMock()
    run = _run()
    db["live_winner_session_runs"].find_one.return_value = run
    db["live_winner_session_runs"].find_one_and_update.side_effect = [
        run, {"status": "recalibrated_no_orders", "action": "buy"},
    ]
    broker = MagicMock()
    broker.get_account.return_value.id = "acct"
    broker.get_orders.return_value = []
    broker.get_clock.side_effect = [
        _clock(31, second=37),
        _clock(33, second=12),
        final_clock,
    ]
    model = {"family": "lightgbm_utility", "settings_snapshot": {}}
    profile = {"id": "winner", "revision": 2, "configuration_hash": "winner-hash"}
    strategy = MagicMock()
    strategy.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
    strategy.assets = ["ADEA"]
    strategy.model_copy.return_value = strategy

    with ExitStack() as stack:
        def fake(method: str, **kwargs):
            return stack.enter_context(patch(PREFIX + method, **kwargs))

        fake("create_live_readonly_client", return_value=broker)
        fake("verified_live_state", return_value={
            "managed_symbol": "CASH", "holding_sessions": 0, "managed_quantity": 0.0,
            "revision": 1,
        })
        fake("get_trader_winner_context", return_value=(strategy, profile))
        fake("get_trader_winner_model_snapshot", return_value=model)
        fake("apply_training_runtime_settings", return_value=strategy)
        fake("apply_execution_profile", return_value=strategy)
        fake("assert_tcc_u77_contract")
        fake("assert_tcc_u67_model_snapshot")
        fake("latest_safe_completed_xnys_session", return_value=pd.Timestamp("2026-10-16T04:00:00Z"))
        fake("download_u67_operational_snapshot", return_value=SimpleNamespace(
            frames={"ADEA": pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2026-10-16T04:00:00Z")])}
        ))
        intraday = fake(
            "append_u67_current_session_intraday",
            return_value=({"ADEA": pd.DataFrame({"close": [1]})}, {
                "live_feed": "iex", "intraday_market_data_cutoff_utc": "2026-10-19T13:33:00+00:00",
            }),
        )
        fake("build_live_model_decision", return_value=SimpleNamespace(
            decision_date=pd.Timestamp("2026-10-19"), target_asset="ADEA"
        ))
        fake("winner_candidate_fields", return_value={})
        fake("append_live_event")
        with patch(PREFIX + "download_u67_operational_snapshot") as snapshot_download:
            snapshot_download.return_value = SimpleNamespace(
                frames={"ADEA": pd.DataFrame({"close": [1]})}
            )
            if final_clock.timestamp.hour == 14:
                with pytest.raises(RuntimeError, match="LiveExecutionWindowExpired"):
                    recalibration.recalibrate_live_cash_winner_signal(db, SESSION)
                assert db["live_winner_session_runs"].update_one.call_args.args[1]["$set"]["phase"] == "execution_window_expired"
            else:
                result = recalibration.recalibrate_live_cash_winner_signal(db, SESSION)
                assert result["status"] == "recalibrated_no_orders"
                assert result["order_submission_enabled"] is False
            snapshot_download.assert_called_once()
        assert intraday.call_args.kwargs["now"] == pd.Timestamp("2026-10-19T13:33:00Z")
    broker.submit_order.assert_not_called()


def test_live_refreshes_clock_after_historical_download():
    _simulate_model_recalibration(_clock(40))


def test_completion_after_deadline_never_creates_executable_plan():
    _simulate_model_recalibration(SimpleNamespace(
        is_open=True, timestamp=datetime(2026, 10, 19, 14, 2, tzinfo=timezone.utc)
    ))


def test_intraday_audit_extracts_only_whitelisted_failure_identifiers():
    exc = RuntimeError(
        "CurrentSessionIntradayCoverageUnavailable: "
        "CurrentSessionIntradayDataMissing: AMS has no IEX regular-session bars "
        "at 2026-10-19. Secret: should not be exported"
    )
    result = recalibration._intraday_error_fields(exc)
    assert result == {
        "error_code": "CurrentSessionIntradayCoverageUnavailable",
        "symbol": "AMS", "feed": "iex",
    }
    unknown = recalibration._intraday_error_fields(
        RuntimeError("API KEY sk-test-not-an-error-code")
    )
    assert unknown == {"error_code": "RuntimeError"}


def test_recovery_waits_until_next_allowed_attempt():
    run = _run()
    run["retry_not_before"] = datetime(2026, 10, 19, 13, 51, tzinfo=timezone.utc)
    db = MagicMock()
    db["live_winner_session_runs"].find_one.return_value = run
    broker = MagicMock()
    broker.get_account.return_value.id = "acct"
    broker.get_clock.return_value = _clock(40)
    with patch(PREFIX + "create_live_readonly_client", return_value=broker), patch(
        PREFIX + "download_u67_operational_snapshot"
    ) as download:
        result = recalibration.recalibrate_live_cash_winner_signal(db, SESSION)
    assert result["status"] == "waiting_for_intraday_data"
    assert result["retry_after"] == "2026-10-19T13:51:00+00:00"
    download.assert_not_called()


def test_executor_refuses_real_order_after_independent_live_deadline():
    db = MagicMock()
    run = _run()
    run.update({"status": "recalibrated_no_orders", "action": "buy", "target_asset": "ADEA",
                "winner_signal_prepared": True})
    db["live_winner_session_runs"].find_one.return_value = run
    broker = MagicMock()
    broker.get_account.return_value = SimpleNamespace(
        id="acct", status="active", account_blocked=False, trading_blocked=False,
        trade_suspended_by_user=False,
    )
    broker.get_clock.return_value = SimpleNamespace(
        is_open=True, timestamp=datetime(2026, 10, 19, 14, 1, tzinfo=timezone.utc)
    )
    prefix = "market_cycle_trader_api.services.live_order_execution."
    with (
        patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
        patch(prefix + "get_trader_winner_summary", return_value={
            "id": "winner", "revision": 2, "configuration_hash": "winner-hash",
        }),
        patch(prefix + "create_live_execution_client", return_value=broker),
        patch(prefix + "_execute_leg") as leg,
    ):
        result = executor.execute_recalibrated_live_winner_session(db, SESSION)
    assert result["status"] == "execution_window_expired"
    assert not result["order_submission_enabled"]
    leg.assert_not_called()
    broker.submit_order.assert_not_called()


def test_each_broker_leg_rechecks_market_deadline():
    broker = MagicMock()
    broker.get_clock.return_value = SimpleNamespace(
        is_open=True, timestamp=datetime(2026, 10, 19, 14, 1, tzinfo=timezone.utc)
    )
    with pytest.raises(RuntimeError, match="LiveExecutionWindowExpired"):
        executor._assert_live_order_window(
            broker, SESSION, pd.Timestamp("2026-10-19T14:00:00Z")
        )
