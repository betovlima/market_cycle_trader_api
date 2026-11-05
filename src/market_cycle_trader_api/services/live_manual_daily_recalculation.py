"""Admin-only request for the canonical Live Winner's completed-daily signal.

This does not run training inside an HTTP request and NEVER submits orders.
The existing observation scheduler processes pending signals using the exact
post-close D-1 contract. An already prepared or executed run is immutable.

Important: preparing a signal may allow the *existing* separately controlled
Live scheduler to execute it at the NEXT regular open when Live is ACTIVE.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .live_observation_scheduler import _post_close_daily_preparation_window
from ..engine.market_data import SAFE_DAILY_BAR_DELAY_MINUTES
from .live_winner_session import LIVE_RUNS, plan_next_live_session


def daily_recalculation_status(db: Any) -> dict[str, Any]:
    """Read-only progress for the canonical Live daily signal.

    The API always reports the actual broker observation and the next XNYS
    decision date while the market is open. It never implies that training
    has begun simply because the service is polling a broker clock.
    """
    control = live_control_status(db)
    base: dict[str, Any] = {
        "market": "live", "status": "unavailable", "can_request": False,
        "execution_session": None, "decision_date": None,
        "order_submission_enabled": False, "method": "previous_completed_session_daily",
        "reason": "unavailable",
        "observed_at": None, "market_close_at": None,
        "expected_daily_bar_check_at": None, "eligible_at": None,
        "run_status": None, "run_phase": None,
        "preparation_attempts": 0, "last_preparation_error_type": None,
        "signal_prepared_at": None,
    }
    # Preserve the operational guard: a paused/stopped Live does not attempt
    # to arm a signal or access brokerage credentials.
    if control.get("control_mode") not in {"active", "exit_only"}:
        return {**base, "status": "inactive", "reason": "live_not_active"}

    broker = create_live_readonly_client()
    account = broker.get_account()
    clock = broker.get_clock()
    account_id = str(getattr(account, "id", "") or "").strip()
    now = getattr(clock, "timestamp", None)
    next_open = getattr(clock, "next_open", None)
    if (not account_id or not isinstance(now, datetime) or now.tzinfo is None
        or not isinstance(next_open, datetime) or next_open.tzinfo is None):
        return {**base, "status": "unavailable", "reason": "broker_clock_unverified"}

    observed_at = now.astimezone(timezone.utc).isoformat()
    calendar = xcals.get_calendar("XNYS")
    target = (pd.Timestamp(next_open.astimezone(timezone.utc))
              .tz_convert("America/New_York").normalize().tz_localize(None))
    if not calendar.is_session(target):
        return {**base, "observed_at": observed_at, "status": "unavailable",
                "reason": "invalid_next_session"}

    session = target.date().isoformat()
    previous_session = calendar.previous_session(target)
    prior = previous_session.date().isoformat()
    closing = calendar.session_close(previous_session).tz_convert("UTC")
    expected_check = closing + pd.Timedelta(minutes=SAFE_DAILY_BAR_DELAY_MINUTES)
    ready, _, ready_at = _post_close_daily_preparation_window(
        session=session, broker_timestamp=now,
    )
    result: dict[str, Any] = {
        **base, "execution_session": session, "decision_date": prior,
        "observed_at": observed_at,
        "market_close_at": closing.isoformat(),
        # This is the first eligible *check*, not proof of data availability.
        "expected_daily_bar_check_at": expected_check.isoformat(),
        "eligible_at": ready_at,
    }
    run = db[LIVE_RUNS].find_one({"_id": f"{account_id}:{session}"}) or {}
    run_status = str(run.get("status") or "")
    result.update({
        "run_status": run_status or None,
        "run_phase": str(run.get("phase") or "") or None,
        "preparation_attempts": int(run.get("preparation_attempts") or 0),
        # Only exception type, never provider exception strings or account IDs.
        "last_preparation_error_type": str(
            run.get("last_preparation_error_type") or ""
        ) or None,
        "signal_prepared_at": (
            run["signal_prepared_at"].isoformat()
            if isinstance(run.get("signal_prepared_at"), datetime) else None
        ),
    })
    if run_status == "recalibrated_no_orders" and (
        run.get("winner_signal_prepared") is True
        and run.get("post_close_signal_ready") is True
        and str(run.get("decision_date") or "")[:10] == prior
    ):
        return {**result, "status": "already_prepared", "reason": "valid_daily_signal"}
    if run_status == "preparing_signal":
        return {**result, "status": "processing", "reason": "daily_model_processing"}
    # No processed or executed session may be reopened or replaced.
    if run_status not in {"", "waiting_for_winner_signal"}:
        return {**result, "status": "blocked",
                "reason": "existing_live_session_" + run_status}
    if bool(getattr(clock, "is_open", False)):
        # The current regular session has not closed; these session/date
        # fields describe the pending NEXT daily plan, not active training.
        return {**result, "status": "market_open", "reason": "wait_for_daily_close"}
    if ready == "waiting_for_daily_bar":
        return {**result, "status": "waiting_daily_bar", "reason": "daily_bar_delay"}
    if ready != "ready":
        return {**result, "status": "outside_window", "reason": "not_post_close_window"}
    if run.get("manual_daily_requested_at"):
        return {**result, "status": "queued",
                "reason": "scheduler_will_prepare_pending_daily_signal"}
    return {**result, "status": "ready", "can_request": True,
            "reason": "pending_daily_signal"}


def request_daily_recalculation(db: Any, *, actor: str) -> dict[str, Any]:
    """Arm missing daily preparation; scheduler does computation, never this POST."""
    if not actor:
        raise ValueError("Authenticated administrator is required.")
    readiness = daily_recalculation_status(db)
    if not readiness["can_request"]:
        return readiness
    # All order-executing and fully prepared states stay immutable.
    armed = plan_next_live_session(db)
    if armed.get("execution_session") != readiness["execution_session"]:
        raise RuntimeError("Live session changed while requesting daily preparation.")
    status = armed.get("status")
    if status != "waiting_for_winner_signal":
        return {
            **readiness, "status": "already_claimed", "can_request": False,
            "reason": "session_status_" + str(status),
        }
    # Bind the audit write to the same verified canonical Alpaca account,
    # never to another account's run with an identical execution date.
    broker = create_live_readonly_client()
    account_id = str(getattr(broker.get_account(), "id", "") or "").strip()
    if not account_id:
        raise RuntimeError("Live account identity is unavailable.")
    db[LIVE_RUNS].update_one(
        {"_id": f'{account_id}:{readiness["execution_session"]}',
         "status": "waiting_for_winner_signal",
         "winner_signal_prepared": False},
        {"$set": {
            "manual_daily_requested_at": datetime.now(timezone.utc),
            "manual_daily_requested_by": actor,
            "manual_daily_source": "administrator_live_panel",
        }},
    )
    return {
        **readiness, "status": "queued", "can_request": False,
        "reason": "scheduler_will_prepare_pending_daily_signal",
        "notice": "Only the pending D-1 signal will be prepared. No order is sent by this request.",
    }
