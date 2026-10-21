"""Independent Live observation worker.

This is a first isolation step, NOT a Winner plan generator or order executor.
No paper scheduler imports, paper collections or broker write methods.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from ..core.runtime import database
from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .live_portfolio import live_portfolio_snapshot
from .strategy_lab import get_trader_winner_summary
from .live_winner_session import plan_next_live_session, LIVE_RUNS
from .live_winner_events import append_live_event
from .live_winner_signal import prepare_live_winner_signal
from .live_order_execution import execute_prepared_live_winner_session
from .live_workflow_monitor import execute_monitored_live_stage

LIVE_OBSERVATIONS = "live_trader_session_observations"
_POLL_SECONDS = 60.0


def _regular_xnys_session_open(
    *, session: str, broker_timestamp: Any, broker_is_open: bool,
    broker_next_close: Any = None,
) -> bool:
    """Fail-closed regular-session gate; no after-hours Live calculation/orders.

    Use the Alpaca clock *and* the XNYS calendar, never host local time.
    Early closes come from XNYS; an earlier valid broker next_close can only
    shorten (never extend) the approved execution window.
    """
    if not broker_is_open or not broker_timestamp:
        return False
    try:
        now = pd.Timestamp(broker_timestamp)
        if pd.isna(now) or now.tzinfo is None:
            return False
        now = now.tz_convert("UTC")
        requested = pd.Timestamp(session)
        if requested.tzinfo is not None:
            return False
        calendar = xcals.get_calendar("XNYS")
        if not calendar.is_session(requested):
            return False
        if now.tz_convert("America/New_York").date() != requested.date():
            return False
        opened = calendar.session_open(requested)
        closed = calendar.session_close(requested)
        if broker_next_close is not None:
            close = pd.Timestamp(broker_next_close)
            if (close.tzinfo is not None
                and close.tz_convert("America/New_York").date() == requested.date()
                and close.tz_convert("UTC") > opened):
                closed = min(closed, close.tz_convert("UTC"))
        return bool(opened <= now < closed)
    except (TypeError, ValueError, OverflowError):
        # Incomplete/unparseable provider timestamps must never arm real orders.
        return False


def _fresh_regular_live_execution_window(session: str) -> bool:
    """Fresh read-only Alpaca clock after lengthy LightGBM computation."""
    broker = create_live_readonly_client()
    clock = broker.get_clock()
    return _regular_xnys_session_open(
        session=session,
        broker_timestamp=getattr(clock, "timestamp", None),
        broker_is_open=bool(getattr(clock, "is_open", False)),
        broker_next_close=getattr(clock, "next_close", None),
    )



def observe_live_cycle(db: Any) -> dict[str, Any]:
    """Record a bounded, independent Live observation only when monitoring is active."""
    state = live_control_status(db)
    if state["control_mode"] not in {"active", "exit_only"}:
        return {"status": "skipped", "reason": "live_control_not_active"}
    snapshot = live_portfolio_snapshot(db)
    winner = get_trader_winner_summary(db)
    broker_stamp = snapshot["market_clock"].get("timestamp")
    if not broker_stamp:
        raise RuntimeError("Live broker timestamp missing from portfolio snapshot.")
    now = datetime.fromisoformat(str(broker_stamp))
    if now.tzinfo is None:
        raise RuntimeError("Live broker timestamp must include timezone.")
    now = now.astimezone(timezone.utc)
    # Winner metadata is informational: the Paper decision is not silently reused.
    document = {
        "recorded_at": now,
        "market": "live",
        "control_mode": state["control_mode"],
        "account_status": snapshot["account_status"],
        "equity": snapshot["portfolio_value"],
        "cash": snapshot["strategy_cash"],
        "position_symbols": sorted(str(p["symbol"]) for p in snapshot["positions"]),
        "market_open": snapshot["market_clock"]["is_open"],
        "winner_name": winner.get("name") if isinstance(winner, dict) else None,
        "phase": "read_only_observation",
        "order_submission_enabled": False,
        "winner_signal_prepared": False,
    }
    db[LIVE_OBSERVATIONS].insert_one(document.copy())
    return {**document, "recorded_at": now.isoformat(), "status": "observed"}


class LiveObservationScheduler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._last_error_fingerprint: str | None = None
        self._last_observed_at: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alpaca-live-observer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "scheduler_alive": bool(self._thread and self._thread.is_alive()),
            "last_observed_at": self._last_observed_at,
            "last_error": self._last_error,
            "poll_seconds": _POLL_SECONDS,
            "phase": "read_only_observation",
            "order_submission_enabled": False,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            db = None
            armed = None
            stage = "observation"
            try:
                db = database()
                result = observe_live_cycle(db)
                if result["status"] == "observed":
                    self._last_observed_at = result["recorded_at"]
                    stage = "session_planning"
                    armed = plan_next_live_session(db)
                    if (not result["market_open"] and armed.get("status") in {
                        "waiting_for_winner_signal", "recalibrated_no_orders",
                        "preparing_signal",
                    }):
                        stage = "signal_preparation"
                        execute_monitored_live_stage(
                            db, session=armed["execution_session"],
                            stage="post_close_calculation",
                            operation=lambda: prepare_live_winner_signal(
                                db, armed["execution_session"]
                            ),
                        )
                    elif (armed.get("status") in {
                        "waiting_for_winner_signal", "recalibrated_no_orders",
                        "waiting_for_open_recalibration",
                    } and _regular_xnys_session_open(
                        session=armed["execution_session"],
                        broker_timestamp=result["recorded_at"],
                        broker_is_open=result["market_open"],
                    )):
                        session = armed["execution_session"]
                        run = db[LIVE_RUNS].find_one(
                            {"execution_session": session}
                        ) or {}
                        previous = xcals.get_calendar("XNYS").previous_session(
                            pd.Timestamp(session)
                        ).date().isoformat()
                        prepared = (
                            run.get("status") == "recalibrated_no_orders"
                            and bool(run.get("winner_signal_prepared"))
                            and bool(run.get("post_close_signal_ready"))
                            and bool(run.get("snapshot_sha256"))
                            and bool(run.get("target_asset"))
                            and str(run.get("decision_date") or "")[:10] == previous
                            and run.get("historical_cutoff") in (None, previous)
                            and run.get("analysis_mode") in (
                                None, "previous_completed_session_daily",
                            )
                        )
                        # Only missing/stale D-1 preparation is retried.
                        # A ready D-1 signal NEVER triggers market data
                        # downloads, intraday bars or LightGBM recalibration.
                        if not prepared:
                            stage = "missing_daily_signal_recovery"
                            recovery = execute_monitored_live_stage(
                                db, session=session, stage="open_daily_recovery",
                                operation=lambda: prepare_live_winner_signal(
                                    db, session, allow_open_recovery=True,
                                ),
                            )
                            prepared = recovery.get("status") == "recalibrated_no_orders"
                        if prepared and _fresh_regular_live_execution_window(session):
                            stage = "order_execution"
                            execute_monitored_live_stage(
                                db, session=session,
                                stage="broker_execution",
                                operation=lambda: execute_prepared_live_winner_session(
                                    db, session,
                                ),
                                brokerage=True,
                            )
                self._last_error = None
                self._last_error_fingerprint = None
            except Exception as exc:
                fingerprint = stage + ":" + type(exc).__name__
                if db is not None and isinstance(armed, dict) and armed.get("execution_session") and self._last_error_fingerprint != fingerprint:
                    try:
                        run = db[LIVE_RUNS].find_one({"execution_session": armed["execution_session"]})
                        if run and run.get("live_account_id"):
                            append_live_event(db, account_id=run["live_account_id"],
                                              session=armed["execution_session"],
                                              stage="scheduler_failed", level="error",
                                              message="Falha no scheduler durante: " + stage,
                                              details={"error_type": type(exc).__name__})
                    except Exception:
                        pass  # Scheduler diagnostics must not hide the original failure.
                self._last_error = type(exc).__name__
                self._last_error_fingerprint = fingerprint
            self._stop.wait(_POLL_SECONDS)


_SCHEDULER = LiveObservationScheduler()


def start_live_observation_scheduler() -> None:
    _SCHEDULER.start()


def stop_live_observation_scheduler() -> None:
    _SCHEDULER.stop()


def live_observation_status() -> dict[str, Any]:
    return _SCHEDULER.snapshot()
