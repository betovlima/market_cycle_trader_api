"""Independent Live observation worker.

This is a first isolation step, NOT a Winner plan generator or order executor.
No paper scheduler imports, paper collections or broker write methods.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from ..core.runtime import database
from .live_control import live_control_status
from .live_portfolio import live_portfolio_snapshot
from .strategy_lab import get_trader_winner_summary
from .live_winner_session import plan_next_live_session, LIVE_RUNS
from .live_winner_events import append_live_event
from .live_winner_signal import prepare_live_cash_winner_signal
from .live_winner_recalibration import recalibrate_live_cash_winner_signal
from .live_order_execution import execute_recalibrated_live_winner_session

LIVE_OBSERVATIONS = "live_trader_session_observations"
_POLL_SECONDS = 60.0


def observe_live_cycle(db: Any) -> dict[str, Any]:
    """Record a bounded, independent Live observation only when monitoring is active."""
    state = live_control_status(db)
    if state["control_mode"] not in {"active", "exit_only"}:
        return {"status": "skipped", "reason": "live_control_not_active"}
    snapshot = live_portfolio_snapshot(db)
    winner = get_trader_winner_summary(db)
    now = datetime.now(timezone.utc)
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
                    if armed.get("status") == "waiting_for_winner_signal" and not result["market_open"]:
                        stage = "signal_preparation"
                        prepare_live_cash_winner_signal(db, armed["execution_session"])
                    elif armed.get("status") == "analysis_only_requested" and result["market_open"]:
                        stage = "late_intraday_analysis"
                        recalibrate_live_cash_winner_signal(db, armed["execution_session"])
                    elif armed.get("status") == "signal_prepared" and result["market_open"]:
                        stage = "market_open_recalibration"
                        recalibrated = recalibrate_live_cash_winner_signal(db, armed["execution_session"])
                        if recalibrated.get("status") == "recalibrated_no_orders":
                            stage = "order_execution"
                            execute_recalibrated_live_winner_session(db, armed["execution_session"])
                    elif armed.get("status") == "recalibrated_no_orders" and result["market_open"]:
                        stage = "order_execution"
                        execute_recalibrated_live_winner_session(db, armed["execution_session"])
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
