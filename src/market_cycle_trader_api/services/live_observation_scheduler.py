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
from .live_winner_session import plan_next_live_session

LIVE_OBSERVATIONS = "live_trader_session_observations"
_POLL_SECONDS = 300.0


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
            try:
                db = database()
                result = observe_live_cycle(db)
                if result["status"] == "observed":
                    plan_next_live_session(db)
                if result["status"] == "observed":
                    self._last_observed_at = result["recorded_at"]
                self._last_error = None
            except Exception as exc:
                self._last_error = type(exc).__name__
            self._stop.wait(_POLL_SECONDS)


_SCHEDULER = LiveObservationScheduler()


def start_live_observation_scheduler() -> None:
    _SCHEDULER.start()


def stop_live_observation_scheduler() -> None:
    _SCHEDULER.stop()


def live_observation_status() -> dict[str, Any]:
    return _SCHEDULER.snapshot()
