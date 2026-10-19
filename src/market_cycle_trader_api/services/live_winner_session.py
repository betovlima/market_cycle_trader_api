"""Independent next-session arming for the Live Winner.

Records a durable schedule and winner identity. It does NOT train a model,
calculate a signal or submit an order. The scheduler is strictly read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .strategy_lab import get_trader_winner_summary

LIVE_RUNS = "live_winner_session_runs"
EASTERN = ZoneInfo("America/New_York")


def plan_next_live_session(db: Any) -> dict[str, Any]:
    control = live_control_status(db)
    if control["control_mode"] not in {"active", "exit_only"}:
        return {"status": "skipped", "reason": "live_not_active"}
    broker = create_live_readonly_client()
    account = broker.get_account()
    clock = broker.get_clock()
    winner = get_trader_winner_summary(db)
    if not isinstance(winner, dict) or not winner.get("id") or not winner.get("configuration_hash"):
        raise RuntimeError("A versioned Winner must be selected before arming Live.")
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("Live account ID is missing.")
    timestamp = getattr(clock, "timestamp", None)
    next_open = getattr(clock, "next_open", None)
    if not timestamp or not next_open:
        raise RuntimeError("Live broker clock is incomplete.")
    if next_open.tzinfo is None or timestamp.tzinfo is None:
        raise RuntimeError("Live broker timestamps must be timezone-aware.")
    # During an open session, preserve the current session rather than arming
    # tomorrow merely because Alpaca next_open already points forward.
    if bool(getattr(clock, "is_open", False)):
        session = timestamp.astimezone(EASTERN).date().isoformat()
        expected_open = timestamp.astimezone(EASTERN).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    else:
        session = next_open.astimezone(EASTERN).date().isoformat()
        expected_open = next_open
    now = datetime.now(timezone.utc)
    identity = {
        "live_account_id": account_id,
        "winner_strategy_id": str(winner["id"]),
        "winner_revision": int(winner.get("revision") or 0),
        "winner_configuration_hash": str(winner["configuration_hash"]),
    }
    doc = {
        "market": "live",
        "execution_session": session,
        "expected_market_open": expected_open,
        "status": "waiting_for_winner_signal",
        "phase": "session_armed_without_model_decision",
        "winner_name": winner.get("name"),
        "decision_date": None,
        "action": None,
        "target_asset": None,
        "winner_signal_prepared": False,
        "order_submission_enabled": False,
        "updated_at": now,
        **identity,
    }
    db[LIVE_RUNS].update_one(
        {"_id": f"{account_id}:{session}"},
        {"$setOnInsert": {**doc, "created_at": now}},
        upsert=True,
    )
    saved = db[LIVE_RUNS].find_one({"_id": f"{account_id}:{session}"}) or {}
    # Fail closed if a Winner change occurs after arming.
    if any(saved.get(key) != value for key, value in identity.items()):
        db[LIVE_RUNS].update_one(
            {"_id": f"{account_id}:{session}"},
            {"$set": {"status": "review_required", "phase": "winner_identity_changed",
                       "order_submission_enabled": False, "updated_at": now}},
        )
        raise RuntimeError("Winner identity changed since Live session was armed.")
    return {
        "status": saved.get("status", doc["status"]),
        "execution_session": session,
        "next_market_open": expected_open.isoformat(),
        "winner_name": saved.get("winner_name"),
        "phase": saved.get("phase", doc["phase"]),
        "winner_signal_prepared": bool(saved.get("winner_signal_prepared")),
        "target_asset": saved.get("target_asset"),
        "action": saved.get("action"),
        "order_submission_enabled": False,
    }


def latest_live_session(db: Any) -> dict[str, Any] | None:
    doc = db[LIVE_RUNS].find_one({}, sort=[("created_at", -1)])
    if not doc:
        return None
    return {
        "execution_session": doc.get("execution_session"),
        "expected_market_open": (
            doc["expected_market_open"].isoformat()
            if hasattr(doc.get("expected_market_open"), "isoformat")
            else None
        ),
        "status": doc.get("status"),
        "phase": doc.get("phase"),
        "winner_name": doc.get("winner_name"),
        "winner_signal_prepared": bool(doc.get("winner_signal_prepared")),
        "decision_date": doc.get("decision_date"),
        "action": doc.get("action"),
        "target_asset": doc.get("target_asset"),
        "calibration_duration_seconds": doc.get("calibration_duration_seconds"),
        "window_seconds": doc.get("window_seconds"),
        "window_deadline_at": doc["window_deadline_at"].isoformat() if hasattr(doc.get("window_deadline_at"), "isoformat") else None,
        "window_expired_at": doc["window_expired_at"].isoformat() if hasattr(doc.get("window_expired_at"), "isoformat") else None,
        "last_calibration_error_type": doc.get("last_calibration_error_type"),
        "last_calibration_error_code": doc.get("last_calibration_error_code"),
        "last_intraday_error_symbol": doc.get("last_intraday_error_symbol"),
        "last_intraday_error_feed": doc.get("last_intraday_error_feed"),
        "retry_not_before": (
            doc["retry_not_before"].isoformat()
            if hasattr(doc.get("retry_not_before"), "isoformat")
            else None
        ),
        "order_submission_enabled": False,
    }
