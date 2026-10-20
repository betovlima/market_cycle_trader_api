"""Bounded, broker-account-scoped Winner Live diagnostics.

Records only curated lifecycle metadata. No credentials, exception messages,
account balances, broker payloads, or order operations are captured.
Instrumentation must never change the trading state machine.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from typing import Any

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_winner_session import LIVE_RUNS

LIVE_EVENTS = "live_winner_process_events"
_LOGGER = logging.getLogger(__name__)
_LEVELS = frozenset({"info", "warning", "error"})
_DETAILS = frozenset({"assets", "cutoff", "target_asset", "action", "error_type", "error_code", "stage", "retry_allowed", "symbol", "feed", "duration_seconds", "retry_after"})


def append_live_event(
    db: Any, *, account_id: str, session: str, stage: str,
    message: str, level: str = "info", details: dict[str, Any] | None = None,
) -> None:
    """Best-effort, non-executable audit event with an intentionally small schema."""
    if not account_id or not session:
        return
    safe_details = {}
    for key, value in (details or {}).items():
        if key in _DETAILS and isinstance(value, (str, int, float, bool)) and not isinstance(value, complex):
            safe_details[key] = value if not isinstance(value, str) else value[:100]
    doc = {
        "account_id": str(account_id), "execution_session": str(session),
        "recorded_at": datetime.now(timezone.utc),
        "stage": str(stage)[:64], "level": level if level in _LEVELS else "info",
        "message": str(message)[:160], "details": safe_details,
    }
    try:
        db[LIVE_EVENTS].insert_one(doc)
    except Exception:
        # Loss of diagnostics must never cancel or modify a Winner decision.
        _LOGGER.warning("Unable to persist a Live Winner lifecycle event", exc_info=False)


def list_live_events(
    db: Any, *, session: str | None = None, level: str = "all", limit: int = 100,
) -> dict[str, Any]:
    """Query the authenticated operator's *current Live broker account* only."""
    broker = create_live_readonly_client()
    account_id = str(getattr(broker.get_account(), "id", "") or "")
    if not account_id:
        raise RuntimeError("Live broker account identity is unavailable.")
    if level not in _LEVELS | {"all"}:
        raise ValueError("Unsupported Live event severity.")
    if session is None:
        latest = db[LIVE_RUNS].find_one(
            {"live_account_id": account_id}, sort=[("execution_session", -1)],
        )
        session = str(latest["execution_session"]) if latest else None
    if not session:
        return {"execution_session": None, "events": [], "limit": min(max(limit, 1), 200)}
    match = {"account_id": account_id, "execution_session": session}
    if level != "all":
        match["level"] = level
    bounded = min(max(int(limit), 1), 200)
    cursor = db[LIVE_EVENTS].find(match, {"_id": 0, "account_id": 0}).sort("recorded_at", -1).limit(bounded)
    events = []
    for doc in cursor:
        stamp = doc.get("recorded_at")
        events.append({
            "recorded_at": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp or ""),
            "execution_session": doc.get("execution_session"),
            "stage": doc.get("stage"),
            "level": doc.get("level"),
            "message": doc.get("message"),
            "details": doc.get("details") or {},
        })
    events.reverse()
    return {"execution_session": session, "events": events, "limit": bounded}


def export_live_events(db: Any, *, session: str | None = None) -> bytes:
    """Provide a session-scoped, sanitized ZIP for operational support."""
    payload = list_live_events(db, session=session, limit=200)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("live_winner_events.json", json.dumps(payload, ensure_ascii=False, indent=2))
        archive.writestr(
            "README.txt",
            "Winner Live lifecycle events only; latest 200 events for the selected "
            "account and session. Earlier Railway logs cannot be reconstructed. "
            "No broker orders are submitted.\n",
        )
    return output.getvalue()
