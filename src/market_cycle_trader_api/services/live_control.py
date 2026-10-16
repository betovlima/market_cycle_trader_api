"""Live Winner operational controls. Start/Pause/Exit only/Stop govern Alpaca Live."""
from __future__ import annotations

from typing import Any

from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import utc_now
from ..infrastructure.trading.alpaca_live import live_credentials

LIVE_CONTROL_COLLECTION = "live_trader_control"
LIVE_CONTROL_HISTORY_COLLECTION = "live_trader_control_history"
VALID_MODES = frozenset({"active", "paused", "exit_only", "stopped"})
CONTROL_ID = "default"


class LiveControlConflict(RuntimeError):
    pass


def _current(db: Any) -> dict[str, Any]:
    # $setOnInsert ensures never writing into Paper collections.
    db[LIVE_CONTROL_COLLECTION].update_one(
        {"_id": CONTROL_ID},
        {"$setOnInsert": {
            "control_mode": "stopped",
            "revision": 0,
            "updated_at": utc_now(),
            "updated_by": "system",
            "reason": "Live initial state is stopped; order execution disabled.",
        }},
        upsert=True,
    )
    return db[LIVE_CONTROL_COLLECTION].find_one({"_id": CONTROL_ID})


def live_control_status(db: Any) -> dict[str, Any]:
    state = _current(db)
    mode = str(state.get("control_mode") or "stopped")
    return {
        "control_mode": mode,
        "revision": int(state.get("revision") or 0),
        "status": mode,
        "phase": "live_operational" if mode in {"active", "exit_only"} else "live_inactive",
        "enabled": mode in {"active", "exit_only"},
        "monitoring_enabled": mode in {"active", "exit_only"},
        "execution_enabled": mode in {"active", "exit_only"},
        "scheduler_alive": False,
        "order_submission_enabled": mode in {"active", "exit_only"},
        "last_control_update_at": state.get("updated_at"),
        "last_control_updated_by": state.get("updated_by"),
        "last_control_reason": state.get("reason"),
    }


def set_live_control_mode(
    db: Any, *, mode: str, expected_revision: int,
    actor_email: str, reason: str,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError("Invalid Live mode.")
    if not actor_email:
        raise ValueError("An authenticated administrator is required.")
    if mode in {"active", "exit_only"}:
        # Validate presence of an independent Live account configuration.
        # Never look up Paper API keys as a fallback.
        live_credentials()
    previous = _current(db)
    if int(previous.get("revision", 0)) != int(expected_revision):
        raise LiveControlConflict("Live mode revision changed. Refresh and retry.")
    if previous.get("control_mode") == mode:
        return live_control_status(db)
    now = utc_now()
    latest = db[LIVE_CONTROL_COLLECTION].find_one_and_update(
        {
            "_id": CONTROL_ID,
            "revision": expected_revision,
            "control_mode": previous["control_mode"],
        },
        {"$set": {
            "control_mode": mode,
            "updated_at": now,
            "updated_by": actor_email,
            "reason": reason[:500],
        }, "$inc": {"revision": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if latest is None:
        raise LiveControlConflict("Live mode changed concurrently. Refresh and retry.")
    db[LIVE_CONTROL_HISTORY_COLLECTION].insert_one({
        "created_at": now, "updated_by": actor_email,
        "old_mode": previous["control_mode"],
        "new_mode": mode,
        "revision": int(latest["revision"]),
        "reason": reason[:500],
        "execution_enabled": mode in {"active", "exit_only"},
        "market": "live",
    })
    return live_control_status(db)
