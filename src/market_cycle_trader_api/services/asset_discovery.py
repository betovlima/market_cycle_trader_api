from __future__ import annotations

import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    ASSET_DISCOVERY_RUNS_COLLECTION,
    ASSET_DISCOVERY_STATE_COLLECTION,
    utc_now,
)
from .asset_discovery_behavior import ASSET_DISCOVERY_EVALUATION_POLICY_VERSION
from .asset_discovery_settings import (
    AssetDiscoveryConflict,
    EASTERN,
    get_asset_discovery_settings,
)
from .asset_discovery_store import (
    ACTIVE_KEY,
    append_run_update,
    candidate_counts,
    public_run,
)
from .asset_discovery_worker import STATE_ID, run_asset_discovery_worker

_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def asset_discovery_status(db: Any) -> dict[str, Any]:
    active = db[ASSET_DISCOVERY_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    latest = active or db[ASSET_DISCOVERY_RUNS_COLLECTION].find_one(
        {}, sort=[("created_at", -1)]
    )
    state = db[ASSET_DISCOVERY_STATE_COLLECTION].find_one({"_id": STATE_ID}) or {}
    return {
        "settings": get_asset_discovery_settings(db),
        "run": public_run(latest),
        "counts": candidate_counts(db),
        "cursor_symbol": state.get("cursor_symbol"),
        "last_automatic_slot": state.get("last_automatic_slot"),
    }


def start_asset_discovery(
    db: Any,
    *,
    source: str,
    actor_email: str | None = None,
) -> dict[str, Any]:
    global _WORKER_THREAD
    now = utc_now()
    run_id = f"asset-discovery-{uuid.uuid4().hex[:12]}"
    settings = get_asset_discovery_settings(db)
    document = {
        "run_id": run_id,
        "active_key": ACTIVE_KEY,
        "source": "automatic" if source == "automatic" else "manual",
        "api_version": API_VERSION,
        "evaluation_policy_version": ASSET_DISCOVERY_EVALUATION_POLICY_VERSION,
        "status": "queued",
        "phase": "queued",
        "created_at": now,
        "updated_at": now,
        "requested_by": str(actor_email or "").strip().lower() or None,
        "batch_size": int(settings["batch_size"]),
        "processed_count": 0,
        "attempted_count": 0,
        "candidate_count": 0,
        "watchlist_count": 0,
        "rejected_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "cancel_requested": False,
        "last_message": "Asset Discovery queued.",
        "logs": [f"{now.isoformat()} — Asset Discovery queued."],
    }
    try:
        db[ASSET_DISCOVERY_RUNS_COLLECTION].insert_one(document)
    except DuplicateKeyError as exc:
        active = db[ASSET_DISCOVERY_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
        active_id = str((active or {}).get("run_id") or "unknown")
        raise AssetDiscoveryConflict(
            f"Another Asset Discovery run is already active: {active_id}."
        ) from exc

    with _WORKER_LOCK:
        _STOP_EVENT.clear()
        _WORKER_THREAD = threading.Thread(
            target=run_asset_discovery_worker,
            args=(db, run_id, _STOP_EVENT),
            name="asset-discovery-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()
    return public_run(document) or {}


def stop_asset_discovery(db: Any) -> dict[str, Any]:
    active = db[ASSET_DISCOVERY_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    if active is None:
        return asset_discovery_status(db)
    _STOP_EVENT.set()
    append_run_update(
        db,
        str(active["run_id"]),
        message="Stop requested. The current safe unit will finish before the worker exits.",
        changes={"status": "stopping", "phase": "stopping", "cancel_requested": True},
    )
    return asset_discovery_status(db)


def automatic_slot_due(db: Any) -> str | None:
    settings = get_asset_discovery_settings(db)
    if not settings["automatic_enabled"]:
        return None
    now_et = utc_now().astimezone(EASTERN)
    if now_et.weekday() >= 5 or now_et.hour not in settings["schedule_hours_et"]:
        return None
    slot = now_et.strftime("%Y-%m-%dT%H")
    state = db[ASSET_DISCOVERY_STATE_COLLECTION].find_one({"_id": STATE_ID}) or {}
    return None if state.get("last_automatic_slot") == slot else slot


def mark_automatic_slot(db: Any, slot: str) -> None:
    db[ASSET_DISCOVERY_STATE_COLLECTION].update_one(
        {"_id": STATE_ID},
        {"$set": {"last_automatic_slot": slot, "updated_at": utc_now()}},
        upsert=True,
    )
