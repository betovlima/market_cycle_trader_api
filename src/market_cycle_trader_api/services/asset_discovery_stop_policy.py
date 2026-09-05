from __future__ import annotations

from typing import Any, Callable

from pymongo.database import Database

from . import asset_discovery as discovery


_INSTALLED = False
_ORIGINAL_STOP: Callable[[Database], dict[str, Any]] | None = None
_CHILD_ACTIVE_STATUSES = frozenset({"queued", "running", "stopping"})


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _stop_asset_discovery_with_child_operations(db: Database) -> dict[str, Any]:
    original = _ORIGINAL_STOP
    if original is None:
        raise RuntimeError("Asset Discovery stop policy is not installed.")

    document = discovery._campaign(db)
    if not isinstance(document, dict):
        return discovery.get_asset_discovery_status(db)

    campaign_status = _status(document.get("status"))
    marginal = document.get("marginal_replay") if isinstance(document.get("marginal_replay"), dict) else {}
    validation = document.get("full_strategy_validation") if isinstance(document.get("full_strategy_validation"), dict) else {}
    marginal_status = _status(marginal.get("status"))
    validation_status = _status(validation.get("status"))

    # The original stop path remains authoritative while the main Discovery
    # worker itself is active. Child operations need their own stop handling
    # because the parent campaign may already be completed/interrupted.
    if campaign_status in discovery.ACTIVE_STATUSES:
        return original(db)

    child_updates: dict[str, Any] = {}
    if marginal_status in _CHILD_ACTIVE_STATUSES:
        child_updates.update({
            "marginal_replay.status": "stopping",
            "marginal_replay.current_stage": "Stop requested; finishing the active replay batch",
        })
    if validation_status in _CHILD_ACTIVE_STATUSES:
        child_updates.update({
            "full_strategy_validation.status": "stopping",
        })

    if not child_updates:
        return discovery.get_asset_discovery_status(db)

    run_id = str(document.get("run_id") or "").strip()
    now = discovery.utc_now()
    child_updates.update({
        "cancel_requested": True,
        "stop_requested_at": now,
        "message": "Stop requested. Active child processing will stop at the next safe checkpoint.",
        "updated_at": now,
    })
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {
            "$set": discovery.bson_value(child_updates),
            "$push": {
                "events": {
                    "$each": [{
                        "at": now,
                        "message": "Stop requested for active Marginal Replay/validation; no new batch will start.",
                    }],
                    "$slice": -24,
                }
            },
        },
    )
    return discovery.get_asset_discovery_status(db)


def install_asset_discovery_stop_policy() -> None:
    global _INSTALLED, _ORIGINAL_STOP
    if _INSTALLED:
        return
    original = discovery.stop_asset_discovery
    if getattr(original, "_asset_discovery_child_stop_policy", False):
        _INSTALLED = True
        return
    _ORIGINAL_STOP = original
    setattr(_stop_asset_discovery_with_child_operations, "_asset_discovery_child_stop_policy", True)
    discovery.stop_asset_discovery = _stop_asset_discovery_with_child_operations
    _INSTALLED = True
