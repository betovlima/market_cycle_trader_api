from __future__ import annotations

from typing import Any, Callable

from pymongo.database import Database

from . import asset_discovery as discovery
from .asset_discovery_marginal_process import terminate_marginal_process


_INSTALLED = False
_ORIGINAL_STOP: Callable[[Database], dict[str, Any]] | None = None
_ACTIVE = frozenset({"queued", "running", "stopping"})


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _predictive_optional_replay(document: dict[str, Any]) -> bool:
    return str(document.get("discovery_mode") or "").strip().lower() == "predictive_only"


def _stop_with_immediate_marginal_termination(db: Database) -> dict[str, Any]:
    original = _ORIGINAL_STOP
    if original is None:
        raise RuntimeError("Immediate Asset Discovery stop policy is not installed.")

    document = discovery._campaign(db)
    if not isinstance(document, dict):
        return discovery.get_asset_discovery_status(db)

    marginal = document.get("marginal_replay") if isinstance(document.get("marginal_replay"), dict) else {}
    if _status(marginal.get("status")) not in _ACTIVE:
        return original(db)

    run_id = str(document.get("run_id") or "").strip()
    predictive_optional = _predictive_optional_replay(document)
    now = discovery.utc_now()
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {
            "$set": {
                "cancel_requested": True,
                "status": "stopping",
                "phase": "marginal_replay",
                "marginal_replay.status": "stopping",
                "marginal_replay.current_stage": "Stopping optional Marginal Capital Replay now" if predictive_optional else "Stopping Marginal Capital Replay now",
                "stop_requested_at": now,
                "updated_at": now,
                "message": "Stopping optional Marginal Capital Replay now." if predictive_optional else "Stopping Marginal Capital Replay now.",
            },
            "$push": {
                "events": {
                    "$each": [{"at": now, "message": "Immediate stop requested for optional Marginal Capital Replay." if predictive_optional else "Immediate stop requested for Marginal Capital Replay."}],
                    "$slice": -24,
                }
            },
        },
    )

    terminated = terminate_marginal_process(run_id, timeout_seconds=2.0)
    worker = discovery._worker_thread
    if terminated and worker is not None and worker.is_alive():
        worker.join(timeout=2.0)

    finished_at = discovery.utc_now()
    campaign_status = "completed" if predictive_optional else "stopped"
    campaign_phase = "completed" if predictive_optional else "stopped"
    message = (
        "Optional Marginal Capital Replay stopped; predictive candidates were preserved."
        if predictive_optional
        else "Marginal Capital Replay stopped. A new Asset Discovery run can be started."
    )
    update_set: dict[str, Any] = {
        "cancel_requested": False if predictive_optional else True,
        "status": campaign_status,
        "phase": campaign_phase,
        "marginal_replay.status": "stopped",
        "marginal_replay.current_stage": "Optional Marginal Capital Replay stopped" if predictive_optional else "Stopped by user",
        "worker_active": False,
        "worker_finished_at": finished_at,
        "updated_at": finished_at,
        "message": message,
    }
    if predictive_optional:
        update_set.update({
            "progress_step": "completed",
            "stage_progress_percent": 100.0,
            "current_stage": "Predictive Asset Discovery completed",
        })

    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {
            "$set": update_set,
            "$unset": {"worker_process_id": ""},
            "$push": {
                "events": {
                    "$each": [{"at": finished_at, "message": message}],
                    "$slice": -24,
                }
            },
        },
    )
    return discovery.get_asset_discovery_status(db)


def install_asset_discovery_immediate_stop() -> None:
    global _INSTALLED, _ORIGINAL_STOP
    if _INSTALLED:
        return
    current = discovery.stop_asset_discovery
    if getattr(current, "_asset_discovery_immediate_stop", False):
        _INSTALLED = True
        return
    _ORIGINAL_STOP = current
    setattr(_stop_with_immediate_marginal_termination, "_asset_discovery_immediate_stop", True)
    discovery.stop_asset_discovery = _stop_with_immediate_marginal_termination
    _INSTALLED = True
