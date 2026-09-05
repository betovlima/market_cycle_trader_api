from __future__ import annotations

import os
import socket
import threading
from typing import Any, Callable
from uuid import uuid4

from . import asset_discovery


_INSTALLED = False
_ORIGINAL_RUN_EXISTING_MARGINAL_WORKER: Callable[..., Any] | None = None


def _run_existing_marginal_worker_with_heartbeat(db: Any, run_id: str) -> None:
    original = _ORIGINAL_RUN_EXISTING_MARGINAL_WORKER
    if original is None:
        raise RuntimeError("Asset Discovery marginal replay heartbeat is not installed.")

    worker_id = f"{socket.gethostname()}:{os.getpid()}:marginal:{uuid4().hex[:8]}"
    heartbeat_stop = threading.Event()
    now = asset_discovery.utc_now()
    db[asset_discovery.COLLECTION].update_one(
        {"_id": asset_discovery.CURRENT_ID, "run_id": run_id},
        {"$set": {
            "worker_id": worker_id,
            "worker_active": True,
            "worker_started_at": now,
            "worker_finished_at": None,
            "worker_heartbeat_at": now,
            "updated_at": now,
        }},
    )

    heartbeat_thread = threading.Thread(
        target=asset_discovery._heartbeat_worker,
        args=(db, run_id, worker_id, heartbeat_stop),
        name="asset-discovery-marginal-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        original(db, run_id)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        finished_at = asset_discovery.utc_now()
        db[asset_discovery.COLLECTION].update_one(
            {"_id": asset_discovery.CURRENT_ID, "run_id": run_id, "worker_id": worker_id},
            {"$set": {
                "worker_active": False,
                "worker_finished_at": finished_at,
                "worker_heartbeat_at": finished_at,
                "updated_at": finished_at,
            }},
        )


def install_asset_discovery_marginal_heartbeat() -> None:
    global _INSTALLED, _ORIGINAL_RUN_EXISTING_MARGINAL_WORKER
    if _INSTALLED:
        return

    original = asset_discovery._run_existing_marginal_worker
    if getattr(original, "_asset_discovery_marginal_heartbeat", False):
        _INSTALLED = True
        return

    _ORIGINAL_RUN_EXISTING_MARGINAL_WORKER = original
    setattr(_run_existing_marginal_worker_with_heartbeat, "_asset_discovery_marginal_heartbeat", True)
    asset_discovery._run_existing_marginal_worker = _run_existing_marginal_worker_with_heartbeat
    _INSTALLED = True
