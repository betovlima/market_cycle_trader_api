from __future__ import annotations

import os
import socket
import threading
from typing import Any, Callable
from uuid import uuid4


_INSTALLED = False
_ORIGINAL_EXISTING_WORKER: Callable[..., Any] | None = None
_ORIGINAL_HEARTBEAT_FRESH: Callable[..., bool] | None = None


def install_asset_discovery_marginal_replay_heartbeat() -> None:
    global _INSTALLED, _ORIGINAL_EXISTING_WORKER, _ORIGINAL_HEARTBEAT_FRESH
    if _INSTALLED:
        return

    from . import asset_discovery as service

    current_worker = service._run_existing_marginal_worker
    if getattr(current_worker, "_asset_discovery_marginal_replay_heartbeat", False):
        _INSTALLED = True
        return

    _ORIGINAL_EXISTING_WORKER = current_worker
    _ORIGINAL_HEARTBEAT_FRESH = service._worker_heartbeat_fresh

    def heartbeat_fresh_with_local_worker_guard(document: dict[str, Any] | None) -> bool:
        # A local worker that is still alive is authoritative during the very short
        # interval before its heartbeat thread publishes the first MongoDB heartbeat.
        if service._worker_alive():
            return True
        original = _ORIGINAL_HEARTBEAT_FRESH
        return bool(original(document)) if original is not None else False

    def marginal_worker_with_heartbeat(db: Any, run_id: str) -> None:
        original = _ORIGINAL_EXISTING_WORKER
        if original is None:
            raise RuntimeError("Marginal Capital Replay worker is unavailable.")

        worker_id = f"{socket.gethostname()}:{os.getpid()}:marginal:{uuid4().hex[:8]}"
        now = service.utc_now()
        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {"$set": {
                "worker_id": worker_id,
                "worker_active": True,
                "worker_heartbeat_at": now,
                "worker_started_at": now,
                "worker_finished_at": None,
                "updated_at": now,
            }},
        )

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=service._heartbeat_worker,
            args=(db, run_id, worker_id, heartbeat_stop),
            name="asset-discovery-marginal-replay-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            original(db, run_id)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            db[service.COLLECTION].update_one(
                {"_id": service.CURRENT_ID, "run_id": run_id, "worker_id": worker_id},
                {"$set": {
                    "worker_active": False,
                    "worker_finished_at": service.utc_now(),
                    "updated_at": service.utc_now(),
                }},
            )

    setattr(marginal_worker_with_heartbeat, "_asset_discovery_marginal_replay_heartbeat", True)
    setattr(heartbeat_fresh_with_local_worker_guard, "_asset_discovery_local_worker_guard", True)
    service._run_existing_marginal_worker = marginal_worker_with_heartbeat
    service._worker_heartbeat_fresh = heartbeat_fresh_with_local_worker_guard
    _INSTALLED = True
