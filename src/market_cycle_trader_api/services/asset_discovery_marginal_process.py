from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Any

from . import asset_discovery as discovery


CHILD_ENV = "MCT_ASSET_DISCOVERY_MARGINAL_CHILD"
_PROCESS_LOCK = threading.RLock()
_PROCESS: subprocess.Popen[Any] | None = None
_PROCESS_RUN_ID = ""
_INSTALLED = False


def _set_process(process: subprocess.Popen[Any] | None, run_id: str = "") -> None:
    global _PROCESS, _PROCESS_RUN_ID
    with _PROCESS_LOCK:
        _PROCESS = process
        _PROCESS_RUN_ID = str(run_id or "")


def terminate_marginal_process(run_id: str, *, timeout_seconds: float = 2.0) -> bool:
    with _PROCESS_LOCK:
        process = _PROCESS
        process_run_id = _PROCESS_RUN_ID
    if process is None or process.poll() is not None or str(run_id or "") != process_run_id:
        return False

    process.terminate()
    try:
        process.wait(timeout=max(0.1, float(timeout_seconds)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.1, float(timeout_seconds)))
    return True


def _supervise_marginal_process(db: Any, run_id: str) -> None:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "market_cycle_trader_api.services.asset_discovery_marginal_process_worker",
            str(run_id),
        ],
        env=env,
    )
    _set_process(process, run_id)
    now = discovery.utc_now()
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {"$set": {"worker_process_id": int(process.pid), "updated_at": now}},
    )

    try:
        return_code = process.wait()
    finally:
        _set_process(None, "")

    document = discovery._campaign(db) or {}
    if str(document.get("run_id") or "") != str(run_id):
        return
    marginal = document.get("marginal_replay") if isinstance(document.get("marginal_replay"), dict) else {}
    marginal_status = str(marginal.get("status") or "").strip().lower()
    if marginal_status not in discovery.ACTIVE_STATUSES:
        db[discovery.COLLECTION].update_one(
            {"_id": discovery.CURRENT_ID, "run_id": run_id},
            {"$unset": {"worker_process_id": ""}},
        )
        return

    stopped = bool(document.get("cancel_requested")) or int(return_code or 0) < 0
    finished_at = discovery.utc_now()
    status_value = "stopped" if stopped else "failed"
    stage = "Stopped by user" if stopped else "Marginal replay worker process exited unexpectedly"
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {
            "$set": {
                "status": status_value,
                "phase": status_value,
                "marginal_replay.status": status_value,
                "marginal_replay.current_stage": stage,
                "worker_active": False,
                "worker_finished_at": finished_at,
                "updated_at": finished_at,
                "message": stage,
            },
            "$unset": {"worker_process_id": ""},
        },
    )


def install_asset_discovery_marginal_process() -> None:
    global _INSTALLED
    if _INSTALLED or os.getenv(CHILD_ENV) == "1":
        _INSTALLED = True
        return
    current = discovery._run_existing_marginal_worker
    if getattr(current, "_asset_discovery_marginal_process", False):
        _INSTALLED = True
        return
    setattr(_supervise_marginal_process, "_asset_discovery_marginal_process", True)
    discovery._run_existing_marginal_worker = _supervise_marginal_process
    _INSTALLED = True
