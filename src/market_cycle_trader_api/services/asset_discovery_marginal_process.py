from __future__ import annotations

import os
from pathlib import Path
import re
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


def _child_environment() -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    source_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else str(source_root) + os.pathsep + existing_pythonpath
    )
    return env, str(source_root.parent)


def _sanitize_worker_error(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"(?i)\b(api[_\s-]?key|secret|token|password)\b\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=<redacted>",
        text,
    )
    text = re.sub(r"mongodb(?:\+srv)?://[^\s]+", "mongodb://<redacted>", text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[-1600:]


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


def _predictive_optional_replay(document: dict[str, Any]) -> bool:
    return str(document.get("discovery_mode") or "").strip().lower() == "predictive_only"


def _supervise_marginal_process(db: Any, run_id: str) -> None:
    env, cwd = _child_environment()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "market_cycle_trader_api.services.asset_discovery_marginal_process_worker",
            str(run_id),
        ],
        env=env,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _set_process(process, run_id)
    now = discovery.utc_now()
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {"$set": {"worker_process_id": int(process.pid), "updated_at": now}},
    )

    stderr = ""
    try:
        _stdout, stderr = process.communicate()
        return_code = int(process.returncode or 0)
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

    stopped = bool(document.get("cancel_requested"))
    predictive_optional = _predictive_optional_replay(document)
    finished_at = discovery.utc_now()
    worker_error = _sanitize_worker_error(stderr)

    if stopped:
        campaign_status = "completed" if predictive_optional else "stopped"
        campaign_phase = "completed" if predictive_optional else "stopped"
        message = (
            "Optional Marginal Capital Replay stopped; predictive candidates were preserved."
            if predictive_optional
            else "Marginal Capital Replay stopped by user."
        )
        marginal_status_value = "stopped"
        stage = "Optional Marginal Capital Replay stopped" if predictive_optional else "Stopped by user"
    else:
        campaign_status = "completed" if predictive_optional else "failed"
        campaign_phase = "completed" if predictive_optional else "failed"
        message = (
            "Optional Marginal Capital Replay failed; predictive candidates were preserved."
            if predictive_optional
            else "Marginal Capital Replay worker process exited unexpectedly."
        )
        marginal_status_value = "failed"
        stage = "Optional Marginal Capital Replay failed" if predictive_optional else "Marginal replay worker process exited unexpectedly"

    update_set: dict[str, Any] = {
        "status": campaign_status,
        "phase": campaign_phase,
        "cancel_requested": False if predictive_optional else bool(document.get("cancel_requested")),
        "marginal_replay.status": marginal_status_value,
        "marginal_replay.current_stage": stage,
        "marginal_replay.worker_exit_code": return_code,
        "marginal_replay.worker_error": worker_error or None,
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

    event_message = message
    if worker_error and not stopped:
        event_message = f"{message} Worker error: {worker_error[:700]}"

    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {
            "$set": update_set,
            "$unset": {"worker_process_id": ""},
            "$push": {
                "events": {
                    "$each": [{"at": finished_at, "message": event_message[:1000]}],
                    "$slice": -24,
                }
            },
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
