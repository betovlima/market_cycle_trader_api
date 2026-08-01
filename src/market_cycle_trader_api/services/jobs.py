from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from fastapi import HTTPException

from ..core.config import ENGINE_MODULE, SOURCE_ROOT
from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, JOBS_COLLECTION, RUNS_COLLECTION, utc_now
from .serialization import iso_value


def public_job(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: iso_value(value) for key, value in document.items() if key not in {"_id", "process_id"}}


def require_job(job_id: str) -> dict[str, Any]:
    job = database()[JOBS_COLLECTION].find_one({"id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    return job


def append_log(job_id: str, raw_line: str) -> None:
    line = raw_line.rstrip()
    if not line:
        return
    db = database()
    job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"completed_runs": 1, "total_runs": 1, "live_trade_count": 1}) or {}
    stripped = line.strip()
    if stripped.startswith("JOB_TRADE|"):
        try:
            trade = json.loads(stripped.removeprefix("JOB_TRADE|"))
        except json.JSONDecodeError:
            return
        if isinstance(trade, dict):
            allowed = {"backend", "model", "timestamp", "asset", "action", "reason", "execution_price", "quantity", "total_fee", "realized_pnl", "position_return", "cash_after_trade", "walk_forward_fold", "model_family", "random_seed", "repetition_index"}
            clean = {key: iso_value(value) for key, value in trade.items() if key in allowed}
            clean["received_at"] = utc_now()
            db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"updated_at": utc_now()}, "$inc": {"live_trade_count": 1}, "$push": {"live_trades": {"$each": [clean], "$slice": -300}}})
        return

    changes: dict[str, Any] = {"updated_at": utc_now()}
    log_line = line
    if stripped.startswith("JOB_PROGRESS|"):
        parts = stripped.split("|", 3)
        if len(parts) == 4:
            try:
                percent = max(0.0, min(99.9, float(parts[1])))
            except (TypeError, ValueError):
                percent = float(job.get("progress", 0) or 0)
            try:
                completed = max(0, int(parts[2]))
            except (TypeError, ValueError):
                completed = int(job.get("completed_runs", 0) or 0)
            total = int(job.get("total_runs", 0) or 0)
            changes["progress"] = percent
            changes["completed_runs"] = min(total, completed) if total else completed
            changes["stage"] = parts[3].strip() or "Running backtest"
            log_line = changes["stage"]
    elif stripped.startswith("Loading "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("Running "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("ERROR"):
        changes["stage"] = "Backtest error"

    db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": changes, "$push": {"logs": {"$each": [log_line], "$slice": -400}}})


def run_job(job_id: str) -> None:
    db = database()
    db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "running", "stage": "Starting backtest", "started_at": utc_now(), "updated_at": utc_now(), "progress": 0}})
    python_path = str(SOURCE_ROOT)
    existing_python_path = os.environ.get("PYTHONPATH", "")
    if existing_python_path:
        python_path = python_path + os.pathsep + existing_python_path
    command = [sys.executable, "-u", "-m", ENGINE_MODULE, "--job-id", job_id]
    try:
        process = subprocess.Popen(command, cwd=str(SOURCE_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env={**os.environ, "PYTHONPATH": python_path, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"process_id": process.pid, "updated_at": utc_now()}})
        assert process.stdout is not None
        for line in process.stdout:
            append_log(job_id, line)
        return_code = process.wait()
        run_count = db[RUNS_COLLECTION].count_documents({"job_id": job_id})
        comparison_exists = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 1}) is not None
        if return_code == 0 and comparison_exists and run_count > 0:
            db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "completed", "stage": "Completed", "progress": 100, "completed_runs": run_count, "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}})
            return
        stored = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 0, "failures": 1}) or {}
        for failure in stored.get("failures", []):
            append_log(job_id, f"ERROR {failure.get('symbol', 'unknown')}/{failure.get('backend', 'unknown')}: {failure.get('error', 'Unknown error')}")
        if return_code != 0 and not stored.get("failures"):
            append_log(job_id, f"ERROR: Backtest engine exited with code {return_code}.")
        db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "failed", "stage": "Backtest failed", "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}})
    except Exception as exc:
        append_log(job_id, f"ERROR: {exc}")
        db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "failed", "stage": "Backtest failed", "finished_at": utc_now(), "updated_at": utc_now(), "error": str(exc)}, "$unset": {"process_id": ""}})
