from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any

from fastapi import HTTPException

from ..core.config import ENGINE_MODULE, SOURCE_ROOT
from ..core.environment import load_project_environment
from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, JOBS_COLLECTION, RUNS_COLLECTION, utc_now
from .serialization import iso_value

PUBLIC_JOB_FIELDS = frozenset(
    {
        "id",
        "status",
        "stage",
        "progress",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "return_code",
        "error",
    }
)

_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(ALPACA_API_KEY_ID|ALPACA_SECRET_KEY|MONGO_URL)\b\s*[:=]\s*([^\s,;]+)"
)
_MONGODB_CREDENTIAL_PATTERN = re.compile(r"(mongodb(?:\+srv)?://)([^@\s]+)@", flags=re.IGNORECASE)


def _generic_stage(line: str) -> str:
    text = line.lower()
    if "loading market data" in text:
        return "Loading market data"
    if "loaded market data" in text:
        return "Market data loaded"
    if "building aligned" in text or "prepared" in text and "assets" in text:
        return "Preparing analysis"
    if "run " in text and ("fold" in text or "completed" in text):
        return "Running analysis"
    if "saving" in text:
        return "Saving results"
    if "finalizing" in text:
        return "Finalizing reports"
    if "completed" in text:
        return "Completed"
    if "error" in text or "traceback" in text:
        return "Analysis failed"
    return "Running analysis"


def _public_log_line(raw_line: Any) -> str | None:
    line = str(raw_line).strip()
    if not line:
        return None
    line = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1=***", line)
    line = _MONGODB_CREDENTIAL_PATTERN.sub(r"\1***@", line)
    return _generic_stage(line)


def public_job(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    payload = {
        key: iso_value(value)
        for key, value in document.items()
        if key in PUBLIC_JOB_FIELDS
    }
    raw_logs = document.get("logs")
    if isinstance(raw_logs, list):
        public_logs: list[str] = []
        for raw_line in raw_logs:
            line = _public_log_line(raw_line)
            if line and (not public_logs or public_logs[-1] != line):
                public_logs.append(line)
        payload["logs"] = public_logs[-20:]
    else:
        payload["logs"] = []
    payload["stage"] = _generic_stage(str(document.get("stage") or "")) if document.get("status") == "running" else payload.get("stage")
    if document.get("status") == "failed":
        payload["error"] = "The analysis could not be completed."
    return payload


def require_job(job_id: str) -> dict[str, Any]:
    job = database()[JOBS_COLLECTION].find_one({"id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return job


def append_log(job_id: str, raw_line: str) -> None:
    line = raw_line.rstrip()
    if not line:
        return
    db = database()
    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {"completed_runs": 1, "total_runs": 1, "live_trade_count": 1},
    ) or {}
    stripped = line.strip()
    if stripped.startswith("JOB_TRADE|"):
        try:
            trade = json.loads(stripped.removeprefix("JOB_TRADE|"))
        except json.JSONDecodeError:
            return
        if isinstance(trade, dict):
            allowed = {
                "timestamp",
                "asset",
                "action",
                "reason",
                "execution_price",
                "quantity",
                "total_fee",
                "realized_pnl",
                "position_return",
                "cash_after_trade",
            }
            clean = {key: iso_value(value) for key, value in trade.items() if key in allowed}
            clean["received_at"] = utc_now()
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {"updated_at": utc_now()},
                    "$inc": {"live_trade_count": 1},
                    "$push": {"live_trades": {"$each": [clean], "$slice": -300}},
                },
            )
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
            changes["stage"] = _generic_stage(parts[3])
            log_line = parts[3].strip() or "Running analysis"
    elif stripped.startswith("ERROR") or "Traceback" in stripped:
        changes["stage"] = "Analysis failed"
    else:
        changes["stage"] = _generic_stage(stripped)

    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$set": changes,
            "$push": {"logs": {"$each": [log_line], "$slice": -400}},
        },
    )


def run_job(job_id: str) -> None:
    load_project_environment()
    db = database()
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$set": {
                "status": "running",
                "stage": "Starting analysis",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "progress": 0,
            }
        },
    )
    python_path = str(SOURCE_ROOT)
    existing_python_path = os.environ.get("PYTHONPATH", "")
    if existing_python_path:
        python_path = python_path + os.pathsep + existing_python_path
    command = [sys.executable, "-u", "-m", ENGINE_MODULE, "--job-id", job_id]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(SOURCE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={
                **os.environ,
                "PYTHONPATH": python_path,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {"$set": {"process_id": process.pid, "updated_at": utc_now()}},
        )
        assert process.stdout is not None
        for output in process.stdout:
            append_log(job_id, output)
        return_code = process.wait()
        run_count = db[RUNS_COLLECTION].count_documents({"job_id": job_id})
        comparison_exists = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 1}) is not None
        if return_code == 0 and comparison_exists and run_count > 0:
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "completed",
                        "stage": "Completed",
                        "progress": 100,
                        "completed_runs": run_count,
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "return_code": return_code,
                    },
                    "$unset": {"process_id": ""},
                },
            )
            return
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "stage": "Analysis failed",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                    "return_code": return_code,
                },
                "$unset": {"process_id": ""},
            },
        )
    except Exception as exc:
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "stage": "Analysis failed",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                    "error": str(exc),
                },
                "$unset": {"process_id": ""},
            },
        )
