from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from fastapi import HTTPException

from ..core.config import ENGINE_PATH, PACKAGE_DIR
from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    RUNS_COLLECTION,
    utc_now,
)
from .serialization import iso_value


def public_job(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {
        key: iso_value(value)
        for key, value in document.items()
        if key not in {"_id", "process_id"}
    }


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
    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {
            "completed_runs": 1,
            "total_runs": 1,
            "request.model_backends": 1,
            "request.strategy_mode": 1,
            "request.exit_risk_compare_models": 1,
            "request.exit_risk_model_backends": 1,
            "live_trade_count": 1,
        },
    ) or {}

    changes: dict[str, Any] = {"updated_at": utc_now()}
    stripped = line.strip()
    completed_increment = 0
    log_line = line
    request = job.get("request") or {}

    if stripped.startswith("JOB_TRADE|"):
        payload_text = stripped.removeprefix("JOB_TRADE|")
        try:
            trade = json.loads(payload_text)
        except json.JSONDecodeError:
            trade = None
        if isinstance(trade, dict):
            allowed_fields = {
                "backend", "model", "timestamp", "asset", "action", "reason",
                "execution_price", "quantity", "total_fee", "realized_pnl",
                "position_return", "cash_after_trade", "walk_forward_fold",
                "model_family", "random_seed", "repetition_index",
            }
            clean_trade = {
                key: iso_value(value)
                for key, value in trade.items()
                if key in allowed_fields
            }
            clean_trade["received_at"] = utc_now()
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {"updated_at": utc_now()},
                    "$inc": {"live_trade_count": 1},
                    "$push": {"live_trades": {"$each": [clean_trade], "$slice": -300}},
                },
            )
        return

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
            if total:
                completed = min(total, completed)
            stage = parts[3].strip() or "Running backtest"
            changes["progress"] = percent
            changes["completed_runs"] = completed
            changes["stage"] = stage
            log_line = stage
    elif stripped.startswith("Loading "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("Running "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("Strategy=") or ": Strategy=" in stripped:
        if request.get("strategy_mode") not in {
            "COMPOUND_ROTATION_SWING_1W",
            "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
        }:
            completed_increment = 1
    elif stripped.startswith("ERROR ") and "/" in stripped.split(":", 1)[0]:
        completed_increment = 1
        changes["stage"] = stripped.split(":", 1)[0]
    elif stripped.startswith("ERROR loading "):
        backends = request.get("model_backends") or []
        multiplier = 1
        if (
            request.get("strategy_mode") in {"BOTTOM_ENTRY_EXIT_RISK_V1", "BOTTOM_ENTRY_EXIT_RISK_SWING_1D"}
            and request.get("exit_risk_compare_models", False)
        ):
            multiplier = max(1, len(request.get("exit_risk_model_backends") or []))
        completed_increment = max(1, len(backends) * multiplier)
        changes["stage"] = stripped.split(":", 1)[0]

    if completed_increment:
        completed = min(
            int(job.get("total_runs", 0)),
            int(job.get("completed_runs", 0)) + completed_increment,
        )
        total = int(job.get("total_runs", 0))
        changes["completed_runs"] = completed
        changes["progress"] = round(completed / total * 100, 1) if total else 0

    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$set": changes,
            "$push": {"logs": {"$each": [log_line], "$slice": -400}},
        },
    )


def run_job(job_id: str) -> None:
    db = database()
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {"$set": {
            "status": "running",
            "stage": "Starting backtest",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "progress": 0,
        }},
    )

    command = [sys.executable, "-u", str(ENGINE_PATH), "--job-id", job_id]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PACKAGE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {"$set": {"process_id": process.pid, "updated_at": utc_now()}},
        )
        assert process.stdout is not None
        for line in process.stdout:
            append_log(job_id, line)

        return_code = process.wait()
        run_count = db[RUNS_COLLECTION].count_documents({"job_id": job_id})
        comparison_exists = db[COMPARISONS_COLLECTION].find_one(
            {"job_id": job_id}, {"_id": 1}
        ) is not None

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
        else:
            stored_comparison = db[COMPARISONS_COLLECTION].find_one(
                {"job_id": job_id}, {"_id": 0, "failures": 1}
            ) or {}
            for failure in stored_comparison.get("failures", []):
                append_log(
                    job_id,
                    "ERROR "
                    f"{failure.get('symbol', 'unknown')}/"
                    f"{failure.get('backend', 'unknown')}: "
                    f"{failure.get('error', 'Unknown error')}",
                )
            if return_code != 0 and not stored_comparison.get("failures"):
                append_log(job_id, f"ERROR: Backtest engine exited with code {return_code}.")
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "stage": "Backtest failed",
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "return_code": return_code,
                    },
                    "$unset": {"process_id": ""},
                },
            )
    except Exception as exc:  # noqa: BLE001
        append_log(job_id, f"ERROR: {exc}")
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "stage": "Backtest failed",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                    "error": str(exc),
                },
                "$unset": {"process_id": ""},
            },
        )
