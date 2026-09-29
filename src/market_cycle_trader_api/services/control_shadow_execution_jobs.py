"""Admin-only, no-order execution feasibility jobs on completed v10.8.41 baseline."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_execution_research import run_control_execution_feasibility
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_validation_jobs import COLLECTION as VALIDATION_COLLECTION

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_execution_jobs"
ACTIVE_KEY = "control-shadow-execution-feasibility"
_THREADS: dict[str, threading.Thread] = {}


class ExecutionConflict(RuntimeError):
    pass


class ExecutionNotFound(LookupError):
    pass


class ExecutionInvalid(ValueError):
    pass


def _public(record: dict[str, Any], *, logs_only: bool = False) -> dict[str, Any]:
    result = {
        "job_id": record["_id"],
        "source_validation_job_id": record["source_validation_job_id"],
        "source_job_id": record["source_job_id"],
        "snapshot_sha256": record["snapshot_sha256"],
        "status": record["status"],
        "stage": record.get("stage"),
        "progress": record.get("progress", 0),
        "logs": list(record.get("logs") or []),
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    if not logs_only:
        result.update({
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "result": record.get("result"),
            "error": record.get("error"),
        })
    return result


def _log(db: Any, job_id: str, message: str, *,
         level: str = "INFO", stage: str | None = None,
         progress: int | None = None) -> None:
    now = utc_now()
    change = {"updated_at": now}
    if stage is not None:
        change["stage"] = stage
    if progress is not None:
        change["progress"] = progress
    db[COLLECTION].update_one(
        {"_id": job_id},
        {"$set": change, "$push": {
            "logs": {"$each": [{
                "at": now, "level": level, "message": str(message)[:400],
            }], "$slice": -300},
        }},
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Execution %s | %s", job_id, message,
    )


def _run_job(db: Any, job_id: str, *,
             source_job_id: str,
             validation_job_id: str,
             expected_sha256: str,
             baseline: dict[str, Any]) -> None:
    try:
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(db, job_id, "Reopening immutable v10.8.41 Control snapshot: no Alpaca or orders.",
             stage="verify_immutable_snapshot", progress=1)
        marks: dict[str, int] = {}

        def progress(stage: str, completed: int, total: int) -> None:
            pct = (5 + int(94 * completed / max(1, total))) if total else 5
            if stage == "verify_immutable_snapshot":
                pct = 3
            if stage == "completed":
                pct = 99
            key = "feasibility_oos" if stage.startswith("Run ") else stage[:90]
            prior = marks.get(key, -10)
            if pct - prior >= 5 or completed == total or stage == "completed":
                marks[key] = pct
                _log(db, job_id, f"{stage}: {completed}/{total}",
                     stage=key, progress=min(99, pct))

        report = run_control_execution_feasibility(
            source_job_id=source_job_id,
            validation_job_id=validation_job_id,
            execution_job_id=job_id,
            expected_sha256=expected_sha256,
            baseline=baseline,
            progress=progress,
        )
        _log(
            db, job_id,
            "Completed: state-aware policy; 3 OOS folds, liquidity-capped "
            "partial fills and residual CASH; research only.",
            stage="completed", progress=100,
        )
        now = utc_now()
        db[COLLECTION].update_one({"_id": job_id}, {
            "$set": {
                "status": "completed", "stage": "completed",
                "progress": 100, "updated_at": now, "finished_at": now,
                "result": report,
            },
            "$unset": {"active_key": ""},
        })
    except Exception as exc:
        LOGGER.exception("Control Execution %s | stopped without any trading action", job_id)
        _log(db, job_id, f"{type(exc).__name__}: {str(exc)[:330]}",
             level="ERROR", stage="failed")
        now = utc_now()
        db[COLLECTION].update_one({"_id": job_id}, {
            "$set": {
                "status": "failed", "stage": "failed",
                "updated_at": now, "finished_at": now,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            },
            "$unset": {"active_key": ""},
        })
    finally:
        _THREADS.pop(job_id, None)


def start_execution_feasibility(db: Any, *, source_validation_job_id: str,
                                expected_snapshot_sha256: str) -> dict[str, Any]:
    _require_enabled()
    if not re.fullmatch(r"control-validation-[a-f0-9]{16}", source_validation_job_id):
        raise ExecutionInvalid("Expected existing, completed Control validation ID.")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_snapshot_sha256):
        raise ExecutionInvalid("An independent exact snapshot SHA-256 is required.")
    source = db[VALIDATION_COLLECTION].find_one({"_id": source_validation_job_id})
    if source is None:
        raise ExecutionNotFound("Completed v10.8.41 validation was not found in current MongoDB.")
    baseline = source.get("result") or {}
    integrity = baseline.get("numeric_input_integrity") or {}
    if (
        source.get("status") != "completed"
        or source.get("error")
        or not baseline.get("source_unchanged")
        or integrity.get("status") != "verified"
        or not (baseline.get("original_shadow") or {}).get("reproduced")
        or baseline.get("order_submission") != "never"
        or str(source.get("snapshot_sha256")) != expected_snapshot_sha256
        or str(baseline.get("source_snapshot_sha256")) != expected_snapshot_sha256
        or not baseline.get("oos")
    ):
        raise ExecutionInvalid(
            "Source must be completed, SHA-matched and numerically reproduced "
            "Control v10.8.41 with a full OOS baseline."
        )
    source_job_id = str(source.get("source_job_id") or "")
    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    job_id = "control-execution-" + uuid.uuid4().hex[:16]
    now = utc_now()
    record = {
        "_id": job_id, "status": "queued", "stage": "queued", "progress": 0,
        "active_key": ACTIVE_KEY,
        "source_validation_job_id": source_validation_job_id,
        "source_job_id": source_job_id, "snapshot_sha256": expected_snapshot_sha256,
        "created_at": now, "updated_at": now, "started_at": None, "finished_at": None,
        "logs": [], "result": None, "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise ExecutionConflict("A Control execution feasibility job is already active.") from exc
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db, "job_id": job_id, "source_job_id": source_job_id,
            "validation_job_id": source_validation_job_id,
            "expected_sha256": expected_snapshot_sha256,
            "baseline": baseline,
        }, name=job_id, daemon=True,
    )
    try:
        _THREADS[job_id] = thread
        thread.start()
    except RuntimeError:
        _THREADS.pop(job_id, None)
        collection.update_one({"_id": job_id}, {
            "$set": {"status": "failed", "stage": "thread_start_failed",
                     "finished_at": utc_now()},
            "$unset": {"active_key": ""},
        })
        raise
    LOGGER.info("Control Execution %s | queued on verified snapshot %s",
                job_id, source_job_id)
    return _public(record)


def get_execution_feasibility(db: Any, job_id: str, *, logs_only: bool = False) -> dict[str, Any]:
    record = db[COLLECTION].find_one({"_id": job_id})
    if record is None:
        raise ExecutionNotFound("Control execution feasibility job was not found.")
    return _public(record, logs_only=logs_only)
