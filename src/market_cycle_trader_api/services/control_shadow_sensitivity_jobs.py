"""Admin-only, no-order fixed Control execution sensitivity jobs."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_execution_sensitivity import run_control_execution_sensitivity
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_validation_jobs import COLLECTION as VALIDATION_COLLECTION
from .control_shadow_execution_jobs import COLLECTION as EXECUTION_COLLECTION

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_sensitivity_jobs"
ACTIVE_KEY = "control-shadow-sensitivity-fixed-v1043"
_THREADS: dict[str, threading.Thread] = {}


class SensitivityConflict(RuntimeError):
    pass


class SensitivityNotFound(LookupError):
    pass


class SensitivityInvalid(ValueError):
    pass


def _public(record: dict[str, Any], *, logs_only: bool = False) -> dict[str, Any]:
    public = {
        "job_id": record["_id"],
        "source_validation_job_id": record["source_validation_job_id"],
        "source_execution_job_id": record["source_execution_job_id"],
        "source_job_id": record["source_job_id"],
        "snapshot_sha256": record["snapshot_sha256"],
        "status": record["status"],
        "stage": record.get("stage"),
        "progress": record.get("progress", 0),
        "logs": list(record.get("logs") or []),
        "source_download": "never",
        "order_eligible": False,
        "order_submission": "never",
    }
    if not logs_only:
        public.update({
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "result": record.get("result"),
            "error": record.get("error"),
        })
    return public


def _log(db: Any, job_id: str, message: str, *,
         level: str = "INFO", stage: str | None = None,
         progress: int | None = None) -> None:
    now = utc_now()
    updates = {"updated_at": now}
    if stage is not None:
        updates["stage"] = stage
    if progress is not None:
        updates["progress"] = progress
    db[COLLECTION].update_one(
        {"_id": job_id}, {"$set": updates, "$push": {
            "logs": {"$each": [{
                "at": now, "level": level, "message": str(message)[:400],
            }], "$slice": -300},
        }},
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Sensitivity %s | %s", job_id, message,
    )


def _run_job(db: Any, job_id: str, *, source_job_id: str,
             validation_job_id: str, feasibility_job_id: str,
             expected_sha256: str, baseline41: dict[str, Any],
             baseline42: dict[str, Any]) -> None:
    try:
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(
            db, job_id,
            "Verifying v10.8.41 and v10.8.42 results and immutable snapshot; no Alpaca or orders.",
            stage="verify_immutable_snapshot", progress=1,
        )
        last_percentage = -10
        def progress(stage: str, completed: int, total: int) -> None:
            nonlocal last_percentage
            if stage == "verify_immutable_snapshot":
                pct = 3
            elif stage == "completed":
                pct = 99
            else:
                pct = 5 + int(94 * completed / max(1, total))
            if pct - last_percentage >= 5 or stage == "completed":
                last_percentage = pct
                _log(db, job_id, f"{stage}: {completed}/{total}",
                     stage=str(stage)[:100], progress=min(99, pct))

        report = run_control_execution_sensitivity(
            source_job_id=source_job_id,
            validation_job_id=validation_job_id,
            feasibility_job_id=feasibility_job_id,
            sensitivity_job_id=job_id,
            expected_sha256=expected_sha256,
            baseline41=baseline41,
            baseline42=baseline42,
            progress=progress,
        )
        _log(
            db, job_id,
            "Completed five predeclared sensitivity cases; v10.8.42 10% reference reproduced.",
            stage="completed", progress=100,
        )
        now = utc_now()
        db[COLLECTION].update_one({"_id": job_id}, {
            "$set": {
                "status": "completed", "stage": "completed", "progress": 100,
                "updated_at": now, "finished_at": now, "result": report,
            },
            "$unset": {"active_key": ""},
        })
    except Exception as exc:
        LOGGER.exception("Control Sensitivity %s | failed without any trading action", job_id)
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


def start_control_sensitivity(db: Any, *, source_validation_job_id: str,
                              source_execution_job_id: str,
                              expected_snapshot_sha256: str) -> dict[str, Any]:
    _require_enabled()
    if not re.fullmatch(r"control-validation-[a-f0-9]{16}", source_validation_job_id):
        raise SensitivityInvalid("Expected exact completed Control validation ID.")
    if not re.fullmatch(r"control-execution-[a-f0-9]{16}", source_execution_job_id):
        raise SensitivityInvalid("Expected exact completed Control execution job ID.")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_snapshot_sha256):
        raise SensitivityInvalid("Independent snapshot SHA-256 confirmation is required.")
    validation = db[VALIDATION_COLLECTION].find_one({"_id": source_validation_job_id})
    execution = db[EXECUTION_COLLECTION].find_one({"_id": source_execution_job_id})
    if validation is None or execution is None:
        raise SensitivityNotFound("Both completed baseline jobs must exist in this MongoDB.")
    base41, base42 = validation.get("result") or {}, execution.get("result") or {}
    source_job_id = str(validation.get("source_job_id") or "")
    if (
        validation.get("status") != "completed"
        or validation.get("error")
        or execution.get("status") != "completed"
        or execution.get("error")
        or source_job_id != str(execution.get("source_job_id") or "")
        or source_validation_job_id != str(execution.get("source_validation_job_id") or "")
        or validation.get("snapshot_sha256") != expected_snapshot_sha256
        or execution.get("snapshot_sha256") != expected_snapshot_sha256
        or base41.get("source_snapshot_sha256") != expected_snapshot_sha256
        or base42.get("source_snapshot_sha256") != expected_snapshot_sha256
        or base42.get("research_kind") != "control_execution_feasibility_scenario"
        or base42.get("execution_job_id") != source_execution_job_id
        or not (base41.get("original_shadow") or {}).get("reproduced")
        or (base41.get("numeric_input_integrity") or {}).get("status") != "verified"
        or (base42.get("numeric_input_integrity") or {}).get("status") != "verified"
        or base41.get("order_submission") != "never"
        or base42.get("order_submission") != "never"
    ):
        raise SensitivityInvalid("Sources do not form a matched SHA-verified v10.8.41/v10.8.42 research pair.")
    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    now = utc_now()
    job_id = "control-sensitivity-" + uuid.uuid4().hex[:16]
    record = {
        "_id": job_id, "status": "queued", "stage": "queued", "progress": 0,
        "active_key": ACTIVE_KEY,
        "source_validation_job_id": source_validation_job_id,
        "source_execution_job_id": source_execution_job_id,
        "source_job_id": source_job_id,
        "snapshot_sha256": expected_snapshot_sha256,
        "created_at": now, "updated_at": now,
        "started_at": None, "finished_at": None,
        "logs": [], "result": None, "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise SensitivityConflict("Another Control sensitivity job is active.") from exc
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db, "job_id": job_id, "source_job_id": source_job_id,
            "validation_job_id": source_validation_job_id,
            "feasibility_job_id": source_execution_job_id,
            "expected_sha256": expected_snapshot_sha256,
            "baseline41": base41, "baseline42": base42,
        },
        name=job_id, daemon=True,
    )
    try:
        _THREADS[job_id] = thread
        thread.start()
    except RuntimeError:
        _THREADS.pop(job_id, None)
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "failed", "stage": "thread_start_failed",
                "finished_at": utc_now(),
            }, "$unset": {"active_key": ""},
        })
        raise
    LOGGER.info("Control Sensitivity %s | queued on immutable source %s",
                job_id, source_job_id)
    return _public(record)


def get_control_sensitivity(db: Any, job_id: str, *, logs_only: bool = False) -> dict[str, Any]:
    record = db[COLLECTION].find_one({"_id": job_id})
    if record is None:
        raise SensitivityNotFound("Control sensitivity job not found.")
    return _public(record, logs_only=logs_only)
