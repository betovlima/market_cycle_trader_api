"""Admin-only, research-only Control candidate liquidity experiment jobs."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_liquidity_research import run_control_liquidity_research
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_validation_jobs import COLLECTION as VALIDATION_COLLECTION
from .control_shadow_execution_jobs import COLLECTION as EXECUTION_COLLECTION
from .control_shadow_sensitivity_jobs import COLLECTION as SENSITIVITY_COLLECTION

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_liquidity_research_jobs"
ACTIVE_KEY = "control-shadow-liquidity-aware-v1044"
_THREADS: dict[str, threading.Thread] = {}


class LiquidityConflict(RuntimeError):
    pass


class LiquidityNotFound(LookupError):
    pass


class LiquidityInvalid(ValueError):
    pass


def _public(record: dict[str, Any], *, logs_only: bool = False) -> dict[str, Any]:
    payload = {
        "job_id": record["_id"],
        "source_job_id": record["source_job_id"],
        "source_validation_job_id": record["source_validation_job_id"],
        "source_execution_job_id": record["source_execution_job_id"],
        "source_sensitivity_job_id": record["source_sensitivity_job_id"],
        "snapshot_sha256": record["snapshot_sha256"],
        "status": record["status"], "stage": record.get("stage"),
        "progress": record.get("progress", 0),
        "logs": list(record.get("logs") or []),
        "source_download": "never", "order_eligible": False,
        "order_submission": "never",
    }
    if not logs_only:
        payload.update({
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "result": record.get("result"), "error": record.get("error"),
        })
    return payload


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
        {"$set": change, "$push": {"logs": {"$each": [{
            "at": now, "level": level, "message": str(message)[:400],
        }], "$slice": -300}}},
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Liquidity %s | %s", job_id, message,
    )


def _run_job(db: Any, job_id: str, *, source_job_id: str,
             validation_id: str, execution_id: str, sensitivity_id: str,
             expected_sha256: str,
             baseline41: dict[str, Any], baseline42: dict[str, Any],
             baseline43: dict[str, Any]) -> None:
    try:
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(
            db, job_id,
            "Reopening exact immutable Control snapshot; reference and liquidity-aware "
            "policies share training and execution scenario. No Alpaca/orders.",
            stage="verify_immutable_source", progress=1,
        )
        last_pct = -10
        def progress(stage: str, done: int, total: int) -> None:
            nonlocal last_pct
            pct = 3 if stage.startswith("verify") else (
                99 if stage == "completed" else
                5 + int(94 * done / max(1, total))
            )
            if pct - last_pct >= 5 or stage == "completed":
                last_pct = pct
                _log(db, job_id, f"{stage}: {done}/{total}",
                     stage=str(stage)[:100], progress=min(99, pct))

        result = run_control_liquidity_research(
            source_job_id=source_job_id,
            validation_job_id=validation_id,
            execution_job_id=execution_id,
            sensitivity_job_id=sensitivity_id,
            research_job_id=job_id,
            expected_sha256=expected_sha256,
            baseline41=baseline41, baseline42=baseline42,
            baseline43=baseline43, progress=progress,
        )
        _log(
            db, job_id,
            "Paired experiment complete: original Control reproduced and "
            "prior-close liquidity-aware candidate policy audited. Research only.",
            stage="completed", progress=100,
        )
        now = utc_now()
        db[COLLECTION].update_one({"_id": job_id}, {
            "$set": {
                "status": "completed", "stage": "completed", "progress": 100,
                "result": result, "updated_at": now, "finished_at": now,
            }, "$unset": {"active_key": ""},
        })
    except Exception as exc:
        LOGGER.exception("Control Liquidity %s | research stopped without trading", job_id)
        _log(db, job_id, f"{type(exc).__name__}: {str(exc)[:330]}",
             level="ERROR", stage="failed")
        now = utc_now()
        db[COLLECTION].update_one({"_id": job_id}, {
            "$set": {
                "status": "failed", "stage": "failed", "finished_at": now,
                "updated_at": now,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }, "$unset": {"active_key": ""},
        })
    finally:
        _THREADS.pop(job_id, None)


def start_liquidity_research(
    db: Any, *, source_validation_job_id: str,
    source_execution_job_id: str, source_sensitivity_job_id: str,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    _require_enabled()
    requested = (
        (source_validation_job_id, r"control-validation-[a-f0-9]{16}"),
        (source_execution_job_id, r"control-execution-[a-f0-9]{16}"),
        (source_sensitivity_job_id, r"control-sensitivity-[a-f0-9]{16}"),
        (expected_snapshot_sha256, r"[a-f0-9]{64}"),
    )
    if any(not re.fullmatch(pattern, value) for value, pattern in requested):
        raise LiquidityInvalid("Exact three completed Control source IDs and SHA-256 are required.")
    validation = db[VALIDATION_COLLECTION].find_one({"_id": source_validation_job_id})
    execution = db[EXECUTION_COLLECTION].find_one({"_id": source_execution_job_id})
    sensitivity = db[SENSITIVITY_COLLECTION].find_one({"_id": source_sensitivity_job_id})
    if any(row is None for row in (validation, execution, sensitivity)):
        raise LiquidityNotFound("All three prior source jobs must exist in the same MongoDB.")
    v41, v42, v43 = (
        validation.get("result") or {}, execution.get("result") or {},
        sensitivity.get("result") or {},
    )
    source_job_id = str(validation.get("source_job_id") or "")
    if (
        any(item.get("status") != "completed" or item.get("error")
            for item in (validation, execution, sensitivity))
        or any(item.get("snapshot_sha256") != expected_snapshot_sha256
               or item.get("source_job_id") != source_job_id
               for item in (validation, execution, sensitivity))
        or execution.get("source_validation_job_id") != source_validation_job_id
        or sensitivity.get("source_validation_job_id") != source_validation_job_id
        or sensitivity.get("source_execution_job_id") != source_execution_job_id
        or v41.get("source_snapshot_sha256") != expected_snapshot_sha256
        or v42.get("source_snapshot_sha256") != expected_snapshot_sha256
        or v43.get("source_snapshot_sha256") != expected_snapshot_sha256
        or v42.get("research_kind") != "control_execution_feasibility_scenario"
        or v43.get("research_kind") != "control_execution_fixed_sensitivity"
        or v43.get("v1042_scenario_regression") != "verified"
        or not (v41.get("original_shadow") or {}).get("reproduced")
        or any((v.get("numeric_input_integrity") or {}).get("status") != "verified"
               for v in (v41, v42, v43))
        or any(v.get("order_submission") != "never" for v in (v41, v42, v43))
    ):
        raise LiquidityInvalid("The three completed original Control research jobs do not match.")
    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    now = utc_now()
    job_id = "control-liquidity-" + uuid.uuid4().hex[:16]
    record = {
        "_id": job_id, "status": "queued", "stage": "queued",
        "progress": 0, "active_key": ACTIVE_KEY,
        "source_job_id": source_job_id,
        "source_validation_job_id": source_validation_job_id,
        "source_execution_job_id": source_execution_job_id,
        "source_sensitivity_job_id": source_sensitivity_job_id,
        "snapshot_sha256": expected_snapshot_sha256,
        "created_at": now, "updated_at": now,
        "started_at": None, "finished_at": None,
        "logs": [], "result": None, "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise LiquidityConflict("Another Control liquidity research job is active.") from exc
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db, "job_id": job_id,
            "source_job_id": source_job_id,
            "validation_id": source_validation_job_id,
            "execution_id": source_execution_job_id,
            "sensitivity_id": source_sensitivity_job_id,
            "expected_sha256": expected_snapshot_sha256,
            "baseline41": v41, "baseline42": v42, "baseline43": v43,
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
    LOGGER.info("Control Liquidity %s | queued on frozen source %s",
                job_id, source_job_id)
    return _public(record)


def get_liquidity_research(db: Any, job_id: str, *, logs_only: bool = False) -> dict[str, Any]:
    record = db[COLLECTION].find_one({"_id": job_id})
    if record is None:
        raise LiquidityNotFound("Control liquidity research job not found.")
    return _public(record, logs_only=logs_only)
