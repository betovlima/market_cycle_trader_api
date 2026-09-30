"""Public research-only v10.8.53 one-shot reduced-signature meta-veto jobs."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_reduced_signature_meta_veto_research import (
    run_reduced_signature_meta_veto_research,
)
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_reduced_signature_jobs import COLLECTION as REDUCED_COLLECTION
from .control_shadow_rollout_signature_jobs import COLLECTION as SIGNATURE_COLLECTION
from .control_shadow_policy_rollout_jobs import COLLECTION as ROLLOUT_COLLECTION

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_reduced_signature_meta_veto_jobs"
ACTIVE_KEY = "control-reduced-signature-meta-veto-v1053"
_THREADS: dict[str, threading.Thread] = {}


class MetaVetoInvalid(ValueError):
    pass


class MetaVetoNotFound(LookupError):
    pass


class MetaVetoConflict(RuntimeError):
    pass


def _public(record: dict[str, Any], *, logs_only: bool = False) -> dict[str, Any]:
    payload = {
        "job_id": record["_id"],
        "status": record["status"],
        "stage": record.get("stage"),
        "progress": record.get("progress", 0),
        "logs": list(record.get("logs") or []),
        "source_job_id": record["source_job_id"],
        "source_rollout_job_id": record["source_rollout_job_id"],
        "source_signature_job_id": record["source_signature_job_id"],
        "source_reduced_job_id": record["source_reduced_job_id"],
        "snapshot_sha256": record["snapshot_sha256"],
        "source_download": "never",
        "order_eligible": False,
        "order_submission": "never",
    }
    if not logs_only:
        payload.update({
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "result": record.get("result"),
            "error": record.get("error"),
        })
    return payload


def _log(db: Any, job_id: str, message: str, *, level="INFO",
         stage: str | None = None, progress: int | None = None) -> None:
    now = utc_now()
    values = {"updated_at": now}
    if stage is not None:
        values["stage"] = stage
    if progress is not None:
        values["progress"] = progress
    db[COLLECTION].update_one(
        {"_id": job_id},
        {"$set": values, "$push": {"logs": {"$each": [{
            "at": now, "level": level, "message": str(message)[:400],
        }], "$slice": -300}}},
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Reduced Meta-Veto %s | %s", job_id, message,
    )


def _run_job(
    db: Any,
    job_id: str,
    *,
    source_id: str,
    rollout_id: str,
    signature_id: str,
    reduced_id: str,
    expected_sha256: str,
    rollout_result: dict[str, Any],
    signature_result: dict[str, Any],
    reduced_result: dict[str, Any],
) -> None:
    try:
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(
            db, job_id,
            "Running v10.8.53 one-shot reduced-signature meta-veto; no orders.",
            stage="verify_and_replay", progress=1,
        )

        def progress(stage: str, done: int, total: int) -> None:
            _log(
                db, job_id, f"{stage}: {done}/{total}",
                stage=str(stage)[:100],
                progress=max(1, min(100, int(done))),
            )

        report = run_reduced_signature_meta_veto_research(
            source_job_id=source_id,
            rollout_job_id=rollout_id,
            signature_job_id=signature_id,
            reduced_job_id=reduced_id,
            meta_job_id=job_id,
            expected_sha256=expected_sha256,
            rollout_result=rollout_result,
            signature_result=signature_result,
            reduced_result=reduced_result,
            progress=progress,
        )
        now = utc_now()
        _log(
            db, job_id,
            "v10.8.53 one-shot meta-veto research completed; no order path touched.",
            stage="completed", progress=100,
        )
        db[COLLECTION].update_one(
            {"_id": job_id},
            {"$set": {
                "status": "completed", "stage": "completed", "progress": 100,
                "result": report, "updated_at": now, "finished_at": now,
            }, "$unset": {"active_key": ""}},
        )
    except Exception as exc:
        LOGGER.exception("Control Reduced Meta-Veto %s | failed", job_id)
        _log(
            db, job_id, f"{type(exc).__name__}: {str(exc)[:330]}",
            level="ERROR", stage="failed",
        )
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id},
            {"$set": {
                "status": "failed", "stage": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "updated_at": now, "finished_at": now,
            }, "$unset": {"active_key": ""}},
        )
    finally:
        _THREADS.pop(job_id, None)


def start_reduced_signature_meta_veto_research(
    db: Any,
    *,
    source_reduced_job_id: str,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    _require_enabled()
    if (
        not re.fullmatch(r"control-reduced-[a-f0-9]{16}", source_reduced_job_id)
        or not re.fullmatch(r"[a-f0-9]{64}", expected_snapshot_sha256)
    ):
        raise MetaVetoInvalid("Exact v10.8.50 reduced job ID and SHA are required.")

    reduced = db[REDUCED_COLLECTION].find_one({"_id": source_reduced_job_id})
    if reduced is None:
        raise MetaVetoNotFound("Completed v10.8.50 reduced-signature job is required.")
    reduced_result = reduced.get("result") or {}
    signature_id = str(reduced.get("source_signature_job_id") or "")
    rollout_id = str(reduced.get("source_rollout_job_id") or "")
    source_id = str(reduced.get("source_job_id") or "")

    signature = db[SIGNATURE_COLLECTION].find_one({"_id": signature_id})
    rollout = db[ROLLOUT_COLLECTION].find_one({"_id": rollout_id})
    if signature is None or rollout is None:
        raise MetaVetoNotFound("Verified v10.8.49 and v10.8.48 source jobs are required.")
    signature_result = signature.get("result") or {}
    rollout_result = rollout.get("result") or {}

    records = (reduced, signature, rollout)
    if (
        any(
            record.get("status") != "completed"
            or record.get("error")
            or record.get("snapshot_sha256") != expected_snapshot_sha256
            or record.get("source_job_id") != source_id
            for record in records
        )
        or reduced_result.get("research_kind")
        != "control_reduced_rollout_signature_confirmation"
        or reduced_result.get("reduced_job_id") != source_reduced_job_id
        or reduced_result.get("source_signature_job_id") != signature_id
        or reduced_result.get("source_rollout_job_id") != rollout_id
        or not (reduced_result.get("reduced_signature") or {}).get(
            "reduced_signature_confirmed"
        )
        or signature_result.get("research_kind")
        != "control_rollout_decision_signature_diagnostic"
        or signature_result.get("signature_job_id") != signature_id
        or signature_result.get("source_rollout_job_id") != rollout_id
        or rollout_result.get("research_kind")
        != "control_policy_rollout_advantage_meta_veto"
        or rollout_result.get("rollout_job_id") != rollout_id
        or any(
            result.get("source_snapshot_sha256") != expected_snapshot_sha256
            or result.get("source_unchanged") is not True
            or result.get("order_submission") != "never"
            or (result.get("numeric_input_integrity") or {}).get("status")
            != "verified"
            for result in (reduced_result, signature_result, rollout_result)
        )
    ):
        raise MetaVetoInvalid("Supplied v10.8.48/49/50 research chain is not exact.")

    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    now = utc_now()
    job_id = "control-meta-" + uuid.uuid4().hex[:16]
    record = {
        "_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "active_key": ACTIVE_KEY,
        "source_job_id": source_id,
        "source_rollout_job_id": rollout_id,
        "source_signature_job_id": signature_id,
        "source_reduced_job_id": source_reduced_job_id,
        "snapshot_sha256": expected_snapshot_sha256,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "logs": [],
        "result": None,
        "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise MetaVetoConflict("Another v10.8.53 meta-veto job is active.") from exc

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db,
            "job_id": job_id,
            "source_id": source_id,
            "rollout_id": rollout_id,
            "signature_id": signature_id,
            "reduced_id": source_reduced_job_id,
            "expected_sha256": expected_snapshot_sha256,
            "rollout_result": rollout_result,
            "signature_result": signature_result,
            "reduced_result": reduced_result,
        },
        name=job_id,
        daemon=True,
    )
    try:
        _THREADS[job_id] = thread
        thread.start()
    except RuntimeError:
        _THREADS.pop(job_id, None)
        collection.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "failed", "stage": "thread_start_failed",
                "finished_at": utc_now(),
            }, "$unset": {"active_key": ""}},
        )
        raise
    return _public(record)


def get_reduced_signature_meta_veto_research(
    db: Any, job_id: str, *, logs_only: bool = False,
) -> dict[str, Any]:
    record = db[COLLECTION].find_one({"_id": job_id})
    if record is None:
        raise MetaVetoNotFound("v10.8.53 meta-veto job not found.")
    return _public(record, logs_only=logs_only)
