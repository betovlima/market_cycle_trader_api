"""Background, admin-only validation of an already completed Control snapshot.

Uses the original job's recorded SHA and source identity, never fetches Alpaca
again, writes solely to the dedicated diagnostics collection and local files.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_snapshot_validation import (
    read_verified_control_snapshot,
    run_control_snapshot_validation,
)
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import (
    COLLECTION as SOURCE_COLLECTION,
    _require_enabled,
)

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_validation_jobs"
ACTIVE_KEY = "control-shadow-validation"
_THREADS: dict[str, threading.Thread] = {}


class SnapshotValidationConflict(RuntimeError):
    pass


class SnapshotValidationNotFound(LookupError):
    pass


class SnapshotValidationInvalid(ValueError):
    pass


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(job["_id"]),
        "source_job_id": job["source_job_id"],
        "source_registry_mode": job.get("source_registry_mode", "mongodb"),
        "snapshot_sha256": job["snapshot_sha256"],
        "completed_session": job["completed_session"],
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress", 0),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result"),
        "error": job.get("error"),
        "logs": list(job.get("logs") or []),
        "source_download": "never",
        "order_eligible": False,
        "order_submission": "never",
    }


def _log(
    db: Any,
    job_id: str,
    message: str,
    *,
    level: str = "INFO",
    stage: str | None = None,
    progress: int | None = None,
) -> None:
    stamp = utc_now()
    update = {"updated_at": stamp}
    if stage is not None:
        update["stage"] = stage
    if progress is not None:
        update["progress"] = progress
    db[COLLECTION].update_one(
        {"_id": job_id},
        {
            "$set": update,
            "$push": {
                "logs": {
                    "$each": [{"at": stamp, "level": level, "message": message}],
                    "$slice": -300,
                },
            },
        },
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Validation %s | %s", job_id, message,
    )


def _run_job(
    db: Any,
    job_id: str,
    *,
    source_job_id: str,
    expected_sha256: str,
    original_calibration_score: float | None,
    original_candidate_margin: float | None,
    source_registry_mode: str = "mongodb",
) -> None:
    collection = db[COLLECTION]
    try:
        now = utc_now()
        collection.update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(db, job_id, "Verifying existing MCT RAW/SIP snapshot; NO download or orders.",
             stage="verify_snapshot", progress=1)
        if source_registry_mode == "local_snapshot_sha_pinned":
            _log(
                db, job_id,
                "Source MongoDB job unavailable; local snapshot accepted only "
                "after independently pinned SHA-256 and all source files were checked. "
                "Original calibration reference is unavailable; it will not be "
                "reported as a reproduction failure.",
                stage="verify_snapshot",
            )
        last_mark: dict[str, int] = {}
        def progress(stage: str, completed: int, total: int) -> None:
            fraction = completed / max(1, total)
            if stage == "calibration_fit":
                percentage = 5 + int(15 * fraction)
            elif stage == "margin_diagnostics":
                percentage = 20 + int(8 * fraction)
            elif stage.startswith("oos_replay"):
                percentage = 29 + int(70 * fraction)
            elif stage == "completed":
                percentage = 99
            else:
                percentage = 2
            key = stage.split(":", 1)[0]
            last = last_mark.get(key, -10)
            if percentage - last >= 5 or completed == total or key != "oos_replay" and completed == 0:
                last_mark[key] = percentage
                _log(db, job_id, f"{stage}: {completed}/{total}",
                     stage=key, progress=percentage)

        report = run_control_snapshot_validation(
            source_job_id=source_job_id,
            validation_job_id=job_id,
            expected_sha256=expected_sha256,
            original_calibration_score=original_calibration_score,
            original_candidate_margin=original_candidate_margin,
            progress=progress,
        )
        report["source_registry_mode"] = source_registry_mode
        _log(
            db, job_id,
            f"Completed: source={source_job_id}, margins=4, "
            f"folds={report['oos']['walk_forward_fold_count']}, "
            f"calibration_reproduced={report['original_shadow']['reproduced']}; "
            "all results are research-only.",
            stage="completed", progress=100,
        )
        now = utc_now()
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "completed", "stage": "completed", "progress": 100,
                "result": report, "finished_at": now, "updated_at": now,
            },
            "$unset": {"active_key": ""},
        })
    except Exception as exc:
        LOGGER.exception("Control Validation %s | failed, no trading action", job_id)
        _log(db, job_id, f"{type(exc).__name__}: {str(exc)[:600]}",
             level="ERROR", stage="failed")
        now = utc_now()
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "failed", "stage": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                "finished_at": now, "updated_at": now,
            },
            "$unset": {"active_key": ""},
        })
    finally:
        _THREADS.pop(job_id, None)


def start_snapshot_validation(
    db: Any,
    *,
    source_job_id: str,
    expected_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Accept Mongo source metadata or a locally pinned immutable snapshot.

    A disk-only recovery MUST supply an independent checksum previously
    recorded by the original job. A failed Mongo source must never fall back.
    The source document is not reconstructed or modified.
    """
    _require_enabled()
    provided_sha = str(expected_snapshot_sha256 or "").strip().lower()
    if provided_sha and not re.fullmatch(r"[a-f0-9]{64}", provided_sha):
        raise SnapshotValidationInvalid(
            "expected_snapshot_sha256 must be an independent 64-character SHA-256."
        )
    original = db[SOURCE_COLLECTION].find_one({"_id": source_job_id})
    source_registry_mode = "mongodb"
    if original is None:
        if not provided_sha:
            raise SnapshotValidationNotFound(
                "Control Shadow source job is not in this MongoDB. Supply "
                "expected_snapshot_sha256 from the original completed result "
                "to verify its existing local snapshot without downloading."
            )
        # This loads and verifies manifest, RAW bars, corporate actions and
        # normalized CSVs before accepting a disk-only recovered source.
        try:
            _, manifest, _ = read_verified_control_snapshot(
                source_job_id, expected_sha256=provided_sha,
            )
        except FileNotFoundError as exc:
            raise SnapshotValidationNotFound(
                "The source MongoDB record and MCT local snapshot are missing. "
                "Check the API's dados/control_shadow/snapshots directory; "
                "do not redownload or silently substitute a different source."
            ) from exc
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise SnapshotValidationInvalid(
                "Local source snapshot failed independent SHA-256 or data-contract "
                f"verification: {type(exc).__name__}: {str(exc)[:240]}"
            ) from exc
        sha = str(manifest["snapshot_sha256"])
        completed = str(manifest["completed_session"])
        original_result = {}
        source_registry_mode = "local_snapshot_sha_pinned"
    else:
        if (
            original.get("status") != "completed"
            or original.get("source_kind") not in (
                None, "fresh_alpaca_raw_sip_local_mct_snapshot",
            )
            or original.get("error")
        ):
            raise SnapshotValidationInvalid(
                "A successfully completed fresh-Alpaca source job is required."
            )
        original_result = original.get("result") or {}
        audit = original_result.get("input_audit") or {}
        if (
            original_result.get("status") != "shadow_only"
            or original_result.get("order_submission") != "never"
            or audit.get("source_kind") != "fresh_alpaca_raw_sip_local_mct_snapshot"
        ):
            raise SnapshotValidationInvalid(
                "Original job does not contain the expected fresh-Alpaca audit."
            )
        sha = str(audit.get("snapshot_sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", sha) or sha != str(original.get("snapshot_sha256")):
            raise SnapshotValidationInvalid(
                "Source job snapshot SHA is missing or inconsistent."
            )
        if provided_sha and provided_sha != sha:
            raise SnapshotValidationInvalid(
                "Provided independent SHA-256 disagrees with original MongoDB source."
            )
        completed = str(original.get("completed_session") or "")
        if completed != str(original_result.get("decision_date")):
            raise SnapshotValidationInvalid(
                "Source job cutoff differs from its Control decision date."
            )

    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    job_id = "control-validation-" + uuid.uuid4().hex[:16]
    now = utc_now()
    record = {
        "_id": job_id, "active_key": ACTIVE_KEY, "status": "queued",
        "stage": "queued", "progress": 0, "source_job_id": source_job_id,
        "snapshot_sha256": sha, "completed_session": completed,
        "source_registry_mode": source_registry_mode,
        "created_at": now, "updated_at": now, "logs": [],
        "result": None, "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise SnapshotValidationConflict(
            "A Control snapshot validation is already active."
        ) from exc
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db, "job_id": job_id, "source_job_id": source_job_id,
            "expected_sha256": sha,
            "original_calibration_score": original_result.get("calibration_score"),
            "original_candidate_margin": original_result.get("calibrated_candidate_margin"),
            "source_registry_mode": source_registry_mode,
        },
        name=job_id, daemon=True,
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
    LOGGER.info(
        "Control Validation %s | queued for source %s; no Alpaca refresh, no orders.",
        job_id, source_job_id,
    )
    return _public(record)


def get_snapshot_validation(db: Any, job_id: str, *, logs_only: bool = False) -> dict[str, Any]:
    doc = db[COLLECTION].find_one({"_id": job_id})
    if doc is None:
        raise SnapshotValidationNotFound("Control validation job not found.")
    response = _public(doc)
    if logs_only:
        return {
            "job_id": response["job_id"],
            "source_job_id": response["source_job_id"],
            "status": response["status"],
            "stage": response["stage"],
            "progress": response["progress"],
            "logs": response["logs"],
            "order_submission": "never",
        }
    return response
