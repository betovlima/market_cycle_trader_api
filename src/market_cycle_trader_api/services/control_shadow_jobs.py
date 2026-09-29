"""Admin-only current-Alpaca Control shadow jobs with durable console logs.

One request downloads a fresh independent RAW/SIP + corporate-actions snapshot
under the MCT API's own dados/ directory, then fits TCC Control and computes a
shadow decision. Trading endpoints, paper plans, Winner and research strategy
selection are deliberately absent from this module.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_shadow_market_data import (
    DATA_DIRECTORY,
    SOURCE_CONTRACT,
    download_current_control_snapshot,
)
from ..engine.market_data import latest_safe_completed_xnys_session
from ..engine.operational_control_preview import build_control_shadow_decision
from ..infrastructure.persistence.mongo_repository import utc_now
from ..tcc_v106_reference.config import ASSETS

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_jobs"
ACTIVE_KEY = "fresh-control-shadow"
ENABLED_ENV = "MCT_CONTROL_SHADOW_API_ENABLED"
MAX_LOGS = 300
_ACTIVE_THREADS: dict[str, threading.Thread] = {}


class ControlShadowConflict(RuntimeError):
    pass


class ControlShadowNotFound(LookupError):
    pass


class ControlShadowUnavailable(RuntimeError):
    pass


def _require_enabled() -> None:
    if str(os.getenv(ENABLED_ENV) or "").strip().lower() != "true":
        raise ControlShadowUnavailable(
            f"Control shadow API is disabled; set {ENABLED_ENV}=true "
            "only in your isolated development API."
        )


def public_control_shadow_job(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(document["_id"]),
        "status": document["status"],
        "stage": document.get("stage"),
        "progress": document.get("progress", 0),
        "completed_session": document["completed_session"],
        "current_asset": document["current_asset"],
        "holding_sessions": document["holding_sessions"],
        "created_at": document.get("created_at"),
        "started_at": document.get("started_at"),
        "updated_at": document.get("updated_at"),
        "finished_at": document.get("finished_at"),
        "logs": list(document.get("logs") or []),
        "result": document.get("result"),
        "error": document.get("error"),
        "source_kind": "fresh_alpaca_raw_sip_local_mct_snapshot",
        "snapshot_directory": document.get("snapshot_directory"),
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
    now = utc_now()
    record = {"at": now, "level": level, "message": message}
    change: dict[str, Any] = {"updated_at": now}
    if stage is not None:
        change["stage"] = stage
    if progress is not None:
        change["progress"] = progress
    db[COLLECTION].update_one(
        {"_id": job_id},
        {
            "$set": change,
            "$push": {"logs": {"$each": [record], "$slice": -MAX_LOGS}},
        },
    )
    getattr(LOGGER, level.lower(), LOGGER.info)(
        "Control Shadow %s | %s", job_id, message,
    )


def _run_control_shadow_job(
    db: Any,
    job_id: str,
    *,
    completed_session: str,
    current_asset: str,
    holding_sessions: int,
) -> None:
    collection = db[COLLECTION]
    try:
        collection.update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {
                "status": "running",
                "started_at": utc_now(),
                "updated_at": utc_now(),
            }},
        )
        _log(
            db, job_id,
            f"Downloading fresh Alpaca RAW/SIP daily history and corporate actions "
            f"through completed XNYS session {completed_session}. "
            "The local MCT dados directory will be created automatically.",
            stage="market_data_download", progress=1,
        )

        def data_progress(phase: str, completed: int, total: int, symbol: str) -> None:
            if phase == "excluded":
                _log(
                    db, job_id, f"{symbol}: structural identity exclusion; see snapshot manifest.",
                    level="WARNING", stage="market_data_download",
                    progress=2 + int(43 * completed / max(1, total)),
                )
            elif phase == "download":
                # The callback runs before and after every symbol to make
                # a slow provider request visible in the PyCharm console.
                _log(
                    db, job_id,
                    f"RAW/SIP + actions {completed}/{total}: {symbol}",
                    stage="market_data_download",
                    progress=2 + int(43 * completed / max(1, total)),
                )

        snapshot = download_current_control_snapshot(
            job_id=job_id,
            completed_session=completed_session,
            progress_callback=data_progress,
        )
        _log(
            db, job_id,
            f"Fresh snapshot published: {snapshot.directory}; "
            f"eligible={len(snapshot.frames)}/{len(ASSETS)}, "
            f"excluded={len(snapshot.manifest['structural_exclusions'])}, "
            f"sha256={snapshot.manifest['snapshot_sha256']}.",
            stage="snapshot_verified", progress=45,
        )
        collection.update_one({"_id": job_id}, {
            "$set": {
                "snapshot_directory": str(snapshot.directory),
                "snapshot_sha256": snapshot.manifest["snapshot_sha256"],
                "updated_at": utc_now(),
            }
        })

        def training_progress(phase: str, completed: int, total: int) -> None:
            if completed % 5 == 0 or completed == total:
                fraction = completed / max(1, total)
                progress = (
                    46 + int(fraction * 24) if phase == "calibration"
                    else 71 + int(fraction * 24)
                )
                _log(
                    db, job_id,
                    f"LightGBM Control {phase}: {completed}/{total} assets.",
                    stage=f"lightgbm_{phase}", progress=progress,
                )

        _log(
            db, job_id, "Training Control and calibrating switch margin; no orders.",
            stage="lightgbm_calibration", progress=46,
        )
        result = build_control_shadow_decision(
            snapshot.frames,
            completed_session=completed_session,
            current_asset=current_asset,
            holding_sessions=holding_sessions,
            progress_callback=training_progress,
        )
        result["input_audit"]["snapshot_sha256"] = snapshot.manifest["snapshot_sha256"]
        result["input_audit"]["structural_exclusions"] = snapshot.manifest["structural_exclusions"]
        result["input_audit"]["source_kind"] = "fresh_alpaca_raw_sip_local_mct_snapshot"
        result["input_audit"]["source_contract"] = SOURCE_CONTRACT
        result["input_audit"]["eligible_assets"] = len(snapshot.frames)
        result["input_audit"]["snapshot_directory"] = str(snapshot.directory)
        result["source_validation"] = "fresh_raw_sip_and_corporate_actions_split_normalized"
        result["order_eligible"] = False
        result["order_submission"] = "never"

        _log(
            db, job_id,
            f"Decision: {current_asset} -> {result['target_asset']}; "
            f"margin={result['effective_switch_margin']:.6f}. "
            "Shadow only; no order created.",
            stage="completed", progress=100,
        )
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "result": result,
                "finished_at": utc_now(),
                "updated_at": utc_now(),
            },
            "$unset": {"active_key": ""},
        })
    except Exception as exc:
        LOGGER.exception(
            "Control Shadow %s | Job failed; no orders were submitted.", job_id
        )
        _log(
            db, job_id,
            f"Shadow failed: {type(exc).__name__}: {str(exc)[:700]}",
            level="ERROR", stage="failed",
        )
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "failed",
                "stage": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:700]}",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
            },
            "$unset": {"active_key": ""},
        })
    finally:
        _ACTIVE_THREADS.pop(job_id, None)


def start_control_shadow_job(
    db: Any,
    *,
    current_asset: str,
    holding_sessions: int,
) -> dict[str, Any]:
    _require_enabled()
    current = str(current_asset or "CASH").strip().upper()
    if current != "CASH" and current not in ASSETS:
        raise ValueError("current_asset must be CASH or an asset of the Control universe.")
    if current == "CASH" and holding_sessions:
        raise ValueError("holding_sessions must be 0 when current_asset is CASH.")
    if holding_sessions < 0:
        raise ValueError("holding_sessions must be nonnegative.")

    completed_session = latest_safe_completed_xnys_session().date().isoformat()
    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    now = utc_now()
    job_id = f"control-shadow-{uuid.uuid4().hex[:16]}"
    document = {
        "_id": job_id,
        "active_key": ACTIVE_KEY,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "completed_session": completed_session,
        "current_asset": current,
        "holding_sessions": holding_sessions,
        "created_at": now,
        "updated_at": now,
        "logs": [],
        "result": None,
        "error": None,
    }
    try:
        collection.insert_one(document)
    except DuplicateKeyError as exc:
        raise ControlShadowConflict(
            "A Control shadow job is already active. Check its status before creating another."
        ) from exc

    thread = threading.Thread(
        target=_run_control_shadow_job,
        args=(db, job_id),
        kwargs={
            "completed_session": completed_session,
            "current_asset": current,
            "holding_sessions": holding_sessions,
        },
        daemon=True,
        name=f"control-shadow-{job_id}",
    )
    try:
        _ACTIVE_THREADS[job_id] = thread
        thread.start()
    except RuntimeError:
        _ACTIVE_THREADS.pop(job_id, None)
        collection.update_one({"_id": job_id}, {
            "$set": {
                "status": "failed",
                "stage": "thread_start_failed",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
            },
            "$unset": {"active_key": ""},
        })
        raise
    LOGGER.info(
        "Control Shadow %s | Queued from /docs: Alpaca RAW/SIP refresh to %s in %s; no orders.",
        job_id, completed_session, DATA_DIRECTORY,
    )
    return public_control_shadow_job(document)


def get_control_shadow_job(db: Any, job_id: str) -> dict[str, Any]:
    doc = db[COLLECTION].find_one({"_id": job_id})
    if doc is None:
        raise ControlShadowNotFound(f"Control shadow job {job_id} was not found.")
    return public_control_shadow_job(doc)


def get_control_shadow_logs(db: Any, job_id: str) -> dict[str, Any]:
    job = get_control_shadow_job(db, job_id)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "logs": job["logs"],
        "snapshot_directory": job["snapshot_directory"],
        "order_submission": "never",
    }
