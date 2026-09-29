"""Admin-only, read-only Control shadow jobs.

A job verifies the pinned CSV snapshot, trains the frozen scientific Control
model and emits a prospective decision to MongoDB and the Uvicorn console.
It has no paper/live execution imports, order methods or Winner writes.

This first endpoint is deliberately frozen-data only, disabled by default
and intended for a single-process development API. It does not run against
current Alpaca data and is not a production Trader integration.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.operational_control_preview import build_control_shadow_decision
from ..engine.research_market_data import StructuralResearchAssetExclusion
from ..engine.tcc_frozen_reference_source import (
    FROZEN_TCC_END,
    FROZEN_TCC_MAIN_SHA256,
    load_frozen_tcc_main_symbol,
    validate_frozen_tcc_main,
)
from ..infrastructure.persistence.mongo_repository import utc_now
from ..tcc_v106_reference.config import ASSETS

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_jobs"
ACTIVE_KEY = "frozen-control-shadow"
ENABLED_ENV = "MCT_CONTROL_SHADOW_API_ENABLED"
ROOT_ENV = "MCT_CONTROL_SHADOW_FROZEN_ROOT"
MAX_LOGS = 300
_ACTIVE_THREADS: dict[str, threading.Thread] = {}


class ControlShadowConflict(RuntimeError):
    pass


class ControlShadowNotFound(LookupError):
    pass


class ControlShadowUnavailable(RuntimeError):
    pass


def _server_frozen_root() -> Path:
    if str(os.getenv(ENABLED_ENV) or "").strip().lower() != "true":
        raise ControlShadowUnavailable(
            f"Control shadow API is disabled; set {ENABLED_ENV}=true only in an isolated development API."
        )
    raw = str(os.getenv(ROOT_ENV) or "").strip()
    if not raw:
        raise ControlShadowUnavailable(
            f"Configure {ROOT_ENV} on the API server; a filesystem path is not accepted through HTTP."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir() or not (root / "manifest.json").is_file():
        raise ControlShadowUnavailable(
            "Configured Control shadow snapshot directory or manifest is unavailable on this API server."
        )
    return root


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
        "source_kind": "verified_frozen_tcc_snapshot",
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
    root: Path,
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
        _log(db, job_id, "Validating manifest and SHA-256 of frozen RAW/SIP and corporate-actions CSVs.",
             stage="snapshot_validation", progress=1)
        manifest = validate_frozen_tcc_main(root, assets=ASSETS)
        frames: dict[str, Any] = {}
        exclusions: list[dict[str, str]] = []
        for index, symbol in enumerate(ASSETS, start=1):
            try:
                frames[symbol] = load_frozen_tcc_main_symbol(root, symbol, manifest)
            except StructuralResearchAssetExclusion as exc:
                exclusions.append({"symbol": symbol, "reason": str(exc)})
                _log(db, job_id, f"Structural exclusion: {symbol} — {exc}",
                     stage="market_data")
            if index % 5 == 0 or index == len(ASSETS):
                _log(db, job_id, f"Verified {index}/{len(ASSETS)} frozen assets.",
                     stage="market_data", progress=5 + int(15 * index / len(ASSETS)))

        def training_progress(phase: str, completed: int, total: int) -> None:
            # The scientific trainer calls this after every asset.
            if completed % 5 == 0 or completed == total:
                fraction = completed / max(1, total)
                progress = (
                    20 + int(fraction * 35) if phase == "calibration"
                    else 60 + int(fraction * 35)
                )
                _log(
                    db, job_id,
                    f"LightGBM {phase}: {completed}/{total} assets.",
                    stage=f"lightgbm_{phase}", progress=progress,
                )

        _log(db, job_id, "Training frozen Control and calibrating switch margin; no orders.",
             stage="lightgbm_calibration", progress=20)
        result = build_control_shadow_decision(
            frames,
            completed_session=FROZEN_TCC_END,
            current_asset=current_asset,
            holding_sessions=holding_sessions,
            progress_callback=training_progress,
        )
        result["input_audit"]["snapshot_sha256"] = FROZEN_TCC_MAIN_SHA256
        result["input_audit"]["structural_exclusions"] = exclusions
        result["input_audit"]["source_kind"] = "frozen_scientific_reference_not_current_alpaca"
        result["input_audit"]["eligible_assets"] = len(frames)
        result["source_validation"] = "sha256_verified_frozen_tcc_main_raw_sip_and_corporate_actions"
        result["order_eligible"] = False
        result["order_submission"] = "never"

        _log(db, job_id, (
            f"Decision: {current_asset} -> {result['target_asset']}; "
            f"margin={result['effective_switch_margin']:.6f}. "
            "Shadow only; no order created."
        ), stage="completed", progress=100)
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
        LOGGER.exception("Control Shadow %s | Job failed; no orders were submitted.", job_id)
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
    root = _server_frozen_root()
    current = str(current_asset or "CASH").strip().upper()
    if current != "CASH" and current not in ASSETS:
        raise ValueError("current_asset must be CASH or an asset of the frozen TCC universe.")
    if current == "CASH" and holding_sessions:
        raise ValueError("holding_sessions must be 0 when current_asset is CASH.")
    if holding_sessions < 0:
        raise ValueError("holding_sessions must be nonnegative.")

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
        "completed_session": FROZEN_TCC_END,
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
        kwargs={"root": root, "current_asset": current, "holding_sessions": holding_sessions},
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
    LOGGER.info("Control Shadow %s | Queued from /docs; frozen-data, no-order execution.", job_id)
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
        "order_submission": "never",
    }
