"""Admin-only v10.8.47 counterfactual advantage research jobs."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..engine.control_counterfactual_advantage_research import (
    run_counterfactual_advantage_research,
)
from ..infrastructure.persistence.mongo_repository import utc_now
from .control_shadow_jobs import _require_enabled
from .control_shadow_validation_jobs import COLLECTION as VALIDATION_COLLECTION
from .control_shadow_execution_jobs import COLLECTION as EXECUTION_COLLECTION
from .control_shadow_liquidity_jobs import COLLECTION as LIQUIDITY_COLLECTION
from .control_shadow_tcn_jobs import COLLECTION as TCN_COLLECTION
from .control_shadow_deep_rank_jobs import COLLECTION as RANK_COLLECTION

LOGGER = logging.getLogger("uvicorn.error")
COLLECTION = "control_shadow_counterfactual_advantage_jobs"
ACTIVE_KEY = "control-counterfactual-advantage-v1047"
_THREADS: dict[str, threading.Thread] = {}


class AdvantageInvalid(ValueError):
    pass


class AdvantageNotFound(LookupError):
    pass


class AdvantageConflict(RuntimeError):
    pass


def _public(record: dict[str, Any], *, logs_only: bool = False) -> dict[str, Any]:
    payload = {
        "job_id": record["_id"],
        "status": record["status"],
        "stage": record.get("stage"),
        "progress": record.get("progress", 0),
        "logs": list(record.get("logs") or []),
        "source_job_id": record["source_job_id"],
        "source_validation_job_id": record["source_validation_job_id"],
        "source_execution_job_id": record["source_execution_job_id"],
        "source_liquidity_job_id": record["source_liquidity_job_id"],
        "source_tcn_job_id": record["source_tcn_job_id"],
        "source_ranking_job_id": record["source_ranking_job_id"],
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
        "Control Advantage %s | %s", job_id, message,
    )


def _run_job(db: Any, job_id: str, *, source_id: str, validation_id: str,
             execution_id: str, liquidity_id: str, tcn_id: str,
             ranking_id: str, expected_sha256: str,
             baselines: tuple[dict[str, Any], ...]) -> None:
    try:
        now = utc_now()
        db[COLLECTION].update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": now, "updated_at": now}},
        )
        _log(
            db, job_id,
            "Running v10.8.44 parity gate and counterfactual meta-veto; no Alpaca/orders.",
            stage="verify_and_train", progress=1,
        )
        last = [-10]
        def progress(stage: str, done: int, total: int) -> None:
            pct = 99 if stage == "completed" else min(98, max(2, int(done)))
            if pct - last[0] >= 4 or stage == "completed":
                last[0] = pct
                _log(db, job_id, f"{stage}: {done}/{total}",
                     stage=str(stage)[:100], progress=pct)

        report = run_counterfactual_advantage_research(
            source_job_id=source_id,
            validation_job_id=validation_id,
            execution_job_id=execution_id,
            liquidity_job_id=liquidity_id,
            tcn_job_id=tcn_id,
            ranking_job_id=ranking_id,
            advantage_job_id=job_id,
            expected_sha256=expected_sha256,
            baseline41=baselines[0],
            baseline42=baselines[1],
            baseline44=baselines[2],
            baseline45=baselines[3],
            baseline46=baselines[4],
            progress=progress,
        )
        now = utc_now()
        _log(db, job_id,
             "v10.8.47 research completed; baseline parity verified; no orders.",
             stage="completed", progress=100)
        db[COLLECTION].update_one(
            {"_id": job_id},
            {"$set": {
                "status": "completed", "stage": "completed", "progress": 100,
                "result": report, "updated_at": now, "finished_at": now,
            }, "$unset": {"active_key": ""}},
        )
    except Exception as exc:
        LOGGER.exception("Control Advantage %s | failed", job_id)
        _log(db, job_id, f"{type(exc).__name__}: {str(exc)[:330]}",
             level="ERROR", stage="failed")
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


def start_counterfactual_advantage_research(
    db: Any, *,
    source_validation_job_id: str,
    source_execution_job_id: str,
    source_liquidity_job_id: str,
    source_tcn_job_id: str,
    source_ranking_job_id: str,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    _require_enabled()
    requested = (
        (source_validation_job_id, r"control-validation-[a-f0-9]{16}"),
        (source_execution_job_id, r"control-execution-[a-f0-9]{16}"),
        (source_liquidity_job_id, r"control-liquidity-[a-f0-9]{16}"),
        (source_tcn_job_id, r"control-tcn-[a-f0-9]{16}"),
        (source_ranking_job_id, r"control-rank-[a-f0-9]{16}"),
        (expected_snapshot_sha256, r"[a-f0-9]{64}"),
    )
    if any(not re.fullmatch(pattern, value) for value, pattern in requested):
        raise AdvantageInvalid("Exact v10.8.41/42/44/45/46 IDs and SHA are required.")

    records = (
        db[VALIDATION_COLLECTION].find_one({"_id": source_validation_job_id}),
        db[EXECUTION_COLLECTION].find_one({"_id": source_execution_job_id}),
        db[LIQUIDITY_COLLECTION].find_one({"_id": source_liquidity_job_id}),
        db[TCN_COLLECTION].find_one({"_id": source_tcn_job_id}),
        db[RANK_COLLECTION].find_one({"_id": source_ranking_job_id}),
    )
    if any(record is None for record in records):
        raise AdvantageNotFound("All completed v10.8.41/42/44/45/46 jobs are required.")
    source_id = str(records[0].get("source_job_id") or "")
    results = tuple((record.get("result") or {}) for record in records)
    if (
        any(record.get("status") != "completed" or record.get("error") for record in records)
        or any(
            record.get("source_job_id") != source_id
            or record.get("snapshot_sha256") != expected_snapshot_sha256
            for record in records
        )
        or records[1].get("source_validation_job_id") != source_validation_job_id
        or records[2].get("source_validation_job_id") != source_validation_job_id
        or records[2].get("source_execution_job_id") != source_execution_job_id
        or records[3].get("source_liquidity_job_id") != source_liquidity_job_id
        or records[4].get("source_liquidity_job_id") != source_liquidity_job_id
        or records[4].get("source_tcn_job_id") != source_tcn_job_id
        or any(
            result.get("source_snapshot_sha256") != expected_snapshot_sha256
            or result.get("source_job_id") != source_id
            or result.get("order_submission") != "never"
            or (result.get("numeric_input_integrity") or {}).get("status") != "verified"
            for result in results
        )
        or not (results[0].get("original_shadow") or {}).get("reproduced")
        or results[2].get("control_reference_parity") != "verified"
        or results[3].get("research_kind") != "control_tcn_fixed_temporal_baseline"
        or results[4].get("research_kind") != "control_deep_pairwise_rank_hybrid"
    ):
        raise AdvantageInvalid("Supplied v10.8.47 source chain is not exact and SHA-matched.")

    collection = db[COLLECTION]
    collection.create_index("active_key", unique=True, sparse=True)
    now = utc_now()
    job_id = "control-advantage-" + uuid.uuid4().hex[:16]
    record = {
        "_id": job_id, "status": "queued", "stage": "queued", "progress": 0,
        "active_key": ACTIVE_KEY,
        "source_job_id": source_id,
        "source_validation_job_id": source_validation_job_id,
        "source_execution_job_id": source_execution_job_id,
        "source_liquidity_job_id": source_liquidity_job_id,
        "source_tcn_job_id": source_tcn_job_id,
        "source_ranking_job_id": source_ranking_job_id,
        "snapshot_sha256": expected_snapshot_sha256,
        "created_at": now, "updated_at": now,
        "started_at": None, "finished_at": None,
        "logs": [], "result": None, "error": None,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as exc:
        raise AdvantageConflict("Another v10.8.47 research job is active.") from exc

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "db": db, "job_id": job_id, "source_id": source_id,
            "validation_id": source_validation_job_id,
            "execution_id": source_execution_job_id,
            "liquidity_id": source_liquidity_job_id,
            "tcn_id": source_tcn_job_id,
            "ranking_id": source_ranking_job_id,
            "expected_sha256": expected_snapshot_sha256,
            "baselines": results,
        },
        name=job_id, daemon=True,
    )
    try:
        _THREADS[job_id] = thread
        thread.start()
    except RuntimeError:
        _THREADS.pop(job_id, None)
        collection.update_one(
            {"_id": job_id},
            {"$set": {"status": "failed", "stage": "thread_start_failed",
                      "finished_at": utc_now()},
             "$unset": {"active_key": ""}},
        )
        raise
    return _public(record)


def get_counterfactual_advantage_research(
    db: Any, job_id: str, *, logs_only: bool = False,
) -> dict[str, Any]:
    record = db[COLLECTION].find_one({"_id": job_id})
    if record is None:
        raise AdvantageNotFound("v10.8.47 Control advantage job not found.")
    return _public(record, logs_only=logs_only)
