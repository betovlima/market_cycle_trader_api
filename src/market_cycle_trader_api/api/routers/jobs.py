from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ...core.config import strategy_lifecycle
from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    bson_value,
    get_alpaca_credentials,
    utc_now,
)
from ...schemas.requests import BacktestExecutionRequest
from ...services.jobs import public_job, require_job, run_job
from ...services.system_settings import apply_training_runtime_settings, get_system_settings
from ...services.strategy_lab import get_research_strategy_context
from ...services.results import build_results

router = APIRouter(tags=["jobs"])


@router.post("/api/jobs", status_code=202)
def create_job() -> dict[str, Any]:
    """Queue a job from the Administrator-selected research strategy snapshot.

    The public client supplies no dates or strategy parameters. Selection and
    editing happen only in the Administrator strategy workspace. Trader continues
    using its separately promoted immutable winner snapshot.
    """
    db = database()
    runtime_settings = get_system_settings(db)
    training_settings = runtime_settings["training"]
    if not bool(training_settings["enabled"]):
        raise HTTPException(status_code=409, detail="Model training is disabled in System Settings.")
    active_jobs = db[JOBS_COLLECTION].count_documents({"status": {"$in": ["queued", "running"]}})
    if active_jobs >= 1:
        raise HTTPException(
            status_code=409,
            detail="Wait for the active backtest to finish before starting another one.",
        )

    try:
        selected_configuration, selected_strategy = get_research_strategy_context(db)
        locked_configuration = apply_training_runtime_settings(
            db,
            selected_configuration,
        )
    except (RuntimeError, ValidationError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Selected backtest strategy is invalid: {exc}",
        ) from exc

    if locked_configuration.market_data_provider == "alpaca":
        try:
            get_alpaca_credentials()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        request = BacktestExecutionRequest.model_validate(
            {
                **locked_configuration.model_dump(mode="python"),
                "analysis_start_date": locked_configuration.start_date,
                "analysis_end_date": locked_configuration.end_date,
            }
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Locked execution period is invalid: {exc}",
        ) from exc

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    request_payload = request.model_dump(mode="python")
    payload = bson_value(request_payload)
    lifecycle = strategy_lifecycle(payload["strategy_mode"])
    total_runs = int(payload["rotation_xgb_repetitions"])
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "completed_runs": 0,
        "strategy_lifecycle": lifecycle,
        "total_runs": total_runs,
        "request": payload,
        "strategy_profile_id": selected_strategy["id"],
        "strategy_profile_name": selected_strategy["name"],
        "strategy_profile_revision": selected_strategy["revision"],
        "strategy_configuration_hash": selected_strategy["configuration_hash"],
        "configuration_locked": True,
        "execution_period_locked": True,
        "live_trades": [],
        "live_trade_count": 0,
        "logs": ["Backtest queued."],
        "system_settings_revision": int(runtime_settings["revision"]),
        "training_timeout_seconds": int(training_settings["timeout_seconds"]),
        "winner_engine_compatibility": "api-v1.13.16",
    }
    db[JOBS_COLLECTION].insert_one(job)
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return public_job(job) or {}


@router.get("/api/jobs/latest")
def get_latest_job() -> dict[str, Any] | None:
    return public_job(database()[JOBS_COLLECTION].find_one({}, sort=[("created_at", -1)]))


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return public_job(require_job(job_id)) or {}


@router.get("/api/jobs/{job_id}/results")
def get_results(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The backtest has not completed.")
    return build_results(job_id)
