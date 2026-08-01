from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    bson_value,
    get_alpaca_credentials,
    get_settings,
    get_strategy_policy,
    utc_now,
)
from ...schemas.requests import BacktestExecutionRequest, BacktestRequest, PublicBacktestRequest
from ...schemas.strategy_policy import StrategyPolicy
from ...services.jobs import public_job, require_job, run_job
from ...services.results import build_results

router = APIRouter(tags=["jobs"])


@router.post("/api/jobs", status_code=202)
def create_job(date_range: PublicBacktestRequest) -> dict[str, Any]:
    db = database()
    if db[JOBS_COLLECTION].find_one({"status": {"$in": ["queued", "running"]}}, {"_id": 1}) is not None:
        raise HTTPException(status_code=409, detail="Another analysis is already running.")

    try:
        configuration = BacktestRequest.model_validate(get_settings(db))
        policy = StrategyPolicy.model_validate(get_strategy_policy(db))
        get_alpaca_credentials()
    except (RuntimeError, ValidationError) as exc:
        raise HTTPException(status_code=503, detail="The analysis service is not ready.") from exc

    try:
        request = BacktestExecutionRequest.model_validate(
            {
                **configuration.model_dump(mode="python"),
                "start_date": policy.training_start_date.isoformat(),
                "end_date": policy.training_end_date.isoformat() if policy.training_end_date else None,
                "market_data_provider": policy.market_data_provider,
                "alpaca_historical_feed": policy.historical_feed,
                "alpaca_live_feed": policy.live_feed,
                "analysis_start_date": date_range.start_date.isoformat(),
                "analysis_end_date": date_range.end_date.isoformat() if date_range.end_date else None,
            }
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="The requested analysis window is invalid.") from exc

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    payload = bson_value(request.model_dump(mode="python"))
    repetitions = int(payload["rotation_xgb_repetitions"])
    total_runs = repetitions + int(repetitions > 1)
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
        "total_runs": total_runs,
        "request": payload,
        "public_date_range": {
            "start_date": payload["analysis_start_date"],
            "end_date": payload["analysis_end_date"],
        },
        "configuration_locked": True,
        "live_trades": [],
        "live_trade_count": 0,
        "logs": ["Analysis queued."],
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
        raise HTTPException(status_code=409, detail="The analysis has not completed.")
    return build_results(job_id)
