from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from ...core.config import ACTIVE_STRATEGY_MODE, strategy_lifecycle
from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import DEFAULT_SETTINGS, JOBS_COLLECTION, bson_value, update_settings, utc_now
from ...schemas.requests import BacktestRequest
from ...services.jobs import public_job, require_job, run_job
from ...services.results import build_results

router = APIRouter(tags=["jobs"])


@router.post("/api/jobs", status_code=202)
def create_job(request: BacktestRequest) -> dict[str, Any]:
    db = database()
    if db[JOBS_COLLECTION].find_one({"status": {"$in": ["queued", "running"]}}, {"_id": 1}) is not None:
        raise HTTPException(status_code=409, detail="Another backtest is already running.")

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    request_payload = request.model_dump(mode="python")
    update_settings(db, {key: value for key, value in request_payload.items() if key in DEFAULT_SETTINGS})
    payload = bson_value(request_payload)
    lifecycle = strategy_lifecycle(payload.get("strategy_mode", ACTIVE_STRATEGY_MODE))
    total_runs = (
        (int(payload.get("rotation_xgb_repetitions", 1)) if "xgboost_utility" in payload.get("rotation_models", []) else 0)
        + (int(payload.get("rotation_qrdqn_repetitions", 1)) if "qrdqn" in payload.get("rotation_models", []) else 0)
    )
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
        "live_trades": [],
        "live_trade_count": 0,
        "logs": [
            "Execution snapshot queued: "
            f"strategy={payload.get('strategy_mode')}, "
            f"assets={','.join(payload.get('assets', []))}, "
            f"timeframe={payload.get('timeframe')}, "
            f"rotation_models={','.join(payload.get('rotation_models', []))}, "
            f"rotation_horizon_days={payload.get('rotation_horizon_days')}, "
            f"rotation_purge_days={payload.get('rotation_purge_days')}, "
            f"xgb_repetitions={payload.get('rotation_xgb_repetitions')}, "
            f"qrdqn_repetitions={payload.get('rotation_qrdqn_repetitions')}"
        ],
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
