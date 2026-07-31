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
    running = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 1},
    )
    if running is not None:
        raise HTTPException(
            status_code=409,
            detail="Another backtest is already running.",
        )

    job_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    # This validated request is the single source of truth for both
    # persistence and execution. Never rebuild it from the stored settings
    # document, because that can reintroduce stale values.
    request_payload = request.model_dump(mode="python")
    settings_payload = {
        key: value
        for key, value in request_payload.items()
        if key in DEFAULT_SETTINGS
    }

    # Persist the current screen values automatically. The return value is
    # intentionally ignored: the job must execute the original snapshot.
    update_settings(db, settings_payload)
    payload = bson_value(request_payload)
    lifecycle = strategy_lifecycle(
        payload.get("strategy_mode", ACTIVE_STRATEGY_MODE)
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
        "total_runs": (
            (
                (
                    int(payload.get("rotation_xgb_repetitions", 1))
                    if "xgboost_utility" in payload.get("rotation_models", [])
                    else 0
                )
                + (
                    int(payload.get("rotation_qrdqn_repetitions", 1))
                    if "qrdqn" in payload.get("rotation_models", [])
                    else 0
                )
            )
            if payload.get("strategy_mode") in {
                "COMPOUND_ROTATION_SWING_1W",
                "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
            }
            else (
                len(payload["assets"])
                * len(payload["model_backends"])
                * (
                    len(payload.get("exit_risk_model_backends", []))
                    if (
                        payload.get("strategy_mode") in {
                            "BOTTOM_ENTRY_EXIT_RISK_V1",
                            "BOTTOM_ENTRY_EXIT_RISK_SWING_1D",
                        }
                        and payload.get("exit_risk_compare_models", False)
                    )
                    else 1
                )
            )
        ),
        "request": payload,
        "live_trades": [],
        "live_trade_count": 0,
        "logs": [
            (
                "Execution snapshot queued: "
                f"strategy={payload.get('strategy_mode')}, "
                f"strategy_status={lifecycle.get('status')}, "
                f"assets={','.join(payload.get('assets', []))}, "
                f"timeframe={payload.get('timeframe')}, "
                f"top_timeframe={payload.get('mtf_top_signal_timeframe')}, "
                f"confirmation_timeframe={payload.get('mtf_top_confirmation_timeframe')}, "
                f"exit_risk_backends={','.join(payload.get('exit_risk_model_backends', [])) if payload.get('exit_risk_compare_models') else payload.get('exit_risk_model_backend')}, "
                f"exit_tolerance_weeks={payload.get('exit_risk_event_tolerance_weeks')}, "
                f"rotation_models={','.join(payload.get('rotation_models', []))}, "
                f"rotation_horizon_days={payload.get('rotation_horizon_days')}, "
                f"rotation_walk_forward_test_days={payload.get('rotation_walk_forward_test_days')}, "
                f"rotation_purge_days={payload.get('rotation_purge_days')}, "
                f"rotation_downside_penalty={payload.get('rotation_downside_penalty')}, "
                f"rotation_drawdown_penalty={payload.get('rotation_drawdown_penalty')}, "
                f"rotation_parallel_models={payload.get('rotation_parallel_models')}, "
                f"xgb_repetitions={payload.get('rotation_xgb_repetitions')}, "
                f"qrdqn_repetitions={payload.get('rotation_qrdqn_repetitions')}, "
                f"qrdqn_parallel_folds={payload.get('qrdqn_parallel_folds')}"
            )
        ],
    }
    db[JOBS_COLLECTION].insert_one(job)

    thread = threading.Thread(
        target=run_job,
        args=(job_id,),
        daemon=True,
    )
    thread.start()
    return public_job(job) or {}


@router.get("/api/jobs/latest")
def get_latest_job() -> dict[str, Any] | None:
    job = database()[JOBS_COLLECTION].find_one(
        {},
        sort=[("created_at", -1)],
    )
    return public_job(job)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return public_job(require_job(job_id)) or {}


@router.get("/api/jobs/{job_id}/results")
def get_results(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The backtest has not completed.",
        )
    return build_results(job_id)
