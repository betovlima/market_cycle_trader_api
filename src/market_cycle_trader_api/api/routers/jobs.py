from __future__ import annotations

from copy import deepcopy
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
    MODEL_TUNING_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from ...schemas.requests import BacktestExecutionRequest
from ...engine.market_data import resolve_backtest_analysis_end_date
from ...services.jobs import public_job, require_job, run_job
from ...services.system_settings import apply_training_runtime_settings, get_system_settings
from ...services.strategy_lab import (
    get_research_reference_context,
    get_research_strategy_context,
    get_research_strategy_model_snapshot,
    get_trader_winner_context,
)
from ...services.results import build_results
from ...services.model_research import (
    apply_execution_profile,
    execution_settings_from_values,
    model_execution_snapshot,
    model_label,
)

router = APIRouter(tags=["jobs"])


def queue_backtest_job(
    *,
    model_values_override: dict[str, Any] | None = None,
    start_thread: bool = True,
    certify_strategy: bool = True,
    tuning_run_id: str | None = None,
    tuning_candidate_id: int | None = None,
    runtime_thread_limit: int | None = None,
    execution_worker_id: str | None = None,
    execution_request_override: dict[str, Any] | None = None,
    execution_metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    




    db = database()
    runtime_settings = get_system_settings(db)
    training_settings = runtime_settings["training"]
    if not bool(training_settings["enabled"]):
        raise HTTPException(status_code=409, detail="Model training is disabled in System Settings.")
    if tuning_run_id is None:
        
        
        
        active_jobs = db[JOBS_COLLECTION].count_documents(
            {"status": {"$in": ["queued", "running"]}, "internal_job": {"$ne": True}}
        )
        if active_jobs >= 1:
            raise HTTPException(
                status_code=409,
                detail="Wait for the active backtest to finish before starting another one.",
            )
        active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
            {"status": {"$in": ["queued", "running", "stop_requested"]}},
            {"_id": 0, "id": 1},
        )
        if active_tuning is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Wait for model tuning {active_tuning.get('id', 'unknown')} to finish before starting a backtest.",
            )

    try:
        if execution_request_override is not None:
            if tuning_run_id is None or model_values_override is None:
                raise RuntimeError("An immutable execution snapshot is valid only for an internal tuning candidate.")
            request_payload = deepcopy(execution_request_override)
            research_model_family = str(request_payload.get("research_model_family") or "")
            if research_model_family not in {"xgboost_utility", "lightgbm_utility", "iqn"}:
                raise RuntimeError("The immutable tuning snapshot has an unsupported research model family.")
            research_model_settings = execution_settings_from_values(
                research_model_family,
                model_values_override,
                settings_revision=1,
                profile_id=f"tuning-{tuning_run_id}-{tuning_candidate_id}",
            )
            request_payload["research_model_family"] = research_model_family
            request_payload["research_model_settings"] = research_model_settings
            
            
            request_payload["research_market_data_mode"] = "database_only"
            
            
            if "repetitions" in model_values_override:
                request_payload["rotation_xgb_repetitions"] = int(model_values_override["repetitions"])
            if "seed_step" in model_values_override:
                request_payload["rotation_seed_step"] = int(model_values_override["seed_step"])
            if "random_state" in model_values_override:
                request_payload["random_state"] = int(model_values_override["random_state"])
            request = BacktestExecutionRequest.model_validate(request_payload)
            selected_model_snapshot = model_execution_snapshot(research_model_family, research_model_settings)
            metadata = dict(execution_metadata_override or {})
            selected_strategy = {
                "id": str(metadata.get("strategy_profile_id") or "tuning-snapshot"),
                "name": str(metadata.get("strategy_profile_name") or "Tuning Snapshot"),
                "revision": int(metadata.get("strategy_profile_revision") or 0),
                "configuration_hash": metadata.get("strategy_configuration_hash"),
            }
            try:
                _, reference_profile = get_research_reference_context(db)
            except Exception:
                reference_profile = {}
            try:
                _, winner_profile = get_trader_winner_context(db)
            except Exception:
                winner_profile = {}
            research_reference_assets = list(request.research_reference_assets)
            research_candidate_assets = list(request.research_candidate_assets)
        else:
            selected_configuration, selected_strategy = get_research_strategy_context(db)
            selected_model_snapshot = get_research_strategy_model_snapshot(db)
            research_model_family = str(selected_model_snapshot["family"])
            research_model_settings = (
                dict(selected_model_snapshot.get("settings_snapshot") or {})
                if isinstance(selected_model_snapshot.get("settings_snapshot"), dict)
                else {}
            )
            if model_values_override is not None:
                research_model_settings = execution_settings_from_values(
                    research_model_family,
                    model_values_override,
                    settings_revision=int(selected_model_snapshot.get("settings_revision") or 1),
                    profile_id=(
                        f"tuning-{tuning_run_id}-{tuning_candidate_id}"
                        if tuning_run_id is not None and tuning_candidate_id is not None
                        else "isolated-research"
                    ),
                )
                selected_model_snapshot = model_execution_snapshot(
                    research_model_family,
                    research_model_settings,
                )
            reference_assets_snapshot, reference_profile = get_research_reference_context(db)
            winner_configuration, winner_profile = get_trader_winner_context(db)
            locked_configuration = apply_training_runtime_settings(db, selected_configuration)
            locked_configuration = apply_execution_profile(
                locked_configuration,
                research_model_family,
                research_model_settings,
            )
            selected_assets = set(locked_configuration.assets)
            calendar_anchor_assets = [symbol for symbol in winner_configuration.assets if symbol in selected_assets]
            if len(calendar_anchor_assets) < 2:
                calendar_anchor_assets = list(locked_configuration.assets)
            research_reference_assets = [symbol for symbol in reference_assets_snapshot if symbol in selected_assets]
            if len(research_reference_assets) < 2:
                research_reference_assets = list(locked_configuration.assets)
            research_reference_set = set(research_reference_assets)
            research_candidate_assets = [symbol for symbol in locked_configuration.assets if symbol not in research_reference_set]
            resolved_analysis_end = resolve_backtest_analysis_end_date(locked_configuration)
            request = BacktestExecutionRequest.model_validate(
                {
                    **locked_configuration.model_dump(mode="python"),
                    "analysis_start_date": locked_configuration.start_date,
                    "analysis_end_date": resolved_analysis_end,
                    "calendar_anchor_assets": calendar_anchor_assets,
                    "research_reference_assets": research_reference_assets,
                    "research_candidate_assets": research_candidate_assets,
                    "research_model_family": research_model_family,
                    "research_model_settings": dict(research_model_settings or {}),
                    "research_market_data_mode": "backtest_bootstrap_missing",
                }
            )
    except (RuntimeError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail=f"Selected backtest strategy is invalid: {exc}") from exc

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    request_payload = request.model_dump(mode="python")
    payload = bson_value(request_payload)
    lifecycle = strategy_lifecycle(payload["strategy_mode"])
    total_runs = int(payload["rotation_xgb_repetitions"])
    research_label = model_label(research_model_family)
    model_snapshot = selected_model_snapshot
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
        "research_model_family": research_model_family,
        "research_model_label": research_label,
        "research_model_profile_id": model_snapshot["profile_id"],
        "research_model_settings_revision": model_snapshot["settings_revision"],
        "research_model_settings_hash": model_snapshot["settings_hash"],
        "strategy_configuration_hash": selected_strategy["configuration_hash"],
        "configuration_locked": True,
        "execution_period_locked": True,
        "live_trades": [],
        "live_trade_count": 0,
        "logs": ["Backtest queued."],
        "progress_detail": {},
        "system_settings_revision": int(runtime_settings["revision"]),
        "training_timeout_seconds": int(training_settings["timeout_seconds"]),
        "winner_engine_compatibility": "api-v1.13.16",
        "trader_winner_strategy_id_at_queue": winner_profile.get("id"),
        "trader_winner_strategy_name_at_queue": winner_profile.get("name"),
        "trader_winner_configuration_hash_at_queue": winner_profile.get("configuration_hash"),
        "trader_winner_api_version_at_queue": winner_profile.get("winner_api_version"),
        "research_reference_strategy_id_at_queue": reference_profile.get("id"),
        "research_reference_strategy_name_at_queue": reference_profile.get("name"),
        "research_reference_configuration_hash_at_queue": reference_profile.get("configuration_hash"),
        "research_reference_assets": research_reference_assets,
        "research_candidate_assets": research_candidate_assets,
        "certifies_strategy": bool(certify_strategy),
        "internal_job": tuning_run_id is not None,
        "tuning_summary_only": tuning_run_id is not None,
        "tuning_run_id": tuning_run_id,
        "tuning_candidate_id": tuning_candidate_id,
        "runtime_thread_limit": max(1, int(runtime_thread_limit)) if runtime_thread_limit else None,
        "execution_worker_id": str(execution_worker_id or "").strip() or None,
    }
    db[JOBS_COLLECTION].insert_one(job)
    if start_thread:
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return public_job(job) or {}


@router.post("/api/jobs", status_code=202)
def create_job() -> dict[str, Any]:
    
    return queue_backtest_job()


@router.get("/api/jobs/latest")
def get_latest_job(
    strategy_profile_id: str | None = None,
    strategy_profile_revision: int | None = None,
    strategy_configuration_hash: str | None = None,
    reusable: bool = False,
) -> dict[str, Any] | None:
    query: dict[str, Any] = {"internal_job": {"$ne": True}}
    strategy_id = str(strategy_profile_id or "").strip()
    configuration_hash = str(strategy_configuration_hash or "").strip()
    if strategy_id:
        query["strategy_profile_id"] = strategy_id
    if strategy_profile_revision is not None:
        query["strategy_profile_revision"] = int(strategy_profile_revision)
    if configuration_hash:
        query["strategy_configuration_hash"] = configuration_hash
    if reusable:
        query["status"] = {"$in": ["queued", "running", "completed"]}
    return public_job(database()[JOBS_COLLECTION].find_one(query, sort=[("created_at", -1)]))


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return public_job(require_job(job_id)) or {}


@router.get("/api/jobs/{job_id}/results")
def get_results(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The backtest has not completed.")
    return build_results(job_id)
