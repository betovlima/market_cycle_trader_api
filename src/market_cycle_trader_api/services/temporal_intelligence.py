from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import zlib
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from ..core.config import SOURCE_ROOT
from ..core.environment import build_subprocess_environment, load_project_environment
from ..engine.market_data import refresh_market_data_to_live_cutoff
from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    MODEL_TUNING_RUNS_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION,
    TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestExecutionRequest
from .analytics import stateful_strategy_processing_id, temporal_strategy_processing_id
from .model_research import apply_execution_profile, model_execution_snapshot
from .model_tuning_market_snapshot import freeze_tuning_market_snapshot, market_snapshot_exists
from .strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    get_research_strategy_context,
    get_research_strategy_model_snapshot,
    materialize_temporal_strategy,
    update_trader_live_market_cutoff,
)
from .system_settings import apply_training_runtime_settings, get_system_settings

TEMPORAL_ENGINE_MODULE = "market_cycle_trader_api.engine.temporal_intelligence"
TEMPORAL_EXPERIMENT = "temporal_decision_intelligence_v8_winner_anchored_timing"
_NUMERIC_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PIPELINE_WORKERS: set[str] = set()
_ACTIVE_PIPELINE_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

STRATEGY_RESEARCH_PIPELINE_STAGES = ("reference", "temporal", "clustering", "fragile_incumbent", "emerging_trend", "risk", "confidence", "stateful", "milp", "validation")
STRATEGY_RESEARCH_PIPELINE_STAGE_STATES = frozenset({"waiting", "running", "completed", "paused", "stopped", "failed", "skipped"})
STRATEGY_RESEARCH_PIPELINE_STATUSES = frozenset({"idle", "running", "pause_requested", "paused", "stop_requested", "stopped", "completed", "failed"})
STRATEGY_RESEARCH_HISTORY_KEEP = 5


class TemporalIntelligenceConflict(RuntimeError):
    pass


class TemporalIntelligenceNotFound(RuntimeError):
    pass


def _register(run_id: str, process: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES[run_id] = process


def _unregister(run_id: str, process: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        if _ACTIVE_PROCESSES.get(run_id) is process:
            _ACTIVE_PROCESSES.pop(run_id, None)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except (ProcessLookupError, OSError):
        return


def _public_temporal_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    public_result = deepcopy(result)
    for horizon in public_result.get("horizon_metrics") or []:
        if not isinstance(horizon, dict):
            continue
        capital = horizon.get("shadow_capital")
        if isinstance(capital, dict):
            capital.pop("decision_diagnostics", None)
    multi_horizon = public_result.get("multi_horizon_metrics")
    if isinstance(multi_horizon, dict) and isinstance(multi_horizon.get("shadow_capital"), dict):
        multi_horizon["shadow_capital"].pop("decision_diagnostics", None)
    return public_result


def public_temporal_run(document: dict[str, Any] | None, *, include_result: bool = True) -> dict[str, Any] | None:
    if document is None:
        return None
    result = document.get("result") if isinstance(document.get("result"), dict) else None
    return {
        "id": str(document.get("id") or ""),
        "status": str(document.get("status") or "queued"),
        "stage": str(document.get("stage") or "Queued"),
        "progress": float(document.get("progress") or 0.0),
        "created_at": bson_value(document.get("created_at")),
        "updated_at": bson_value(document.get("updated_at")),
        "started_at": bson_value(document.get("started_at")),
        "finished_at": bson_value(document.get("finished_at")),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": document.get("strategy_profile_revision"),
        "strategy_configuration_hash": document.get("strategy_configuration_hash"),
        "strategy_kind": document.get("strategy_kind") or "standard",
        "temporal_strategy_variant": document.get("temporal_strategy_variant"),
        "research_processing_id": document.get("research_processing_id"),
        "research_processing_kind": document.get("research_processing_kind"),
        "research_processing_label": document.get("research_processing_label"),
        "model_family": document.get("model_family"),
        "model_label": document.get("model_label"),
        "model_settings_hash": document.get("model_settings_hash"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "market_data_snapshot_source": document.get("market_data_snapshot_source"),
        "market_data_snapshot_source_run_id": document.get("market_data_snapshot_source_run_id"),
        "deterministic_execution": bool((document.get("request") or {}).get("deterministic_execution")),
        "analysis_end_date": document.get("analysis_end_date"),
        "certified_backtest_cutoff": document.get("certified_backtest_cutoff"),
        "live_market_cutoff": document.get("live_market_cutoff"),
        "research_snapshot_cutoff": document.get("research_snapshot_cutoff") or document.get("analysis_end_date"),
        "horizons": list(document.get("horizons") or []),
        "failure_message": document.get("failure_message"),
        "experiment": document.get("experiment") or (result.get("experiment") if isinstance(result, dict) else None),
        "materialized_strategy_id": document.get("materialized_strategy_id"),
        "materialized_strategy_name": document.get("materialized_strategy_name"),
        "materialized_strategy_at": bson_value(document.get("materialized_strategy_at")),
        "shadow_only": True,
        "strategy_research_pipeline": bson_value(document.get("strategy_research_pipeline")) if isinstance(document.get("strategy_research_pipeline"), dict) else None,
        **({"result": bson_value(_public_temporal_result(result)) if result is not None else None} if include_result else {}),
    }


def _build_execution_request(
    db: Any,
) -> tuple[BacktestExecutionRequest, dict[str, Any], dict[str, Any], dict[str, Any]]:
    winner_configuration, winner_strategy = get_research_strategy_context(db)
    model_snapshot = get_research_strategy_model_snapshot(db)
    model_family = str(model_snapshot.get("family") or "")
    if model_family != "lightgbm_utility":
        raise TemporalIntelligenceConflict(
            "Temporal Decision Intelligence v8 requires the selected Strategy Research baseline to use LightGBM."
        )
    model_settings = deepcopy(model_snapshot.get("settings_snapshot") or {})

    locked = apply_training_runtime_settings(db, winner_configuration)
    locked = apply_execution_profile(locked, model_family, model_settings)
    anchors = list(locked.assets)
    reference = list(locked.assets)
    candidates: list[str] = []

    # Research starts from current operational market data, not from the historical
    # certification cutoff. Refresh once, then freeze that exact cutoff for the run.
    live_refresh = refresh_market_data_to_live_cutoff(locked)
    resolved_end = str(live_refresh["live_market_cutoff"])
    update_trader_live_market_cutoff(
        db,
        cutoff=resolved_end,
        source="temporal_research_boundary_refresh",
    )

    request = BacktestExecutionRequest.model_validate({
        **locked.model_dump(mode="python"),
        "analysis_start_date": locked.start_date,
        "analysis_end_date": resolved_end,
        "calendar_anchor_assets": anchors,
        "research_reference_assets": reference,
        "research_candidate_assets": candidates,
        "research_model_family": model_family,
        "research_model_settings": model_settings,
        "research_market_data_mode": "database_only",
        "deterministic_execution": True,
        "xgb_n_jobs": 1,
        "numeric_thread_limit": 1,
    })
    market_context = {
        "certified_backtest_cutoff": winner_strategy.get("certified_backtest_cutoff"),
        "live_market_cutoff": resolved_end,
        "research_snapshot_cutoff": resolved_end,
    }
    return request, winner_strategy, model_snapshot, market_context


def _stable_temporal_market_snapshot(
    db: Any,
    *,
    strategy_configuration_hash: str,
    model_settings_hash: str,
    analysis_end_date: str | None,
) -> tuple[str | None, str | None]:
    cursor = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find(
        {
            "status": "completed",
            "experiment": TEMPORAL_EXPERIMENT,
            "strategy_configuration_hash": str(strategy_configuration_hash or ""),
            "model_settings_hash": str(model_settings_hash or ""),
            "analysis_end_date": analysis_end_date,
            "market_data_snapshot_id": {"$type": "string"},
        },
        {"_id": 0, "id": 1, "market_data_snapshot_id": 1, "created_at": 1},
    ).sort("created_at", 1)
    for item in cursor:
        snapshot_id = str(item.get("market_data_snapshot_id") or "").strip().lower()
        if snapshot_id and market_snapshot_exists(db, snapshot_id):
            return snapshot_id, str(item.get("id") or "").strip() or None
    return None, None


def _research_processing_context(db: Any, strategy: dict[str, Any]) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    strategy_id = str(strategy.get("id") or "").strip()
    from ..milp_decision.processing import resolve_research_processing_context
    try:
        milp_context = resolve_research_processing_context(strategy)
    except ValueError as exc:
        raise TemporalIntelligenceConflict(str(exc)) from exc
    if milp_context is not None:
        return milp_context
    is_stateful = (
        str(strategy.get("strategy_kind") or "") == "temporal_intelligence"
        and str(strategy.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
        and str(strategy.get("tuning_target") or "") == "stateful_transition"
    )
    if is_stateful:
        from .temporal_winner_transition_stateful import build_stateful_live_runtime_bundle
        try:
            bundle = build_stateful_live_runtime_bundle(db, strategy)
        except Exception as exc:
            raise TemporalIntelligenceConflict(
                f"Unable to execute the selected Stateful Strategy Research snapshot: {exc}"
            ) from exc
        return (
            stateful_strategy_processing_id(strategy_id),
            "strategy_research_stateful",
            "Strategy Research · Stateful",
            bundle,
        )

    if str(strategy.get("strategy_kind") or "") == "temporal_intelligence":
        source_run_id = str(strategy.get("source_temporal_run_id") or "").strip()
        policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
        source_run_id = source_run_id or str(policy.get("source_run_id") or "").strip()
        if source_run_id:
            source_run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
                {"id": source_run_id, "status": "completed"}, {"_id": 0, "id": 1}
            )
            if source_run is None:
                raise TemporalIntelligenceConflict(
                    "The selected Temporal Strategy Research source run is unavailable."
                )
            return (
                temporal_strategy_processing_id(strategy_id),
                "strategy_research_temporal",
                "Strategy Research · Temporal",
                None,
            )

    exact_job = db[JOBS_COLLECTION].find_one(
        {
            "status": "completed",
            "strategy_profile_id": strategy_id,
            "strategy_profile_revision": int(strategy.get("revision") or 1),
            "strategy_configuration_hash": str(strategy.get("configuration_hash") or ""),
        },
        {"_id": 0, "id": 1},
        sort=[("finished_at", -1), ("created_at", -1)],
    )
    processing_id = str((exact_job or {}).get("id") or "").strip() or None
    return processing_id, ("backtest" if processing_id else None), ("Simulation Backtest" if processing_id else None), None


def _strategy_research_run_is_protected(db: Any, run_id: str) -> bool:
    return db[STRATEGY_PROFILES_COLLECTION].find_one(
        {"source_temporal_run_id": str(run_id)},
        {"_id": 1},
    ) is not None


def _delete_strategy_research_run_data(db: Any, run_id: str, *, delete_run: bool) -> dict[str, int]:
    run_key = str(run_id)
    deleted = {
        "observations": int(db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "artifacts": int(db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "risk": int(db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "alternative_action": int(db[TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "operational_policy_qualification": int(db[TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "intervention": int(db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "confidence": int(db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
        "decision_policy": int(db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].delete_many({"run_id": run_key}).deleted_count or 0),
    }
    from ..milp_decision.persistence import delete_run_results
    from ..leadership_regime.service import delete_run_results as delete_leadership_results
    deleted["decision_optimization"] = delete_run_results(db, run_key)
    deleted["leadership_regime"] = delete_leadership_results(db, run_key)
    from ..opportunity_drought.service import delete_run_results as delete_opportunity_drought_results
    deleted["opportunity_drought"] = delete_opportunity_drought_results(db, run_key)
    from ..fragile_incumbent.service import delete_run_results as delete_fragile_incumbent_results
    deleted["fragile_incumbent"] = delete_fragile_incumbent_results(db, run_key)
    from ..regime_clustering.service import delete_run_results as delete_regime_clustering_results
    deleted["clustering"] = delete_regime_clustering_results(db, run_key)
    from ..emerging_trend.service import delete_run_results as delete_emerging_trend_results
    deleted["emerging_trend"] = delete_emerging_trend_results(db, run_key)
    deleted["runs"] = int(db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].delete_many({"id": run_key}).deleted_count or 0) if delete_run else 0
    return deleted


def purge_strategy_research_history(
    db: Any,
    *,
    strategy_profile_id: str | None,
    keep: int = STRATEGY_RESEARCH_HISTORY_KEEP,
    exclude_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    strategy_id = str(strategy_profile_id or "").strip()
    if not strategy_id:
        return {"purged_runs": 0, "deleted_total": 0, "run_ids": []}
    excluded = {str(value) for value in (exclude_run_ids or set()) if str(value)}
    cursor = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find(
        {
            "strategy_profile_id": strategy_id,
            "status": {"$in": ["completed", "failed", "interrupted", "cancelled", "stopped"]},
        },
        {"_id": 0, "id": 1, "created_at": 1},
    ).sort("created_at", -1)
    retained_unprotected = 0
    purged_ids: list[str] = []
    deleted_total = 0
    for item in cursor:
        candidate_id = str(item.get("id") or "").strip()
        if not candidate_id or candidate_id in excluded or _strategy_research_run_is_protected(db, candidate_id):
            continue
        if retained_unprotected < max(0, int(keep)):
            retained_unprotected += 1
            continue
        counts = _delete_strategy_research_run_data(db, candidate_id, delete_run=True)
        purged_ids.append(candidate_id)
        deleted_total += sum(counts.values())
    return {"purged_runs": len(purged_ids), "deleted_total": int(deleted_total), "run_ids": purged_ids}


def start_temporal_intelligence(db: Any, *, actor_email: str | None, start_thread: bool = True) -> dict[str, Any]:
    active = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active is not None:
        raise TemporalIntelligenceConflict(
            f"Wait for Temporal Intelligence {active.get('id', 'unknown')} to finish before starting another run."
        )
    active_backtest = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active_backtest is not None:
        raise TemporalIntelligenceConflict("Wait for the active Simulation Backtest to finish before starting Temporal Intelligence.")
    active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active_tuning is not None:
        raise TemporalIntelligenceConflict("Wait for the active Model Tuning campaign to finish before starting Temporal Intelligence.")

    runtime_settings = get_system_settings(db)
    if not bool(runtime_settings["training"]["enabled"]):
        raise TemporalIntelligenceConflict("Model training is disabled in System Settings.")

    try:
        request, strategy, model_snapshot, market_context = _build_execution_request(db)
        research_processing_id, research_processing_kind, research_processing_label, stateful_reference_bundle = _research_processing_context(db, strategy)
        snapshot_id, source_run_id = _stable_temporal_market_snapshot(
            db,
            strategy_configuration_hash=str(strategy.get("configuration_hash") or ""),
            model_settings_hash=str(model_snapshot.get("settings_hash") or ""),
            analysis_end_date=request.analysis_end_date,
        )
        snapshot_source = "temporal_baseline_reuse" if snapshot_id else "new_frozen_snapshot"
        if not snapshot_id:
            frozen = freeze_tuning_market_snapshot(db, request.model_dump(mode="python"))
            snapshot_id = str(frozen.get("snapshot_id") or frozen.get("signature") or "").strip().lower()
        if not snapshot_id:
            raise RuntimeError("Unable to freeze the Temporal Intelligence market-data snapshot.")
        request = request.model_copy(update={
            "research_market_data_mode": "database_only",
            "research_market_data_snapshot_id": snapshot_id,
            "expected_market_data_signature_sha256": snapshot_id,
            "deterministic_execution": True,
            "xgb_n_jobs": 1,
            "numeric_thread_limit": 1,
        })
    except TemporalIntelligenceConflict:
        raise
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise TemporalIntelligenceConflict(str(exc)) from exc

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-temporal-" + uuid.uuid4().hex[:8]
    purge_strategy_research_history(
        db,
        strategy_profile_id=str(strategy.get("id") or ""),
        keep=STRATEGY_RESEARCH_HISTORY_KEEP,
        exclude_run_ids={value for value in (run_id, source_run_id) if value},
    )
    now = utc_now()
    document = {
        "id": run_id,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "actor_email": (actor_email or "").strip().lower() or None,
        "strategy_profile_id": strategy.get("id"),
        "strategy_profile_name": strategy.get("name"),
        "strategy_profile_revision": strategy.get("revision"),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "strategy_kind": strategy.get("strategy_kind") or "standard",
        "temporal_strategy_variant": strategy.get("temporal_strategy_variant"),
        "research_processing_id": research_processing_id,
        "research_processing_kind": research_processing_kind,
        "research_processing_label": research_processing_label,
        "stateful_reference_bundle": bson_value(stateful_reference_bundle) if isinstance(stateful_reference_bundle, dict) else None,
        "model_family": model_snapshot.get("family"),
        "model_label": model_snapshot.get("label"),
        "model_settings_hash": model_snapshot.get("settings_hash"),
        "model_settings_revision": model_snapshot.get("settings_revision"),
        "market_data_snapshot_id": snapshot_id,
        "market_data_snapshot_source": snapshot_source,
        "market_data_snapshot_source_run_id": source_run_id,
        "analysis_end_date": request.analysis_end_date,
        "certified_backtest_cutoff": market_context.get("certified_backtest_cutoff"),
        "live_market_cutoff": market_context.get("live_market_cutoff"),
        "research_snapshot_cutoff": market_context.get("research_snapshot_cutoff"),
        "horizons": list(request.rotation_target_horizons),
        "request": bson_value(request.model_dump(mode="python")),
        "experiment": TEMPORAL_EXPERIMENT,
        "result": None,
        "failure_message": None,
        "technical_error": None,
        "shadow_only": True,
        "system_settings_revision": int(runtime_settings["revision"]),
        "training_timeout_seconds": int(runtime_settings["training"]["timeout_seconds"]),
    }
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].insert_one(deepcopy(document))
    if start_thread:
        threading.Thread(target=_run_temporal_process, args=(db, run_id), daemon=True).start()
    return public_temporal_run(document) or {}


def _run_temporal_process(db: Any, run_id: str) -> None:
    load_project_environment()
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}) or {}
    timeout_seconds = max(300, int(run.get("training_timeout_seconds") or 21_600))
    python_path = str(SOURCE_ROOT)
    existing_python_path = os.environ.get("PYTHONPATH", "")
    if existing_python_path:
        python_path = python_path + os.pathsep + existing_python_path
    request_payload = run.get("request") if isinstance(run.get("request"), dict) else {}
    numeric_environment: dict[str, str] = {}
    if bool(request_payload.get("deterministic_execution")):
        numeric_threads = max(1, int(request_payload.get("numeric_thread_limit") or 1))
        numeric_environment = {key: str(numeric_threads) for key in _NUMERIC_THREAD_ENVIRONMENT_KEYS}
        numeric_environment["MCT_MODEL_THREADS_OVERRIDE"] = str(numeric_threads)
    environment = build_subprocess_environment({
        "PYTHONPATH": python_path,
        **numeric_environment,
    })
    command = [sys.executable, "-u", "-m", TEMPORAL_ENGINE_MODULE, "--run-id", run_id]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(SOURCE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        _register(run_id, process)
        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
            {"id": run_id}, {"$set": {"process_id": process.pid, "updated_at": utc_now()}}
        )
        timed_out = threading.Event()

        def kill_for_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        timer = threading.Timer(timeout_seconds, kill_for_timeout)
        timer.daemon = True
        timer.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stripped = str(line).strip()
                if stripped.startswith("ERROR"):
                    print(stripped, file=sys.stderr, flush=True)
            return_code = process.wait()
        finally:
            timer.cancel()
            _unregister(run_id, process)

        current = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}) or {}
        if str(current.get("status")) == "stop_requested":
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {"status": "cancelled", "stage": "Stopped", "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}},
            )
            return
        if timed_out.is_set():
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {"status": "failed", "stage": "Temporal Intelligence failed", "failure_message": "Temporal Intelligence exceeded the configured training time limit.", "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}},
            )
            return
        refreshed = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}) or {}
        if return_code != 0 and str(refreshed.get("status")) != "failed":
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {"status": "failed", "stage": "Temporal Intelligence failed", "failure_message": "Temporal Intelligence execution failed. Check protected server logs.", "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}},
            )
        else:
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
                {"id": run_id}, {"$set": {"return_code": return_code, "updated_at": utc_now()}, "$unset": {"process_id": ""}}
            )
    except Exception as exc:
        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {"$set": {"status": "failed", "stage": "Temporal Intelligence failed", "failure_message": "Temporal Intelligence execution failed. Check protected server logs.", "technical_error": str(exc)[:2000], "finished_at": utc_now(), "updated_at": utc_now()}, "$unset": {"process_id": ""}},
        )


def validate_temporal_research_processing(db: Any, run_id: str, processing_id: str) -> None:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {"_id": 0, "research_processing_id": 1, "strategy_profile_name": 1},
    )
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    expected = str(document.get("research_processing_id") or "").strip()
    supplied = str(processing_id or "").strip()
    if not expected:
        raise TemporalIntelligenceConflict(
            "This Temporal Intelligence run predates the unified Strategy Research source binding and cannot start a new full analysis."
        )
    if supplied != expected:
        strategy_name = str(document.get("strategy_profile_name") or "the selected Strategy Research")
        raise TemporalIntelligenceConflict(
            f"Full analysis must use the processing bound to {strategy_name}. Expected {expected}."
        )


def get_temporal_intelligence_run(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    return public_temporal_run(document) or {}


def get_latest_temporal_intelligence_run(db: Any) -> dict[str, Any] | None:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({}, sort=[("created_at", -1)])
    return public_temporal_run(document)


def list_temporal_intelligence_history(db: Any, *, limit: int = 30) -> list[dict[str, Any]]:
    items = (
        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION]
        .find({}, {"request": 0, "result": 0, "technical_error": 0})
        .sort("created_at", -1)
        .limit(max(1, min(100, int(limit))))
    )
    return [public_temporal_run(item, include_result=False) or {} for item in items]



def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _load_temporal_artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        collection = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION]
    except (AssertionError, KeyError):
        return rows
    cursor = collection.find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        artifact_rows = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            decoded = zlib.decompress(bytes(item["payload"])).decode("utf-8")
            artifact_rows = json.loads(decoded)
        for row in artifact_rows:
            if isinstance(row, dict):
                rows.append(bson_value(dict(row)))
    return rows


def _temporal_strategy_label(experiment: str) -> str:
    labels = {
        "temporal_decision_intelligence_v8_winner_anchored_timing": "Temporal Intelligence v8 — Winner-Anchored Timing",
        "temporal_decision_intelligence_v7_rotation_before_cash": "Temporal Intelligence v7 — Rotation Before CASH",
        "temporal_decision_intelligence_v6_adaptive_trend_capture": "Temporal Intelligence v6 — Adaptive Trend Capture",
        "temporal_decision_intelligence_v5_trend_capture_hysteresis": "Temporal Intelligence v5 — Trend Capture + Hysteresis",
        "temporal_decision_intelligence_v4_multi_horizon": "Temporal Intelligence v4 — Multi-Horizon",
    }
    return labels.get(str(experiment), "Temporal Intelligence Strategy")


def _temporal_policy_strategy_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    result = document.get("result") if isinstance(document.get("result"), dict) else {}
    experiment = str(document.get("experiment") or result.get("experiment") or "")
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    winner = result.get("winner_reference") if isinstance(result.get("winner_reference"), dict) else {}
    parameters = {
        "decision_policy": capital.get("decision_policy") or (result.get("decision_policy") or {}).get("policy"),
        "timing_base_weak_threshold": capital.get("timing_base_weak_threshold"),
        "timing_challenger_minimum": capital.get("timing_challenger_minimum"),
        "timing_minimum_advantage": capital.get("timing_minimum_advantage"),
        "timing_maximum_advantage": capital.get("timing_maximum_advantage"),
        "entry_horizons": list(multi.get("entry_horizons") or []),
        "hold_horizons": list(multi.get("hold_horizons") or []),
        "risk_horizons": list(multi.get("risk_horizons") or []),
        "horizons": list(result.get("horizons") or document.get("horizons") or []),
    }
    validation = {
        "initial_capital": capital.get("initial_capital"),
        "ending_capital": capital.get("ending_capital"),
        "total_return": capital.get("total_return"),
        "cagr": capital.get("cagr"),
        "sharpe": capital.get("sharpe"),
        "max_drawdown": capital.get("max_drawdown"),
        "exposure": capital.get("exposure"),
        "switch_count": capital.get("switch_count"),
        "timing_override_count": capital.get("timing_override_count"),
        "capital_vs_winner": multi.get("capital_vs_winner"),
        "capital_vs_benchmark": multi.get("capital_vs_benchmark"),
        "winner_ending_capital": winner.get("ending_capital"),
        "winner_cagr": winner.get("cagr"),
        "winner_sharpe": winner.get("sharpe"),
        "winner_max_drawdown": winner.get("max_drawdown"),
        "folds": bson_value(result.get("multi_horizon_fold_metrics") or []),
        "cost_stress": bson_value(capital.get("cost_stress") or []),
    }
    return {
        "schema_version": 1,
        "family": "winner_anchored_temporal_timing" if experiment == "temporal_decision_intelligence_v8_winner_anchored_timing" else "temporal_decision_policy",
        "label": _temporal_strategy_label(experiment),
        "experiment": experiment,
        "source_run_id": str(document.get("id") or ""),
        "source_strategy_id": document.get("strategy_profile_id"),
        "source_strategy_revision": document.get("strategy_profile_revision"),
        "source_strategy_configuration_hash": document.get("strategy_configuration_hash"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "market_data_snapshot_source": document.get("market_data_snapshot_source"),
        "market_data_snapshot_source_run_id": document.get("market_data_snapshot_source_run_id"),
        "analysis_end_date": document.get("analysis_end_date"),
        "certified_backtest_cutoff": document.get("certified_backtest_cutoff"),
        "live_market_cutoff": document.get("live_market_cutoff"),
        "research_snapshot_cutoff": document.get("research_snapshot_cutoff") or document.get("analysis_end_date"),
        "model_family": document.get("model_family"),
        "model_settings_hash": document.get("model_settings_hash"),
        "parameters": bson_value(parameters),
        "validation": bson_value(validation),
    }


def materialize_temporal_intelligence_strategy(db: Any, run_id: str, *, actor_email: str | None) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    if str(document.get("status") or "") != "completed" or not isinstance(document.get("result"), dict):
        raise TemporalIntelligenceConflict("Only a successfully completed Temporal Intelligence run can create a Strategy.")

    existing_strategy_id = str(document.get("materialized_strategy_id") or "").strip()
    if existing_strategy_id:
        try:
            from .strategy_lab import get_strategy
            return {"created": False, "strategy": get_strategy(db, existing_strategy_id)}
        except StrategyLabNotFound:
            pass

    snapshot = _temporal_policy_strategy_snapshot(document)
    experiment = str(snapshot.get("experiment") or "")
    label = str(snapshot.get("label") or "Temporal Intelligence Strategy")
    analysis_end = str(document.get("analysis_end_date") or "").strip()
    run_suffix = str(run_id).split("-")[-1][:8]
    name_suffix = analysis_end[:10] if analysis_end else run_suffix
    name = f"{label} — {name_suffix}"
    description = f"Generated from Temporal Intelligence run {run_id}."

    request_payload = document.get("request") if isinstance(document.get("request"), dict) else {}
    research_model_snapshot = None
    research_settings = request_payload.get("research_model_settings") if isinstance(request_payload.get("research_model_settings"), dict) else {}
    model_family = str(document.get("model_family") or request_payload.get("research_model_family") or "")
    if model_family and research_settings:
        research_model_snapshot = model_execution_snapshot(model_family, research_settings)

    try:
        materialized = materialize_temporal_strategy(
            db,
            run_id=str(run_id),
            source_strategy_id=str(document.get("strategy_profile_id") or ""),
            source_strategy_revision=int(document.get("strategy_profile_revision") or 0) or None,
            source_configuration_hash=str(document.get("strategy_configuration_hash") or "") or None,
            name=name,
            description=description,
            experiment=experiment,
            policy_snapshot=snapshot,
            actor_email=actor_email,
            research_model_snapshot=research_model_snapshot,
        )
    except (StrategyLabConflict, StrategyLabNotFound, StrategyLabError, ValueError) as exc:
        raise TemporalIntelligenceConflict(str(exc)) from exc

    strategy = materialized["strategy"]
    now = utc_now()
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {
            "materialized_strategy_id": strategy.get("id"),
            "materialized_strategy_name": strategy.get("name"),
            "materialized_strategy_at": now,
            "updated_at": now,
        }},
    )
    return materialized


def build_temporal_intelligence_export(
    db: Any,
    run_id: str,
    *,
    start_month: str | None = None,
    end_month: str | None = None,
) -> bytes:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    result = document.get("result") if isinstance(document.get("result"), dict) else None
    if result is None:
        raise TemporalIntelligenceConflict("Temporal Intelligence results are not available until the execution finishes successfully.")

    horizon_rows = [
        bson_value({key: value for key, value in item.items() if key not in {"confidence_bins", "risk_buckets", "signal_metrics", "shadow_capital"}})
        for item in (result.get("horizon_metrics") or [])
        if isinstance(item, dict)
    ]
    fold_rows = [
        bson_value({key: value for key, value in item.items() if key not in {"signal_metrics", "shadow_capital"}})
        for item in (result.get("fold_metrics") or [])
        if isinstance(item, dict)
    ]
    forecast_rows = [bson_value(dict(item)) for item in (result.get("latest_forecasts") or []) if isinstance(item, dict)]
    multi_horizon_latest_rows = [
        bson_value(dict(item)) for item in (result.get("multi_horizon_latest_forecasts") or []) if isinstance(item, dict)
    ]

    multi_horizon = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    multi_horizon_capital = multi_horizon.get("shadow_capital") if isinstance(multi_horizon.get("shadow_capital"), dict) else {}
    multi_horizon_rows: list[dict[str, Any]] = []
    if multi_horizon:
        multi_horizon_rows.append(bson_value({
            **{key: value for key, value in multi_horizon.items() if key != "shadow_capital"},
            **{key: value for key, value in multi_horizon_capital.items() if key not in {"action_counts", "decision_diagnostics", "economic_curve", "cost_stress"}},
            **{f"actions_{key}": value for key, value in (multi_horizon_capital.get("action_counts") or {}).items()},
        }))
    multi_horizon_fold_rows: list[dict[str, Any]] = []
    for fold in result.get("multi_horizon_fold_metrics") or []:
        if not isinstance(fold, dict):
            continue
        capital = fold.get("shadow_capital") if isinstance(fold.get("shadow_capital"), dict) else {}
        multi_horizon_fold_rows.append(bson_value({
            **{key: value for key, value in fold.items() if key != "shadow_capital"},
            **{key: value for key, value in capital.items() if key not in {"action_counts", "decision_diagnostics", "economic_curve", "cost_stress"}},
            **{f"actions_{key}": value for key, value in (capital.get("action_counts") or {}).items()},
        }))
    multi_horizon_diagnostic_rows = [
        bson_value(dict(item))
        for item in (multi_horizon_capital.get("decision_diagnostics") or [])
        if isinstance(item, dict)
    ]
    external_diagnostic_rows = _load_temporal_artifact_rows(db, run_id, "decision_diagnostics")
    if external_diagnostic_rows:
        multi_horizon_diagnostic_rows.extend(
            {key: value for key, value in row.items() if key != "artifact_kind"}
            for row in external_diagnostic_rows
            if row.get("artifact_kind") == "multi_horizon_decision_diagnostics"
        )
    multi_horizon_equity_rows = [
        {key: value for key, value in row.items() if key != "artifact_kind"}
        for row in external_diagnostic_rows
        if row.get("artifact_kind") == "multi_horizon_equity_curve"
    ]
    timing_override_attribution_rows = [
        dict(row)
        for row in multi_horizon_equity_rows
        if bool(row.get("temporal_timing_candidate"))
    ]

    temporal_cost_stress = {
        float(item.get("one_side_cost_bps")): item
        for item in (multi_horizon_capital.get("cost_stress") or [])
        if isinstance(item, dict) and item.get("one_side_cost_bps") is not None
    }
    anchor_capital = multi_horizon.get("winner_anchor_replay") if isinstance(multi_horizon.get("winner_anchor_replay"), dict) else {}
    anchor_cost_stress = {
        float(item.get("one_side_cost_bps")): item
        for item in (anchor_capital.get("cost_stress") or [])
        if isinstance(item, dict) and item.get("one_side_cost_bps") is not None
    }
    cost_stress_rows: list[dict[str, Any]] = []
    for cost_bps in sorted(set(temporal_cost_stress) | set(anchor_cost_stress)):
        temporal_row = temporal_cost_stress.get(cost_bps) or {}
        anchor_row = anchor_cost_stress.get(cost_bps) or {}
        temporal_ending = temporal_row.get("ending_capital")
        anchor_ending = anchor_row.get("ending_capital")
        cost_stress_rows.append(bson_value({
            "one_side_cost_bps": cost_bps,
            "temporal_ending_capital": temporal_ending,
            "winner_anchor_ending_capital": anchor_ending,
            "capital_lift_vs_winner_anchor": (
                float(temporal_ending) / float(anchor_ending) - 1.0
                if temporal_ending is not None and anchor_ending not in {None, 0, 0.0}
                else None
            ),
            "temporal_total_return": temporal_row.get("total_return"),
            "winner_anchor_total_return": anchor_row.get("total_return"),
            "temporal_sharpe": temporal_row.get("sharpe"),
            "winner_anchor_sharpe": anchor_row.get("sharpe"),
            "temporal_max_drawdown": temporal_row.get("max_drawdown"),
            "winner_anchor_max_drawdown": anchor_row.get("max_drawdown"),
            "temporal_switch_cost_events": temporal_row.get("switch_cost_events"),
            "winner_anchor_switch_cost_events": anchor_row.get("switch_cost_events"),
        }))

    winner_reference_daily_rows = _load_temporal_artifact_rows(db, run_id, "winner_reference_daily")
    winner_reference_trade_rows = _load_temporal_artifact_rows(db, run_id, "winner_reference_trades")
    multi_horizon_daily_asset_rows: list[dict[str, Any]] = []
    observation_cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)}, {"_id": 0, "timestamp": 1, "rows": 1, "encoding": 1, "payload": 1}
    ).sort("timestamp", 1)
    for document_item in observation_cursor:
        timestamp = document_item.get("timestamp")
        observation_rows = document_item.get("rows") or []
        if document_item.get("encoding") == "zlib-json-v1" and document_item.get("payload"):
            decoded = zlib.decompress(bytes(document_item["payload"])).decode("utf-8")
            observation_rows = json.loads(decoded)
        for item in observation_rows:
            if isinstance(item, dict):
                multi_horizon_daily_asset_rows.append(bson_value({"timestamp": timestamp, **dict(item)}))

    market_replay_columns = (
        "timestamp", "fold_id", "symbol", "decision_close", "execution_date", "next_execution_date",
        "execution_open", "execution_high", "execution_low", "execution_close", "execution_volume",
        "next_open", "open_to_open_return",
    )
    multi_horizon_market_replay_rows = [
        {key: row.get(key) for key in market_replay_columns}
        for row in multi_horizon_daily_asset_rows
        if any(row.get(key) is not None for key in ("execution_open", "next_open", "open_to_open_return"))
    ]

    confidence_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    capital_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for horizon in result.get("horizon_metrics") or []:
        if not isinstance(horizon, dict):
            continue
        horizon_value = horizon.get("horizon")
        for item in horizon.get("confidence_bins") or []:
            if isinstance(item, dict):
                confidence_rows.append(bson_value({"horizon": horizon_value, **dict(item)}))
        for item in horizon.get("risk_buckets") or []:
            if isinstance(item, dict):
                risk_rows.append(bson_value({"horizon": horizon_value, **dict(item)}))
        for item in horizon.get("signal_metrics") or []:
            if isinstance(item, dict):
                signal_rows.append(bson_value({"horizon": horizon_value, **dict(item)}))
        capital = horizon.get("shadow_capital")
        if isinstance(capital, dict):
            capital_rows.append(bson_value({
                "horizon": horizon_value,
                **{key: value for key, value in capital.items() if key not in {"action_counts", "decision_diagnostics", "economic_curve", "cost_stress"}},
                **{f"actions_{key}": value for key, value in (capital.get("action_counts") or {}).items()},
            }))
            for item in capital.get("decision_diagnostics") or []:
                if isinstance(item, dict):
                    diagnostic_rows.append(bson_value({"horizon": horizon_value, **dict(item)}))

    if external_diagnostic_rows:
        diagnostic_rows.extend(
            {key: value for key, value in row.items() if key != "artifact_kind"}
            for row in external_diagnostic_rows
            if row.get("artifact_kind") == "horizon_decision_diagnostics"
        )

    fold_capital_rows: list[dict[str, Any]] = []
    for fold in result.get("fold_metrics") or []:
        if not isinstance(fold, dict) or not isinstance(fold.get("shadow_capital"), dict):
            continue
        capital = fold["shadow_capital"]
        fold_capital_rows.append(bson_value({
            "fold_id": fold.get("fold_id"),
            "horizon": fold.get("horizon"),
            "test_start": fold.get("test_start"),
            "test_end": fold.get("test_end"),
            **{key: value for key, value in capital.items() if key not in {"action_counts", "decision_diagnostics", "economic_curve", "cost_stress"}},
            **{f"actions_{key}": value for key, value in (capital.get("action_counts") or {}).items()},
        }))

    winner_reference = result.get("winner_reference") if isinstance(result.get("winner_reference"), dict) else {}
    winner_reference_row = bson_value({key: value for key, value in winner_reference.items() if key != "folds"}) if winner_reference else {}
    winner_reference_fold_rows = [bson_value(dict(item)) for item in (winner_reference.get("folds") or []) if isinstance(item, dict)]

    summary_row = bson_value({
        "run_id": document.get("id"),
        "status": document.get("status"),
        "experiment": result.get("experiment") or document.get("experiment"),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": document.get("strategy_profile_revision"),
        "strategy_configuration_hash": document.get("strategy_configuration_hash"),
        "model_family": document.get("model_family"),
        "model_label": result.get("model_label") or document.get("model_label"),
        "model_settings_hash": document.get("model_settings_hash"),
        "model_settings_revision": document.get("model_settings_revision"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "market_data_snapshot_source": document.get("market_data_snapshot_source"),
        "market_data_snapshot_source_run_id": document.get("market_data_snapshot_source_run_id"),
        "deterministic_execution": bool((document.get("request") or {}).get("deterministic_execution")),
        "analysis_end_date": document.get("analysis_end_date"),
        "certified_backtest_cutoff": document.get("certified_backtest_cutoff"),
        "live_market_cutoff": document.get("live_market_cutoff"),
        "research_snapshot_cutoff": document.get("research_snapshot_cutoff") or document.get("analysis_end_date"),
        "horizons": json.dumps(result.get("horizons") or document.get("horizons") or []),
        "asset_count": result.get("asset_count"),
        "feature_count": result.get("feature_count"),
        "walk_forward_fold_count": result.get("walk_forward_fold_count"),
        "purge_sessions": result.get("purge_sessions"),
        "oos_start": result.get("oos_start"),
        "oos_end": result.get("oos_end"),
        "latest_as_of": result.get("latest_as_of"),
        "duration_seconds": result.get("duration_seconds"),
        "lightgbm_version": result.get("lightgbm_version"),
        "shadow_only": result.get("shadow_only"),
        "created_at": document.get("created_at"),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
    })

    manifest_result = deepcopy(result)
    for horizon in manifest_result.get("horizon_metrics") or []:
        if isinstance(horizon, dict) and isinstance(horizon.get("shadow_capital"), dict):
            horizon["shadow_capital"].pop("decision_diagnostics", None)
    manifest_multi_horizon = manifest_result.get("multi_horizon_metrics")
    if isinstance(manifest_multi_horizon, dict) and isinstance(manifest_multi_horizon.get("shadow_capital"), dict):
        manifest_multi_horizon["shadow_capital"].pop("decision_diagnostics", None)
        manifest_multi_horizon["shadow_capital"].pop("economic_curve", None)
    manifest = bson_value({
        "schema_version": (
            "temporal_intelligence_export_v12"
            if result.get("experiment") == TEMPORAL_EXPERIMENT
            else "temporal_intelligence_export_v9"
            if result.get("experiment") == "temporal_decision_intelligence_v7_rotation_before_cash"
            else "temporal_intelligence_export_v7"
            if result.get("experiment") == "temporal_decision_intelligence_v6_adaptive_trend_capture"
            else "temporal_intelligence_export_v6"
            if result.get("experiment") == "temporal_decision_intelligence_v5_trend_capture_hysteresis"
            else "temporal_intelligence_export_v5"
            if result.get("experiment") == "temporal_decision_intelligence_v4_multi_horizon"
            else "temporal_intelligence_export_v4"
            if result.get("experiment") == "temporal_decision_intelligence_v3"
            else "temporal_intelligence_export_v3"
            if result.get("experiment") == "temporal_decision_intelligence_v2"
            else "temporal_intelligence_export_v2"
            if result.get("experiment") == "temporal_decision_intelligence_v1"
            else "temporal_intelligence_export_v1"
        ),
        "run": {
            key: deepcopy(document.get(key))
            for key in (
                "id", "status", "stage", "progress", "created_at", "updated_at", "started_at", "finished_at",
                "experiment", "strategy_profile_id", "strategy_profile_name", "strategy_profile_revision", "strategy_configuration_hash",
                "model_family", "model_label", "model_settings_hash", "model_settings_revision",
                "market_data_snapshot_id", "market_data_snapshot_source", "market_data_snapshot_source_run_id",
                "analysis_end_date", "certified_backtest_cutoff", "live_market_cutoff", "research_snapshot_cutoff",
                "horizons", "system_settings_revision", "shadow_only",
            )
        },
        "request": deepcopy(document.get("request") or {}),
        "result": manifest_result,
    })

    pipeline_period_start = str(start_month or "").strip() or None
    pipeline_period_end = str(end_month or "").strip() or None
    processing_id = str(document.get("research_processing_id") or "").strip() or None
    pipeline_query: dict[str, Any] = {"run_id": str(run_id)}
    if processing_id:
        pipeline_query["processing_id"] = processing_id
    if pipeline_period_start:
        pipeline_query["period_start"] = pipeline_period_start
    if pipeline_period_end:
        pipeline_query["period_end"] = pipeline_period_end

    def latest_pipeline_document(collection_name: str) -> dict[str, Any] | None:
        row = db[collection_name].find_one(pipeline_query, {"_id": 0}, sort=[("created_at", -1)])
        return bson_value(row) if row is not None else None

    pipeline_risk = latest_pipeline_document(TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION)
    pipeline_alternative_action = latest_pipeline_document(TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION)
    pipeline_operational_qualification = latest_pipeline_document(TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION)
    pipeline_intervention = latest_pipeline_document(TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION)
    pipeline_confidence = latest_pipeline_document(TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION)
    pipeline_stateful = latest_pipeline_document(TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION)
    pipeline_milp = None
    if processing_id and pipeline_period_start and pipeline_period_end:
        from ..milp_decision.persistence import latest_raw as latest_milp_decision
        pipeline_milp = latest_milp_decision(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
    pipeline_validation = (
        bson_value(document.get("strategy_research_final_validation"))
        if isinstance(document.get("strategy_research_final_validation"), dict)
        else None
    )
    pipeline_leadership = None
    pipeline_opportunity_drought = None
    pipeline_clustering = None
    pipeline_fragile_incumbent = None
    pipeline_emerging_trend = None
    if processing_id and pipeline_period_start and pipeline_period_end:
        from ..leadership_regime.service import get_persisted as get_leadership_regime
        pipeline_leadership = get_leadership_regime(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
        from ..opportunity_drought.service import get_persisted as get_opportunity_drought
        pipeline_opportunity_drought = get_opportunity_drought(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
        from ..regime_clustering.service import get_persisted as get_regime_clustering
        pipeline_clustering = get_regime_clustering(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
        from ..fragile_incumbent.service import get_persisted as get_fragile_incumbent
        pipeline_fragile_incumbent = get_fragile_incumbent(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
        from ..emerging_trend.service import get_persisted as get_emerging_trend
        pipeline_emerging_trend = get_emerging_trend(
            db, run_id, processing_id=processing_id, start_month=pipeline_period_start, end_month=pipeline_period_end
        )
    leadership_session_rows: list[dict[str, Any]] = []
    leadership_monthly_rows: list[dict[str, Any]] = []
    if pipeline_leadership is not None:
        for item in pipeline_leadership.get("sessions") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "signals", "thresholds"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            row.update({f"signal_{key}": value for key, value in (item.get("signals") or {}).items()})
            row.update({f"threshold_{key}": value for key, value in (item.get("thresholds") or {}).items()})
            leadership_session_rows.append(bson_value(row))
        for item in pipeline_leadership.get("monthly") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"state_counts", "state_shares"}}
            row.update({f"state_count_{key}": value for key, value in (item.get("state_counts") or {}).items()})
            row.update({f"state_share_{key}": value for key, value in (item.get("state_shares") or {}).items()})
            leadership_monthly_rows.append(bson_value(row))
    opportunity_drought_monthly_rows: list[dict[str, Any]] = []
    opportunity_drought_fold_rows: list[dict[str, Any]] = []
    opportunity_drought_feature_rows: list[dict[str, Any]] = []
    opportunity_drought_oos_rows: list[dict[str, Any]] = []
    if pipeline_opportunity_drought is not None:
        for item in pipeline_opportunity_drought.get("monthly") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "top_drivers"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            for index, driver in enumerate((item.get("top_drivers") or [])[:5], start=1):
                if isinstance(driver, dict):
                    row[f"driver_{index}_feature"] = driver.get("feature")
                    row[f"driver_{index}_contribution"] = driver.get("contribution")
            opportunity_drought_monthly_rows.append(bson_value(row))
        for item in pipeline_opportunity_drought.get("folds") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"session_metrics", "monthly_metrics"}}
            row.update({f"session_{key}": value for key, value in (item.get("session_metrics") or {}).items()})
            row.update({f"monthly_{key}": value for key, value in (item.get("monthly_metrics") or {}).items()})
            opportunity_drought_fold_rows.append(bson_value(row))
        opportunity_drought_feature_rows = [
            bson_value(dict(item)) for item in (pipeline_opportunity_drought.get("feature_importance") or []) if isinstance(item, dict)
        ]
        for item in pipeline_opportunity_drought.get("oos_sessions") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "contributions"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            row.update({f"contribution_{key}": value for key, value in (item.get("contributions") or {}).items()})
            opportunity_drought_oos_rows.append(bson_value(row))

    clustering_monthly_rows: list[dict[str, Any]] = []
    clustering_cluster_rows: list[dict[str, Any]] = []
    clustering_feature_rows: list[dict[str, Any]] = []
    if pipeline_clustering is not None:
        for item in pipeline_clustering.get("monthly") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "feature_zscores", "similar_months"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            row.update({f"zscore_{key}": value for key, value in (item.get("feature_zscores") or {}).items()})
            for index, similar in enumerate((item.get("similar_months") or [])[:5], start=1):
                if isinstance(similar, dict):
                    row[f"similar_{index}_month"] = similar.get("month")
                    row[f"similar_{index}_distance"] = similar.get("distance")
                    row[f"similar_{index}_return"] = similar.get("official_return")
            clustering_monthly_rows.append(bson_value(row))
        for item in pipeline_clustering.get("clusters") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "feature_zscores", "months_list"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            row.update({f"zscore_{key}": value for key, value in (item.get("feature_zscores") or {}).items()})
            row["months_list"] = ", ".join(str(v) for v in (item.get("months_list") or []))
            clustering_cluster_rows.append(bson_value(row))
        clustering_feature_rows = [bson_value(dict(item)) for item in (pipeline_clustering.get("feature_importance") or []) if isinstance(item, dict)]

    emerging_trend_monthly_rows: list[dict[str, Any]] = []
    emerging_trend_fold_rows: list[dict[str, Any]] = []
    emerging_trend_feature_rows: list[dict[str, Any]] = []
    emerging_trend_session_rows: list[dict[str, Any]] = []
    if pipeline_emerging_trend is not None:
        for item in pipeline_emerging_trend.get("monthly") or []:
            if isinstance(item, dict):
                emerging_trend_monthly_rows.append(bson_value(dict(item)))
        for item in pipeline_emerging_trend.get("folds") or []:
            if isinstance(item, dict):
                emerging_trend_fold_rows.append(bson_value(dict(item)))
        emerging_trend_feature_rows = [bson_value(dict(item)) for item in (pipeline_emerging_trend.get("feature_importance") or []) if isinstance(item, dict)]
        for item in pipeline_emerging_trend.get("sessions") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "top_drivers"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            for index, driver in enumerate((item.get("top_drivers") or [])[:6], start=1):
                if isinstance(driver, dict):
                    row[f"driver_{index}_feature"] = driver.get("feature")
                    row[f"driver_{index}_contribution"] = driver.get("contribution")
            emerging_trend_session_rows.append(bson_value(row))

    fragile_incumbent_monthly_rows: list[dict[str, Any]] = []
    fragile_incumbent_fold_rows: list[dict[str, Any]] = []
    fragile_incumbent_feature_rows: list[dict[str, Any]] = []
    fragile_incumbent_oos_rows: list[dict[str, Any]] = []
    if pipeline_fragile_incumbent is not None:
        for item in pipeline_fragile_incumbent.get("monthly") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "top_drivers"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            for index, driver in enumerate((item.get("top_drivers") or [])[:6], start=1):
                if isinstance(driver, dict):
                    row[f"driver_{index}_feature"] = driver.get("feature")
                    row[f"driver_{index}_contribution"] = driver.get("contribution")
            fragile_incumbent_monthly_rows.append(bson_value(row))
        for item in pipeline_fragile_incumbent.get("folds") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"session_metrics", "monthly_metrics"}}
            row.update({f"session_{key}": value for key, value in (item.get("session_metrics") or {}).items()})
            row.update({f"monthly_{key}": value for key, value in (item.get("monthly_metrics") or {}).items()})
            fragile_incumbent_fold_rows.append(bson_value(row))
        fragile_incumbent_feature_rows = [
            bson_value(dict(item)) for item in (pipeline_fragile_incumbent.get("feature_importance") or []) if isinstance(item, dict)
        ]
        for item in pipeline_fragile_incumbent.get("oos_sessions") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"features", "contributions"}}
            row.update({f"feature_{key}": value for key, value in (item.get("features") or {}).items()})
            row.update({f"contribution_{key}": value for key, value in (item.get("contributions") or {}).items()})
            fragile_incumbent_oos_rows.append(bson_value(row))

    alternative_action_rows: list[dict[str, Any]] = []
    alternative_action_yearly_rows: list[dict[str, Any]] = []
    if pipeline_alternative_action is not None:
        alternative_action_rows = [bson_value(dict(item)) for item in (pipeline_alternative_action.get("alerts") or []) if isinstance(item, dict)]
        for item in pipeline_alternative_action.get("yearly_oos") or []:
            if not isinstance(item, dict):
                continue
            row = {key: value for key, value in item.items() if key not in {"best_action_counts", "average_returns"}}
            row.update({f"best_{key.lower()}": value for key, value in (item.get("best_action_counts") or {}).items()})
            row.update({f"avg_{key.lower()}_return": value for key, value in (item.get("average_returns") or {}).items()})
            alternative_action_yearly_rows.append(bson_value(row))

    operational_qualification_prediction_rows: list[dict[str, Any]] = []
    operational_qualification_yearly_rows: list[dict[str, Any]] = []
    operational_qualification_gate_rows: list[dict[str, Any]] = []
    if pipeline_operational_qualification is not None:
        operational_qualification_prediction_rows = [bson_value(dict(item)) for item in (pipeline_operational_qualification.get("predictions") or []) if isinstance(item, dict)]
        operational_qualification_yearly_rows = [bson_value(dict(item)) for item in (pipeline_operational_qualification.get("yearly_oos") or []) if isinstance(item, dict)]
        operational_qualification_gate_rows = [bson_value(dict(item)) for item in (pipeline_operational_qualification.get("gates") or []) if isinstance(item, dict)]

    pipeline_state_export = _strategy_research_pipeline_state(document)
    pipeline_reference_analytics = None
    if processing_id:
        try:
            from .analytics import processing_analytics
            pipeline_reference_analytics = bson_value(processing_analytics(db, processing_id))
        except Exception as exc:
            pipeline_reference_analytics = {"status": "unavailable", "failure_message": str(exc)}

    pipeline_attribution = None
    if pipeline_period_start and pipeline_period_end:
        try:
            from .temporal_winner_transition_attribution import get_winner_transition_attribution
            pipeline_attribution = get_winner_transition_attribution(
                db,
                run_id,
                start_month=pipeline_period_start,
                end_month=pipeline_period_end,
            )
        except Exception as exc:
            pipeline_attribution = {"status": "unavailable", "failure_message": str(exc)}

    pipeline_manifest = bson_value({
        "schema_version": 6,
        "run_id": str(run_id),
        "pipeline_status": pipeline_state_export.get("status"),
        "current_stage": pipeline_state_export.get("current_stage"),
        "pipeline_failure_message": pipeline_state_export.get("failure_message"),
        "stage_states": pipeline_state_export.get("stage_states"),
        "processing_id": processing_id,
        "period_start": pipeline_period_start,
        "period_end": pipeline_period_end,
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": document.get("strategy_profile_revision"),
        "strategy_configuration_hash": document.get("strategy_configuration_hash"),
        "strategy_kind": document.get("strategy_kind"),
        "temporal_strategy_variant": document.get("temporal_strategy_variant"),
        "risk_status": (pipeline_risk or {}).get("status"),
        "alternative_action_status": (pipeline_alternative_action or {}).get("status"),
        "alternative_action_readiness": ((pipeline_alternative_action or {}).get("readiness") or {}).get("status"),
        "operational_policy_qualification_status": (pipeline_operational_qualification or {}).get("status"),
        "operational_policy_decision": ((pipeline_operational_qualification or {}).get("decision") or {}).get("status"),
        "risk_failure_message": (pipeline_risk or {}).get("failure_message"),
        "intervention_status": (pipeline_intervention or {}).get("status"),
        "confidence_status": (pipeline_confidence or {}).get("status"),
        "stateful_status": (pipeline_stateful or {}).get("status"),
        "leadership_regime_status": (pipeline_leadership or {}).get("status"),
        "clustering_status": (pipeline_clustering or {}).get("status"),
        "clustering_readiness": ((pipeline_clustering or {}).get("readiness") or {}).get("status"),
        "opportunity_drought_status": (pipeline_opportunity_drought or {}).get("status"),
        "opportunity_drought_readiness": ((pipeline_opportunity_drought or {}).get("readiness") or {}).get("status"),
        "fragile_incumbent_status": (pipeline_fragile_incumbent or {}).get("status"),
        "fragile_incumbent_readiness": ((pipeline_fragile_incumbent or {}).get("readiness") or {}).get("status"),
        "emerging_trend_status": (pipeline_emerging_trend or {}).get("status"),
        "emerging_trend_readiness": ((pipeline_emerging_trend or {}).get("readiness") or {}).get("status"),
        "milp_decision_status": (pipeline_milp or {}).get("status"),
        "validation_status": (pipeline_validation or {}).get("status") or pipeline_state_export.get("stage_states", {}).get("validation"),
    })

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("temporal_intelligence_summary.csv", _csv_text([summary_row]))
        archive.writestr("temporal_intelligence_horizons.csv", _csv_text(horizon_rows))
        archive.writestr("temporal_intelligence_folds.csv", _csv_text(fold_rows))
        archive.writestr("temporal_intelligence_confidence_bins.csv", _csv_text(confidence_rows))
        archive.writestr("temporal_intelligence_risk_buckets.csv", _csv_text(risk_rows))
        archive.writestr("temporal_intelligence_signal_metrics.csv", _csv_text(signal_rows))
        archive.writestr("temporal_intelligence_shadow_capital.csv", _csv_text(capital_rows))
        archive.writestr("temporal_intelligence_shadow_capital_folds.csv", _csv_text(fold_capital_rows))
        archive.writestr("temporal_intelligence_decision_diagnostics.csv", _csv_text(diagnostic_rows))
        archive.writestr("temporal_intelligence_winner_reference.csv", _csv_text([winner_reference_row] if winner_reference_row else []))
        archive.writestr("temporal_intelligence_winner_reference_folds.csv", _csv_text(winner_reference_fold_rows))
        archive.writestr("temporal_intelligence_multi_horizon.csv", _csv_text(multi_horizon_rows))
        archive.writestr("temporal_intelligence_multi_horizon_folds.csv", _csv_text(multi_horizon_fold_rows))
        archive.writestr("temporal_intelligence_multi_horizon_decision_diagnostics.csv", _csv_text(multi_horizon_diagnostic_rows))
        archive.writestr("temporal_intelligence_multi_horizon_equity_curve.csv", _csv_text(multi_horizon_equity_rows))
        archive.writestr("temporal_intelligence_timing_override_attribution.csv", _csv_text(timing_override_attribution_rows))
        archive.writestr("temporal_intelligence_cost_stress.csv", _csv_text(cost_stress_rows))
        archive.writestr("temporal_intelligence_multi_horizon_latest_forecasts.csv", _csv_text(multi_horizon_latest_rows))
        archive.writestr("temporal_intelligence_multi_horizon_daily_assets.csv", _csv_text(multi_horizon_daily_asset_rows))
        archive.writestr("temporal_intelligence_multi_horizon_market_replay.csv", _csv_text(multi_horizon_market_replay_rows))
        archive.writestr("temporal_intelligence_winner_reference_daily.csv", _csv_text(winner_reference_daily_rows))
        archive.writestr("temporal_intelligence_winner_reference_trades.csv", _csv_text(winner_reference_trade_rows))
        archive.writestr("temporal_intelligence_latest_forecasts.csv", _csv_text(forecast_rows))
        archive.writestr("temporal_intelligence_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
        archive.writestr("strategy_research_pipeline_manifest.json", json.dumps(pipeline_manifest, indent=2, ensure_ascii=False, default=str))
        if pipeline_reference_analytics is not None:
            archive.writestr("strategy_research_reference_analytics.json", json.dumps(pipeline_reference_analytics, indent=2, ensure_ascii=False, default=str))
        if pipeline_attribution is not None:
            archive.writestr("strategy_research_transition_attribution.json", json.dumps(bson_value(pipeline_attribution), indent=2, ensure_ascii=False, default=str))
        if pipeline_risk is not None:
            archive.writestr("strategy_research_risk.json", json.dumps(pipeline_risk, indent=2, ensure_ascii=False, default=str))
        if pipeline_alternative_action is not None:
            archive.writestr("strategy_research_alternative_action.json", json.dumps(pipeline_alternative_action, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_alternative_action_alerts.csv", _csv_text(alternative_action_rows))
            archive.writestr("strategy_research_alternative_action_yearly.csv", _csv_text(alternative_action_yearly_rows))
        if pipeline_operational_qualification is not None:
            archive.writestr("strategy_research_operational_policy_qualification.json", json.dumps(pipeline_operational_qualification, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_operational_policy_predictions.csv", _csv_text(operational_qualification_prediction_rows))
            archive.writestr("strategy_research_operational_policy_yearly.csv", _csv_text(operational_qualification_yearly_rows))
            archive.writestr("strategy_research_operational_policy_gates.csv", _csv_text(operational_qualification_gate_rows))
        if pipeline_intervention is not None:
            archive.writestr("strategy_research_intervention.json", json.dumps(pipeline_intervention, indent=2, ensure_ascii=False, default=str))
        if pipeline_confidence is not None:
            archive.writestr("strategy_research_confidence.json", json.dumps(pipeline_confidence, indent=2, ensure_ascii=False, default=str))
        if pipeline_stateful is not None:
            archive.writestr("strategy_research_stateful.json", json.dumps(pipeline_stateful, indent=2, ensure_ascii=False, default=str))
        if pipeline_leadership is not None:
            archive.writestr("strategy_research_leadership_regime.json", json.dumps(pipeline_leadership, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_leadership_regime_sessions.csv", _csv_text(leadership_session_rows))
            archive.writestr("strategy_research_leadership_regime_monthly.csv", _csv_text(leadership_monthly_rows))
        if pipeline_clustering is not None:
            archive.writestr("strategy_research_regime_clustering.json", json.dumps(pipeline_clustering, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_regime_clustering_monthly.csv", _csv_text(clustering_monthly_rows))
            archive.writestr("strategy_research_regime_clustering_clusters.csv", _csv_text(clustering_cluster_rows))
            archive.writestr("strategy_research_regime_clustering_feature_importance.csv", _csv_text(clustering_feature_rows))
        if pipeline_opportunity_drought is not None:
            archive.writestr("strategy_research_opportunity_drought.json", json.dumps(pipeline_opportunity_drought, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_opportunity_drought_monthly.csv", _csv_text(opportunity_drought_monthly_rows))
            archive.writestr("strategy_research_opportunity_drought_folds.csv", _csv_text(opportunity_drought_fold_rows))
            archive.writestr("strategy_research_opportunity_drought_feature_importance.csv", _csv_text(opportunity_drought_feature_rows))
            archive.writestr("strategy_research_opportunity_drought_oos_sessions.csv", _csv_text(opportunity_drought_oos_rows))
        if pipeline_fragile_incumbent is not None:
            archive.writestr("strategy_research_fragile_incumbent.json", json.dumps(pipeline_fragile_incumbent, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_fragile_incumbent_monthly.csv", _csv_text(fragile_incumbent_monthly_rows))
            archive.writestr("strategy_research_fragile_incumbent_folds.csv", _csv_text(fragile_incumbent_fold_rows))
            archive.writestr("strategy_research_fragile_incumbent_feature_importance.csv", _csv_text(fragile_incumbent_feature_rows))
            archive.writestr("strategy_research_fragile_incumbent_oos_sessions.csv", _csv_text(fragile_incumbent_oos_rows))
        if pipeline_emerging_trend is not None:
            archive.writestr("strategy_research_emerging_trend.json", json.dumps(pipeline_emerging_trend, indent=2, ensure_ascii=False, default=str))
            archive.writestr("strategy_research_emerging_trend_monthly.csv", _csv_text(emerging_trend_monthly_rows))
            archive.writestr("strategy_research_emerging_trend_folds.csv", _csv_text(emerging_trend_fold_rows))
            archive.writestr("strategy_research_emerging_trend_feature_importance.csv", _csv_text(emerging_trend_feature_rows))
            archive.writestr("strategy_research_emerging_trend_sessions.csv", _csv_text(emerging_trend_session_rows))
        if pipeline_milp is not None:
            archive.writestr("strategy_research_milp_decision.json", json.dumps(pipeline_milp, indent=2, ensure_ascii=False, default=str))
        if pipeline_validation is not None:
            archive.writestr("strategy_research_final_validation.json", json.dumps(pipeline_validation, indent=2, ensure_ascii=False, default=str))
    return archive_buffer.getvalue()

def _default_strategy_research_stage_states() -> dict[str, str]:
    return {stage: "waiting" for stage in STRATEGY_RESEARCH_PIPELINE_STAGES}


def _strategy_research_pipeline_state(document: dict[str, Any] | None) -> dict[str, Any]:
    stored = deepcopy((document or {}).get("strategy_research_pipeline") or {})
    stage_states = _default_strategy_research_stage_states()
    for stage, value in dict(stored.get("stage_states") or {}).items():
        if stage in stage_states and str(value) in STRATEGY_RESEARCH_PIPELINE_STAGE_STATES:
            stage_states[stage] = str(value)
    status_value = str(stored.get("status") or "idle")
    if status_value not in STRATEGY_RESEARCH_PIPELINE_STATUSES:
        status_value = "idle"
    stored_stage_states = dict(stored.get("stage_states") or {})
    if status_value == "completed" and "clustering" not in stored_stage_states:
        stage_states["clustering"] = "skipped"
    if status_value == "completed" and "fragile_incumbent" not in stored_stage_states:
        stage_states["fragile_incumbent"] = "skipped"
    if status_value == "completed" and "emerging_trend" not in stored_stage_states:
        stage_states["emerging_trend"] = "skipped"
    return {
        "status": status_value,
        "current_stage": stored.get("current_stage") if stored.get("current_stage") in STRATEGY_RESEARCH_PIPELINE_STAGES else None,
        "stage_states": stage_states,
        "start_month": stored.get("start_month"),
        "end_month": stored.get("end_month"),
        "failure_message": stored.get("failure_message"),
        "updated_at": bson_value(stored.get("updated_at")),
    }


def _persist_strategy_research_pipeline_state(
    db: Any,
    run_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    now = utc_now()
    update = {
        "status": state.get("status") or "idle",
        "current_stage": state.get("current_stage"),
        "stage_states": dict(state.get("stage_states") or _default_strategy_research_stage_states()),
        "start_month": state.get("start_month"),
        "end_month": state.get("end_month"),
        "failure_message": state.get("failure_message"),
        "updated_at": now,
    }
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {"strategy_research_pipeline": update, "updated_at": now}},
    )
    return {"run_id": str(run_id), **_strategy_research_pipeline_state({"strategy_research_pipeline": update})}


def _pipeline_stop_requested(db: Any, run_id: str) -> bool:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)}, {"_id": 0, "strategy_research_pipeline.status": 1}
    ) or {}
    pipeline = document.get("strategy_research_pipeline") if isinstance(document.get("strategy_research_pipeline"), dict) else {}
    return str(pipeline.get("status") or "").lower() in {"stop_requested", "stopped"}


def _compact_validation_metrics(analytics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = analytics.get("metrics") if isinstance(analytics, dict) and isinstance(analytics.get("metrics"), dict) else {}
    keys = (
        "initial_capital", "ending_capital", "total_return", "cagr", "sharpe",
        "maximum_drawdown", "max_drawdown", "market_exposure", "exposure",
        "capital_rotations", "switch_count", "cash_days", "equity_sessions",
    )
    return bson_value({key: metrics.get(key) for key in keys if key in metrics})


def _persist_strategy_research_final_validation(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    from .analytics import processing_analytics
    from .temporal_winner_transition_stateful import get_latest_winner_transition_stateful_replay
    from ..milp_decision.persistence import latest_raw as latest_milp_decision
    from ..operational_policy_qualification.service import get_persisted as get_operational_policy_qualification, public_summary as operational_policy_public_summary

    control_analytics = processing_analytics(db, processing_id)
    stateful = get_latest_winner_transition_stateful_replay(
        db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
    )
    milp = latest_milp_decision(
        db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
    )
    if stateful is None:
        raise TemporalIntelligenceConflict("Final Validation requires a completed Decision Policy Replay.")
    if milp is None or str(milp.get("status") or "").lower() != "completed":
        raise TemporalIntelligenceConflict("Final Validation requires a completed MILP Decision Optimization.")
    parity = milp.get("control_parity") if isinstance(milp.get("control_parity"), dict) else {}
    if str(parity.get("status") or "").lower() != "passed":
        raise TemporalIntelligenceConflict("Final Validation requires MILP exact Control replay parity.")

    candidate_a = stateful.get("candidate_a") if isinstance(stateful.get("candidate_a"), dict) else {}
    candidate_a_analytics = candidate_a.get("analytics") if isinstance(candidate_a.get("analytics"), dict) else {}
    milp_analytics = milp.get("analytics") if isinstance(milp.get("analytics"), dict) else {}
    from ..leadership_regime.service import get_persisted as get_leadership_regime, public_summary as leadership_public_summary
    leadership = leadership_public_summary(
        get_leadership_regime(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
    )
    operational_qualification = operational_policy_public_summary(
        get_operational_policy_qualification(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
    )
    if operational_qualification is None:
        raise TemporalIntelligenceConflict("Final Validation requires Operational Policy Qualification.")
    now = utc_now()
    validation = bson_value({
        "schema_version": 1,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(start_month),
        "period_end": str(end_month),
        "control": {
            "metrics": _compact_validation_metrics(control_analytics),
        },
        "stateful": {
            "id": stateful.get("id"),
            "status": stateful.get("status"),
            "candidate_a_label": candidate_a.get("label"),
            "metrics": _compact_validation_metrics(candidate_a_analytics),
        },
        "milp": {
            "id": milp.get("id"),
            "status": milp.get("status"),
            "control_parity_status": parity.get("status"),
            "metrics": _compact_validation_metrics(milp_analytics),
            "attribution": bson_value(milp.get("attribution") or {}),
        },
        "leadership_regime": {
            "status": (leadership or {}).get("status"),
            "summary": bson_value((leadership or {}).get("summary") or {}),
        },
        "operational_policy_qualification": {
            "id": (operational_qualification or {}).get("id"),
            "status": (operational_qualification or {}).get("status"),
            "decision": bson_value((operational_qualification or {}).get("decision") or {}),
            "summary": bson_value((operational_qualification or {}).get("summary") or {}),
            "gates": bson_value((operational_qualification or {}).get("gates") or []),
        },
        "created_at": now,
        "updated_at": now,
    })
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {"strategy_research_final_validation": validation, "updated_at": now}},
    )
    return validation


def _pipeline_stage_start(db: Any, run_id: str, stage: str) -> dict[str, Any]:
    return control_strategy_research_pipeline(db, run_id, action="stage_start", stage=stage)


def _pipeline_stage_complete(db: Any, run_id: str, stage: str) -> dict[str, Any]:
    return control_strategy_research_pipeline(db, run_id, action="stage_complete", stage=stage)


def _run_strategy_research_pipeline_worker(db: Any, run_id: str) -> None:
    current_stage = "temporal"
    try:
        while True:
            if _pipeline_stop_requested(db, run_id):
                return
            run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0}) or {}
            temporal_status = str(run.get("status") or "").lower()
            if temporal_status == "completed":
                break
            if temporal_status in {"failed", "interrupted"}:
                raise TemporalIntelligenceConflict(str(run.get("failure_message") or "Temporal Intelligence failed."))
            if temporal_status in {"cancelled", "stopped"}:
                return
            time.sleep(1.0)

        state = get_strategy_research_pipeline_state(db, run_id)
        if state["stage_states"].get("reference") != "completed":
            _pipeline_stage_complete(db, run_id, "reference")
        if state["stage_states"].get("temporal") != "completed":
            _pipeline_stage_complete(db, run_id, "temporal")

        run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0}) or {}
        pipeline = _strategy_research_pipeline_state(run)
        processing_id = str(run.get("research_processing_id") or "").strip()
        start_month = str(pipeline.get("start_month") or "").strip()
        end_month = str(pipeline.get("end_month") or "").strip()
        if not processing_id or not start_month or not end_month:
            raise TemporalIntelligenceConflict("Strategy Research pipeline is missing its processing or period binding.")

        from .temporal_winner_transition_risk import (
            get_latest_winner_transition_risk_search,
            run_winner_transition_risk_search,
        )
        from .temporal_winner_transition_intervention import (
            get_latest_winner_transition_confidence_calibration,
            get_latest_winner_transition_intervention_search,
            run_winner_transition_confidence_calibration,
            run_winner_transition_intervention_search,
        )
        from .temporal_winner_transition_stateful import (
            get_latest_winner_transition_stateful_replay,
            run_winner_transition_stateful_replay,
        )
        from ..milp_decision.persistence import latest_raw as latest_milp_decision
        from ..milp_decision.service import run as run_milp_decision

        current_stage = "clustering"
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_start(db, run_id, current_stage)
        try:
            from ..leadership_regime.service import build_and_persist as build_leadership_regime, unavailable as leadership_regime_unavailable
            leadership = build_leadership_regime(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
        except Exception as leadership_exc:
            leadership = None
            try:
                leadership_regime_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, message=str(leadership_exc)
                )
            except Exception:
                pass
        try:
            from ..regime_clustering.service import build_and_persist as build_regime_clustering, unavailable as regime_clustering_unavailable
            if leadership and str(leadership.get("status") or "").lower() == "completed":
                build_regime_clustering(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
                )
            else:
                regime_clustering_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month,
                    message="Leadership Regime diagnostics are unavailable for Regime Clustering.",
                )
        except Exception as clustering_exc:
            try:
                regime_clustering_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, message=str(clustering_exc)
                )
            except Exception:
                pass
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "fragile_incumbent"
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_start(db, run_id, current_stage)
        try:
            from ..fragile_incumbent.service import build_and_persist as build_fragile_incumbent, unavailable as fragile_incumbent_unavailable
            if leadership and str(leadership.get("status") or "").lower() == "completed":
                build_fragile_incumbent(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
                )
            else:
                fragile_incumbent_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month,
                    message="Leadership Regime diagnostics are unavailable for Fragile Incumbent Research.",
                )
        except Exception as fragile_exc:
            try:
                fragile_incumbent_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, message=str(fragile_exc)
                )
            except Exception:
                pass
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "emerging_trend"
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_start(db, run_id, current_stage)
        try:
            from ..emerging_trend.service import build_and_persist as build_emerging_trend, unavailable as emerging_trend_unavailable
            emerging_result = build_emerging_trend(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
            if str((emerging_result or {}).get("status") or "").lower() != "completed":
                raise TemporalIntelligenceConflict(
                    str((emerging_result or {}).get("failure_message") or "Emerging Trend Research did not produce a completed result.")
                )
        except Exception as emerging_exc:
            try:
                emerging_trend_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, message=str(emerging_exc)
                )
            except Exception:
                pass
            raise TemporalIntelligenceConflict(f"Emerging Trend Research failed: {emerging_exc}") from emerging_exc
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "risk"
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_start(db, run_id, current_stage)
        risk = get_latest_winner_transition_risk_search(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if risk is None:
            run_winner_transition_risk_search(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, seed=42
            )
        if _pipeline_stop_requested(db, run_id):
            return
        intervention = get_latest_winner_transition_intervention_search(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if intervention is None:
            run_winner_transition_intervention_search(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, seed=42
            )
        if _pipeline_stop_requested(db, run_id):
            return
        from ..alternative_action.service import build_and_persist as build_alternative_action
        alternative_action = build_alternative_action(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if str((alternative_action or {}).get("status") or "").lower() != "completed":
            raise TemporalIntelligenceConflict(
                str((alternative_action or {}).get("failure_message") or "Risk-Aware Alternative Action did not produce a completed result.")
            )
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "confidence"
        _pipeline_stage_start(db, run_id, current_stage)
        confidence = get_latest_winner_transition_confidence_calibration(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if confidence is None:
            run_winner_transition_confidence_calibration(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "stateful"
        _pipeline_stage_start(db, run_id, current_stage)
        stateful = get_latest_winner_transition_stateful_replay(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if stateful is None:
            run_winner_transition_stateful_replay(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
        if _pipeline_stop_requested(db, run_id):
            return
        try:
            from ..leadership_regime.service import build_and_persist as build_leadership_regime, unavailable as leadership_regime_unavailable
            build_leadership_regime(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
        except Exception as leadership_exc:
            try:
                leadership_regime_unavailable(
                    db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month, message=str(leadership_exc)
                )
            except Exception:
                pass
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "milp"
        _pipeline_stage_start(db, run_id, current_stage)
        milp = latest_milp_decision(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        valid_milp = bool(
            milp
            and str(milp.get("status") or "").lower() == "completed"
            and int(milp.get("schema_version") or 0) >= 2
            and str(((milp.get("control_parity") or {}).get("status") or "")).lower() == "passed"
        )
        if not valid_milp:
            run_milp_decision(
                db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
            )
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)

        current_stage = "validation"
        _pipeline_stage_start(db, run_id, current_stage)
        from ..operational_policy_qualification.service import build_and_persist as build_operational_policy_qualification
        operational_qualification = build_operational_policy_qualification(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if str((operational_qualification or {}).get("status") or "").lower() != "completed":
            raise TemporalIntelligenceConflict("Operational Policy Qualification did not produce a completed result.")
        _persist_strategy_research_final_validation(
            db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month
        )
        if _pipeline_stop_requested(db, run_id):
            return
        _pipeline_stage_complete(db, run_id, current_stage)
    except Exception as exc:
        if not _pipeline_stop_requested(db, run_id):
            _logger.exception(
                "Strategy Research pipeline failed at stage %s for run %s: %s",
                current_stage,
                run_id,
                exc,
            )
            try:
                control_strategy_research_pipeline(
                    db, run_id, action="stage_failed", stage=current_stage, message=str(exc)
                )
            except Exception:
                _logger.exception(
                    "Failed to persist Strategy Research pipeline failure at stage %s for run %s.",
                    current_stage,
                    run_id,
                )
    finally:
        with _ACTIVE_PIPELINE_LOCK:
            _ACTIVE_PIPELINE_WORKERS.discard(str(run_id))


def _start_strategy_research_pipeline_worker(db: Any, run_id: str) -> None:
    run_key = str(run_id)
    with _ACTIVE_PIPELINE_LOCK:
        if run_key in _ACTIVE_PIPELINE_WORKERS:
            return
        _ACTIVE_PIPELINE_WORKERS.add(run_key)
    threading.Thread(target=_run_strategy_research_pipeline_worker, args=(db, run_key), daemon=True).start()


def _reconcile_detached_strategy_research_pipeline(
    db: Any,
    document: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(document.get("id") or "")
    state = _strategy_research_pipeline_state(document)
    pipeline_status = str(state.get("status") or "idle").lower()
    current_stage = state.get("current_stage")
    temporal_status = str(document.get("status") or "").lower()

    if pipeline_status == "paused" and temporal_status == "completed":
        stage_states = dict(state.get("stage_states") or _default_strategy_research_stage_states())
        if stage_states.get("reference") == "waiting":
            stage_states["reference"] = "completed"
        stage_states["temporal"] = "completed"
        repaired = _persist_strategy_research_pipeline_state(
            db,
            run_id,
            {
                **state,
                "status": "running",
                "current_stage": None,
                "stage_states": stage_states,
                "failure_message": None,
            },
        )
        _start_strategy_research_pipeline_worker(db, run_id)
        return repaired

    if pipeline_status == "running":
        _start_strategy_research_pipeline_worker(db, run_id)

    if current_stage != "temporal" or pipeline_status not in {"running", "stop_requested"}:
        return {"run_id": run_id, **state}
    if temporal_status in {"queued", "running", "stop_requested"}:
        return {"run_id": run_id, **state}

    stage_states = dict(state.get("stage_states") or _default_strategy_research_stage_states())
    if temporal_status == "completed":
        stage_states["temporal"] = "completed"
        next_status = "stopped" if pipeline_status == "stop_requested" else "running"
        next_stage = None
        failure_message = None
    elif pipeline_status == "stop_requested" or temporal_status in {"cancelled", "stopped"}:
        stage_states["temporal"] = "stopped"
        next_status = "stopped"
        next_stage = "temporal"
        failure_message = None
    else:
        stage_states["temporal"] = "failed"
        next_status = "failed"
        next_stage = "temporal"
        failure_message = str(document.get("failure_message") or "Temporal Intelligence failed.")

    reconciled = _persist_strategy_research_pipeline_state(
        db,
        run_id,
        {
            **state,
            "status": next_status,
            "current_stage": next_stage,
            "stage_states": stage_states,
            "failure_message": failure_message,
        },
    )
    if next_status == "running":
        _start_strategy_research_pipeline_worker(db, run_id)
    return reconciled

def get_strategy_research_pipeline_state(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {"_id": 0, "id": 1, "status": 1, "stage": 1, "failure_message": 1, "strategy_research_pipeline": 1},
    )
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    return _reconcile_detached_strategy_research_pipeline(db, document)


def get_strategy_research_pipeline_snapshot(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {
            "_id": 0, "id": 1, "status": 1, "research_processing_id": 1,
            "strategy_research_pipeline": 1, "strategy_research_final_validation": 1, "result": 1, "analysis_end_date": 1,
        },
    )
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    processing_id = str(document.get("research_processing_id") or "").strip()
    pipeline = _strategy_research_pipeline_state(document)
    result = document.get("result") if isinstance(document.get("result"), dict) else {}
    period_start = str(pipeline.get("start_month") or "").strip()
    period_end = str(pipeline.get("end_month") or "").strip()
    if not period_start:
        period_start = str(result.get("oos_start") or "")[:7]
    if not period_end:
        period_end = str(result.get("oos_end") or document.get("analysis_end_date") or "")[:7]

    query: dict[str, Any] = {"run_id": str(run_id)}
    if processing_id:
        query["processing_id"] = processing_id
    if period_start:
        query["period_start"] = period_start
    if period_end:
        query["period_end"] = period_end

    def latest(collection: str, statuses: tuple[str, ...] = ("completed",)) -> dict[str, Any] | None:
        row = db[collection].find_one(
            {**query, "status": {"$in": list(statuses)}},
            {"_id": 0},
            sort=[("created_at", -1)],
        )
        return bson_value(row) if row is not None else None

    pipeline_milp = None
    if processing_id and period_start and period_end:
        from ..milp_decision.persistence import latest_raw as latest_milp_decision
        pipeline_milp = latest_milp_decision(
            db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
        )
    validation = document.get("strategy_research_final_validation") if isinstance(document.get("strategy_research_final_validation"), dict) else None
    leadership = None
    clustering = None
    opportunity_drought = None
    fragile_incumbent = None
    emerging_trend = None
    alternative_action = None
    operational_policy_qualification = None
    if processing_id and period_start and period_end:
        from ..leadership_regime.service import get_persisted as get_leadership_regime, public_summary as leadership_public_summary
        leadership = leadership_public_summary(
            get_leadership_regime(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..regime_clustering.service import get_persisted as get_regime_clustering, public_summary as clustering_public_summary
        clustering = clustering_public_summary(
            get_regime_clustering(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..opportunity_drought.service import get_persisted as get_opportunity_drought, public_summary as opportunity_drought_public_summary
        opportunity_drought = opportunity_drought_public_summary(
            get_opportunity_drought(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..fragile_incumbent.service import get_persisted as get_fragile_incumbent, public_summary as fragile_incumbent_public_summary
        fragile_incumbent = fragile_incumbent_public_summary(
            get_fragile_incumbent(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..emerging_trend.service import get_persisted as get_emerging_trend, public_summary as emerging_trend_public_summary
        emerging_trend = emerging_trend_public_summary(
            get_emerging_trend(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..alternative_action.service import get_persisted as get_alternative_action, public_summary as alternative_action_public_summary
        alternative_action = alternative_action_public_summary(
            get_alternative_action(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )
        from ..operational_policy_qualification.service import get_persisted as get_operational_policy_qualification, public_summary as operational_policy_public_summary
        operational_policy_qualification = operational_policy_public_summary(
            get_operational_policy_qualification(
                db, run_id, processing_id=processing_id, start_month=period_start, end_month=period_end
            )
        )

    return bson_value({
        "schema_version": 6,
        "run_id": str(run_id),
        "processing_id": processing_id or None,
        "period_start": period_start or None,
        "period_end": period_end or None,
        "pipeline": pipeline,
        "risk": latest(TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION),
        "alternative_action": alternative_action,
        "operational_policy_qualification": operational_policy_qualification,
        "intervention": latest(TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION),
        "confidence": latest(TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION),
        "stateful": latest(TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION, ("completed", "blocked")),
        "milp": pipeline_milp,
        "leadership_regime": leadership,
        "clustering": clustering,
        "opportunity_drought": opportunity_drought,
        "fragile_incumbent": fragile_incumbent,
        "emerging_trend": emerging_trend,
        "validation": bson_value(validation) if validation is not None else None,
    })


def control_strategy_research_pipeline(
    db: Any,
    run_id: str,
    *,
    action: str,
    stage: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    current = _strategy_research_pipeline_state(document)
    stage_states = dict(current["stage_states"])
    now = utc_now()
    next_status = current["status"]
    current_stage = current.get("current_stage")
    failure_message = current.get("failure_message")
    start_worker = False

    if action == "start":
        if not start_month or not end_month:
            raise TemporalIntelligenceConflict("Strategy Research start requires start_month and end_month.")
        if str(end_month) < str(start_month):
            raise TemporalIntelligenceConflict("Strategy Research end_month must be greater than or equal to start_month.")
        stage_states = _default_strategy_research_stage_states()
        stage_states["reference"] = "completed"
        temporal_status = str(document.get("status") or "").lower()
        if temporal_status == "completed":
            stage_states["temporal"] = "completed"
            current_stage = None
        elif temporal_status in {"failed", "interrupted", "cancelled", "stopped"}:
            raise TemporalIntelligenceConflict("A failed or stopped Temporal Intelligence run cannot start Strategy Research. Restart the pipeline.")
        else:
            stage_states["temporal"] = "running"
            current_stage = "temporal"
        next_status = "running"
        failure_message = None
        start_worker = True
    elif action == "resume":
        if next_status not in {"paused", "failed", "running"}:
            raise TemporalIntelligenceConflict("Only a legacy paused or failed Strategy Research pipeline can be recovered.")
        next_status = "running"
        failure_message = None
        if current_stage and stage_states.get(current_stage) in {"paused", "stopped", "failed"}:
            stage_states[current_stage] = "waiting"
        current_stage = None
        start_worker = True
    elif action == "stage_start":
        if stage not in STRATEGY_RESEARCH_PIPELINE_STAGES:
            raise TemporalIntelligenceConflict("Unknown Strategy Research stage.")
        if next_status in {"stop_requested", "stopped"}:
            return {"run_id": str(run_id), **current}
        next_status = "running"
        current_stage = stage
        stage_states[stage] = "running"
        failure_message = None
    elif action == "stage_complete":
        if stage not in STRATEGY_RESEARCH_PIPELINE_STAGES:
            raise TemporalIntelligenceConflict("Unknown Strategy Research stage.")
        stage_states[stage] = "completed"
        current_stage = None
        if next_status in {"stop_requested", "stopped"}:
            next_status = "stopped"
        elif stage == "validation":
            next_status = "completed"
        else:
            next_status = "running"
    elif action == "checkpoint":
        if stage not in STRATEGY_RESEARCH_PIPELINE_STAGES:
            raise TemporalIntelligenceConflict("Unknown Strategy Research stage.")
        if next_status in {"stop_requested", "stopped"}:
            next_status = "stopped"
            current_stage = stage
            if stage_states.get(stage) != "completed":
                stage_states[stage] = "stopped"
    elif action == "stage_failed":
        if stage not in STRATEGY_RESEARCH_PIPELINE_STAGES:
            raise TemporalIntelligenceConflict("Unknown Strategy Research stage.")
        next_status = "failed"
        current_stage = stage
        stage_states[stage] = "failed"
        failure_message = str(message or "Strategy Research stage failed.")
    else:
        raise TemporalIntelligenceConflict("Unsupported Strategy Research pipeline action.")

    update = {
        "status": next_status,
        "current_stage": current_stage,
        "stage_states": stage_states,
        "start_month": start_month or current.get("start_month"),
        "end_month": end_month or current.get("end_month"),
        "failure_message": failure_message,
        "updated_at": now,
    }
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {"strategy_research_pipeline": update, "updated_at": now}},
    )
    result = {"run_id": str(run_id), **_strategy_research_pipeline_state({"strategy_research_pipeline": update})}
    if start_worker:
        _start_strategy_research_pipeline_worker(db, run_id)
    return result

def request_strategy_research_pipeline_pause(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0, "id": 1})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    raise TemporalIntelligenceConflict("Strategy Research does not pause between stages. Use Stop Pipeline to interrupt the full pipeline.")

def request_strategy_research_pipeline_stop(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    current = _strategy_research_pipeline_state(document)
    if current["status"] == "stopped":
        return {"run_id": str(run_id), **current}

    temporal_status = str(document.get("status") or "").lower()
    temporal_active = temporal_status in {"queued", "running", "stop_requested"}
    current_stage = current.get("current_stage")

    if temporal_active and current["status"] not in {"running", "pause_requested", "paused", "stop_requested"}:
        stage_states = dict(current.get("stage_states") or _default_strategy_research_stage_states())
        if stage_states.get("reference") == "waiting":
            stage_states["reference"] = "completed"
        stage_states["temporal"] = "running"
        for stage in ("clustering", "fragile_incumbent", "emerging_trend", "risk", "confidence", "stateful", "milp", "validation"):
            stage_states[stage] = "waiting"
        repaired = _persist_strategy_research_pipeline_state(
            db,
            run_id,
            {
                **current,
                "status": "running",
                "current_stage": "temporal",
                "stage_states": stage_states,
                "failure_message": None,
            },
        )
        current = {key: value for key, value in repaired.items() if key != "run_id"}
        current_stage = "temporal"

    if current["status"] == "completed":
        return {"run_id": str(run_id), **current}
    if current["status"] not in {"running", "pause_requested", "paused", "stop_requested"}:
        raise TemporalIntelligenceConflict("Only an unfinished Strategy Research pipeline can be stopped.")

    if current_stage == "temporal" and temporal_active:
        try:
            stop_temporal_intelligence(db, run_id)
        except TemporalIntelligenceConflict:
            pass

    stage_states = dict(current.get("stage_states") or _default_strategy_research_stage_states())
    if current_stage and stage_states.get(current_stage) not in {"completed", "skipped"}:
        stage_states[current_stage] = "stopped"
    return _persist_strategy_research_pipeline_state(
        db,
        run_id,
        {
            **current,
            "status": "stopped",
            "current_stage": current_stage,
            "stage_states": stage_states,
            "failure_message": None,
        },
    )


def reset_strategy_research_pipeline(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {"_id": 0, "id": 1, "status": 1, "strategy_profile_id": 1},
    )
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    status = str(document.get("status") or "").lower()
    if status in {"queued", "running", "stop_requested"}:
        raise TemporalIntelligenceConflict("Stop the active Temporal Intelligence run before restarting Strategy Research.")

    protected = _strategy_research_run_is_protected(db, run_id)
    if protected:
        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
            {"id": str(run_id)},
            {"$unset": {"strategy_research_pipeline": "", "strategy_research_final_validation": ""}, "$set": {"updated_at": utc_now()}},
        )
        from ..leadership_regime.service import delete_run_results as delete_leadership_results
        from ..opportunity_drought.service import delete_run_results as delete_opportunity_drought_results
        from ..fragile_incumbent.service import delete_run_results as delete_fragile_incumbent_results
        from ..regime_clustering.service import delete_run_results as delete_regime_clustering_results
        from ..emerging_trend.service import delete_run_results as delete_emerging_trend_results
        from ..alternative_action.service import delete_run_results as delete_alternative_action_results
        from ..operational_policy_qualification.service import delete_run_results as delete_operational_policy_qualification_results
        deleted = {
            "observations": 0, "artifacts": 0, "risk": 0, "intervention": 0, "confidence": 0,
            "decision_policy": 0, "decision_optimization": 0,
            "leadership_regime": delete_leadership_results(db, run_id),
            "opportunity_drought": delete_opportunity_drought_results(db, run_id),
            "clustering": delete_regime_clustering_results(db, run_id),
            "emerging_trend": delete_emerging_trend_results(db, run_id),
            "alternative_action": delete_alternative_action_results(db, run_id),
            "operational_policy_qualification": delete_operational_policy_qualification_results(db, run_id),
            "fragile_incumbent": delete_fragile_incumbent_results(db, run_id), "runs": 0,
        }
    else:
        deleted = _delete_strategy_research_run_data(db, run_id, delete_run=True)

    retention = purge_strategy_research_history(
        db,
        strategy_profile_id=str(document.get("strategy_profile_id") or ""),
        keep=STRATEGY_RESEARCH_HISTORY_KEEP,
        exclude_run_ids=set(),
    )
    return {
        "run_id": str(run_id),
        "status": "reset",
        "protected": bool(protected),
        "deleted": deleted,
        "deleted_total": int(sum(deleted.values())),
        "history_purge": retention,
    }


def stop_temporal_intelligence(db: Any, run_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise TemporalIntelligenceNotFound("Temporal Intelligence run not found.")
    status = str(document.get("status") or "")
    if status not in {"queued", "running", "stop_requested"}:
        raise TemporalIntelligenceConflict("Only an active Temporal Intelligence run can be stopped.")
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
        {"id": str(run_id)}, {"$set": {"status": "stop_requested", "stage": "Stopping", "updated_at": utc_now()}}
    )
    with _ACTIVE_LOCK:
        process = _ACTIVE_PROCESSES.get(str(run_id))
    if process is not None:
        _terminate(process)
    return get_temporal_intelligence_run(db, run_id)


def recover_temporal_intelligence_runs(db: Any) -> int:
    now = utc_now()
    result = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_many(
        {"status": {"$in": ["queued", "running", "stop_requested"]}},
        {"$set": {"status": "interrupted", "stage": "Interrupted after API restart", "failure_message": "The Temporal Intelligence process was interrupted by an API restart.", "finished_at": now, "updated_at": now}, "$unset": {"process_id": ""}},
    )
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_many(
        {
            "status": "interrupted",
            "strategy_research_pipeline.current_stage": "temporal",
            "strategy_research_pipeline.status": "stop_requested",
        },
        {
            "$set": {
                "strategy_research_pipeline.status": "stopped",
                "strategy_research_pipeline.stage_states.temporal": "stopped",
                "strategy_research_pipeline.updated_at": now,
            }
        },
    )
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_many(
        {
            "status": "interrupted",
            "strategy_research_pipeline.current_stage": "temporal",
            "strategy_research_pipeline.status": {"$in": ["running", "pause_requested", "paused"]},
        },
        {
            "$set": {
                "strategy_research_pipeline.status": "failed",
                "strategy_research_pipeline.stage_states.temporal": "failed",
                "strategy_research_pipeline.failure_message": "Temporal Intelligence was interrupted by an API restart.",
                "strategy_research_pipeline.updated_at": now,
            }
        },
    )
    resumable = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find(
        {
            "status": "completed",
            "strategy_research_pipeline.status": "running",
        },
        {"_id": 0, "id": 1},
    )
    for item in resumable:
        run_id = str(item.get("id") or "").strip()
        if run_id:
            _start_strategy_research_pipeline_worker(db, run_id)
    return int(getattr(result, "modified_count", 0) or 0)
