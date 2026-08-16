from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import io
import json
import os
import subprocess
import sys
import threading
import uuid
import zipfile
import zlib
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from ..core.config import SOURCE_ROOT
from ..core.environment import build_subprocess_environment, load_project_environment
from ..engine.market_data import resolve_backtest_analysis_end_date
from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    MODEL_TUNING_RUNS_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestExecutionRequest
from .model_research import apply_execution_profile, model_execution_snapshot
from .model_tuning_market_snapshot import freeze_tuning_market_snapshot, market_snapshot_exists
from .strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    get_trader_winner_context,
    get_trader_winner_model_snapshot,
    materialize_temporal_strategy,
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
        "model_family": document.get("model_family"),
        "model_label": document.get("model_label"),
        "model_settings_hash": document.get("model_settings_hash"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "market_data_snapshot_source": document.get("market_data_snapshot_source"),
        "market_data_snapshot_source_run_id": document.get("market_data_snapshot_source_run_id"),
        "deterministic_execution": bool((document.get("request") or {}).get("deterministic_execution")),
        "analysis_end_date": document.get("analysis_end_date"),
        "horizons": list(document.get("horizons") or []),
        "failure_message": document.get("failure_message"),
        "experiment": document.get("experiment") or (result.get("experiment") if isinstance(result, dict) else None),
        "materialized_strategy_id": document.get("materialized_strategy_id"),
        "materialized_strategy_name": document.get("materialized_strategy_name"),
        "materialized_strategy_at": bson_value(document.get("materialized_strategy_at")),
        "shadow_only": True,
        **({"result": bson_value(_public_temporal_result(result)) if result is not None else None} if include_result else {}),
    }


def _build_execution_request(db: Any) -> tuple[BacktestExecutionRequest, dict[str, Any], dict[str, Any]]:
    winner_configuration, winner_strategy = get_trader_winner_context(db)
    model_snapshot = get_trader_winner_model_snapshot(db)
    model_family = str(model_snapshot.get("family") or "")
    if model_family != "lightgbm_utility":
        raise TemporalIntelligenceConflict(
            "Temporal Decision Intelligence v8 requires the immutable Trader Winner to use LightGBM."
        )
    model_settings = deepcopy(model_snapshot.get("settings_snapshot") or {})

    locked = apply_training_runtime_settings(db, winner_configuration)
    locked = apply_execution_profile(locked, model_family, model_settings)
    anchors = list(locked.assets)
    reference = list(locked.assets)
    candidates: list[str] = []
    resolved_end = resolve_backtest_analysis_end_date(locked)

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
    return request, winner_strategy, model_snapshot


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
        request, strategy, model_snapshot = _build_execution_request(db)
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
        "model_family": model_snapshot.get("family"),
        "model_label": model_snapshot.get("label"),
        "model_settings_hash": model_snapshot.get("settings_hash"),
        "model_settings_revision": model_snapshot.get("settings_revision"),
        "market_data_snapshot_id": snapshot_id,
        "market_data_snapshot_source": snapshot_source,
        "market_data_snapshot_source_run_id": source_run_id,
        "analysis_end_date": request.analysis_end_date,
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


def build_temporal_intelligence_export(db: Any, run_id: str) -> bytes:
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
                "analysis_end_date", "horizons", "system_settings_revision", "shadow_only",
            )
        },
        "request": deepcopy(document.get("request") or {}),
        "result": manifest_result,
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
    return archive_buffer.getvalue()

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
    return int(getattr(result, "modified_count", 0) or 0)
