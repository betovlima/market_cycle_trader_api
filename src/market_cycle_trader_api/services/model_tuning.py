from __future__ import annotations

from copy import deepcopy
import csv
import io
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import threading
import traceback
import uuid
import zipfile
from typing import Any

from scipy.stats import qmc
from pymongo import ReturnDocument

from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    FAILURES_COLLECTION,
    JOBS_COLLECTION,
    MODEL_TUNING_RUNS_COLLECTION,
    MODEL_TUNING_VALIDATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
    bson_value,
    utc_now,
)
from .jobs import _redact_sensitive_text, request_job_cancel, run_job
from .model_research import model_values_from_snapshot
from .reproducibility import market_data_research_signature_from_manifests
from .model_tuning_probability import (
    PROBABILITY_MODEL,
    champion_gate_evaluation,
    evolve_probability_search,
    initial_probability_state,
    propose_champion_probability_candidate,
    propose_unified_space_filling_candidate,
    unified_caro_next_mode,
)
from .model_tuning_space import normalize_tuning_values as _normalize_tuning_values, settings_from_unit_point
from .model_tuning_ranking import candidate_economic_sort_key
from .model_tuning_market_snapshot import (
    TuningMarketSnapshotMismatch,
    freeze_tuning_market_snapshot,
    market_snapshot_exists,
    require_tuning_market_snapshot,
)
from .serialization import downsample_documents, iso_value
from .temporal_policy_tuning import (
    TEMPORAL_POLICY_MODEL_FAMILY,
    TEMPORAL_POLICY_TUNING_SCOPE,
    derived_temporal_policy_snapshot,
    evaluate_temporal_policy_candidate,
    is_temporal_policy_strategy,
    temporal_policy_baseline,
    temporal_policy_plan,
)
from .temporal_model_tuning import (
    TEMPORAL_MODEL_FAMILY,
    TEMPORAL_MODEL_TUNING_SCOPE,
    TemporalModelTuningCancelled,
    evaluate_temporal_model_candidate,
    persist_temporal_model_champion_cache,
    prepare_temporal_model_campaign_context,
    temporal_model_baseline,
    temporal_model_plan,
)
from ..schemas.requests import BacktestRequest
from .strategy_lab import (
    create_strategy,
    create_tuned_temporal_strategy,
    get_strategy,
    get_strategy_control,
    get_strategy_model_snapshot,
    prepare_strategy_for_backtest_candidate,
    select_model_tuning_strategy,
    select_research_strategy,
    update_strategy,
    update_strategy_model,
    _configuration_hash as _strategy_configuration_hash,
)

TUNING_METHOD = "latin_hypercube"
PROBABILITY_METHOD = "champion_probability"
PIPELINE_METHOD = "latin_hypercube_then_caro"
_ADAPTIVE_METHODS = {PROBABILITY_METHOD, PIPELINE_METHOD}
TUNING_MODEL_FAMILY = "lightgbm_utility"
TUNING_SCHEMA_VERSION = 16
DEFAULT_CANDIDATE_COUNT = 20
DEFAULT_SEED = 42
TECHNICAL_RESEARCH_SEGMENT_MAX = 2000
DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT = 100
DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT = 0.0025
DEFAULT_RESEARCH_FOLDS = 3
DEFAULT_VALIDATION_FOLDS = 5
DEFAULT_CERTIFICATION_FOLDS = 7


def _normalized_fold_protocol(value: dict[str, Any] | None) -> dict[str, int]:
    payload = dict(value or {})
    research = int(payload.get("research_folds") or DEFAULT_RESEARCH_FOLDS)
    validation = int(payload.get("validation_folds") or DEFAULT_VALIDATION_FOLDS)
    certification = int(payload.get("certification_folds") or DEFAULT_CERTIFICATION_FOLDS)
    if research < 2:
        raise ModelTuningConflict("Research folds must be at least 2.")
    if validation < 2:
        raise ModelTuningConflict("Validation folds must be at least 2.")
    if certification < 2:
        raise ModelTuningConflict("Certification folds must be at least 2.")
    if validation < research:
        raise ModelTuningConflict("Validation folds must be greater than or equal to research folds.")
    if certification < validation:
        raise ModelTuningConflict("Certification folds must be greater than or equal to validation folds.")
    return {
        "research_folds": research,
        "validation_folds": validation,
        "certification_folds": certification,
    }



_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"name": "n_estimators", "type": "integer", "min": 220, "max": 380},
    {"name": "learning_rate", "type": "number", "min": 0.020, "max": 0.050, "precision": 6},
    {"name": "max_depth", "type": "integer", "min": 2, "max": 4},
    {"name": "num_leaves", "type": "integer", "min": 4, "max": 12},
    {"name": "min_child_samples", "type": "integer", "min": 15, "max": 30},
    {"name": "colsample_bytree", "type": "number", "min": 0.75, "max": 0.95, "precision": 6},
    {"name": "reg_alpha", "type": "number", "min": 0.0, "max": 0.50, "precision": 6},
    {"name": "reg_lambda", "type": "number", "min": 1.0, "max": 4.0, "precision": 6},
)
_TUNED_NAMES = tuple(item["name"] for item in _SEARCH_SPACE)
MODEL_PARAMETER_TUNING_SCOPE = "model_parameters"
LEGACY_ABSOLUTE_UTILITY_TUNING_SCOPE = "absolute_utility_cash_gate"
ABSOLUTE_UTILITY_TUNING_SCOPE = "joint_model_absolute_utility_cash_gate"
ABSOLUTE_UTILITY_STRATEGY_MODE = "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE"
_ABSOLUTE_UTILITY_STRATEGY_PARAMETER_NAMES = (
    "opportunity_utility_entry_threshold",
    "opportunity_utility_exit_threshold",
)
_ABSOLUTE_UTILITY_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    *_SEARCH_SPACE,
    {"name": "opportunity_utility_entry_threshold", "type": "number", "min": 0.22, "max": 0.38, "precision": 6},
    {"name": "opportunity_utility_exit_threshold", "type": "number", "min": 0.20, "max": 0.36, "precision": 6},
)
_ACTIVE_STATUSES = ("queued", "running", "stop_requested")
_TUNING_LOG_MAX_EVENTS = 250
_TUNING_CANDIDATE_LOG_MAX_LINES = 400
_AUTHORIZATION_PATTERN = re.compile(r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[^\s]+")
_GENERIC_SECRET_PATTERN = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|access[_-]?token|refresh[_-]?token)\b(\s*[:=]\s*)([^\s,;]+)")
_GENERIC_CREDENTIAL_URI_PATTERN = re.compile(r"([a-z][a-z0-9+.-]*://)([^/@\s:]+):([^@\s]+)@", flags=re.IGNORECASE)


class ModelTuningConflict(RuntimeError):
    pass


class ModelTuningNotFound(RuntimeError):
    pass


def _tuning_target_strategy(db: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    control = get_strategy_control(db)
    strategy_id = str(
        control.get("strategy_research_strategy_id")
        or control.get("research_strategy_id")
        or ""
    ).strip()
    if not strategy_id:
        raise ModelTuningConflict("No Strategy is selected for Strategy Research.")
    strategy = get_strategy(db, strategy_id)
    model_snapshot = get_strategy_model_snapshot(db, strategy_id)
    return strategy, model_snapshot, "strategy_research_selection"


def _tuning_target_allows_locked_strategy(strategy: dict[str, Any]) -> bool:
    # Lifecycle status/locked protects governance and editing, not research eligibility.
    # Technical compatibility is validated by the tuning plan and baseline checks.
    return True


def _sanitize_tuning_log_line(raw_line: Any) -> str:
    line = _redact_sensitive_text(raw_line)
    line = _AUTHORIZATION_PATTERN.sub(r"\1***", line)
    line = _GENERIC_SECRET_PATTERN.sub(r"\1\2***", line)
    line = _GENERIC_CREDENTIAL_URI_PATTERN.sub(r"\1***:***@", line)
    return line.strip()


def _append_campaign_event(
    db: Any,
    run_id: str,
    *,
    message: str,
    level: str = "info",
    stage: str | None = None,
    candidate_id: int | None = None,
    job_id: str | None = None,
) -> None:
    event = {
        "at": utc_now(),
        "level": str(level or "info").lower(),
        "stage": str(stage or "").strip() or None,
        "message": _sanitize_tuning_log_line(message),
        "candidate_id": int(candidate_id) if candidate_id is not None else None,
        "job_id": str(job_id or "").strip() or None,
    }
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {
            "$push": {"event_log": {"$each": [bson_value(event)], "$slice": -_TUNING_LOG_MAX_EVENTS}},
            "$set": {"updated_at": utc_now()},
        },
    )


def _infer_failure_details(job: dict[str, Any] | None, exc: Exception) -> tuple[str, str]:
    job = job or {}
    if bool(job.get("timed_out")):
        return "TrainingTimeout", "Training exceeded the configured time limit."

    raw_logs = job.get("logs") if isinstance(job.get("logs"), list) else []
    exception_pattern = re.compile(r"(?:^|\b)([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Mismatch)):\s*(.+)$")
    for raw_line in reversed(raw_logs):
        line = _sanitize_tuning_log_line(raw_line)
        match = exception_pattern.search(line)
        if match:
            failure_type = match.group(1).split(".")[-1]
            return failure_type[:120], line[:500]
        if line.startswith("ERROR") and len(line) > 6:
            return "BacktestEngineError", line[:500]

    raw_error = _sanitize_tuning_log_line(job.get("error") or "")
    if raw_error:
        return "BacktestWorkerError", raw_error[:500]
    if job.get("return_code") not in (None, 0):
        return "BacktestEngineExit", f"Backtest engine exited with code {job.get('return_code')}."
    return type(exc).__name__, _sanitize_tuning_log_line(str(exc))[:500]


def _diagnostic_traceback_lines() -> list[str]:
    lines = [_sanitize_tuning_log_line(line) for line in traceback.format_exc().splitlines() if line.strip()]
    return lines[-120:]


def _format_campaign_log_text(payload: dict[str, Any]) -> str:
    lines = [
        "MODEL TUNING CAMPAIGN LOG",
        f"Campaign: {payload.get('run_id') or '—'}",
        f"Status: {payload.get('status') or '—'}",
        f"Phase: {payload.get('phase') or '—'}",
        f"Method: {payload.get('method') or '—'}",
        f"Failure: {payload.get('failure_type') or '—'}",
        f"Failure message: {payload.get('failure_message') or '—'}",
        "",
        "EVENTS",
    ]
    events = payload.get("events") or []
    if not events:
        lines.append("No campaign events were recorded.")
    for event in events:
        at = event.get("at") or "—"
        level = str(event.get("level") or "info").upper()
        candidate = f" candidate=#{event.get('candidate_id')}" if event.get("candidate_id") is not None else ""
        job = f" job={event.get('job_id')}" if event.get("job_id") else ""
        stage = f" stage={event.get('stage')}" if event.get("stage") else ""
        lines.append(f"[{at}] {level}{stage}{candidate}{job} | {event.get('message') or ''}")
    return "\n".join(lines).strip() + "\n"


def _format_candidate_log_text(payload: dict[str, Any]) -> str:
    lines = [
        "MODEL TUNING CANDIDATE LOG",
        f"Campaign: {payload.get('run_id') or '—'}",
        f"Candidate: {payload.get('candidate_id')}",
        f"Kind: {payload.get('kind') or '—'}",
        f"Status: {payload.get('status') or '—'}",
        f"Job: {payload.get('job_id') or '—'}",
        f"Job status: {payload.get('job_status') or '—'}",
        f"Job stage: {payload.get('job_stage') or '—'}",
        f"Return code: {payload.get('return_code') if payload.get('return_code') is not None else '—'}",
        f"Failure: {payload.get('failure_type') or '—'}",
        f"Failure message: {payload.get('failure_message') or '—'}",
        "",
        "CAMPAIGN EVENTS",
    ]
    events = payload.get("events") or []
    if not events:
        lines.append("No candidate-specific campaign events were recorded.")
    for event in events:
        at = event.get("at") or "—"
        level = str(event.get("level") or "info").upper()
        stage = f" stage={event.get('stage')}" if event.get("stage") else ""
        lines.append(f"[{at}] {level}{stage} | {event.get('message') or ''}")
    lines.extend(["", "DIAGNOSTIC TRACEBACK"])
    diagnostics = payload.get("diagnostic_lines") or []
    lines.extend(diagnostics or ["No tuning-worker traceback was recorded."])
    lines.extend(["", "BACKTEST JOB LOG"])
    job_lines = payload.get("job_log_lines") or []
    lines.extend(job_lines or ["No backtest job log lines were recorded."])
    return "\n".join(str(item) for item in lines).strip() + "\n"


def get_model_tuning_campaign_log(db: Any, run_id: str) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    events = []
    for raw_event in list(document.get("event_log") or [])[-_TUNING_LOG_MAX_EVENTS:]:
        if not isinstance(raw_event, dict):
            continue
        event = deepcopy(raw_event)
        event.pop("_id", None)
        event["message"] = _sanitize_tuning_log_line(event.get("message") or "")
        events.append(bson_value(event))
    payload = bson_value({
        "scope": "campaign",
        "run_id": run_id,
        "status": str(document.get("status") or "unknown"),
        "phase": str(document.get("phase") or "unknown"),
        "method": str(document.get("method") or TUNING_METHOD),
        "failure_type": document.get("failure_type"),
        "failure_message": _sanitize_tuning_log_line(document.get("failure_message") or "") or None,
        "events": events,
    })
    payload["log_text"] = _format_campaign_log_text(payload)
    return payload


def get_model_tuning_candidate_log(db: Any, run_id: str, candidate_id: int) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    candidate = next(
        (item for item in list(document.get("candidates") or []) if int(item.get("candidate_id") or 0) == int(candidate_id)),
        None,
    )
    if candidate is None:
        raise ModelTuningNotFound("Model tuning candidate not found.")

    job_id = str(candidate.get("job_id") or "")
    job = db[JOBS_COLLECTION].find_one({"id": job_id}) if job_id else None
    raw_logs = list((job or {}).get("logs") or [])[-_TUNING_CANDIDATE_LOG_MAX_LINES:]
    job_log_lines = [_sanitize_tuning_log_line(line) for line in raw_logs if str(line).strip()]
    diagnostic_lines = [
        _sanitize_tuning_log_line(line)
        for line in list(candidate.get("diagnostic_log") or [])[-120:]
        if str(line).strip()
    ]
    events = []
    for raw_event in list(document.get("event_log") or [])[-_TUNING_LOG_MAX_EVENTS:]:
        if not isinstance(raw_event, dict):
            continue
        if raw_event.get("candidate_id") is None or int(raw_event.get("candidate_id")) != int(candidate_id):
            continue
        event = deepcopy(raw_event)
        event.pop("_id", None)
        event["message"] = _sanitize_tuning_log_line(event.get("message") or "")
        events.append(bson_value(event))

    fallback_exc = RuntimeError(str(candidate.get("error") or "Candidate execution failed."))
    inferred_type, inferred_message = _infer_failure_details(job, fallback_exc)
    failure_type = str(candidate.get("failure_type") or "").strip() or (inferred_type if str(candidate.get("status")) == "failed" else None)
    failure_message = _sanitize_tuning_log_line(candidate.get("failure_message") or candidate.get("error") or "") or None
    if str(candidate.get("status")) == "failed" and not failure_message:
        failure_message = inferred_message

    payload = bson_value({
        "scope": "candidate",
        "run_id": run_id,
        "candidate_id": int(candidate_id),
        "kind": str(candidate.get("kind") or "latin_hypercube"),
        "status": str(candidate.get("status") or "unknown"),
        "job_id": job_id or None,
        "job_status": (job or {}).get("status"),
        "job_stage": (job or {}).get("stage"),
        "return_code": (job or {}).get("return_code"),
        "failure_type": failure_type,
        "failure_message": failure_message,
        "events": events,
        "diagnostic_lines": diagnostic_lines,
        "job_log_lines": job_log_lines,
    })
    payload["log_text"] = _format_candidate_log_text(payload)
    return payload


def _tuning_plan(
    strategy: dict[str, Any],
    model_snapshot: dict[str, Any],
    requested_target: str | None = None,
) -> dict[str, Any]:
    if str(strategy.get("strategy_kind") or "") == "temporal_intelligence":
        if str(requested_target or "") == TEMPORAL_MODEL_TUNING_SCOPE:
            return temporal_model_plan(strategy, model_snapshot)
        return temporal_policy_plan(strategy)
    configuration = strategy.get("configuration") if isinstance(strategy.get("configuration"), dict) else {}
    mode = str(configuration.get("strategy_mode") or "")
    frozen_model_values = model_values_from_snapshot(model_snapshot)
    if mode == ABSOLUTE_UTILITY_STRATEGY_MODE:
        search_space = [dict(item) for item in _ABSOLUTE_UTILITY_SEARCH_SPACE]
        base_values = deepcopy(frozen_model_values)
        base_values.update({
            "opportunity_utility_entry_threshold": float(configuration.get("opportunity_utility_entry_threshold", 0.28)),
            "opportunity_utility_exit_threshold": float(configuration.get("opportunity_utility_exit_threshold", 0.27)),
        })
        return {
            "scope": ABSOLUTE_UTILITY_TUNING_SCOPE,
            "scope_label": "Joint LightGBM + Absolute Utility Cash Gate",
            "description": "Jointly tune the LightGBM ranking hyperparameters and MARKET/CASH hysteresis thresholds from the active Candidate Backtest, under one frozen validation and market-data protocol.",
            "search_space": search_space,
            "tuned_parameters": [item["name"] for item in search_space],
            "tuned_model_parameters": list(_TUNED_NAMES),
            "tuned_strategy_parameters": list(_ABSOLUTE_UTILITY_STRATEGY_PARAMETER_NAMES),
            "base_values": _normalize_tuning_values(base_values, search_space),
            "base_model_values": frozen_model_values,
            "frozen_model_values": frozen_model_values,
            "fixed_model_values": {
                name: value for name, value in frozen_model_values.items() if name not in _TUNED_NAMES
            },
            "strategy_mode": mode,
        }
    search_space = [dict(item) for item in _SEARCH_SPACE]
    return {
        "scope": MODEL_PARAMETER_TUNING_SCOPE,
        "scope_label": "LightGBM model parameters",
        "description": "Tune the saved LightGBM model hyperparameters under the frozen Strategy and market-data protocol.",
        "search_space": search_space,
        "tuned_parameters": [item["name"] for item in search_space],
        "tuned_model_parameters": [item["name"] for item in search_space],
        "tuned_strategy_parameters": [],
        "base_values": frozen_model_values,
        "base_model_values": frozen_model_values,
        "frozen_model_values": frozen_model_values,
        "fixed_model_values": {
            name: value for name, value in frozen_model_values.items() if name not in _TUNED_NAMES
        },
        "strategy_mode": mode,
    }


def _current_tuning_plan(db: Any | None) -> dict[str, Any] | None:
    if db is None:
        return None
    try:
        strategy, model_snapshot, _ = _tuning_target_strategy(db)
        return _tuning_plan(strategy, model_snapshot)
    except Exception:
        return None


def tuning_catalog(db: Any | None = None) -> dict[str, Any]:
    plan = _current_tuning_plan(db)
    search_space = list((plan or {}).get("search_space") or [dict(item) for item in _SEARCH_SPACE])
    scope = str((plan or {}).get("scope") or MODEL_PARAMETER_TUNING_SCOPE)
    scope_label = str((plan or {}).get("scope_label") or "LightGBM model parameters")
    scope_description = str((plan or {}).get("description") or "Tune the saved LightGBM model hyperparameters under the frozen Strategy and market-data protocol.")
    temporal_scope = scope in {TEMPORAL_POLICY_TUNING_SCOPE, TEMPORAL_MODEL_TUNING_SCOPE}
    temporal_modes: list[dict[str, Any]] = []
    if db is not None:
        try:
            temporal_strategy, temporal_model_snapshot, _ = _tuning_target_strategy(db)
            if (
                str(temporal_strategy.get("strategy_kind") or "") == "temporal_intelligence"
                and str(temporal_strategy.get("tuning_target") or "") != "decision_optimization"
            ):
                for target in (TEMPORAL_MODEL_TUNING_SCOPE, TEMPORAL_POLICY_TUNING_SCOPE):
                    target_plan = _tuning_plan(temporal_strategy, temporal_model_snapshot, target)
                    temporal_modes.append({
                        "id": target,
                        "label": target_plan["scope_label"],
                        "description": target_plan["description"],
                        "search_space": deepcopy(target_plan["search_space"]),
                        "tuned_parameters": list(target_plan["tuned_parameters"]),
                        "tuned_model_parameters": list(target_plan.get("tuned_model_parameters") or []),
                        "tuned_strategy_parameters": list(target_plan.get("tuned_strategy_parameters") or []),
                        "execution_mode": "full_temporal_lightgbm_retrain" if target == TEMPORAL_MODEL_TUNING_SCOPE else "frozen_temporal_replay",
                    })
        except Exception:
            temporal_modes = []
    strategy_compatibility = {"eligible": True, "reason": None}
    if db is not None:
        try:
            selected_strategy, _, _ = _tuning_target_strategy(db)
            if str(selected_strategy.get("tuning_target") or "") == "decision_optimization":
                strategy_compatibility = {
                    "eligible": False,
                    "reason": "MILP Decision Optimization is research-only and is not supported by the current Model Tuning engine.",
                }
            elif (
                str(selected_strategy.get("strategy_kind") or "") == "temporal_intelligence"
                and str(selected_strategy.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
            ):
                strategy_compatibility = {
                    "eligible": False,
                    "reason": "Stateful-transition tuning is not implemented in the current Model Tuning engine. The Strategy remains the shared Strategy Research selection.",
                }
        except Exception as exc:
            strategy_compatibility = {"eligible": False, "reason": str(exc)}
    default_startup_trials = max(4, min(24, len(search_space) + 2))
    return {
        "schema_version": TUNING_SCHEMA_VERSION,
        "start_request_contract_version": 1,
        "strategy_selection_source": "strategy_research_selection",
        "strategy_compatibility": strategy_compatibility,
        "method": PROBABILITY_METHOD,
        "methods": [
            {
                "id": PROBABILITY_METHOD,
                "label": "Unified Adaptive CARO — recommended",
                "description": "One sequential research engine that automatically alternates Latin-Hypercube space filling, global exploration and Champion-focused probabilistic refinement. There is no fixed Hypercube → CARO boundary.",
            },
            {
                "id": TUNING_METHOD,
                "label": "Latin Hypercube — diagnostics only",
                "description": "Static space-filling design retained for sensitivity analysis and diagnostics. It does not learn from candidate outcomes and is not required before Unified CARO.",
            },
        ],
        "recommended_method": PROBABILITY_METHOD,
        "model_family": (TEMPORAL_MODEL_FAMILY if scope == TEMPORAL_MODEL_TUNING_SCOPE else TEMPORAL_POLICY_MODEL_FAMILY) if temporal_scope else TUNING_MODEL_FAMILY,
        "model_label": ("LightGBM Temporal Intelligence" if scope == TEMPORAL_MODEL_TUNING_SCOPE else "Temporal Policy") if temporal_scope else "LightGBM Utility",
        "temporal_tuning_modes": temporal_modes,
        "default_temporal_tuning_target": TEMPORAL_MODEL_TUNING_SCOPE if temporal_modes else None,
        "tuning_scope": scope,
        "tuning_scope_label": scope_label,
        "tuning_scope_description": scope_description,
        "baseline_policy": "materialized_temporal_run" if temporal_scope else "active_candidate_certified_backtest",
        "joint_optimization": scope == ABSOLUTE_UTILITY_TUNING_SCOPE,
        "default_candidate_count": DEFAULT_CANDIDATE_COUNT,
        "candidate_count_min": 4,
        "candidate_count_max": TECHNICAL_RESEARCH_SEGMENT_MAX,
        "research_budget_min": 4,
        "research_budget_technical_segment_max": TECHNICAL_RESEARCH_SEGMENT_MAX,
        "research_budget_unbounded_across_continuations": True,
        "default_seed": DEFAULT_SEED,
        "control_candidate_included": True,
        "control_execution_mode": "reuse_materialized_temporal_result" if temporal_scope else "reuse_certified_candidate_backtest",
        "baseline_execution_required": True,
        "campaign_export_available": True,
        "validation": "chronological_walk_forward",
        "selection_metric": "champion_gate_then_realized_economic_quality",
        "legacy_compound_score_role": "diagnostic_tiebreaker",
        "eligibility_gate": "all_walk_forward_folds_positive",
        "search_space": search_space,
        "tuned_parameters": list((plan or {}).get("tuned_parameters") or [item["name"] for item in search_space]),
        "tuned_model_parameters": list((plan or {}).get("tuned_model_parameters") or [item["name"] for item in search_space]),
        "tuned_strategy_parameters": list((plan or {}).get("tuned_strategy_parameters") or []),
        "raw_artifacts": "summary_only",
        "adoption_requires_final_backtest": False,
        "dedicated_worker": False,
        "execution_mode": ("full_temporal_lightgbm_retrain" if scope == TEMPORAL_MODEL_TUNING_SCOPE else "frozen_temporal_replay") if temporal_scope else "integrated_api_worker",
        "market_data_access": "frozen_temporal_snapshot_only" if temporal_scope else "database_only",
        "prior_campaign_reuse": True,
        "automatic_compatible_prior_observation_reuse": not temporal_scope,
        "continue_research_available": True,
        "historical_trial_retention": "active_campaign_only",
        "durable_candidate_retention": "adopted_or_validated_only",
        "technical_failure_retention": "campaign_counters_and_summary_only",
        "historical_cleanup_trigger": "next_tuning_campaign_start",
        "fold_protocol": {
            "supported": bool(temporal_scope),
            "research_default": DEFAULT_RESEARCH_FOLDS,
            "validation_default": DEFAULT_VALIDATION_FOLDS,
            "certification_default": DEFAULT_CERTIFICATION_FOLDS,
            "minimum": 2,
            "technical_maximum": None,
            "maximum_constraint": "available_out_of_sample_history_and_minimum_test_rows_per_fold",
            "ordering_rule": "research <= validation <= certification",
            "research_role": "candidate_search",
            "validation_role": "full_temporal_retrain_finalist_validation",
            "certification_role": "full_temporal_retrain_candidate_certification",
        },
        "automatic_lhs_to_caro_handoff": False,
        "unified_caro": True,
        "dynamic_exploration": True,
        "legacy_pipeline_supported": True,
        "reproducibility_guard": "materialized_temporal_strategy_and_frozen_replay" if temporal_scope else "frozen_execution_snapshot_and_market_data_signature",
        "restart_recovery": "invalidate_unfinished_campaign_after_restart",
        "probability": {
            "label": "Unified Adaptive CARO",
            "probability_model": PROBABILITY_MODEL,
            "default_minimum_exploration_trials": default_startup_trials,
            "default_startup_trials": default_startup_trials,
            "search_policy": "dynamic_space_filling_plus_sequential_adaptive_trust_region",
            "exploration_strategy": "automatic",
            "space_filling_sampler": "latin_hypercube_maximin",
            "global_exploration_never_zero": True,
            "default_initial_exploration_fraction": 0.45,
            "default_minimum_exploration_fraction": 0.20,
            "default_stagnation_recovery_trials": 4,
            "champion_policy": "promote_only_after_observed_champion_gate_pass",
            "default_min_capital_improvement": 0.03,
            "default_sharpe_tolerance": 0.05,
            "default_drawdown_tolerance": 0.03,
            "default_min_worst_fold_return": 0.0,
            "default_candidate_pool_size": 2048,
            "default_exploration_weight": 0.15,
            "default_adaptive_stopping_enabled": True,
            "default_no_improvement_trial_limit": DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT,
            "default_minimum_meaningful_improvement": DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT,
            "interpretation": "Estimated probability of outperforming the current research Champion under the frozen validation protocol; not a probability of future market profit.",
        },
    }


def _settings_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_latin_hypercube_candidates(
    base_values: dict[str, Any],
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
    search_space: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> list[dict[str, Any]]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    if not base_values:
        raise ValueError("A tuning baseline parameter snapshot is required.")
    active_space = [dict(item) for item in (search_space or _SEARCH_SPACE)]
    if not active_space:
        raise ValueError("A non-empty tuning search space is required.")
    control_values = _normalize_tuning_values(base_values, active_space)
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": 0,
            "kind": "control",
            "is_control": True,
            "settings": deepcopy(control_values),
            "settings_hash": _settings_hash(control_values),
            "status": "pending",
        }
    ]
    seen = {candidates[0]["settings_hash"]}

    def add_points(points: Any) -> None:
        for point in points:
            if len(candidates) >= candidate_count + 1:
                return
            values = settings_from_unit_point(control_values, active_space, point)
            fingerprint = _settings_hash(values)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "kind": "latin_hypercube",
                    "is_control": False,
                    "settings": values,
                    "settings_hash": fingerprint,
                    "status": "pending",
                }
            )

    add_points(qmc.LatinHypercube(d=len(active_space), seed=seed).random(n=candidate_count))
    attempt = 1
    while len(candidates) < candidate_count + 1 and attempt <= 12:
        missing = candidate_count + 1 - len(candidates)
        auxiliary = qmc.LatinHypercube(d=len(active_space), seed=seed + attempt)
        add_points(auxiliary.random(n=max(8, missing * 2)))
        attempt += 1

    if len(candidates) != candidate_count + 1:
        raise RuntimeError("Unable to generate the requested number of unique tuning candidates.")
    return candidates


def _find_portfolio_metrics(db: Any, job_id: str) -> dict[str, Any] | None:
    run = db[RUNS_COLLECTION].find_one(
        {"job_id": job_id, "symbol": "PORTFOLIO", "backend": TUNING_MODEL_FAMILY},
        {"_id": 0, "metrics": 1},
    )
    metrics = (run or {}).get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else None


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    folds = metrics.get("walk_forward_folds") if isinstance(metrics.get("walk_forward_folds"), list) else []
    fold_rows = [
        {
            "fold_id": int(item.get("fold_id") or index + 1),
            "strategy_return": float(item.get("strategy_return") or 0.0),
            "maximum_drawdown": float(item.get("maximum_drawdown") or 0.0),
            "benchmark_return": float(item.get("benchmark_return") or 0.0),
        }
        for index, item in enumerate(folds)
        if isinstance(item, dict)
    ]
    fold_returns = [row["strategy_return"] for row in fold_rows]
    eligible = bool(fold_returns) and all(value > 0 for value in fold_returns)
    signatures = metrics.get("market_data_signatures") if isinstance(metrics.get("market_data_signatures"), dict) else {}
    last_timestamps = [
        str(item.get("last_timestamp"))
        for item in signatures.values()
        if isinstance(item, dict) and item.get("last_timestamp")
    ]
    market_data_last_timestamp = max(last_timestamps) if last_timestamps else None
    return {
        "initial_capital": float(metrics.get("initial_capital") or 0.0),
        "ending_capital": float(metrics.get("strategy_ending_capital") or 0.0),
        "strategy_return": float(metrics.get("strategy_return") or 0.0),
        "cagr": float(metrics.get("strategy_cagr") or 0.0),
        "sharpe": float(metrics.get("strategy_sharpe") or 0.0),
        "maximum_drawdown": float(metrics.get("strategy_maximum_drawdown") or 0.0),
        "risk_adjusted_compound_score": float(metrics.get("risk_adjusted_compound_score") or 0.0),
        "turnover_ratio": float(metrics.get("turnover_ratio") or 0.0),
        "capital_rotations": int(metrics.get("capital_rotations") or 0),
        "average_holding_days": float(metrics.get("average_holding_days") or 0.0),
        "market_exposure": float(metrics.get("market_exposure") or 0.0),
        "cash_days": int(metrics.get("cash_days") or 0),
        "absolute_utility_entry_threshold": (
            float(metrics.get("absolute_utility_entry_threshold"))
            if metrics.get("absolute_utility_entry_threshold") is not None else None
        ),
        "absolute_utility_exit_threshold": (
            float(metrics.get("absolute_utility_exit_threshold"))
            if metrics.get("absolute_utility_exit_threshold") is not None else None
        ),
        "absolute_utility_gate_acceptance_rate": (
            float(metrics.get("absolute_utility_gate_acceptance_rate"))
            if metrics.get("absolute_utility_gate_acceptance_rate") is not None else None
        ),
        "cash_gate_changed_base_action_sessions": int(metrics.get("cash_gate_changed_base_action_sessions") or 0),
        "cash_gate_net_avoided_return_sum": float(metrics.get("cash_gate_net_avoided_return_sum") or 0.0),
        "benchmark_ending_capital": float(metrics.get("buy_hold_ending_capital") or 0.0),
        "market_data_signature_sha256": metrics.get("market_data_signature_sha256"),
        "market_data_last_timestamp": market_data_last_timestamp,
        "folds": fold_rows,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": eligible,
    }


def _candidate_equity_preview(db: Any, job_id: str) -> list[dict[str, Any]]:
    

    run = db[RUNS_COLLECTION].find_one(
        {"job_id": job_id, "symbol": "PORTFOLIO"},
        {"_id": 0, "symbol": 1, "backend": 1},
    )
    if run is None:
        run = db[RUNS_COLLECTION].find_one(
            {"job_id": job_id},
            {"_id": 0, "symbol": 1, "backend": 1},
        )
    if run is None:
        return []

    query = {
        "job_id": job_id,
        "symbol": run.get("symbol"),
        "backend": run.get("backend"),
    }
    rows = list(
        db[PREDICTIONS_COLLECTION]
        .find(
            query,
            {
                "_id": 0,
                "timestamp": 1,
                "strategy_equity": 1,
                "buy_hold_equity": 1,
                "selected_asset": 1,
                "trade_action": 1,
                "final_action_cash_edge": 1,
            },
        )
        .sort("timestamp", 1)
    )
    preview = []
    for row in downsample_documents(rows, maximum_points=500):
        preview.append({
            "timestamp": iso_value(row.get("timestamp")),
            "simulation_equity": _as_finite_float(row.get("strategy_equity")),
            "reference_equity": _as_finite_float(row.get("buy_hold_equity")),
            "selected_asset": str(row.get("selected_asset") or "") or None,
            "trade_action": str(row.get("trade_action") or "") or None,
            "cash_edge": _as_finite_float(row.get("final_action_cash_edge")),
        })
    return preview


def _as_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cleanup_job_artifacts(db: Any, job_id: str) -> None:
    
    db[PREDICTIONS_COLLECTION].delete_many({"job_id": job_id})
    db[TRADES_COLLECTION].delete_many({"job_id": job_id})
    db[RUNS_COLLECTION].delete_many({"job_id": job_id})
    db[COMPARISONS_COLLECTION].delete_many({"job_id": job_id})
    db[FAILURES_COLLECTION].delete_many({"job_id": job_id})
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {"$set": {"tuning_summary_only": True, "raw_results_retained": False, "updated_at": utc_now()}},
    )


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = deepcopy(candidates)
    eligible = [
        item for item in result
        if item.get("status") == "completed"
        and isinstance(item.get("metrics"), dict)
        and bool(item["metrics"].get("eligible"))
    ]
    # Very old campaign documents may predate the explicit `eligible` metric but
    # already contain a persisted rank. Preserve that legacy ranking instead of
    # erasing it when there is not enough information to recompute scientifically.
    if not eligible and any(item.get("rank") is not None for item in result):
        return result
    champion_aware = any(item.get("champion_gate_passed") is not None for item in eligible)
    eligible.sort(
        key=lambda item: candidate_economic_sort_key(item, champion_aware=champion_aware),
        reverse=True,
    )
    rank_by_id = {int(item["candidate_id"]): index + 1 for index, item in enumerate(eligible)}
    for item in result:
        item["rank"] = rank_by_id.get(int(item.get("candidate_id") or 0))
    return result


def _refresh_campaign_ranking(db: Any, run_id: str) -> None:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or {}
    candidates = _rank_candidates(document.get("candidates") or [])
    ranked = [item for item in candidates if item.get("rank") is not None]
    ranked.sort(key=lambda item: int(item["rank"]))
    control = next((item for item in candidates if item.get("is_control")), None)
    best = ranked[0] if ranked else None
    best_exploratory = next((item for item in ranked if not item.get("is_control")), None)
    best_champion_beating = next((item for item in ranked if bool(item.get("champion_gate_passed"))), None)
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {
            "$set": {
                "candidates": bson_value(candidates),
                "best_candidate_id": int(best["candidate_id"]) if best else None,
                "best_exploratory_candidate_id": int(best_exploratory["candidate_id"]) if best_exploratory else None,
                "best_champion_beating_candidate_id": int(best_champion_beating["candidate_id"]) if best_champion_beating else None,
                "control_candidate_id": int(control["candidate_id"]) if control else None,
                "updated_at": utc_now(),
            }
        },
    )




def list_model_tuning_baselines(db: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    strategy, model_snapshot, target_source = _tuning_target_strategy(db)
    if str(strategy.get("tuning_target") or "") == "decision_optimization":
        raise ModelTuningConflict(
            "The selected MILP Decision Strategy is research-only and is not a target for the current Model Tuning engine."
        )
    if (
        str(strategy.get("strategy_kind") or "") == "temporal_intelligence"
        and str(strategy.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
    ):
        return []
    if str(strategy.get("strategy_kind") or "") == "temporal_intelligence":
        return [temporal_policy_baseline(strategy)]
    if str(model_snapshot.get("family") or "") != TUNING_MODEL_FAMILY:
        return []

    preferred_job_id = str(strategy.get("candidate_backtest_id") or strategy.get("last_backtest_id") or "")
    query = {
        "status": "completed",
        "internal_job": {"$ne": True},
        "strategy_profile_id": str(strategy["id"]),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "research_model_family": TUNING_MODEL_FAMILY,
        "research_model_settings_hash": model_snapshot.get("settings_hash"),
    }
    if str(strategy.get("status") or "") in {"candidate", "promoted_candidate"} and preferred_job_id:
        query["id"] = preferred_job_id
    jobs = list(
        db[JOBS_COLLECTION]
        .find(
            query,
            {
                "_id": 0,
                "id": 1,
                "created_at": 1,
                "started_at": 1,
                "finished_at": 1,
                "strategy_profile_id": 1,
                "strategy_profile_name": 1,
                "strategy_profile_revision": 1,
                "strategy_configuration_hash": 1,
                "research_model_family": 1,
                "research_model_label": 1,
                "research_model_settings_hash": 1,
                "research_model_settings_revision": 1,
            },
        )
        .sort("finished_at", -1)
        .limit(max(1, min(int(limit), 100)))
    )
    result: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("id") or "")
        metrics = _find_portfolio_metrics(db, job_id)
        if not job_id or metrics is None:
            continue
        result.append(
            bson_value(
                {
                    "job_id": job_id,
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "strategy_profile_id": job.get("strategy_profile_id"),
                    "strategy_profile_name": job.get("strategy_profile_name"),
                    "strategy_profile_revision": job.get("strategy_profile_revision"),
                    "strategy_configuration_hash": job.get("strategy_configuration_hash"),
                    "model_family": job.get("research_model_family"),
                    "model_label": job.get("research_model_label") or "LightGBM Utility",
                    "model_settings_hash": job.get("research_model_settings_hash"),
                    "model_settings_revision": job.get("research_model_settings_revision"),
                    "metrics": _metric_summary(metrics),
                    "tuning_target_source": target_source,
                    "certified_candidate_baseline": bool(preferred_job_id and job_id == preferred_job_id),
                }
            )
        )
    if preferred_job_id:
        preferred = [item for item in result if item.get("job_id") == preferred_job_id]
        others = [item for item in result if item.get("job_id") != preferred_job_id]
        others.sort(key=lambda item: str(item.get("finished_at") or item.get("created_at") or ""), reverse=True)
        result = preferred + others
    else:
        result.sort(key=lambda item: str(item.get("finished_at") or item.get("created_at") or ""), reverse=True)
    return result


def _baseline_by_job_id(db: Any, job_id: str) -> dict[str, Any]:
    normalized = str(job_id or "").strip()
    if not normalized:
        raise ModelTuningConflict("Select a completed compatible baseline Backtest before starting model tuning.")
    baseline = next(
        (item for item in list_model_tuning_baselines(db, limit=100) if str(item.get("job_id")) == normalized),
        None,
    )
    if baseline is None:
        raise ModelTuningConflict(
            "The selected baseline Backtest is not compatible with the current Strategy and saved LightGBM model snapshot."
        )
    return baseline


def _reused_baseline_control_candidate(
    db: Any,
    *,
    baseline: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Materialize candidate #0 from the already-certified Candidate Backtest.

    Control is a reference observation, not a new experiment. Reusing the exact
    certified Backtest saves one complete execution while preserving the frozen
    Strategy/model/market-data validation context used by every challenger.
    """
    job_id = str(baseline.get("job_id") or "").strip()
    metrics = deepcopy(baseline.get("metrics") or {})
    if not job_id or not metrics:
        raise ModelTuningConflict("The certified Candidate Backtest cannot be reused as tuning Control.")
    return {
        "candidate_id": 0,
        "kind": "control",
        "is_control": True,
        "settings": deepcopy(settings),
        "settings_hash": _settings_hash(settings),
        "status": "completed",
        "rank": None,
        "metrics": metrics,
        "equity_preview": _candidate_equity_preview(db, job_id),
        "job_id": job_id,
        "source_job_id": job_id,
        "baseline_reused": True,
        "champion_gate_passed": None,
        "champion_gate": None,
        "strategy_configuration_hash": baseline.get("strategy_configuration_hash"),
        "started_at": baseline.get("started_at"),
        "finished_at": baseline.get("finished_at"),
        "raw_results_retained": True,
    }



def _execution_request_context_hash(request_payload: dict[str, Any]) -> str:
    
    context = deepcopy(request_payload)
    context.pop("research_model_settings", None)
    context.pop("expected_market_data_signature_sha256", None)
    context.pop("research_market_data_snapshot_id", None)
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_execution_context_from_job(db: Any, job_id: str) -> dict[str, Any]:
    job = db[JOBS_COLLECTION].find_one(
        {"id": str(job_id)},
        {
            "_id": 0,
            "id": 1,
            "request": 1,
            "strategy_profile_id": 1,
            "strategy_profile_name": 1,
            "strategy_profile_revision": 1,
            "strategy_configuration_hash": 1,
            "research_model_family": 1,
            "research_model_label": 1,
        },
    )
    if not job or not isinstance(job.get("request"), dict):
        raise ModelTuningConflict("The reference Backtest execution snapshot is no longer available.")
    metrics = _find_portfolio_metrics(db, str(job_id))
    if metrics is None:
        raise ModelTuningConflict("The reference Backtest metrics are no longer available.")
    summary = _metric_summary(metrics)
    signatures = metrics.get("market_data_signatures") if isinstance(metrics.get("market_data_signatures"), dict) else {}
    stable_market_data_signature = (
        market_data_research_signature_from_manifests(signatures)
        if signatures
        else summary.get("market_data_signature_sha256")
    )
    request_snapshot = deepcopy(job["request"])
    last_timestamp = summary.get("market_data_last_timestamp")
    cutoff_date = str(last_timestamp)[:10] if last_timestamp else None
    if cutoff_date:
        
        
        request_snapshot["end_date"] = cutoff_date
        request_snapshot["analysis_end_date"] = cutoff_date
    
    
    
    request_snapshot["research_market_data_mode"] = "database_only"
    return {
        "job_id": str(job_id),
        "request": bson_value(request_snapshot),
        "context_hash": _execution_request_context_hash(request_snapshot),
        "market_data_signature_sha256": stable_market_data_signature,
        "market_data_cutoff_date": cutoff_date,
        "strategy_profile_id": job.get("strategy_profile_id"),
        "strategy_profile_name": job.get("strategy_profile_name"),
        "strategy_profile_revision": job.get("strategy_profile_revision"),
        "strategy_configuration_hash": job.get("strategy_configuration_hash"),
        "model_family": job.get("research_model_family"),
        "model_label": job.get("research_model_label") or "LightGBM Utility",
    }


def _frozen_execution_context_from_campaign(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    





    request_snapshot = document.get("execution_request_snapshot")
    baseline = document.get("baseline_execution") if isinstance(document.get("baseline_execution"), dict) else {}
    baseline_job_id = str(baseline.get("job_id") or "")

    if not isinstance(request_snapshot, dict):
        if not baseline_job_id:
            raise ModelTuningConflict("The source campaign no longer contains a frozen execution snapshot.")
        return _frozen_execution_context_from_job(db, baseline_job_id)

    request_snapshot = deepcopy(request_snapshot)
    request_snapshot["research_market_data_mode"] = "database_only"
    cutoff_date = str(document.get("market_data_cutoff_date") or "").strip() or None
    if cutoff_date:
        request_snapshot["end_date"] = cutoff_date
        request_snapshot["analysis_end_date"] = cutoff_date

    signature = str(document.get("expected_market_data_signature_sha256") or "").strip()
    if int(document.get("schema_version") or 0) < 3 or not signature:
        if baseline_job_id:
            legacy_context = _frozen_execution_context_from_job(db, baseline_job_id)
            signature = str(legacy_context.get("market_data_signature_sha256") or "").strip()
            cutoff_date = cutoff_date or legacy_context.get("market_data_cutoff_date")
        else:
            completed_signatures = {
                str((item.get("metrics") or {}).get("market_data_signature_sha256") or "").strip()
                for item in document.get("candidates") or []
                if item.get("status") == "completed" and isinstance(item.get("metrics"), dict)
            }
            completed_signatures.discard("")
            if len(completed_signatures) == 1:
                signature = completed_signatures.pop()

    if not signature:
        raise ModelTuningConflict("The source campaign market-data signature cannot be reconstructed safely.")

    return {
        "job_id": baseline_job_id or None,
        "request": bson_value(request_snapshot),
        "context_hash": str(document.get("execution_context_hash") or "") or _execution_request_context_hash(request_snapshot),
        "market_data_signature_sha256": signature,
        "market_data_snapshot_id": str(document.get("market_data_snapshot_id") or request_snapshot.get("research_market_data_snapshot_id") or "").strip().lower() or None,
        "market_data_cutoff_date": cutoff_date,
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": document.get("strategy_profile_revision"),
        "strategy_configuration_hash": document.get("strategy_configuration_hash"),
        "model_family": document.get("model_family") or TUNING_MODEL_FAMILY,
        "model_label": document.get("model_label") or "LightGBM Utility",
    }


def _is_pristine_completed_latin_campaign(document: dict[str, Any]) -> bool:
    





    if str(document.get("status") or "") != "completed":
        return False
    if str(document.get("method") or "") != TUNING_METHOD:
        return False
    if int(document.get("failed_candidates") or 0) != 0:
        return False
    if int(document.get("cancelled_candidates") or 0) != 0:
        return False
    candidates = list(document.get("candidates") or [])
    total = int(document.get("total_candidates") or len(candidates))
    completed = int(document.get("completed_candidates") or 0)
    if total <= 0 or len(candidates) != total or completed != total:
        return False
    return all(str(item.get("status") or "") == "completed" for item in candidates)


def list_model_tuning_sources(db: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    try:
        current_strategy, current_model_snapshot, _ = _tuning_target_strategy(db)
        current_plan = _tuning_plan(current_strategy, current_model_snapshot)
    except Exception:
        current_plan = None
    expected_scope = str((current_plan or {}).get("scope") or MODEL_PARAMETER_TUNING_SCOPE)
    expected_space = list((current_plan or {}).get("search_space") or [dict(item) for item in _SEARCH_SPACE])

    documents = list(
        db[MODEL_TUNING_RUNS_COLLECTION]
        .find(
            {"status": "completed", "method": TUNING_METHOD, "failed_candidates": 0, "cancelled_candidates": 0},
            {
                "_id": 0,
                "id": 1,
                "finished_at": 1,
                "seed": 1,
                "strategy_profile_id": 1,
                "strategy_profile_name": 1,
                "strategy_profile_revision": 1,
                "strategy_configuration_hash": 1,
                "baseline_execution": 1,
                "best_candidate_id": 1,
                "candidates": 1,
                "search_space": 1,
                "schema_version": 1,
                "execution_request_snapshot": 1,
                "execution_context_hash": 1,
                "expected_market_data_signature_sha256": 1,
                "market_data_snapshot_id": 1,
                "market_data_cutoff_date": 1,
                "model_family": 1,
                "model_label": 1,
                "tuning_scope": 1,
                "tuning_scope_label": 1,
            },
        )
        .sort("finished_at", -1)
        .limit(max(1, min(int(limit), 100)))
    )
    result: list[dict[str, Any]] = []
    for document in documents:
        if not _is_pristine_completed_latin_campaign(document):
            continue
        observations = [
            item for item in document.get("candidates") or []
            if item.get("status") == "completed"
            and isinstance(item.get("settings"), dict)
            and isinstance(item.get("metrics"), dict)
        ]
        document_scope = str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE)
        if document_scope != expected_scope:
            continue
        if len(observations) < 4 or list(document.get("search_space") or []) != expected_space:
            continue
        best_id = document.get("best_candidate_id")
        best = next(
            (item for item in observations if int(item.get("candidate_id") or 0) == int(best_id)),
            None,
        ) if best_id is not None else None
        if best is None or not bool((best.get("metrics") or {}).get("eligible")):
            eligible = [item for item in observations if bool((item.get("metrics") or {}).get("eligible"))]
            eligible = _rank_candidates(eligible)
            eligible.sort(key=lambda item: int(item.get("rank") or 10**9))
            best = eligible[0] if eligible else None
        baseline = document.get("baseline_execution") if isinstance(document.get("baseline_execution"), dict) else {}
        baseline_job_id = str(baseline.get("job_id") or "")
        if not baseline_job_id:
            continue
        
        
        
        try:
            context = _frozen_execution_context_from_campaign(db, document)
        except ModelTuningConflict:
            continue
        result.append(
            bson_value({
                "run_id": document.get("id"),
                "tuning_scope": document_scope,
                "tuning_scope_label": document.get("tuning_scope_label"),
                "finished_at": document.get("finished_at"),
                "seed": document.get("seed"),
                "strategy_profile_id": document.get("strategy_profile_id"),
                "strategy_profile_name": document.get("strategy_profile_name"),
                "strategy_profile_revision": document.get("strategy_profile_revision"),
                "strategy_configuration_hash": document.get("strategy_configuration_hash"),
                "observation_count": len(observations),
                "baseline_job_id": baseline_job_id,
                "baseline_metrics": deepcopy(baseline.get("metrics") or {}),
                "best_candidate": {
                    "candidate_id": best.get("candidate_id"),
                    "settings_hash": best.get("settings_hash"),
                    "metrics": deepcopy(best.get("metrics") or {}),
                } if best else None,
                "eligible_candidates": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "rank": item.get("rank"),
                        "is_control": bool(item.get("is_control")),
                        "settings_hash": item.get("settings_hash"),
                        "metrics": deepcopy(item.get("metrics") or {}),
                    }
                    for item in sorted(
                        [row for row in observations if bool((row.get("metrics") or {}).get("eligible"))],
                        key=lambda row: (int(row.get("rank") or 10**9), int(row.get("candidate_id") or 0)),
                    )
                ],
                "execution_context_hash": context.get("context_hash"),
                "market_data_signature_sha256": context.get("market_data_signature_sha256"),
                "market_data_snapshot_id": context.get("market_data_snapshot_id"),
                "market_data_cutoff_date": context.get("market_data_cutoff_date"),
            })
        )
    return result


def _source_campaign(
    db: Any,
    run_id: str,
    *,
    expected_scope: str,
    expected_search_space: list[dict[str, Any]],
) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise ModelTuningNotFound("Source tuning campaign not found.")
    if not _is_pristine_completed_latin_campaign(document):
        raise ModelTuningConflict(
            "CARO can import observations only from a fully completed Latin Hypercube campaign "
            "with zero failed or cancelled candidates."
        )
    source_scope = str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE)
    if source_scope != str(expected_scope):
        raise ModelTuningConflict("The source tuning campaign targets a different parameter scope.")
    if list(document.get("search_space") or []) != list(expected_search_space):
        raise ModelTuningConflict("The source tuning campaign uses a different parameter search space.")
    return document


def _source_observations(document: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in document.get("candidates") or []:
        if item.get("status") != "completed" or not isinstance(item.get("settings"), dict) or not isinstance(item.get("metrics"), dict):
            continue
        observations.append({
            "candidate_id": int(item.get("candidate_id") or 0),
            "source_candidate_id": int(item.get("candidate_id") or 0),
            "kind": "prior_latin_hypercube_observation",
            "is_control": bool(item.get("is_control")),
            "settings": deepcopy(item.get("settings") or {}),
            "settings_hash": str(item.get("settings_hash") or ""),
            "status": "completed",
            "rank": item.get("rank"),
            "metrics": deepcopy(item.get("metrics") or {}),
        })
    if len(observations) < 4:
        raise ModelTuningConflict("The source campaign does not contain enough completed observations for CARO.")
    return observations


def _continuation_observations(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact, de-duplicated observations for Continue Research.

    Candidate ids are made negative so the next campaign can number its newly
    executed candidates from 1 while preserving the original ids in metadata.
    """
    merged = list(document.get("prior_observations") or []) + list(document.get("candidates") or [])
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in merged:
        if bool(item.get("is_control")):
            continue
        if item.get("status") != "completed" or not isinstance(item.get("settings"), dict) or not isinstance(item.get("metrics"), dict):
            continue
        fingerprint = str(item.get("settings_hash") or "") or _settings_hash(dict(item.get("settings") or {}))
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        source_candidate_id = int(item.get("source_candidate_id") if item.get("source_candidate_id") is not None else item.get("candidate_id") or 0)
        result.append({
            "candidate_id": -(len(result) + 1),
            "source_candidate_id": source_candidate_id,
            "source_tuning_run_id": str(item.get("source_tuning_run_id") or document.get("id") or ""),
            "source": "continue_research",
            "kind": str(item.get("kind") or "historical_observation"),
            "is_control": bool(item.get("is_control")),
            "settings": deepcopy(item.get("settings") or {}),
            "settings_hash": fingerprint,
            "status": "completed",
            "rank": item.get("rank"),
            "metrics": deepcopy(item.get("metrics") or {}),
            "champion_gate_passed": item.get("champion_gate_passed"),
            "champion_gate": deepcopy(item.get("champion_gate") or None),
        })
    if len(result) < 4:
        raise ModelTuningConflict("Continue Research requires at least four completed observations in the source campaign.")
    return result




def _retained_historical_candidate_ids(document: dict[str, Any]) -> set[int]:
    """Candidates explicitly promoted into a durable research/lifecycle state.

    Trial observations are useful while a campaign is active and when Continue
    Research copies a source campaign. They are not permanent business records.
    Once a later campaign starts, only candidates that were explicitly adopted
    or validated are retained in the historical tuning document.
    """
    retained: set[int] = set()
    for field in ("adopted_candidate_id", "validated_candidate_id"):
        value = document.get(field)
        if value is not None:
            retained.add(int(value))
    for item in document.get("adoption_history") or []:
        if isinstance(item, dict) and item.get("candidate_id") is not None:
            retained.add(int(item["candidate_id"]))
    for value in document.get("retained_candidate_ids") or []:
        if value is not None:
            retained.add(int(value))
    return retained


def _compact_historical_tuning_document(document: dict[str, Any], *, next_run_id: str) -> dict[str, Any]:
    retained_ids = _retained_historical_candidate_ids(document)
    candidates = list(document.get("candidates") or [])
    prior = list(document.get("prior_observations") or [])
    retained_candidates = [
        deepcopy(item)
        for item in candidates
        if item.get("candidate_id") is not None and int(item.get("candidate_id")) in retained_ids
    ]
    retained_ids_present = {
        int(item.get("candidate_id")) for item in retained_candidates if item.get("candidate_id") is not None
    }
    champion_history = [
        deepcopy(item)
        for item in document.get("probability_champion_history") or []
        if item.get("candidate_id") is not None and int(item.get("candidate_id")) in retained_ids_present
    ]
    event_log = [
        deepcopy(item)
        for item in document.get("event_log") or []
        if item.get("candidate_id") is None or int(item.get("candidate_id")) in retained_ids_present
    ]
    successful_discarded = sum(
        1
        for item in candidates
        if item.get("status") == "completed"
        and not bool(item.get("is_control"))
        and (item.get("candidate_id") is None or int(item.get("candidate_id")) not in retained_ids_present)
    )
    failed_discarded = sum(1 for item in candidates if item.get("status") == "failed")
    cancelled_discarded = sum(1 for item in candidates if item.get("status") == "cancelled")
    now = utc_now()
    return {
        "candidates": bson_value(retained_candidates),
        "prior_observations": [],
        "probability_champion_history": bson_value(champion_history),
        "event_log": bson_value(event_log[-_TUNING_LOG_MAX_EVENTS:]),
        "historical_trials_compacted": True,
        "historical_trials_compacted_at": now,
        "historical_trials_compacted_for_run_id": str(next_run_id),
        "retained_candidate_ids": sorted(retained_ids_present),
        "retained_candidate_count": len(retained_candidates),
        "discarded_successful_trial_count": int(document.get("discarded_successful_trial_count") or 0) + successful_discarded + len(prior),
        "discarded_failed_trial_count": int(document.get("discarded_failed_trial_count") or 0) + failed_discarded,
        "discarded_cancelled_trial_count": int(document.get("discarded_cancelled_trial_count") or 0) + cancelled_discarded,
        "best_candidate_id": (
            int(document.get("best_candidate_id"))
            if document.get("best_candidate_id") is not None and int(document.get("best_candidate_id")) in retained_ids_present
            else None
        ),
        "best_exploratory_candidate_id": (
            int(document.get("best_exploratory_candidate_id"))
            if document.get("best_exploratory_candidate_id") is not None and int(document.get("best_exploratory_candidate_id")) in retained_ids_present
            else None
        ),
        "best_champion_beating_candidate_id": (
            int(document.get("best_champion_beating_candidate_id"))
            if document.get("best_champion_beating_candidate_id") is not None and int(document.get("best_champion_beating_candidate_id")) in retained_ids_present
            else None
        ),
        "updated_at": now,
    }


def _compact_historical_tuning_runs(db: Any, *, next_run_id: str) -> dict[str, int]:
    """Prune transient research storage when a new campaign is created.

    The active campaign keeps every observation required by CARO. Historical
    campaigns retain only candidates that entered a durable lifecycle state.
    Failed validation records and temporary fold-specific Temporal caches are
    also removed at this boundary so MongoDB growth follows useful research
    outcomes instead of attempted trials.
    """
    compacted_runs = 0
    retained_candidates = 0
    discarded_trials = 0
    discarded_validations = 0
    removed_fold_caches = 0
    cursor = db[MODEL_TUNING_RUNS_COLLECTION].find({
        "id": {"$ne": str(next_run_id)},
        "status": {"$nin": list(_ACTIVE_STATUSES)},
        "historical_trials_compacted": {"$ne": True},
    })
    for document in cursor:
        run_id = str(document.get("id") or "")
        durable_validation_ids: list[int] = []
        validation_rows = list(db[MODEL_TUNING_VALIDATIONS_COLLECTION].find(
            {"tuning_run_id": run_id},
            {"_id": 0, "candidate_id": 1, "validation_passed": 1, "certification_passed": 1},
        ))
        for validation in validation_rows:
            candidate_id = validation.get("candidate_id")
            if candidate_id is None:
                continue
            if bool(validation.get("validation_passed")) or bool(validation.get("certification_passed")):
                durable_validation_ids.append(int(candidate_id))
                continue
            db[MODEL_TUNING_VALIDATIONS_COLLECTION].delete_many({
                "tuning_run_id": run_id,
                "candidate_id": int(candidate_id),
            })
            discarded_validations += 1

        if durable_validation_ids:
            document = deepcopy(document)
            document["retained_candidate_ids"] = sorted(set(
                [int(value) for value in document.get("retained_candidate_ids") or [] if value is not None]
                + durable_validation_ids
            ))

        before_candidates = len(document.get("candidates") or [])
        before_prior = len(document.get("prior_observations") or [])
        update = _compact_historical_tuning_document(document, next_run_id=next_run_id)
        kept = len(update.get("candidates") or [])

        cache_run_id = str(document.get("research_fold_cache_run_id") or "").strip()
        if cache_run_id:
            db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].delete_many({"run_id": cache_run_id})
            db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].delete_many({"run_id": cache_run_id})
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].delete_many({"id": cache_run_id})
            update["research_fold_cache_run_id"] = None
            update["research_fold_cache_cleaned_at"] = utc_now()
            removed_fold_caches += 1

        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {"$set": update},
        )
        compacted_runs += 1
        retained_candidates += kept
        discarded_trials += max(0, before_candidates + before_prior - kept)
    return {
        "compacted_runs": compacted_runs,
        "retained_candidates": retained_candidates,
        "discarded_trials": discarded_trials,
        "discarded_validations": discarded_validations,
        "removed_fold_caches": removed_fold_caches,
    }


def _validate_temporal_continuation_source(
    source: dict[str, Any],
    *,
    strategy: dict[str, Any],
    tuning_scope: str,
    search_space: list[dict[str, Any]],
    source_temporal_run_id: str,
    market_data_snapshot_id: str | None,
    research_folds: int,
) -> None:
    if str(source.get("status") or "") != "completed":
        raise ModelTuningConflict("Continue Research requires a completed source campaign.")
    if str(source.get("method") or "") != PROBABILITY_METHOD:
        raise ModelTuningConflict("Continue Research is available for Unified Adaptive CARO campaigns.")
    if str(source.get("tuning_scope") or "") != str(tuning_scope):
        raise ModelTuningConflict("The source campaign targets a different Temporal tuning scope.")
    if str(source.get("strategy_profile_id") or "") != str(strategy.get("id") or ""):
        raise ModelTuningConflict("The source campaign belongs to a different TEMPORAL Strategy.")
    if int(source.get("strategy_profile_revision") or 0) != int(strategy.get("revision") or 0):
        raise ModelTuningConflict("The source campaign belongs to a different TEMPORAL Strategy revision.")
    if str(source.get("strategy_configuration_hash") or "") != str(strategy.get("configuration_hash") or ""):
        raise ModelTuningConflict("The source campaign Strategy configuration hash does not match the selected TEMPORAL Strategy.")
    if str(source.get("source_temporal_run_id") or "") != str(source_temporal_run_id):
        raise ModelTuningConflict("The source campaign uses a different frozen Temporal Intelligence replay.")
    if list(source.get("search_space") or []) != list(search_space):
        raise ModelTuningConflict("The source campaign uses a different parameter search space.")
    source_snapshot = str(source.get("market_data_snapshot_id") or source.get("expected_market_data_signature_sha256") or "").strip().lower()
    expected_snapshot = str(market_data_snapshot_id or "").strip().lower()
    if source_snapshot and expected_snapshot and source_snapshot != expected_snapshot:
        raise ModelTuningConflict("The source campaign uses a different frozen market-data snapshot.")
    source_protocol = dict(source.get("fold_protocol") or {})
    source_research_folds = int(source_protocol.get("research_folds") or DEFAULT_RESEARCH_FOLDS)
    if source_research_folds != int(research_folds):
        raise ModelTuningConflict(
            "Continue Research requires the same research fold count as the source campaign. "
            "Start a new campaign to change the research fold protocol."
        )


def _meaningful_no_improvement_streak(document: dict[str, Any]) -> int:
    config = dict(document.get("probability_config") or {})
    threshold = max(0.0, float(config.get("minimum_meaningful_improvement") or DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT))
    starting_anchor = document.get("starting_probability_anchor") if isinstance(document.get("starting_probability_anchor"), dict) else document.get("probability_anchor")
    anchor_metrics = (starting_anchor or {}).get("metrics") if isinstance(starting_anchor, dict) else None
    best = float((anchor_metrics or {}).get("ending_capital") or 0.0)
    if best <= 0:
        control = next((item for item in document.get("candidates") or [] if bool(item.get("is_control")) and isinstance(item.get("metrics"), dict)), None)
        best = float(((control or {}).get("metrics") or {}).get("ending_capital") or 0.0)
    streak = 0
    completed = [
        item for item in document.get("candidates") or []
        if not bool(item.get("is_control")) and item.get("status") == "completed" and isinstance(item.get("metrics"), dict)
    ]
    completed.sort(key=lambda item: int(item.get("candidate_id") or 0))
    for item in completed:
        capital = float((item.get("metrics") or {}).get("ending_capital") or 0.0)
        if capital > 0 and (best <= 0 or capital >= best * (1.0 + threshold)):
            best = max(best, capital)
            streak = 0
        else:
            streak += 1
    return streak


def _adaptive_early_stop_reason(document: dict[str, Any]) -> str | None:
    if str(document.get("method") or "") != PROBABILITY_METHOD:
        return None
    config = dict(document.get("probability_config") or {})
    if not bool(config.get("adaptive_stopping_enabled", True)):
        return None
    limit = max(10, int(config.get("no_improvement_trial_limit") or DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT))
    completed_trials = sum(1 for item in document.get("candidates") or [] if not bool(item.get("is_control")) and item.get("status") == "completed")
    minimum_exploration = max(4, int(config.get("minimum_exploration_trials") or config.get("startup_trials") or 4))
    if completed_trials < minimum_exploration:
        return None
    streak = _meaningful_no_improvement_streak(document)
    if streak < limit:
        return None
    threshold = max(0.0, float(config.get("minimum_meaningful_improvement") or DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT))
    return f"Adaptive early stopping: {streak} consecutive trials without at least {threshold * 100:.3f}% meaningful capital improvement."


def _automatic_compatible_prior_observations(
    db: Any,
    *,
    strategy: dict[str, Any],
    tuning_scope: str,
    strategy_mode: str | None,
    search_space: list[dict[str, Any]],
    execution_context: dict[str, Any],
    base_values: dict[str, Any],
    limit: int = 160,
) -> list[dict[str, Any]]:
    """Reuse only scientifically identical historical tuning observations.

    This automatic path is deliberately stricter than the manual source-campaign
    workflow: same Strategy id/configuration, tuning scope/search space, frozen
    execution context and market-data signature. It can therefore reuse a prior
    Unified CARO/LHS campaign without silently mixing different research surfaces.
    """
    strategy_id = str(strategy.get("id") or "")
    configuration_hash = str(strategy.get("configuration_hash") or "")
    context_hash = str(execution_context.get("context_hash") or "")
    market_signature = str(execution_context.get("market_data_signature_sha256") or "").strip().lower()
    if not strategy_id or not configuration_hash or not context_hash or not market_signature:
        return []

    projection = {
        "_id": 0, "id": 1, "status": 1, "method": 1, "tuning_scope": 1,
        "strategy_mode": 1, "strategy_profile_id": 1, "strategy_configuration_hash": 1,
        "execution_context_hash": 1, "expected_market_data_signature_sha256": 1,
        "search_space": 1, "candidates": 1, "finished_at": 1,
    }
    cursor = (
        db[MODEL_TUNING_RUNS_COLLECTION]
        .find(
            {
                "status": "completed",
                "strategy_profile_id": strategy_id,
                "strategy_configuration_hash": configuration_hash,
                "tuning_scope": str(tuning_scope),
                "strategy_mode": str(strategy_mode or ""),
                "execution_context_hash": context_hash,
            },
            projection,
        )
        .sort("finished_at", -1)
        .limit(20)
    )

    seen = {_settings_hash(_normalize_tuning_values(base_values, search_space))}
    imported: list[dict[str, Any]] = []
    for campaign in cursor:
        campaign_signature = str(campaign.get("expected_market_data_signature_sha256") or "").strip().lower()
        if campaign_signature != market_signature:
            continue
        if list(campaign.get("search_space") or []) != list(search_space):
            continue
        campaign_id = str(campaign.get("id") or "")
        for item in campaign.get("candidates") or []:
            if len(imported) >= max(0, int(limit)):
                break
            if bool(item.get("is_control")) or item.get("status") != "completed":
                continue
            if not isinstance(item.get("settings"), dict) or not isinstance(item.get("metrics"), dict):
                continue
            fingerprint = str(item.get("settings_hash") or "") or _settings_hash(dict(item["settings"]))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            imported.append({
                "candidate_id": -len(imported) - 1,
                "source_candidate_id": int(item.get("candidate_id") or 0),
                "source_tuning_run_id": campaign_id,
                "source": "automatic_compatible_prior_campaign",
                "kind": str(item.get("kind") or "historical_observation"),
                "is_control": False,
                "settings": deepcopy(item.get("settings") or {}),
                "settings_hash": fingerprint,
                "status": "completed",
                "metrics": deepcopy(item.get("metrics") or {}),
                "champion_gate_passed": item.get("champion_gate_passed"),
                "champion_gate": deepcopy(item.get("champion_gate") or None),
            })
        if len(imported) >= max(0, int(limit)):
            break
    return imported


def _source_anchor(document: dict[str, Any], anchor_candidate_id: int | None) -> dict[str, Any]:
    probability_anchor = document.get("probability_anchor") if isinstance(document.get("probability_anchor"), dict) else None
    if anchor_candidate_id is None and str(document.get("method") or "") == PROBABILITY_METHOD and probability_anchor:
        if isinstance(probability_anchor.get("settings"), dict) and isinstance(probability_anchor.get("metrics"), dict):
            return {
                "candidate_id": int(probability_anchor.get("candidate_id") or 0),
                "settings_hash": str(probability_anchor.get("settings_hash") or ""),
                "settings": deepcopy(probability_anchor.get("settings") or {}),
                "metrics": deepcopy(probability_anchor.get("metrics") or {}),
                "status": "completed",
                "is_control": False,
                "source_tuning_run_id": probability_anchor.get("source_tuning_run_id"),
            }

    best_candidate_id = document.get("best_candidate_id")
    candidate_id = int(anchor_candidate_id) if anchor_candidate_id is not None else (int(best_candidate_id) if best_candidate_id is not None else -1)

    if probability_anchor and int(probability_anchor.get("candidate_id") or 0) == candidate_id:
        if isinstance(probability_anchor.get("settings"), dict) and isinstance(probability_anchor.get("metrics"), dict):
            return {
                "candidate_id": candidate_id,
                "settings_hash": str(probability_anchor.get("settings_hash") or ""),
                "settings": deepcopy(probability_anchor.get("settings") or {}),
                "metrics": deepcopy(probability_anchor.get("metrics") or {}),
                "status": "completed",
                "is_control": False,
                "source_tuning_run_id": probability_anchor.get("source_tuning_run_id"),
            }

    candidate = next(
        (item for item in document.get("candidates") or [] if int(item.get("candidate_id") or 0) == candidate_id),
        None,
    )
    if candidate is None:
        candidate = next(
            (
                item for item in document.get("prior_observations") or []
                if (
                    (item.get("candidate_id") is not None and int(item.get("candidate_id")) == candidate_id)
                    or (item.get("source_candidate_id") is not None and int(item.get("source_candidate_id")) == candidate_id)
                )
            ),
            None,
        )
    if candidate is None or candidate.get("status") != "completed" or not isinstance(candidate.get("metrics"), dict):
        raise ModelTuningConflict("Select a completed source candidate as the CARO Champion anchor.")
    if not bool(candidate["metrics"].get("eligible")):
        raise ModelTuningConflict("The CARO Champion anchor must pass the positive-fold robustness gate.")
    return candidate


def _candidate_export_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tuned_parameters = list(document.get("tuned_parameters") or _TUNED_NAMES)
    all_setting_names = list(tuned_parameters)
    ranked_candidates = _rank_candidates(list(document.get("candidates") or []))
    for candidate in ranked_candidates:
        settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else {}
        for name in settings:
            if name not in all_setting_names:
                all_setting_names.append(name)
    for candidate in ranked_candidates:
        settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else {}
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        row: dict[str, Any] = {
            "rank": candidate.get("rank"),
            "candidate_id": candidate.get("candidate_id"),
            "kind": candidate.get("kind"),
            "is_control": bool(candidate.get("is_control")),
            "status": candidate.get("status"),
            "eligible": metrics.get("eligible"),
            "champion_gate_passed": candidate.get("champion_gate_passed"),
            "champion_gate": deepcopy(candidate.get("champion_gate") or None),
            "settings_hash": candidate.get("settings_hash"),
            "strategy_configuration_hash": candidate.get("strategy_configuration_hash"),
            "job_id": candidate.get("job_id"),
            "worker_id": candidate.get("worker_id"),
            "worker_cpu_count": candidate.get("worker_cpu_count"),
            "worker_concurrency": candidate.get("worker_concurrency"),
            "runtime_thread_limit": candidate.get("runtime_thread_limit"),
            "retry_count": candidate.get("retry_count"),
            "started_at": candidate.get("started_at"),
            "finished_at": candidate.get("finished_at"),
            "initial_capital": metrics.get("initial_capital"),
            "ending_capital": metrics.get("ending_capital"),
            "strategy_return": metrics.get("strategy_return"),
            "cagr": metrics.get("cagr"),
            "sharpe": metrics.get("sharpe"),
            "maximum_drawdown": metrics.get("maximum_drawdown"),
            "risk_adjusted_compound_score": metrics.get("risk_adjusted_compound_score"),
            "turnover_ratio": metrics.get("turnover_ratio"),
            "capital_rotations": metrics.get("capital_rotations"),
            "average_holding_days": metrics.get("average_holding_days"),
            "market_exposure": metrics.get("market_exposure"),
            "cash_days": metrics.get("cash_days"),
            "absolute_utility_entry_threshold": metrics.get("absolute_utility_entry_threshold"),
            "absolute_utility_exit_threshold": metrics.get("absolute_utility_exit_threshold"),
            "absolute_utility_gate_acceptance_rate": metrics.get("absolute_utility_gate_acceptance_rate"),
            "cash_gate_changed_base_action_sessions": metrics.get("cash_gate_changed_base_action_sessions"),
            "cash_gate_net_avoided_return_sum": metrics.get("cash_gate_net_avoided_return_sum"),
            "benchmark_ending_capital": metrics.get("benchmark_ending_capital"),
            "worst_fold_return": metrics.get("worst_fold_return"),
            "error": candidate.get("error"),
        }
        proposal = candidate.get("proposal") if isinstance(candidate.get("proposal"), dict) else {}
        row.update({
            "probability_model": proposal.get("probability_model"),
            "proposal_observation_count": proposal.get("observation_count"),
            "estimated_probability_beats_champion": proposal.get("estimated_probability_beats_champion"),
            "estimated_expected_improvement": proposal.get("estimated_expected_improvement"),
            "estimated_ending_capital_mean": proposal.get("estimated_ending_capital_mean"),
            "estimated_ending_capital_std": proposal.get("estimated_ending_capital_std"),
            "acquisition_score": proposal.get("acquisition_score"),
            "promising_region_probability_mean": proposal.get("promising_region_probability_mean"),
            "promising_region_expected_improvement_mean": proposal.get("promising_region_expected_improvement_mean"),
            "promising_region": json.dumps(proposal.get("promising_region") or {}, sort_keys=True, separators=(",", ":")),
        })
        for name in all_setting_names:
            row[name] = settings.get(name)
        for fold in metrics.get("folds") or []:
            if not isinstance(fold, dict):
                continue
            fold_id = int(fold.get("fold_id") or 0)
            if fold_id <= 0:
                continue
            row[f"fold_{fold_id}_return"] = fold.get("strategy_return")
            row[f"fold_{fold_id}_maximum_drawdown"] = fold.get("maximum_drawdown")
            row[f"fold_{fold_id}_benchmark_return"] = fold.get("benchmark_return")
        rows.append(row)
    return rows


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


def build_model_tuning_export(db: Any, run_id: str) -> bytes:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 0})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")

    ranked_candidate_docs = _rank_candidates(list(document.get("candidates") or []))
    export_document = deepcopy(document)
    export_document["candidates"] = ranked_candidate_docs
    candidate_rows = _candidate_export_rows(export_document)
    prior_rows = _candidate_export_rows({
        "tuned_parameters": document.get("tuned_parameters") or list(_TUNED_NAMES),
        "candidates": document.get("prior_observations") or [],
    })
    ranked = [item for item in candidate_rows if item.get("rank") is not None]
    ranked.sort(key=lambda item: int(item["rank"]))
    best_row = ranked[0] if ranked else None
    best_exploratory_row = next((item for item in ranked if not bool(item.get("is_control"))), None)
    best_champion_row = next((item for item in ranked if item.get("champion_gate_passed") is True), None)
    baseline = deepcopy(document.get("baseline_execution") or {})
    summary_row = {
        "run_id": document.get("id"),
        "status": document.get("status"),
        "method": document.get("method"),
        "tuning_scope": document.get("tuning_scope"),
        "tuning_scope_label": document.get("tuning_scope_label"),
        "strategy_mode": document.get("strategy_mode"),
        "execution_mode": document.get("execution_mode"),
        "generated_candidates": document.get("generated_candidates"),
        "model_family": document.get("model_family"),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": document.get("strategy_profile_revision"),
        "candidate_count": document.get("candidate_count"),
        "total_candidates": document.get("total_candidates"),
        "completed_candidates": document.get("completed_candidates"),
        "failed_candidates": document.get("failed_candidates"),
        "seed": document.get("seed"),
        "baseline_job_id": baseline.get("job_id"),
        "baseline_ending_capital": (baseline.get("metrics") or {}).get("ending_capital") if isinstance(baseline.get("metrics"), dict) else None,
        "source_tuning_run_id": document.get("source_tuning_run_id"),
        "imported_observation_count": document.get("imported_observation_count"),
        "anchor_candidate_id": (document.get("probability_anchor") or {}).get("candidate_id") if isinstance(document.get("probability_anchor"), dict) else None,
        "anchor_ending_capital": ((document.get("probability_anchor") or {}).get("metrics") or {}).get("ending_capital") if isinstance((document.get("probability_anchor") or {}).get("metrics"), dict) else None,
        "market_data_cutoff_date": document.get("market_data_cutoff_date"),
        "research_snapshot_cutoff": document.get("market_data_cutoff_date"),
        "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "control_candidate_id": document.get("control_candidate_id"),
        "best_candidate_id": best_row.get("candidate_id") if best_row else None,
        "best_exploratory_candidate_id": best_exploratory_row.get("candidate_id") if best_exploratory_row else None,
        "best_champion_beating_candidate_id": best_champion_row.get("candidate_id") if best_champion_row else None,
        "best_ending_capital": ranked[0].get("ending_capital") if ranked else None,
        "best_sharpe": ranked[0].get("sharpe") if ranked else None,
        "best_maximum_drawdown": ranked[0].get("maximum_drawdown") if ranked else None,
        "created_at": document.get("created_at"),
        "created_by": document.get("created_by"),
        "explicit_start_confirmation": bool(document.get("explicit_start_confirmation")),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
    }
    manifest = bson_value(
        {
            "schema_version": document.get("schema_version"),
            "run_id": document.get("id"),
            "status": document.get("status"),
            "phase": document.get("phase"),
            "method": document.get("method"),
            "tuning_scope": document.get("tuning_scope"),
            "tuning_scope_label": document.get("tuning_scope_label"),
            "tuning_scope_description": document.get("tuning_scope_description"),
            "strategy_mode": document.get("strategy_mode"),
            "execution_mode": document.get("execution_mode"),
            "generated_candidates": document.get("generated_candidates"),
            "probability_config": deepcopy(document.get("probability_config") or {}),
            "validation": "chronological_walk_forward",
            "selection_metric": "champion_gate_then_realized_economic_quality",
            "eligibility_gate": "all_walk_forward_folds_positive",
            "model_family": document.get("model_family"),
            "model_label": document.get("model_label"),
            "candidate_count": document.get("candidate_count"),
            "total_candidates": document.get("total_candidates"),
            "completed_candidates": document.get("completed_candidates"),
            "failed_candidates": document.get("failed_candidates"),
            "seed": document.get("seed"),
            "search_space": deepcopy(document.get("search_space") or []),
            "tuned_parameters": deepcopy(document.get("tuned_parameters") or []),
            "tuned_model_parameters": deepcopy(document.get("tuned_model_parameters") or []),
            "tuned_strategy_parameters": deepcopy(document.get("tuned_strategy_parameters") or []),
            "base_tuning_values": deepcopy(document.get("base_tuning_values") or {}),
            "frozen_model_values": deepcopy(document.get("frozen_model_values") or {}),
            "fixed_model_values": deepcopy(document.get("fixed_model_values") or {}),
            "strategy_profile_id": document.get("strategy_profile_id"),
            "strategy_profile_name": document.get("strategy_profile_name"),
            "strategy_profile_revision": document.get("strategy_profile_revision"),
            "strategy_configuration_hash": document.get("strategy_configuration_hash"),
            "strategy_configuration_snapshot": deepcopy(document.get("strategy_configuration_snapshot") or {}),
            "base_model_settings_hash": document.get("base_model_settings_hash"),
            "base_model_settings_revision": document.get("base_model_settings_revision"),
            "baseline_execution": baseline,
            "source_tuning_run_id": document.get("source_tuning_run_id"),
            "source_strategy_profile_id": document.get("source_strategy_profile_id"),
            "source_strategy_profile_revision": document.get("source_strategy_profile_revision"),
            "prior_observations": deepcopy(document.get("prior_observations") or []),
            "imported_observation_count": document.get("imported_observation_count"),
            "probability_anchor": deepcopy(document.get("probability_anchor") or None),
            "probability_state": deepcopy(document.get("probability_state") or None),
            "probability_champion_history": deepcopy(document.get("probability_champion_history") or []),
            "execution_context_hash": document.get("execution_context_hash"),
            "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
            "market_data_snapshot_id": document.get("market_data_snapshot_id"),
            "market_data_cutoff_date": document.get("market_data_cutoff_date"),
            "research_snapshot_cutoff": document.get("market_data_cutoff_date"),
            "adoption_context_compatible": document.get("adoption_context_compatible"),
            "control_candidate_id": document.get("control_candidate_id"),
            "best_candidate_id": best_row.get("candidate_id") if best_row else None,
            "best_exploratory_candidate_id": best_exploratory_row.get("candidate_id") if best_exploratory_row else None,
            "best_champion_beating_candidate_id": best_champion_row.get("candidate_id") if best_champion_row else None,
            "adopted_candidate_id": document.get("adopted_candidate_id"),
            "created_at": document.get("created_at"),
            "started_at": document.get("started_at"),
            "finished_at": document.get("finished_at"),
            "candidates": deepcopy(ranked_candidate_docs),
        }
    )

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model_tuning_summary.csv", _csv_text([bson_value(summary_row)]))
        archive.writestr("model_tuning_candidates.csv", _csv_text([bson_value(row) for row in candidate_rows]))
        if prior_rows:
            archive.writestr("model_tuning_prior_observations.csv", _csv_text([bson_value(row) for row in prior_rows]))
        archive.writestr("model_tuning_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return archive_buffer.getvalue()


def _history_candidate(document: dict[str, Any]) -> dict[str, Any] | None:
    candidates = _rank_candidates(list(document.get("candidates") or []))
    ranked = [item for item in candidates if item.get("rank") is not None]
    if not ranked:
        return None
    return min(ranked, key=lambda item: int(item.get("rank") or 10**9))


def _public_tuning_history_item(document: dict[str, Any]) -> dict[str, Any]:
    reranked = _rank_candidates(list(document.get("candidates") or []))
    best = _history_candidate({**document, "candidates": reranked})
    best_public = _public_candidate(best) if best is not None else None
    ranked = sorted((item for item in reranked if item.get("rank") is not None), key=lambda item: int(item["rank"]))
    best_champion = next((item for item in ranked if item.get("champion_gate_passed") is True), None)
    return bson_value({
        "id": document.get("id"),
        "status": str(document.get("status") or "unknown"),
        "phase": str(document.get("phase") or "unknown"),
        "method": str(document.get("method") or TUNING_METHOD),
        "tuning_scope": str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE),
        "tuning_scope_label": str(document.get("tuning_scope_label") or "LightGBM model parameters"),
        "model_family": str(document.get("model_family") or TUNING_MODEL_FAMILY),
        "model_label": str(document.get("model_label") or "LightGBM Utility"),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": int(document.get("strategy_profile_revision") or 0),
        "strategy_profile_status": document.get("strategy_profile_status"),
        "created_at": document.get("created_at"),
        "created_by": document.get("created_by"),
        "explicit_start_confirmation": bool(document.get("explicit_start_confirmation")),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
        "market_data_cutoff_date": document.get("market_data_cutoff_date"),
        "research_snapshot_cutoff": document.get("market_data_cutoff_date"),
        "candidate_count": int(document.get("candidate_count") or 0),
        "total_candidates": int(document.get("total_candidates") or len(document.get("candidates") or [])),
        "completed_candidates": int(document.get("completed_candidates") or 0),
        "failed_candidates": int(document.get("failed_candidates") or 0),
        "cancelled_candidates": int(document.get("cancelled_candidates") or 0),
        "best_candidate_id": int(best["candidate_id"]) if best is not None else None,
        "best_champion_beating_candidate_id": int(best_champion["candidate_id"]) if best_champion is not None else None,
        "best_candidate": best_public,
        "adopted_candidate_id": document.get("adopted_candidate_id"),
        "adopted_strategy_id": document.get("adopted_strategy_id"),
        "validated_candidate_id": document.get("validated_candidate_id"),
        "validation_processing_id": document.get("validation_processing_id"),
        "validation_strategy_id": document.get("validation_strategy_id"),
        "validated_at": document.get("validated_at"),
        "adaptive_early_stopped": bool(document.get("adaptive_early_stopped")),
        "adaptive_early_stop_reason": document.get("adaptive_early_stop_reason"),
        "research_budget_used": document.get("research_budget_used"),
        "adoption_history": deepcopy(document.get("adoption_history") or []),
    })


def list_model_tuning_history(db: Any, *, limit: int = 100) -> list[dict[str, Any]]:
    safe_limit = max(1, min(100, int(limit)))
    projection = {
        "_id": 0,
        "id": 1,
        "status": 1,
        "phase": 1,
        "method": 1,
        "tuning_scope": 1,
        "tuning_scope_label": 1,
        "model_family": 1,
        "model_label": 1,
        "strategy_profile_id": 1,
        "strategy_profile_name": 1,
        "strategy_profile_revision": 1,
        "strategy_profile_status": 1,
        "created_at": 1,
        "started_at": 1,
        "finished_at": 1,
        "market_data_cutoff_date": 1,
        "candidate_count": 1,
        "total_candidates": 1,
        "completed_candidates": 1,
        "failed_candidates": 1,
        "cancelled_candidates": 1,
        "best_candidate_id": 1,
        "best_champion_beating_candidate_id": 1,
        "adopted_candidate_id": 1,
        "adopted_strategy_id": 1,
        "adoption_history": 1,
        "candidates.candidate_id": 1,
        "candidates.status": 1,
        "candidates.rank": 1,
        "candidates.kind": 1,
        "candidates.is_control": 1,
        "candidates.metrics": 1,
        "candidates.champion_gate_passed": 1,
    }
    cursor = db[MODEL_TUNING_RUNS_COLLECTION].find({}, projection).sort("created_at", -1).limit(safe_limit)
    return [_public_tuning_history_item(document) for document in cursor]



def _temporal_control_candidate(strategy: dict[str, Any], baseline: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    metrics = deepcopy(baseline.get("metrics") or {})
    source_run_id = str(baseline.get("source_temporal_run_id") or baseline.get("job_id") or "")
    if not source_run_id or not metrics:
        raise ModelTuningConflict("The materialized TEMPORAL Strategy does not contain a reusable validation baseline.")
    return {
        "candidate_id": 0,
        "kind": "control",
        "is_control": True,
        "settings": deepcopy(settings),
        "settings_hash": _settings_hash(settings),
        "status": "completed",
        "rank": None,
        "metrics": metrics,
        "equity_preview": [],
        "job_id": None,
        "source_job_id": None,
        "source_temporal_run_id": source_run_id,
        "baseline_reused": True,
        "champion_gate_passed": None,
        "champion_gate": None,
        "raw_results_retained": False,
    }


def _start_temporal_tuning(
    db: Any,
    *,
    strategy: dict[str, Any],
    model_snapshot: dict[str, Any],
    tuning_scope: str,
    method: str,
    candidate_count: int,
    caro_candidate_count: int | None,
    seed: int,
    baseline_job_id: str | None,
    source_tuning_run_id: str | None,
    anchor_candidate_id: int | None,
    probability_config: dict[str, Any] | None,
    fold_protocol: dict[str, Any] | None,
    explicit_start_confirmation: bool,
    actor_email: str | None,
    tuning_target_source: str,
) -> dict[str, Any]:
    temporal_model_scope = str(tuning_scope) == TEMPORAL_MODEL_TUNING_SCOPE
    protocol = _normalized_fold_protocol(fold_protocol)
    plan = temporal_model_plan(strategy, model_snapshot) if temporal_model_scope else temporal_policy_plan(strategy)
    search_space = [dict(item) for item in plan["search_space"]]
    base_values = deepcopy(plan["base_values"])
    baseline = temporal_model_baseline(strategy, model_snapshot) if temporal_model_scope else temporal_policy_baseline(strategy)
    source_run_id = str(baseline.get("source_temporal_run_id") or "")
    if baseline_job_id and str(baseline_job_id) != source_run_id:
        raise ModelTuningConflict("The selected TEMPORAL baseline does not match the Strategy source Temporal run.")
    control = _temporal_control_candidate(strategy, baseline, base_values)
    baseline_fold_count = len((baseline.get("metrics") or {}).get("folds") or [])
    research_cache_required = bool(protocol["research_folds"] != baseline_fold_count and baseline_fold_count > 0)
    if research_cache_required:
        control["status"] = "pending"
        control["metrics"] = None
        control["baseline_reused"] = False
        control["research_fold_rebuild"] = True
    probability_input = dict(probability_config or {})
    probability: dict[str, Any]
    probability_anchor: dict[str, Any] | None = None
    prior_observations: list[dict[str, Any]] = []
    continuation_source_id: str | None = None

    if method == PROBABILITY_METHOD:
        candidates = [control]
        total_candidates = int(candidate_count) + 1
        minimum_exploration = min(max(4, min(24, len(search_space) + 2)), max(4, int(candidate_count)))
        probability_anchor = {
            "source": "materialized_temporal_run",
            "source_temporal_run_id": source_run_id,
            "candidate_id": 0,
            "settings_hash": control["settings_hash"],
            "settings": deepcopy(base_values),
            "metrics": deepcopy(control.get("metrics") or baseline.get("metrics") or {}),
        }
        if source_tuning_run_id:
            source = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": str(source_tuning_run_id)})
            if source is None:
                raise ModelTuningNotFound("Source tuning campaign not found.")
            source_snapshot_id = str((strategy.get("temporal_policy") or {}).get("market_data_snapshot_id") or "").strip().lower() or None
            _validate_temporal_continuation_source(
                source, strategy=strategy, tuning_scope=tuning_scope, search_space=search_space,
                source_temporal_run_id=source_run_id, market_data_snapshot_id=source_snapshot_id,
                research_folds=protocol["research_folds"],
            )
            prior_observations = _continuation_observations(source)
            anchor = _source_anchor(source, anchor_candidate_id)
            probability_anchor = {
                "source": "continued_campaign_champion",
                "source_tuning_run_id": str(source_tuning_run_id),
                "candidate_id": int(anchor.get("candidate_id") or 0),
                "settings_hash": str(anchor.get("settings_hash") or ""),
                "settings": deepcopy(anchor.get("settings") or {}),
                "metrics": deepcopy(anchor.get("metrics") or {}),
            }
            continuation_source_id = str(source_tuning_run_id)
        probability = {
            "startup_trials": minimum_exploration,
            "minimum_exploration_trials": minimum_exploration,
            "imported_observation_count": len(prior_observations),
            "min_capital_improvement": float(probability_input.get("min_capital_improvement", 0.03)),
            "sharpe_tolerance": float(probability_input.get("sharpe_tolerance", 0.05)),
            "drawdown_tolerance": float(probability_input.get("drawdown_tolerance", 0.03)),
            "min_worst_fold_return": float(probability_input.get("min_worst_fold_return", 0.0)),
            "candidate_pool_size": int(probability_input.get("candidate_pool_size", 2048)),
            "exploration_weight": float(probability_input.get("exploration_weight", 0.15)),
            "initial_exploration_fraction": float(probability_input.get("initial_exploration_fraction", 0.45)),
            "minimum_exploration_fraction": float(probability_input.get("minimum_exploration_fraction", 0.20)),
            "stagnation_recovery_trials": int(probability_input.get("stagnation_recovery_trials", 4)),
            "adaptive_stopping_enabled": bool(probability_input.get("adaptive_stopping_enabled", True)),
            "no_improvement_trial_limit": int(probability_input.get("no_improvement_trial_limit", DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT)),
            "minimum_meaningful_improvement": float(probability_input.get("minimum_meaningful_improvement", DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT)),
            "space_filling_pool_size": int(probability_input.get("space_filling_pool_size", 1024)),
            "probability_model": PROBABILITY_MODEL,
            "source_mode": "temporal_lightgbm_retrain_unified" if temporal_model_scope else "temporal_frozen_replay_unified",
            "search_policy": "dynamic_space_filling_plus_sequential_adaptive_trust_region",
        }
    elif method == PIPELINE_METHOD:
        adaptive_trials = max(1, int(caro_candidate_count or DEFAULT_CANDIDATE_COUNT))
        candidates = generate_latin_hypercube_candidates(base_values, candidate_count=candidate_count, seed=seed, search_space=search_space)
        candidates[0] = control
        total_candidates = len(candidates) + adaptive_trials
        probability_anchor = {
            "source": "materialized_temporal_run",
            "source_temporal_run_id": source_run_id,
            "candidate_id": 0,
            "settings_hash": control["settings_hash"],
            "settings": deepcopy(base_values),
            "metrics": deepcopy(baseline.get("metrics") or {}),
        }
        probability = {
            "startup_trials": int(candidate_count),
            "full_lhs_candidate_count": int(candidate_count),
            "adaptive_trials": adaptive_trials,
            "imported_observation_count": 0,
            "min_capital_improvement": float(probability_input.get("min_capital_improvement", 0.03)),
            "sharpe_tolerance": float(probability_input.get("sharpe_tolerance", 0.05)),
            "drawdown_tolerance": float(probability_input.get("drawdown_tolerance", 0.03)),
            "min_worst_fold_return": float(probability_input.get("min_worst_fold_return", 0.0)),
            "candidate_pool_size": int(probability_input.get("candidate_pool_size", 2048)),
            "exploration_weight": float(probability_input.get("exploration_weight", 0.15)),
            "probability_model": PROBABILITY_MODEL,
            "source_mode": "temporal_lightgbm_retrain_lhs_then_caro" if temporal_model_scope else "temporal_frozen_replay_lhs_then_caro",
        }
    else:
        candidates = generate_latin_hypercube_candidates(base_values, candidate_count=candidate_count, seed=seed, search_space=search_space)
        candidates[0] = control
        total_candidates = len(candidates)
        probability = {}

    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    snapshot_id = str(policy.get("market_data_snapshot_id") or "").strip().lower() or None
    analysis_end = str(policy.get("analysis_end_date") or "").strip() or None
    now = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-tune-" + uuid.uuid4().hex[:8]
    initial_completed = sum(1 for item in candidates if item.get("status") == "completed")
    document = {
        "_id": run_id,
        "id": run_id,
        "schema_version": TUNING_SCHEMA_VERSION,
        "status": "queued",
        "phase": "queued",
        "method": method,
        "model_family": TEMPORAL_MODEL_FAMILY if temporal_model_scope else TEMPORAL_POLICY_MODEL_FAMILY,
        "model_label": "LightGBM Temporal Intelligence" if temporal_model_scope else "Temporal Policy",
        "tuning_scope": TEMPORAL_MODEL_TUNING_SCOPE if temporal_model_scope else TEMPORAL_POLICY_TUNING_SCOPE,
        "tuning_scope_label": plan["scope_label"],
        "tuning_scope_description": plan["description"],
        "strategy_mode": plan.get("strategy_mode"),
        "candidate_count": int(candidate_count),
        "caro_candidate_count": (int(caro_candidate_count or DEFAULT_CANDIDATE_COUNT) if method == PIPELINE_METHOD else None),
        "pipeline_mode": ("full_lhs_then_caro" if method == PIPELINE_METHOD else None),
        "pipeline_handoff_completed": False,
        "total_candidates": int(total_candidates),
        "generated_candidates": len(candidates),
        "completed_candidates": int(initial_completed),
        "failed_candidates": 0,
        "cancelled_candidates": 0,
        "seed": int(seed),
        "fold_protocol": bson_value(protocol),
        "source_research_fold_count": int(baseline_fold_count or protocol["research_folds"]),
        "research_fold_cache_required": bool(research_cache_required),
        "research_fold_cache_run_id": None,
        "search_space": search_space,
        "tuned_parameters": list(plan["tuned_parameters"]),
        "tuned_model_parameters": list(plan.get("tuned_model_parameters") or []),
        "tuned_strategy_parameters": list(plan.get("tuned_strategy_parameters") or []),
        "strategy_profile_id": strategy["id"],
        "strategy_profile_name": strategy["name"],
        "strategy_profile_revision": int(strategy["revision"]),
        "strategy_profile_status": str(strategy.get("status") or "draft"),
        "tuning_target_source": tuning_target_source,
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "strategy_configuration_snapshot": bson_value(deepcopy(strategy.get("configuration") or {})),
        "base_model_settings_hash": model_snapshot.get("settings_hash") if temporal_model_scope else None,
        "base_model_settings_revision": int(model_snapshot.get("settings_revision") or 0) if temporal_model_scope else 0,
        "base_model_values": bson_value(plan.get("base_model_values") or {}) if temporal_model_scope else {},
        "base_tuning_values": bson_value(base_values),
        "frozen_model_values": bson_value(plan.get("frozen_model_values") or {}) if temporal_model_scope else {},
        "fixed_model_values": bson_value(plan.get("fixed_model_values") or {}) if temporal_model_scope else {},
        "baseline_execution": bson_value(baseline),
        "probability_config": bson_value(probability),
        "probability_anchor": bson_value(probability_anchor) if probability_anchor else None,
        "starting_probability_anchor": bson_value(deepcopy(probability_anchor)) if probability_anchor else None,
        "probability_state": bson_value(initial_probability_state(prior_observations)) if method in _ADAPTIVE_METHODS else None,
        "probability_champion_history": [],
        "source_tuning_run_id": continuation_source_id,
        "source_strategy_profile_id": strategy["id"],
        "source_strategy_profile_revision": int(strategy["revision"]),
        "source_temporal_run_id": source_run_id,
        "prior_observations": bson_value(prior_observations),
        "imported_observation_count": len(prior_observations),
        "execution_mode": "full_temporal_lightgbm_retrain" if temporal_model_scope else "frozen_temporal_replay",
        "execution_request_snapshot": None,
        "execution_context_hash": None,
        "expected_market_data_signature_sha256": snapshot_id,
        "market_data_snapshot_id": snapshot_id,
        "market_data_signature_source": "materialized_temporal_run",
        "market_data_signature_established_by_candidate_id": None,
        "market_data_cutoff_date": analysis_end,
        "adoption_context_compatible": True,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "created_by": (actor_email or "").strip().lower() or None,
        "explicit_start_confirmation": bool(explicit_start_confirmation),
        "stop_requested": False,
        "current_candidate_id": None,
        "current_job_id": None,
        "best_candidate_id": None,
        "best_exploratory_candidate_id": None,
        "best_champion_beating_candidate_id": None,
        "control_candidate_id": 0,
        "event_log": bson_value([
            {
                "at": now,
                "level": "info",
                "stage": "created",
                "message": _sanitize_tuning_log_line(
                    f"TEMPORAL {'Model' if temporal_model_scope else 'Policy'} campaign created. method={method}; source_temporal_run={source_run_id}; total_candidates={int(total_candidates)}; imported_observations={len(prior_observations)}; continuation_source={continuation_source_id or 'none'}."
                ),
                "candidate_id": None,
                "job_id": None,
            },
            {
                "at": now,
                "level": "info",
                "stage": "control_reused",
                "message": _sanitize_tuning_log_line(
                    (f"Control #0 reused materialized Temporal Intelligence run {source_run_id}; challenger candidates retrain Temporal LightGBM on the same frozen market snapshot." if temporal_model_scope else f"Control #0 reused materialized Temporal Intelligence run {source_run_id}; no model retraining or market-data download was executed.")
                ),
                "candidate_id": 0,
                "job_id": None,
            },
        ]),
        "candidates": bson_value(candidates),
    }
    db[MODEL_TUNING_RUNS_COLLECTION].insert_one(document)
    cleanup = _compact_historical_tuning_runs(db, next_run_id=run_id)
    if cleanup["compacted_runs"]:
        _append_campaign_event(
            db, run_id,
            message=(
                f"Historical tuning storage compacted {cleanup['compacted_runs']} prior campaign(s): "
                f"discarded {cleanup['discarded_trials']} transient trial observation(s) and retained "
                f"{cleanup['retained_candidates']} adopted/validated candidate(s)."
            ),
            stage="historical_storage_cleanup",
        )
    threading.Thread(target=run_model_tuning, args=(run_id,), daemon=True).start()
    return public_model_tuning_run(db, db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or document)


def start_model_tuning(
    db: Any,
    *,
    method: str = TUNING_METHOD,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    caro_candidate_count: int | None = None,
    seed: int = DEFAULT_SEED,
    baseline_job_id: str | None = None,
    source_tuning_run_id: str | None = None,
    anchor_candidate_id: int | None = None,
    tuning_target: str | None = None,
    probability_config: dict[str, Any] | None = None,
    fold_protocol: dict[str, Any] | None = None,
    explicit_start_confirmation: bool = False,
    actor_email: str | None = None,
) -> dict[str, Any]:
    normalized_method = str(method or TUNING_METHOD).strip().lower()
    if normalized_method not in {TUNING_METHOD, PROBABILITY_METHOD, PIPELINE_METHOD}:
        raise ModelTuningConflict("Unsupported model tuning method.")

    active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"status": {"$in": list(_ACTIVE_STATUSES)}}, {"_id": 0, "id": 1}
    )
    if active_tuning is not None:
        raise ModelTuningConflict(f"Model tuning {active_tuning.get('id', 'unknown')} is already active.")

    active_backtest = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}, "internal_job": {"$ne": True}},
        {"_id": 0, "id": 1},
    )
    if active_backtest is not None:
        raise ModelTuningConflict(
            f"Wait for backtest {active_backtest.get('id', 'unknown')} to finish before starting model tuning."
        )

    strategy, model_snapshot, tuning_target_source = _tuning_target_strategy(db)
    if str(strategy.get("tuning_target") or "") == "decision_optimization":
        raise ModelTuningConflict(
            "The selected MILP Decision Strategy is research-only and is not a target for the current Model Tuning engine."
        )
    if (
        str(strategy.get("strategy_kind") or "") == "temporal_intelligence"
        and str(strategy.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
    ):
        raise ModelTuningConflict(
            "The selected Stateful Temporal Strategy is the Strategy Research baseline, but its stateful-transition parameters are not supported by the current Model Tuning engine."
        )
    if str(strategy.get("strategy_kind") or "") == "temporal_intelligence":
        requested_temporal_target = str(tuning_target or TEMPORAL_MODEL_TUNING_SCOPE).strip().lower()
        if requested_temporal_target not in {TEMPORAL_MODEL_TUNING_SCOPE, TEMPORAL_POLICY_TUNING_SCOPE}:
            raise ModelTuningConflict("TEMPORAL Strategies support only Temporal Model Tuning or Temporal Policy Tuning.")
        if requested_temporal_target == TEMPORAL_MODEL_TUNING_SCOPE and not bool(explicit_start_confirmation):
            raise ModelTuningConflict("Temporal Model Tuning requires explicit start confirmation.")
        return _start_temporal_tuning(
            db,
            strategy=strategy,
            model_snapshot=model_snapshot,
            tuning_scope=requested_temporal_target,
            method=normalized_method,
            candidate_count=candidate_count,
            caro_candidate_count=caro_candidate_count,
            seed=seed,
            baseline_job_id=baseline_job_id,
            source_tuning_run_id=source_tuning_run_id,
            anchor_candidate_id=anchor_candidate_id,
            probability_config=probability_config,
            fold_protocol=fold_protocol,
            explicit_start_confirmation=explicit_start_confirmation,
            actor_email=actor_email,
            tuning_target_source=tuning_target_source,
        )
    if str(model_snapshot.get("family") or "") != TUNING_MODEL_FAMILY:
        raise ModelTuningConflict("The current tuning target must use LightGBM before model tuning can start.")
    tuning_plan = _tuning_plan(strategy, model_snapshot)
    tuning_scope = str(tuning_plan["scope"])
    tuning_scope_label = str(tuning_plan["scope_label"])
    tuning_scope_description = str(tuning_plan["description"])
    search_space = [dict(item) for item in tuning_plan["search_space"]]
    tuned_parameters = list(tuning_plan["tuned_parameters"])
    tuned_model_parameters = list(tuning_plan.get("tuned_model_parameters") or [])
    tuned_strategy_parameters = list(tuning_plan.get("tuned_strategy_parameters") or [])
    base_tuning_values = deepcopy(tuning_plan["base_values"])
    base_model_values = deepcopy(tuning_plan.get("base_model_values") or model_values_from_snapshot(model_snapshot))
    frozen_model_values = deepcopy(tuning_plan["frozen_model_values"])
    fixed_model_values = deepcopy(tuning_plan.get("fixed_model_values") or {})

    probability = dict(probability_config or {})
    prior_observations: list[dict[str, Any]] = []
    probability_anchor: dict[str, Any] | None = None
    source_run_id: str | None = None
    source_strategy_profile_id: str | None = None
    source_strategy_profile_revision: int | None = None
    adoption_context_compatible = True
    expected_market_data_signature: str | None = None
    market_data_snapshot_id: str | None = None
    market_data_signature_source = "frozen_campaign_snapshot"

    if normalized_method == PROBABILITY_METHOD and source_tuning_run_id:
        source = _source_campaign(
            db,
            source_tuning_run_id,
            expected_scope=tuning_scope,
            expected_search_space=search_space,
        )
        prior_observations = _source_observations(source)
        anchor = _source_anchor(source, anchor_candidate_id)
        source_run_id = str(source.get("id") or source_tuning_run_id)
        source_strategy_profile_id = str(source.get("strategy_profile_id") or "") or None
        source_strategy_profile_revision = int(source.get("strategy_profile_revision") or 0)
        baseline = deepcopy(source.get("baseline_execution") or {})
        reference_job_id = str(baseline.get("job_id") or "")
        if not reference_job_id:
            raise ModelTuningConflict("The source campaign no longer references its baseline Backtest.")
        execution_context = _frozen_execution_context_from_campaign(db, source)
        expected_market_data_signature = str(execution_context.get("market_data_signature_sha256") or "").strip() or None
        market_data_snapshot_id = str(execution_context.get("market_data_snapshot_id") or "").strip().lower() or None
        market_data_signature_source = "source_campaign_snapshot" if market_data_snapshot_id else "source_campaign_legacy_signature"
        base_values = deepcopy(anchor.get("settings") or {})
        if tuning_scope == ABSOLUTE_UTILITY_TUNING_SCOPE:
            source_base_model = deepcopy(source.get("base_model_values") or base_model_values)
            base_model_values = source_base_model
            frozen_model_values = deepcopy(source.get("frozen_model_values") or source_base_model)
        adoption_context_compatible = False
        if str(strategy.get("id") or "") == str(source.get("strategy_profile_id") or ""):
            current_baselines = list_model_tuning_baselines(db, limit=10)
            if current_baselines:
                try:
                    current_context = _frozen_execution_context_from_job(db, str(current_baselines[0].get("job_id") or ""))
                    adoption_context_compatible = current_context.get("context_hash") == execution_context.get("context_hash")
                except ModelTuningConflict:
                    adoption_context_compatible = False
        probability_anchor = {
            "source": "prior_latin_hypercube_candidate",
            "source_tuning_run_id": source_run_id,
            "candidate_id": int(anchor.get("candidate_id") or 0),
            "settings_hash": str(anchor.get("settings_hash") or ""),
            "settings": deepcopy(anchor.get("settings") or {}),
            "metrics": deepcopy(anchor.get("metrics") or {}),
        }
        candidates: list[dict[str, Any]] = []
        total_candidates = int(candidate_count)
        probability = {
            "startup_trials": 0,
            "imported_observation_count": len(prior_observations),
            "min_capital_improvement": float(probability.get("min_capital_improvement", 0.03)),
            "sharpe_tolerance": float(probability.get("sharpe_tolerance", 0.05)),
            "drawdown_tolerance": float(probability.get("drawdown_tolerance", 0.03)),
            "min_worst_fold_return": float(probability.get("min_worst_fold_return", 0.0)),
            "candidate_pool_size": int(probability.get("candidate_pool_size", 2048)),
            "exploration_weight": float(probability.get("exploration_weight", 0.15)),
            "initial_exploration_fraction": float(probability.get("initial_exploration_fraction", 0.45)),
            "minimum_exploration_fraction": float(probability.get("minimum_exploration_fraction", 0.20)),
            "stagnation_recovery_trials": int(probability.get("stagnation_recovery_trials", 4)),
            "adaptive_stopping_enabled": bool(probability.get("adaptive_stopping_enabled", True)),
            "no_improvement_trial_limit": int(probability.get("no_improvement_trial_limit", DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT)),
            "minimum_meaningful_improvement": float(probability.get("minimum_meaningful_improvement", DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT)),
            "space_filling_pool_size": int(probability.get("space_filling_pool_size", 1024)),
            "probability_model": PROBABILITY_MODEL,
            "source_mode": "prior_campaign_unified",
            "search_policy": "dynamic_space_filling_plus_sequential_adaptive_trust_region",
        }
    else:
        if source_tuning_run_id:
            raise ModelTuningConflict("A prior tuning campaign can seed only CARO Probability.")
        baselines = list_model_tuning_baselines(db, limit=100)
        if not baselines:
            raise ModelTuningConflict(
                "Run and complete a normal Simulation Backtest with the currently saved LightGBM Strategy model before starting tuning."
            )
        baseline = _baseline_by_job_id(db, baseline_job_id) if baseline_job_id else baselines[0]
        reference_job_id = str(baseline.get("job_id") or "")
        execution_context = _frozen_execution_context_from_job(db, reference_job_id)
        expected_market_data_signature = str(execution_context.get("market_data_signature_sha256") or "").strip().lower() or None
        market_data_signature_source = "certified_baseline_backtest"
        base_values = deepcopy(base_tuning_values)

        if normalized_method == PROBABILITY_METHOD:
            default_minimum_exploration = max(4, min(24, len(search_space) + 2))
            legacy_startup = probability.get("startup_trials")
            minimum_exploration = max(
                4,
                int(probability.get("minimum_exploration_trials") or legacy_startup or default_minimum_exploration),
            )
            minimum_exploration = min(minimum_exploration, max(4, int(candidate_count)))
            control_values = _normalize_tuning_values(base_values, search_space)
            candidates = [_reused_baseline_control_candidate(
                db, baseline=baseline, settings=control_values,
            )]
            total_candidates = int(candidate_count) + 1
            probability_anchor = {
                "source": "baseline_backtest",
                "job_id": reference_job_id,
                "settings_hash": _settings_hash(base_values),
                "settings": deepcopy(base_values),
                "metrics": deepcopy(baseline.get("metrics") or {}),
            }
            prior_observations = _automatic_compatible_prior_observations(
                db,
                strategy=strategy,
                tuning_scope=tuning_scope,
                strategy_mode=tuning_plan.get("strategy_mode"),
                search_space=search_space,
                execution_context=execution_context,
                base_values=base_values,
            )
            probability = {
                "startup_trials": minimum_exploration,
                "minimum_exploration_trials": minimum_exploration,
                "imported_observation_count": len(prior_observations),
                "min_capital_improvement": float(probability.get("min_capital_improvement", 0.03)),
                "sharpe_tolerance": float(probability.get("sharpe_tolerance", 0.05)),
                "drawdown_tolerance": float(probability.get("drawdown_tolerance", 0.03)),
                "min_worst_fold_return": float(probability.get("min_worst_fold_return", 0.0)),
                "candidate_pool_size": int(probability.get("candidate_pool_size", 2048)),
                "exploration_weight": float(probability.get("exploration_weight", 0.15)),
                "initial_exploration_fraction": float(probability.get("initial_exploration_fraction", 0.45)),
                "minimum_exploration_fraction": float(probability.get("minimum_exploration_fraction", 0.20)),
                "stagnation_recovery_trials": int(probability.get("stagnation_recovery_trials", 4)),
                "adaptive_stopping_enabled": bool(probability.get("adaptive_stopping_enabled", True)),
                "no_improvement_trial_limit": int(probability.get("no_improvement_trial_limit", DEFAULT_NO_IMPROVEMENT_TRIAL_LIMIT)),
                "minimum_meaningful_improvement": float(probability.get("minimum_meaningful_improvement", DEFAULT_MINIMUM_MEANINGFUL_IMPROVEMENT)),
                "space_filling_pool_size": int(probability.get("space_filling_pool_size", 1024)),
                "probability_model": PROBABILITY_MODEL,
                "source_mode": ("standalone_unified_with_compatible_history" if prior_observations else "standalone_unified"),
                "search_policy": "dynamic_space_filling_plus_sequential_adaptive_trust_region",
            }
        elif normalized_method == PIPELINE_METHOD:
            adaptive_trials = max(1, int(caro_candidate_count or DEFAULT_CANDIDATE_COUNT))
            candidates = generate_latin_hypercube_candidates(base_values, candidate_count=candidate_count, seed=seed, search_space=search_space)
            candidates[0] = _reused_baseline_control_candidate(
                db, baseline=baseline, settings=deepcopy(candidates[0]["settings"]),
            )
            total_candidates = len(candidates) + adaptive_trials
            probability_anchor = {
                "source": "baseline_backtest",
                "job_id": reference_job_id,
                "settings_hash": _settings_hash(base_values),
                "settings": deepcopy(base_values),
                "metrics": deepcopy(baseline.get("metrics") or {}),
            }
            probability = {
                "startup_trials": int(candidate_count),
                "full_lhs_candidate_count": int(candidate_count),
                "adaptive_trials": adaptive_trials,
                "imported_observation_count": 0,
                "min_capital_improvement": float(probability.get("min_capital_improvement", 0.03)),
                "sharpe_tolerance": float(probability.get("sharpe_tolerance", 0.05)),
                "drawdown_tolerance": float(probability.get("drawdown_tolerance", 0.03)),
                "min_worst_fold_return": float(probability.get("min_worst_fold_return", 0.0)),
                "candidate_pool_size": int(probability.get("candidate_pool_size", 2048)),
                "exploration_weight": float(probability.get("exploration_weight", 0.15)),
                "probability_model": PROBABILITY_MODEL,
                "source_mode": "full_lhs_then_caro",
            }
        else:
            candidates = generate_latin_hypercube_candidates(base_values, candidate_count=candidate_count, seed=seed, search_space=search_space)
            candidates[0] = _reused_baseline_control_candidate(
                db, baseline=baseline, settings=deepcopy(candidates[0]["settings"]),
            )
            total_candidates = int(candidate_count) + 1
            probability = {}

    initial_completed_candidates = sum(1 for item in candidates if item.get("status") == "completed")
    now = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-tune-" + uuid.uuid4().hex[:8]
    document = {
        "_id": run_id,
        "id": run_id,
        "schema_version": TUNING_SCHEMA_VERSION,
        "status": "queued",
        "phase": "queued",
        "method": normalized_method,
        "model_family": TUNING_MODEL_FAMILY,
        "model_label": "LightGBM Utility",
        "tuning_scope": tuning_scope,
        "tuning_scope_label": tuning_scope_label,
        "tuning_scope_description": tuning_scope_description,
        "strategy_mode": tuning_plan.get("strategy_mode"),
        "candidate_count": int(candidate_count),
        "caro_candidate_count": (int(caro_candidate_count or DEFAULT_CANDIDATE_COUNT) if normalized_method == PIPELINE_METHOD else None),
        "pipeline_mode": ("full_lhs_then_caro" if normalized_method == PIPELINE_METHOD else None),
        "pipeline_handoff_completed": False,
        "total_candidates": int(total_candidates),
        "generated_candidates": len(candidates),
        "completed_candidates": int(initial_completed_candidates),
        "failed_candidates": 0,
        "cancelled_candidates": 0,
        "seed": int(seed),
        "search_space": search_space,
        "tuned_parameters": tuned_parameters,
        "tuned_model_parameters": tuned_model_parameters,
        "tuned_strategy_parameters": tuned_strategy_parameters,
        "strategy_profile_id": strategy["id"],
        "strategy_profile_name": strategy["name"],
        "strategy_profile_revision": int(strategy["revision"]),
        "strategy_profile_status": str(strategy.get("status") or "draft"),
        "tuning_target_source": tuning_target_source,
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "strategy_configuration_snapshot": bson_value(deepcopy(strategy.get("configuration") or {})),
        "base_model_settings_hash": model_snapshot.get("settings_hash"),
        "base_model_settings_revision": int(model_snapshot.get("settings_revision") or 0),
        "base_model_values": bson_value(base_model_values),
        "base_tuning_values": bson_value(base_values),
        "frozen_model_values": bson_value(frozen_model_values),
        "fixed_model_values": bson_value(fixed_model_values),
        "baseline_execution": bson_value(baseline),
        "probability_config": bson_value(probability),
        "probability_anchor": bson_value(probability_anchor) if probability_anchor else None,
        "starting_probability_anchor": bson_value(deepcopy(probability_anchor)) if probability_anchor else None,
        "probability_state": bson_value(initial_probability_state(prior_observations)) if normalized_method in _ADAPTIVE_METHODS else None,
        "probability_champion_history": [],
        "source_tuning_run_id": source_run_id,
        "source_strategy_profile_id": source_strategy_profile_id,
        "source_strategy_profile_revision": source_strategy_profile_revision,
        "prior_observations": bson_value(prior_observations),
        "imported_observation_count": len(prior_observations),
        "execution_mode": "integrated_api_worker",
        "execution_request_snapshot": bson_value(execution_context["request"]),
        "execution_context_hash": execution_context.get("context_hash"),
        "expected_market_data_signature_sha256": expected_market_data_signature,
        "market_data_snapshot_id": market_data_snapshot_id,
        "market_data_signature_source": market_data_signature_source,
        "market_data_signature_established_by_candidate_id": None,
        "market_data_cutoff_date": execution_context.get("market_data_cutoff_date"),
        "adoption_context_compatible": bool(adoption_context_compatible),
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "created_by": (actor_email or "").strip().lower() or None,
        "stop_requested": False,
        "current_candidate_id": None,
        "current_job_id": None,
        "best_candidate_id": None,
        "best_exploratory_candidate_id": None,
        "best_champion_beating_candidate_id": None,
        "control_candidate_id": 0 if any(item.get("is_control") for item in candidates) else None,
        "event_log": bson_value([
            {
                "at": now,
                "level": "info",
                "stage": "created",
                "message": _sanitize_tuning_log_line(
                    f"Campaign created. method={normalized_method}; scope={tuning_scope}; total_candidates={int(total_candidates)}; "
                    f"imported_observations={len(prior_observations)}; source_campaign={source_run_id or 'none'}."
                ),
                "candidate_id": None,
                "job_id": None,
            },
            *([
                {
                    "at": now,
                    "level": "info",
                    "stage": "control_reused",
                    "message": _sanitize_tuning_log_line(
                        f"Control #0 reused certified Candidate Backtest {reference_job_id}; no duplicate Control Backtest was executed."
                    ),
                    "candidate_id": 0,
                    "job_id": reference_job_id,
                }
            ] if any(item.get("is_control") and item.get("baseline_reused") for item in candidates) else []),
        ]),
        "candidates": bson_value(candidates),
    }
    db[MODEL_TUNING_RUNS_COLLECTION].insert_one(document)
    cleanup = _compact_historical_tuning_runs(db, next_run_id=run_id)
    if cleanup["compacted_runs"]:
        _append_campaign_event(
            db, run_id,
            message=(
                f"Historical tuning storage compacted {cleanup['compacted_runs']} prior campaign(s): "
                f"discarded {cleanup['discarded_trials']} transient trial observation(s) and retained "
                f"{cleanup['retained_candidates']} adopted/validated candidate(s)."
            ),
            stage="historical_storage_cleanup",
        )
    threading.Thread(target=run_model_tuning, args=(run_id,), daemon=True).start()
    return public_model_tuning_run(db, db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or document)


def _ensure_campaign_market_snapshot(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    
    run_id = str(document.get("id") or "")
    expected = str(document.get("expected_market_data_signature_sha256") or "").strip().lower() or None
    snapshot_id = str(document.get("market_data_snapshot_id") or "").strip().lower() or None

    if snapshot_id:
        snapshot = require_tuning_market_snapshot(db, snapshot_id)
        actual = str(snapshot.get("signature") or snapshot.get("snapshot_id") or "").strip().lower()
        if expected and actual != expected:
            raise RuntimeError(
                f"FrozenTuningSnapshotMismatch: campaign expects {expected}, but snapshot {snapshot_id} reports {actual}."
            )
        signature = actual
        source = "source_campaign_snapshot" if document.get("source_tuning_run_id") else "frozen_campaign_snapshot"
    else:
        try:
            snapshot = freeze_tuning_market_snapshot(
                db,
                deepcopy(document.get("execution_request_snapshot") or {}),
                expected_signature=expected or None,
            )
        except TuningMarketSnapshotMismatch:
            raise
        signature = str(snapshot.get("signature") or snapshot.get("snapshot_id") or "").strip().lower()
        snapshot_id = str(snapshot.get("snapshot_id") or signature).strip().lower()
        source = "legacy_source_recovered_snapshot" if document.get("source_tuning_run_id") else "frozen_campaign_snapshot"

    request_snapshot = deepcopy(document.get("execution_request_snapshot") or {})
    request_snapshot["research_market_data_mode"] = "database_only"
    request_snapshot["research_market_data_snapshot_id"] = snapshot_id
    request_snapshot["expected_market_data_signature_sha256"] = signature
    now = utc_now()
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {
            "$set": {
                "execution_request_snapshot": bson_value(request_snapshot),
                "expected_market_data_signature_sha256": signature,
                "market_data_snapshot_id": snapshot_id,
                "market_data_signature_source": source,
                "market_data_signature_established_by_candidate_id": None,
                "updated_at": now,
            }
        },
    )
    document["execution_request_snapshot"] = request_snapshot
    document["expected_market_data_signature_sha256"] = signature
    document["market_data_snapshot_id"] = snapshot_id
    document["market_data_signature_source"] = source
    _append_campaign_event(
        db,
        run_id,
        message=f"Frozen market-data snapshot {snapshot_id} bound to the complete tuning campaign.",
        stage="market_data_snapshot_frozen",
    )
    return document



def _run_temporal_tuning(db: Any, run_id: str) -> None:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        return
    initial_phase = "latin_hypercube_exploration" if str(document.get("method") or "") == PIPELINE_METHOD else "running"
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {"$set": {"status": "running", "phase": initial_phase, "started_at": utc_now(), "updated_at": utc_now()}},
    )
    tuning_scope = str(document.get("tuning_scope") or TEMPORAL_POLICY_TUNING_SCOPE)
    temporal_model_scope = tuning_scope == TEMPORAL_MODEL_TUNING_SCOPE
    _append_campaign_event(
        db, run_id,
        message="Temporal LightGBM model tuning worker started." if temporal_model_scope else "Frozen Temporal Policy tuning worker started.",
        stage="running",
    )
    temporal_model_campaign_context: dict[str, Any] | None = None

    while True:
        document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or {}
        if bool(document.get("stop_requested")):
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {"status": "stopped", "phase": "stopped", "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
            )
            _append_campaign_event(db, run_id, message=("Stop request honored after the current Temporal LightGBM candidate." if temporal_model_scope else "Stop request honored after the current frozen replay unit."), stage="stopped")
            return

        candidates = list(document.get("candidates") or [])
        pending = next((item for item in candidates if item.get("status") == "pending"), None)
        method = str(document.get("method") or "")
        if pending is None:
            early_stop_reason = _adaptive_early_stop_reason(document)
            if early_stop_reason:
                completed_count = int(document.get("completed_candidates") or 0)
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {
                        "status": "completed",
                        "phase": "adaptive_early_stopped",
                        "adaptive_early_stopped": True,
                        "adaptive_early_stop_reason": early_stop_reason,
                        "research_budget_used": max(0, completed_count - 1),
                        "total_candidates": completed_count,
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "current_candidate_id": None,
                        "current_job_id": None,
                    }},
                )
                _refresh_campaign_ranking(db, run_id)
                _append_campaign_event(db, run_id, message=early_stop_reason, stage="adaptive_early_stopped")
                return
        if pending is None and method in _ADAPTIVE_METHODS and len(candidates) < int(document.get("total_candidates") or 0):
            if method == PIPELINE_METHOD and not bool(document.get("pipeline_handoff_completed")):
                failed_count = int(document.get("failed_candidates") or 0)
                cancelled_count = int(document.get("cancelled_candidates") or 0)
                expected_lhs = int((document.get("probability_config") or {}).get("full_lhs_candidate_count") or document.get("candidate_count") or 0) + 1
                completed_count = int(document.get("completed_candidates") or 0)
                if failed_count or cancelled_count or completed_count < expected_lhs:
                    message = (
                        ("Automatic Latin Hypercube → CARO handoff requires a complete Temporal model exploration phase: " if temporal_model_scope else "Automatic Latin Hypercube → CARO handoff requires a complete Temporal replay exploration phase: ")
                        + f"completed={completed_count}, failed={failed_count}, cancelled={cancelled_count}, expected={expected_lhs}."
                    )
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"status": "failed", "phase": "pipeline_handoff_failed", "failure_type": "IncompleteExplorationPhase", "failure_message": message, "finished_at": utc_now(), "updated_at": utc_now()}},
                    )
                    _append_campaign_event(db, run_id, message=message, level="error", stage="pipeline_handoff_failed")
                    return
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"pipeline_handoff_completed": True, "phase": "probabilistic_refinement", "updated_at": utc_now()}},
                )
                document["pipeline_handoff_completed"] = True
                _append_campaign_event(
                    db, run_id,
                    message=(f"Temporal LightGBM Latin Hypercube exploration completed with {completed_count} observations. Adaptive CARO started on the same frozen market snapshot." if temporal_model_scope else f"Temporal Latin Hypercube exploration completed with {completed_count} observations. Adaptive CARO started on the same frozen replay."),
                    stage="pipeline_handoff",
                )
            if method == PROBABILITY_METHOD:
                policy = unified_caro_next_mode(document)
                if str(policy.get("mode")) == "space_filling":
                    candidate = propose_unified_space_filling_candidate(document)
                    next_phase = "adaptive_exploration"
                    message = f"Unified CARO proposed Temporal space-filling candidate #{int(candidate.get('candidate_id') or 0)}."
                else:
                    candidate = propose_champion_probability_candidate(document)
                    next_phase = "probabilistic_refinement"
                    message = f"Unified CARO proposed Temporal adaptive candidate #{int(candidate.get('candidate_id') or 0)}."
            else:
                candidate = propose_champion_probability_candidate(document)
                next_phase = "probabilistic_refinement"
                message = f"CARO proposed Temporal candidate #{int(candidate.get('candidate_id') or 0)}."
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$push": {"candidates": bson_value(candidate)}, "$set": {"phase": next_phase, "generated_candidates": len(candidates) + 1, "updated_at": utc_now()}},
            )
            _append_campaign_event(db, run_id, message=message, stage=next_phase, candidate_id=int(candidate.get("candidate_id") or 0))
            continue

        if pending is None:
            failed_count = int(document.get("failed_candidates") or 0)
            cancelled_count = int(document.get("cancelled_candidates") or 0)
            completed_count = int(document.get("completed_candidates") or 0)
            total_count = int(document.get("total_candidates") or len(candidates))
            if failed_count or cancelled_count or completed_count != total_count:
                message = (
                    ("Temporal Model campaign is incomplete and cannot be certified: " if temporal_model_scope else "Temporal Policy campaign is incomplete and cannot be certified: ")
                    + f"completed={completed_count}, failed={failed_count}, cancelled={cancelled_count}, expected={total_count}."
                )
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"status": "failed", "phase": "incomplete_campaign", "failure_type": "IncompleteTuningCampaign", "failure_message": message, "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                )
                _append_campaign_event(db, run_id, message=message, level="error", stage="incomplete_campaign")
                return
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {"status": "completed", "phase": "completed", "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
            )
            _refresh_campaign_ranking(db, run_id)
            _append_campaign_event(db, run_id, message=("Temporal Model campaign completed with every candidate successful." if temporal_model_scope else "Temporal Policy campaign completed with every candidate successful."), stage="completed")
            return

        candidate_id = int(pending["candidate_id"])
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id, "candidates.candidate_id": candidate_id},
            {"$set": {
                "current_candidate_id": candidate_id,
                "phase": (
                    "adaptive_exploration" if str(pending.get("kind") or "") in {"probability_startup", "unified_exploration"}
                    else "adaptive_refinement" if str(pending.get("kind") or "") == "champion_probability"
                    else "running_candidate"
                ),
                "updated_at": utc_now(),
                "current_candidate_progress": 0.0 if temporal_model_scope else None,
                "current_candidate_stage": "Preparing Temporal LightGBM candidate" if temporal_model_scope else None,
                "candidates.$.status": "running",
                "candidates.$.started_at": utc_now(),
            }},
        )
        _append_campaign_event(db, run_id, message=(f"Temporal candidate #{candidate_id} started full LightGBM retraining." if temporal_model_scope else f"Temporal candidate #{candidate_id} started frozen replay."), stage="running_candidate", candidate_id=candidate_id)
        try:
            strategy = get_strategy(db, str(document.get("strategy_profile_id") or ""))
            protocol = _normalized_fold_protocol(document.get("fold_protocol") if isinstance(document.get("fold_protocol"), dict) else None)
            research_fold_count = int(protocol["research_folds"])
            if temporal_model_scope:
                model_snapshot = get_strategy_model_snapshot(db, str(strategy.get("id") or ""))

                def temporal_cancel_requested() -> bool:
                    state = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}, {"stop_requested": 1}) or {}
                    return bool(state.get("stop_requested"))

                def temporal_progress(percent: float, stage: str) -> None:
                    if temporal_cancel_requested():
                        raise TemporalModelTuningCancelled("Temporal Model Tuning cancelled by user.")
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"current_candidate_progress": max(0.0, min(100.0, float(percent))), "current_candidate_stage": str(stage), "updated_at": utc_now()}},
                    )
                if temporal_model_campaign_context is None:
                    temporal_progress(1.0, "Preparing reusable Temporal campaign context")
                    temporal_model_campaign_context = prepare_temporal_model_campaign_context(
                        db, strategy, model_snapshot,
                        fold_count=research_fold_count,
                        progress_callback=temporal_progress,
                        cancel_check=temporal_cancel_requested,
                    )
                    _append_campaign_event(
                        db, run_id,
                        message="Reusable Temporal LightGBM campaign context prepared.",
                        stage="temporal_model_context_ready",
                        candidate_id=candidate_id,
                    )
                else:
                    temporal_progress(18.0, "Reusing Temporal campaign context")
                evaluation = evaluate_temporal_model_candidate(
                    db, strategy, model_snapshot, dict(pending.get("settings") or {}),
                    progress_callback=temporal_progress,
                    cancel_check=temporal_cancel_requested,
                    fold_count=research_fold_count,
                    prepared_context=temporal_model_campaign_context,
                )
            else:
                research_cache_run_id = str(document.get("research_fold_cache_run_id") or "").strip() or None
                if bool(document.get("research_fold_cache_required")) and research_cache_run_id is None:
                    model_snapshot = get_strategy_model_snapshot(db, str(strategy.get("id") or ""))
                    _append_campaign_event(
                        db, run_id,
                        message=f"Building fold-specific Temporal research cache with {research_fold_count} folds before Policy CARO replay.",
                        stage="research_fold_cache_build", candidate_id=candidate_id,
                    )
                    cache_evaluation = evaluate_temporal_model_candidate(
                        db, strategy, model_snapshot, {}, fold_count=research_fold_count
                    )
                    research_cache_run_id = persist_temporal_model_champion_cache(
                        db, tuning_run_id=run_id, candidate_id=0, strategy=strategy, evaluation=cache_evaluation
                    )
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"research_fold_cache_run_id": research_cache_run_id, "updated_at": utc_now()}},
                    )
                    document["research_fold_cache_run_id"] = research_cache_run_id
                    _append_campaign_event(
                        db, run_id,
                        message=f"Fold-specific Temporal research cache ready: {research_fold_count} folds.",
                        stage="research_fold_cache_ready", candidate_id=candidate_id,
                    )
                evaluation = evaluate_temporal_policy_candidate(
                    db, strategy, dict(pending.get("settings") or {}),
                    source_run_id_override=research_cache_run_id,
                )
            summary = dict(evaluation.get("metrics") or {})
            equity_preview = list(evaluation.get("equity_preview") or [])
            is_control = bool(pending.get("is_control"))
            if is_control and method in _ADAPTIVE_METHODS:
                fresh_anchor = {
                    "source": "research_fold_control",
                    "source_temporal_run_id": str(document.get("research_fold_cache_run_id") or document.get("source_temporal_run_id") or ""),
                    "candidate_id": candidate_id,
                    "settings_hash": str(pending.get("settings_hash") or ""),
                    "settings": deepcopy(pending.get("settings") or {}),
                    "metrics": deepcopy(summary),
                }
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {
                        "probability_anchor": bson_value(fresh_anchor),
                        "starting_probability_anchor": bson_value(fresh_anchor),
                        "updated_at": utc_now(),
                    }},
                )
                document["probability_anchor"] = fresh_anchor
                document["starting_probability_anchor"] = fresh_anchor
            champion_gate = (
                champion_gate_evaluation(document, summary)
                if method in _ADAPTIVE_METHODS and not is_control
                else None
            )
            probability_evolution = None
            if method in _ADAPTIVE_METHODS and not is_control:
                probability_evolution = evolve_probability_search(document, dict(pending), summary, champion_gate)
            now = utc_now()
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id, "candidates.candidate_id": candidate_id},
                {"$set": {
                    "candidates.$.status": "completed",
                    "candidates.$.metrics": bson_value(summary),
                    "candidates.$.equity_preview": [],
                    "candidates.$.champion_gate_passed": (bool(champion_gate.get("passed")) if champion_gate else None),
                    "candidates.$.champion_gate": bson_value(champion_gate) if champion_gate else None,
                    "candidates.$.finished_at": now,
                    "candidates.$.raw_results_retained": False,
                    "updated_at": now,
                    "current_candidate_id": None,
                    "current_job_id": None,
                }, "$inc": {"completed_candidates": 1}},
            )
            if probability_evolution is not None:
                probability_update: dict[str, Any] = {
                    "probability_state": bson_value(probability_evolution["state"]),
                    "updated_at": utc_now(),
                }
                next_anchor = probability_evolution.get("probability_anchor")
                if next_anchor is not None:
                    probability_update["probability_anchor"] = bson_value(next_anchor)
                update_document: dict[str, Any] = {"$set": probability_update}
                if next_anchor is not None:
                    update_document["$push"] = {"probability_champion_history": bson_value({
                        "at": utc_now(),
                        "candidate_id": candidate_id,
                        "settings_hash": pending.get("settings_hash"),
                        "metrics": summary,
                    })}
                db[MODEL_TUNING_RUNS_COLLECTION].update_one({"id": run_id}, update_document)
                if next_anchor is not None:
                    _append_campaign_event(db, run_id, message=f"Temporal candidate #{candidate_id} became the new research Champion.", stage="champion_promoted", candidate_id=candidate_id)
            _refresh_campaign_ranking(db, run_id)
            if temporal_model_scope:
                refreshed = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or {}
                if int(refreshed.get("best_candidate_id") if refreshed.get("best_candidate_id") is not None else -1) == candidate_id:
                    try:
                        cache_run_id = persist_temporal_model_champion_cache(
                            db, tuning_run_id=run_id, candidate_id=candidate_id, strategy=strategy, evaluation=evaluation
                        )
                    except Exception as cache_exc:
                        failure_message = _sanitize_tuning_log_line(str(cache_exc))[:500]
                        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                            {"id": run_id},
                            {"$set": {
                                "status": "failed",
                                "phase": "temporal_model_champion_cache_failed",
                                "failure_type": type(cache_exc).__name__,
                                "failure_message": failure_message,
                                "finished_at": utc_now(),
                                "updated_at": utc_now(),
                                "current_candidate_id": None,
                                "current_job_id": None,
                            }},
                        )
                        _append_campaign_event(db, run_id, message=f"Temporal model Champion cache failed after candidate #{candidate_id}: {failure_message}", level="error", stage="temporal_model_champion_cache_failed", candidate_id=candidate_id)
                        return
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {
                            "temporal_model_champion_cache_run_id": cache_run_id,
                            "temporal_model_champion_candidate_id": candidate_id,
                            "current_candidate_progress": 100.0,
                            "current_candidate_stage": "Completed",
                            "updated_at": utc_now(),
                        }},
                    )
                    _append_campaign_event(db, run_id, message=f"Temporal model candidate #{candidate_id} became the cached model Champion.", stage="temporal_model_champion_cached", candidate_id=candidate_id)
            _append_campaign_event(db, run_id, message=f"Temporal candidate #{candidate_id} completed successfully.", stage="candidate_completed", candidate_id=candidate_id)
        except TemporalModelTuningCancelled:
            now = utc_now()
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id, "candidates.candidate_id": candidate_id},
                {"$set": {
                    "candidates.$.status": "cancelled",
                    "candidates.$.finished_at": now,
                    "candidates.$.error": None,
                    "candidates.$.failure_type": None,
                    "candidates.$.failure_message": None,
                    "status": "stopped",
                    "phase": "stopped",
                    "finished_at": now,
                    "updated_at": now,
                    "current_candidate_id": None,
                    "current_job_id": None,
                    "current_candidate_stage": "Cancelled",
                }, "$inc": {"cancelled_candidates": 1}},
            )
            _refresh_campaign_ranking(db, run_id)
            _append_campaign_event(
                db, run_id,
                message=f"Temporal model candidate #{candidate_id} cancelled and campaign stopped by user request.",
                stage="candidate_cancelled", candidate_id=candidate_id,
            )
            return
        except Exception as exc:
            now = utc_now()
            failure_message = _sanitize_tuning_log_line(str(exc))[:500]
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id, "candidates.candidate_id": candidate_id},
                {"$set": {
                    "candidates.$.status": "failed",
                    "candidates.$.finished_at": now,
                    "candidates.$.error": failure_message,
                    "candidates.$.failure_type": type(exc).__name__,
                    "candidates.$.failure_message": failure_message,
                    "candidates.$.diagnostic_log": bson_value(_diagnostic_traceback_lines()),
                    "updated_at": now,
                    "current_candidate_id": None,
                    "current_job_id": None,
                }, "$inc": {"failed_candidates": 1}},
            )
            _refresh_campaign_ranking(db, run_id)
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {"$set": {
                    "status": "failed",
                    "phase": "candidate_failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": failure_message,
                    "finished_at": now,
                    "updated_at": now,
                }},
            )
            _append_campaign_event(db, run_id, message=f"Temporal candidate #{candidate_id} failed: {failure_message}", level="error", stage="candidate_failed", candidate_id=candidate_id)
            return


def run_model_tuning(run_id: str) -> None:
    db = database()
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is not None and str(document.get("tuning_scope") or "") in {TEMPORAL_POLICY_TUNING_SCOPE, TEMPORAL_MODEL_TUNING_SCOPE}:
        _run_temporal_tuning(db, run_id)
        return
    from ..api.routers.jobs import queue_backtest_job  

    try:
        document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
        if document is None:
            return
        initial_phase = "latin_hypercube_exploration" if str(document.get("method") or "") == PIPELINE_METHOD else "running"
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {"$set": {"status": "running", "phase": initial_phase, "started_at": utc_now(), "updated_at": utc_now()}},
        )
        _append_campaign_event(db, run_id, message="Integrated tuning worker started.", stage="running")
        document = _ensure_campaign_market_snapshot(db, document)

        while True:
            document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}) or {}
            if bool(document.get("stop_requested")):
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"status": "stopped", "phase": "stopped", "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                )
                _append_campaign_event(db, run_id, message="Stop request honored after the current safe execution unit.", stage="stopped")
                return

            candidates = list(document.get("candidates") or [])
            pending = next((item for item in candidates if item.get("status") == "pending"), None)
            method = str(document.get("method") or "")
            if pending is None:
                early_stop_reason = _adaptive_early_stop_reason(document)
                if early_stop_reason:
                    completed_count = int(document.get("completed_candidates") or 0)
                    reused_control_count = sum(
                        1 for item in candidates if bool(item.get("is_control")) and bool(item.get("baseline_reused"))
                    )
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {
                            "status": "completed",
                            "phase": "adaptive_early_stopped",
                            "adaptive_early_stopped": True,
                            "adaptive_early_stop_reason": early_stop_reason,
                            "research_budget_used": max(0, completed_count - reused_control_count),
                            "total_candidates": completed_count,
                            "finished_at": utc_now(),
                            "updated_at": utc_now(),
                            "current_candidate_id": None,
                            "current_job_id": None,
                        }},
                    )
                    _refresh_campaign_ranking(db, run_id)
                    _append_campaign_event(db, run_id, message=early_stop_reason, stage="adaptive_early_stopped")
                    return
            if pending is None and method in _ADAPTIVE_METHODS and len(candidates) < int(document.get("total_candidates") or 0):
                if method == PIPELINE_METHOD and not bool(document.get("pipeline_handoff_completed")):
                    failed_count = int(document.get("failed_candidates") or 0)
                    cancelled_count = int(document.get("cancelled_candidates") or 0)
                    expected_lhs = int((document.get("probability_config") or {}).get("full_lhs_candidate_count") or document.get("candidate_count") or 0) + 1
                    completed_count = int(document.get("completed_candidates") or 0)
                    if failed_count or cancelled_count or completed_count < expected_lhs:
                        message = (
                            "Automatic Latin Hypercube → CARO handoff requires a complete exploration phase: "
                            f"completed={completed_count}, failed={failed_count}, cancelled={cancelled_count}, expected={expected_lhs}."
                        )
                        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                            {"id": run_id},
                            {"$set": {"status": "failed", "phase": "pipeline_handoff_failed", "failure_type": "IncompleteExplorationPhase", "failure_message": message, "finished_at": utc_now(), "updated_at": utc_now()}},
                        )
                        _append_campaign_event(db, run_id, message=message, stage="pipeline_handoff_failed")
                        return
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"pipeline_handoff_completed": True, "phase": "probabilistic_refinement", "updated_at": utc_now()}},
                    )
                    document["pipeline_handoff_completed"] = True
                    _append_campaign_event(
                        db, run_id,
                        message=f"Full Latin Hypercube exploration completed with {completed_count} observations. Adaptive CARO started automatically using the same frozen baseline and market-data snapshot.",
                        stage="pipeline_handoff",
                    )
                if method == PROBABILITY_METHOD:
                    policy = unified_caro_next_mode(document)
                    if str(policy.get("mode")) == "space_filling":
                        candidate = propose_unified_space_filling_candidate(document)
                        next_phase = "adaptive_exploration"
                        message = (
                            f"Unified CARO proposed space-filling candidate #{int(candidate.get('candidate_id') or 0)} "
                            f"({policy.get('reason')}; exploration={int(policy.get('exploration_observations') or 0)}, "
                            f"adaptive={int(policy.get('adaptive_observations') or 0)})."
                        )
                    else:
                        candidate = propose_champion_probability_candidate(document)
                        next_phase = "probabilistic_refinement"
                        message = (
                            f"Unified CARO proposed adaptive candidate #{int(candidate.get('candidate_id') or 0)} "
                            f"using {int((candidate.get('proposal') or {}).get('observation_count') or 0)} observations."
                        )
                else:
                    candidate = propose_champion_probability_candidate(document)
                    next_phase = "probabilistic_refinement"
                    message = (
                        f"CARO proposed candidate #{int(candidate.get('candidate_id') or 0)} "
                        f"using {int((candidate.get('proposal') or {}).get('observation_count') or 0)} observations."
                    )
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {
                        "$push": {"candidates": bson_value(candidate)},
                        "$set": {
                            "phase": next_phase,
                            "generated_candidates": len(candidates) + 1,
                            "updated_at": utc_now(),
                        },
                    },
                )
                _append_campaign_event(
                    db, run_id,
                    message=message,
                    stage=next_phase,
                    candidate_id=int(candidate.get("candidate_id") or 0),
                )
                continue

            if pending is None:
                failed_count = int(document.get("failed_candidates") or 0)
                cancelled_count = int(document.get("cancelled_candidates") or 0)
                completed_count = int(document.get("completed_candidates") or 0)
                total_count = int(document.get("total_candidates") or len(candidates))
                if failed_count or cancelled_count or completed_count != total_count:
                    message = (
                        "Campaign is incomplete and cannot be certified or reused: "
                        f"completed={completed_count}, failed={failed_count}, cancelled={cancelled_count}, "
                        f"expected={total_count}. Start a new campaign."
                    )
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"status": "failed", "phase": "incomplete_campaign", "failure_type": "IncompleteTuningCampaign", "failure_message": message, "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                    )
                    _append_campaign_event(db, run_id, message=message, level="error", stage="incomplete_campaign")
                    return
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"status": "completed", "phase": "completed", "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                )
                _append_campaign_event(db, run_id, message="Campaign completed with every candidate successful.", stage="completed")
                return

            candidate_id = int(pending["candidate_id"])
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id, "candidates.candidate_id": candidate_id},
                {
                    "$set": {
                        "current_candidate_id": candidate_id,
                        "phase": (
                            "adaptive_exploration"
                            if str(pending.get("kind") or "") in {"probability_startup", "unified_exploration"}
                            else "adaptive_refinement"
                            if str(pending.get("kind") or "") == "champion_probability"
                            else "running_candidate"
                        ),
                        "updated_at": utc_now(),
                        "candidates.$.status": "running",
                        "candidates.$.started_at": utc_now(),
                    }
                },
            )
            _append_campaign_event(
                db, run_id, message=f"Candidate #{candidate_id} started.",
                stage="running_candidate", candidate_id=candidate_id,
            )
            stop_state = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}, {"stop_requested": 1}) or {}
            if bool(stop_state.get("stop_requested")):
                now = utc_now()
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id, "candidates.candidate_id": candidate_id},
                    {
                        "$set": {
                            "candidates.$.status": "cancelled",
                            "candidates.$.finished_at": now,
                            "status": "stopped",
                            "phase": "stopped",
                            "finished_at": now,
                            "updated_at": now,
                            "current_candidate_id": None,
                            "current_job_id": None,
                        },
                        "$inc": {"cancelled_candidates": 1},
                    },
                )
                _append_campaign_event(
                    db, run_id, message=f"Candidate #{candidate_id} cancelled before execution because Stop was requested.",
                    stage="candidate_cancelled", candidate_id=candidate_id,
                )
                return
            job_id: str | None = None
            try:
                execution_metadata = {
                    "strategy_profile_id": (document.get("source_strategy_profile_id") or document.get("strategy_profile_id")),
                    "strategy_profile_name": (document.get("baseline_execution") or {}).get("strategy_profile_name") or document.get("strategy_profile_name"),
                    "strategy_profile_revision": (document.get("source_strategy_profile_revision") or document.get("strategy_profile_revision")),
                    "strategy_configuration_hash": (document.get("baseline_execution") or {}).get("strategy_configuration_hash") or document.get("strategy_configuration_hash"),
                }
                candidate_request = deepcopy(document.get("execution_request_snapshot") or {})
                expected_signature = str(document.get("expected_market_data_signature_sha256") or "").strip().lower()
                snapshot_id = str(document.get("market_data_snapshot_id") or "").strip().lower()
                candidate_request["expected_market_data_signature_sha256"] = expected_signature or None
                candidate_request["research_market_data_snapshot_id"] = snapshot_id or None
                tuning_scope = str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE)
                if tuning_scope in {ABSOLUTE_UTILITY_TUNING_SCOPE, LEGACY_ABSOLUTE_UTILITY_TUNING_SCOPE}:
                    candidate_settings = dict(pending["settings"])
                    strategy_overrides = {
                        name: candidate_settings[name]
                        for name in _ABSOLUTE_UTILITY_STRATEGY_PARAMETER_NAMES
                        if name in candidate_settings
                    }
                    candidate_request.update(strategy_overrides)
                    frozen_strategy_configuration = deepcopy(
                        document.get("strategy_configuration_snapshot") or {}
                    )
                    if frozen_strategy_configuration:
                        frozen_strategy_configuration.update(strategy_overrides)
                        validated_candidate_configuration = BacktestRequest.model_validate(
                            frozen_strategy_configuration
                        )
                        execution_metadata["strategy_configuration_hash"] = _strategy_configuration_hash(
                            validated_candidate_configuration.model_dump(mode="json")
                        )
                    if tuning_scope == ABSOLUTE_UTILITY_TUNING_SCOPE:
                        model_values_override = deepcopy(document.get("base_model_values") or {})
                        for name in _TUNED_NAMES:
                            if name in candidate_settings:
                                model_values_override[name] = candidate_settings[name]
                    else:
                        model_values_override = dict(
                            document.get("base_model_values")
                            or document.get("frozen_model_values")
                            or {}
                        )
                else:
                    model_values_override = dict(pending["settings"])
                queued = queue_backtest_job(
                    model_values_override=model_values_override,
                    start_thread=False,
                    certify_strategy=False,
                    tuning_run_id=run_id,
                    tuning_candidate_id=candidate_id,
                    execution_request_override=candidate_request,
                    execution_metadata_override=execution_metadata,
                )
                job_id = str(queued["id"])
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id, "candidates.candidate_id": candidate_id},
                    {"$set": {"current_job_id": job_id, "candidates.$.job_id": job_id, "updated_at": utc_now()}},
                )
                _append_campaign_event(
                    db, run_id, message=f"Backtest job {job_id} queued for candidate #{candidate_id}.",
                    stage="backtest_queued", candidate_id=candidate_id, job_id=job_id,
                )
                stop_state = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id}, {"stop_requested": 1}) or {}
                if bool(stop_state.get("stop_requested")):
                    request_job_cancel(job_id, reason=f"Model tuning {run_id} stopped by user.")
                run_job(job_id)
                job = db[JOBS_COLLECTION].find_one({"id": job_id}) or {}
                if str(job.get("status") or "") == "cancelled":
                    _cleanup_job_artifacts(db, job_id)
                    now = utc_now()
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id, "candidates.candidate_id": candidate_id},
                        {
                            "$set": {
                                "candidates.$.status": "cancelled",
                                "candidates.$.finished_at": now,
                                "candidates.$.error": None,
                                "candidates.$.failure_type": None,
                                "candidates.$.failure_message": None,
                                "status": "stopped",
                                "phase": "stopped",
                                "finished_at": now,
                                "updated_at": now,
                                "current_candidate_id": None,
                                "current_job_id": None,
                            },
                            "$inc": {"cancelled_candidates": 1},
                        },
                    )
                    _refresh_campaign_ranking(db, run_id)
                    _append_campaign_event(
                        db, run_id, message=f"Candidate #{candidate_id} cancelled and campaign stopped by user request.",
                        stage="candidate_cancelled", candidate_id=candidate_id, job_id=job_id,
                    )
                    return
                if job.get("status") != "completed":
                    raise RuntimeError("The candidate backtest did not complete successfully.")
                metrics = _find_portfolio_metrics(db, job_id)
                if metrics is None:
                    raise RuntimeError("Portfolio metrics are missing for the tuning candidate.")
                summary = _metric_summary(metrics)
                equity_preview = _candidate_equity_preview(db, job_id)
                actual_signature = str(summary.get("market_data_signature_sha256") or "").strip().lower()
                expected_signature = str(document.get("expected_market_data_signature_sha256") or "").strip().lower()
                is_control = bool(pending.get("is_control"))
                if not actual_signature:
                    raise RuntimeError("MarketDataSignatureMissing: the candidate did not produce a research market-data signature.")
                if expected_signature and actual_signature != expected_signature:
                    raise RuntimeError(
                        f"MarketDataSignatureMismatch: expected {expected_signature}, got {actual_signature}"
                    )
                if not expected_signature:
                    if not is_control:
                        raise RuntimeError(
                            "MarketDataSignatureMissing: a legacy Control rerun is the only candidate allowed to establish a missing campaign signature."
                        )
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {
                            "$set": {
                                "expected_market_data_signature_sha256": actual_signature,
                                "market_data_signature_source": "legacy_control_candidate",
                                "market_data_signature_established_by_candidate_id": candidate_id,
                                "updated_at": utc_now(),
                            }
                        },
                    )
                    document["expected_market_data_signature_sha256"] = actual_signature
                    document["market_data_signature_source"] = "legacy_control_candidate"
                    document["market_data_signature_established_by_candidate_id"] = candidate_id
                    _append_campaign_event(
                        db, run_id,
                        message=f"Control candidate #{candidate_id} established the frozen research market-data signature.",
                        stage="market_data_signature_frozen", candidate_id=candidate_id, job_id=job_id,
                    )

                
                
                
                if (
                    is_control
                    and str(document.get("method") or "") in _ADAPTIVE_METHODS
                    and str((document.get("probability_config") or {}).get("source_mode") or "") in {"standalone", "standalone_unified", "standalone_unified_with_compatible_history", "full_lhs_then_caro"}
                ):
                    fresh_anchor = {
                        "source": "fresh_control",
                        "job_id": job_id,
                        "candidate_id": candidate_id,
                        "settings_hash": str(pending.get("settings_hash") or ""),
                        "settings": deepcopy(pending.get("settings") or {}),
                        "metrics": deepcopy(summary),
                    }
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"probability_anchor": bson_value(fresh_anchor), "updated_at": utc_now()}},
                    )
                    document["probability_anchor"] = fresh_anchor

                champion_gate = (
                    champion_gate_evaluation(document, summary)
                    if str(document.get("method") or "") in _ADAPTIVE_METHODS and not is_control
                    else None
                )
                probability_evolution = None
                if str(document.get("method") or "") in _ADAPTIVE_METHODS and not is_control:
                    probability_evolution = evolve_probability_search(document, {**pending, "job_id": job_id}, summary, champion_gate)

                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id, "candidates.candidate_id": candidate_id},
                    {
                        "$set": {
                            "candidates.$.status": "completed",
                            "candidates.$.metrics": bson_value(summary),
                            "candidates.$.equity_preview": [],
                            "candidates.$.strategy_configuration_hash": job.get("strategy_configuration_hash"),
                            "candidates.$.champion_gate_passed": (bool(champion_gate.get("passed")) if champion_gate else None),
                            "candidates.$.champion_gate": bson_value(champion_gate) if champion_gate else None,
                            "candidates.$.finished_at": utc_now(),
                            "candidates.$.raw_results_retained": False,
                            "updated_at": utc_now(),
                            "current_candidate_id": None,
                            "current_job_id": None,
                        },
                        "$inc": {"completed_candidates": 1},
                    },
                )
                if probability_evolution is not None:
                    probability_update: dict[str, Any] = {
                        "probability_state": bson_value(probability_evolution["state"]),
                        "updated_at": utc_now(),
                    }
                    next_anchor = probability_evolution.get("probability_anchor")
                    if next_anchor is not None:
                        probability_update["probability_anchor"] = bson_value(next_anchor)
                    update_document: dict[str, Any] = {"$set": probability_update}
                    if next_anchor is not None:
                        update_document["$push"] = {
                            "probability_champion_history": bson_value({
                                "at": utc_now(),
                                "candidate_id": candidate_id,
                                "settings_hash": pending.get("settings_hash"),
                                "metrics": summary,
                            })
                        }
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one({"id": run_id}, update_document)
                    if next_anchor is not None:
                        _append_campaign_event(
                            db, run_id,
                            message=(
                                f"Candidate #{candidate_id} became the new research Champion; "
                                f"future CARO proposals will optimize against this observed result."
                            ),
                            stage="champion_promoted", candidate_id=candidate_id, job_id=job_id,
                        )

                _cleanup_job_artifacts(db, job_id)
                _refresh_campaign_ranking(db, run_id)
                _append_campaign_event(
                    db, run_id, message=f"Candidate #{candidate_id} completed successfully.",
                    stage="candidate_completed", candidate_id=candidate_id, job_id=job_id,
                )
            except Exception as exc:
                job_document = db[JOBS_COLLECTION].find_one({"id": job_id}) if job_id else None
                failure_type, failure_message = _infer_failure_details(job_document, exc)
                diagnostic_log = _diagnostic_traceback_lines()
                if job_id:
                    _cleanup_job_artifacts(db, job_id)
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id, "candidates.candidate_id": candidate_id},
                    {
                        "$set": {
                            "candidates.$.status": "failed",
                            "candidates.$.finished_at": utc_now(),
                            "candidates.$.error": _sanitize_tuning_log_line(str(exc))[:500],
                            "candidates.$.failure_type": failure_type,
                            "candidates.$.failure_message": failure_message,
                            "candidates.$.diagnostic_log": bson_value(diagnostic_log),
                            "updated_at": utc_now(),
                            "current_candidate_id": None,
                            "current_job_id": None,
                        },
                        "$inc": {"failed_candidates": 1},
                    },
                )
                _refresh_campaign_ranking(db, run_id)
                _append_campaign_event(
                    db, run_id,
                    message=f"Candidate #{candidate_id} failed: {failure_type}: {failure_message}",
                    level="error", stage="candidate_failed", candidate_id=candidate_id, job_id=job_id,
                )
                reproducibility_failure = (
                    failure_type in {"MarketDataSignatureMismatch", "MarketDataSignatureMissing"}
                    or "MarketDataSignatureMismatch" in failure_message
                    or "MarketDataSignatureMissing" in failure_message
                    or "MarketDataSignatureMismatch" in str(exc)
                    or "MarketDataSignatureMissing" in str(exc)
                )
                control_failure = bool(pending.get("is_control"))
                terminal_type = (
                    failure_type
                    if reproducibility_failure
                    else ("ControlCandidateFailed" if control_failure else "CandidateFailed")
                )
                terminal_phase = (
                    "reproducibility_guard_failed"
                    if reproducibility_failure
                    else ("control_failed" if control_failure else "candidate_failed_terminal")
                )
                terminal_message = failure_message or _sanitize_tuning_log_line(str(exc))[:500]
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"status": "failed", "phase": terminal_phase, "failure_type": terminal_type, "failure_message": terminal_message, "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                )
                _append_campaign_event(
                    db, run_id, message=(
                        f"Campaign invalidated after candidate #{candidate_id} failed. "
                        "No remaining candidate will run and this campaign cannot seed CARO."
                    ),
                    level="error", stage=terminal_phase, candidate_id=candidate_id, job_id=job_id,
                )
                return
    except Exception as exc:
        failure_message = _sanitize_tuning_log_line(str(exc))[:500]
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {"$set": {"status": "failed", "phase": "failed", "failure_type": type(exc).__name__, "failure_message": failure_message, "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
        )
        _append_campaign_event(
            db, run_id, message=f"Campaign worker failed: {type(exc).__name__}: {failure_message}",
            level="error", stage="failed",
        )


def recover_integrated_model_tuning_runs(db: Any) -> int:
    





    documents = list(
        db[MODEL_TUNING_RUNS_COLLECTION].find(
            {
                "status": {"$in": list(_ACTIVE_STATUSES)},
                "$or": [
                    {"execution_mode": "integrated_api_worker"},
                    {"execution_mode": "frozen_temporal_replay"},
                    {"execution_mode": "full_temporal_lightgbm_retrain"},
                    {"execution_mode": {"$exists": False}},
                ],
            }
        )
    )
    invalidated = 0
    for document in documents:
        run_id = str(document.get("id") or "")
        if not run_id:
            continue
        candidates = deepcopy(document.get("candidates") or [])
        cancelled_now = 0
        job_ids: set[str] = set()
        current_job_id = str(document.get("current_job_id") or "").strip()
        if current_job_id:
            job_ids.add(current_job_id)

        now = utc_now()
        for candidate in candidates:
            status = str(candidate.get("status") or "")
            if status not in {"pending", "running"}:
                continue
            job_id = str(candidate.get("job_id") or "").strip()
            if job_id:
                job_ids.add(job_id)
            candidate["status"] = "cancelled"
            candidate["finished_at"] = now
            candidate["failure_type"] = "CampaignRestarted"
            candidate["failure_message"] = (
                "Candidate invalidated because the API restarted before the campaign completed."
            )
            cancelled_now += 1

        for job_id in job_ids:
            db[JOBS_COLLECTION].update_one(
                {"id": job_id, "status": {"$in": ["queued", "running"]}},
                {"$set": {
                    "status": "cancelled",
                    "stage": "Cancelled after API restart",
                    "cancel_requested": True,
                    "cancel_reason": f"Model tuning {run_id} invalidated after API restart.",
                    "finished_at": now,
                    "updated_at": now,
                }},
            )
            _cleanup_job_artifacts(db, job_id)

        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {
                "$set": {
                    "candidates": bson_value(candidates),
                    "status": "stopped",
                    "phase": "invalidated_after_restart",
                    "failure_type": "CampaignRestarted",
                    "failure_message": (
                        "Unfinished tuning campaigns are not resumed after an API restart. "
                        "Start a new campaign; no partial result will be reused."
                    ),
                    "current_candidate_id": None,
                    "current_job_id": None,
                    "finished_at": now,
                    "updated_at": now,
                },
                "$inc": {"cancelled_candidates": cancelled_now, "restart_invalidation_count": 1},
            },
        )
        _append_campaign_event(
            db,
            run_id,
            message="Campaign invalidated after API restart. Partial results will not be resumed or reused.",
            level="error",
            stage="invalidated_after_restart",
        )
        invalidated += 1
    return invalidated


def request_model_tuning_stop(db: Any, run_id: str) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    if str(document.get("status") or "") not in _ACTIVE_STATUSES:
        return public_model_tuning_run(db, document)

    running = any(item.get("status") == "running" for item in document.get("candidates") or [])
    current_job_id = str(document.get("current_job_id") or "").strip()
    now = utc_now()
    if not running:
        updated = db[MODEL_TUNING_RUNS_COLLECTION].find_one_and_update(
            {"id": run_id},
            {"$set": {
                "stop_requested": True,
                "status": "stopped",
                "phase": "stopped",
                "finished_at": now,
                "updated_at": now,
                "current_candidate_id": None,
                "current_job_id": None,
            }},
            return_document=ReturnDocument.AFTER,
        )
        _append_campaign_event(db, run_id, message="Campaign stopped before another candidate started.", stage="stopped")
    else:
        temporal_model_run = str(document.get("tuning_scope") or "") == TEMPORAL_MODEL_TUNING_SCOPE
        cancel_phase = "cancelling_temporal_model_candidate" if temporal_model_run and not current_job_id else "cancelling_active_candidate"
        updated = db[MODEL_TUNING_RUNS_COLLECTION].find_one_and_update(
            {"id": run_id},
            {"$set": {
                "stop_requested": True,
                "status": "stop_requested",
                "phase": cancel_phase,
                "updated_at": now,
            }},
            return_document=ReturnDocument.AFTER,
        )
        if current_job_id:
            request_job_cancel(current_job_id, reason=f"Model tuning {run_id} stopped by user.")
        _append_campaign_event(
            db, run_id,
            message=(
                f"Stop requested. Cancelling active candidate job {current_job_id}."
                if current_job_id else
                "Stop requested. Cancelling the active Temporal LightGBM candidate at the next model checkpoint."
                if temporal_model_run else
                "Stop requested. The active candidate will be cancelled before execution continues."
            ),
            stage=cancel_phase,
            candidate_id=document.get("current_candidate_id"),
            job_id=current_job_id or None,
        )
    return public_model_tuning_run(db, updated or document)


def _format_adopted_strategy_description(
    document: dict[str, Any],
    candidate: dict[str, Any],
    source_strategy: dict[str, Any],
) -> str:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    method_label = ("Latin Hypercube → Adaptive CARO (legacy)" if str(document.get("method") or "") == PIPELINE_METHOD else ("Unified Adaptive CARO" if str(document.get("method") or "") == PROBABILITY_METHOD else "Latin Hypercube"))
    parts = [
        f"{method_label} campaign {document.get('id')} candidate #{int(candidate.get('candidate_id') or 0)}.",
        f"Source: {source_strategy.get('name') or 'Strategy'} rev {int(source_strategy.get('revision') or 1)}.",
        f"Model: {document.get('model_label') or 'LightGBM Utility'}.",
    ]
    ending_capital = _as_finite_float(metrics.get("ending_capital"))
    strategy_return = _as_finite_float(metrics.get("strategy_return"))
    cagr = _as_finite_float(metrics.get("cagr"))
    sharpe = _as_finite_float(metrics.get("sharpe"))
    maximum_drawdown = _as_finite_float(metrics.get("maximum_drawdown"))
    worst_fold_return = _as_finite_float(metrics.get("worst_fold_return"))
    result_parts = []
    if ending_capital is not None:
        result_parts.append(f"capital ${ending_capital:,.2f}")
    if strategy_return is not None:
        result_parts.append(f"return {strategy_return * 100:+.2f}%")
    if cagr is not None:
        result_parts.append(f"CAGR {cagr * 100:+.2f}%")
    if sharpe is not None:
        result_parts.append(f"Sharpe {sharpe:.3f}")
    if maximum_drawdown is not None:
        result_parts.append(f"Max DD {maximum_drawdown * 100:+.2f}%")
    if worst_fold_return is not None:
        result_parts.append(f"Worst Fold {worst_fold_return * 100:+.2f}%")
    if result_parts:
        parts.append("Tuning result: " + "; ".join(result_parts) + ".")
    cutoff = str(document.get("market_data_cutoff_date") or "").strip()
    if cutoff:
        parts.append(f"Market cutoff: {cutoff}.")
    parts.append("Created automatically for Backtest; a successful Backtest moves it to Candidate.")
    return " ".join(parts)[:500]


def adopt_model_tuning_candidate(
    db: Any,
    run_id: str,
    candidate_id: int,
    *,
    reason: str | None,
    actor_email: str | None,
) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    candidate = next(
        (item for item in document.get("candidates") or [] if int(item.get("candidate_id") if item.get("candidate_id") is not None else -1) == int(candidate_id)),
        None,
    )
    if candidate is None or candidate.get("status") != "completed":
        raise ModelTuningConflict("Only a completed tuning candidate can be adopted.")
    source_strategy = get_strategy(db, str(document["strategy_profile_id"]))
    if str(document.get("tuning_scope") or "") == TEMPORAL_MODEL_TUNING_SCOPE:
        best_candidate_id = int(document.get("best_candidate_id") if document.get("best_candidate_id") is not None else -1)
        if int(candidate_id) != best_candidate_id:
            raise ModelTuningConflict("Temporal Model Tuning can materialize only the final ranked model Champion so its frozen prediction cache remains exact.")
        if bool(candidate.get("is_control")):
            control = get_strategy_control(db)
            select_model_tuning_strategy(
                db,
                str(source_strategy["id"]),
                expected_control_revision=int(control["revision"]),
                note=(reason or f"Keep TEMPORAL model Control from {run_id}.").strip(),
                actor_email=actor_email,
            )
            return {
                "strategy": get_strategy(db, str(source_strategy["id"])),
                "candidate_id": int(candidate_id),
                "derived_strategy_created": False,
                "source_strategy_preserved": True,
                "ready_for_backtest": False,
                "ready_for_model_tuning": True,
                "recommended_tuning_target": TEMPORAL_POLICY_TUNING_SCOPE,
                "auto_candidate_after_backtest": False,
            }
        cache_run_id = str(document.get("temporal_model_champion_cache_run_id") or "").strip()
        cache_candidate_id = int(document.get("temporal_model_champion_candidate_id") if document.get("temporal_model_champion_candidate_id") is not None else -1)
        if not cache_run_id or cache_candidate_id != int(candidate_id):
            raise ModelTuningConflict("The final Temporal model Champion prediction cache is unavailable. Re-run the Temporal Model Tuning campaign before materialization.")
        from .temporal_intelligence import materialize_temporal_intelligence_strategy
        materialized = materialize_temporal_intelligence_strategy(db, cache_run_id, actor_email=actor_email)
        updated_strategy = materialized["strategy"]
        control = get_strategy_control(db)
        select_model_tuning_strategy(
            db,
            str(updated_strategy["id"]),
            expected_control_revision=int(control["revision"]),
            note=(reason or f"Use Temporal Model tuning Champion #{int(candidate_id)} from {run_id} for Policy Tuning.").strip(),
            actor_email=actor_email,
        )
        updated_strategy = get_strategy(db, str(updated_strategy["id"]))
        adopted_at = utc_now()
        adoption_entry = {
            "candidate_id": int(candidate_id),
            "strategy_id": str(updated_strategy.get("id") or ""),
            "strategy_name": str(updated_strategy.get("name") or "TEMPORAL Strategy"),
            "at": adopted_at,
            "by": (actor_email or "").strip().lower() or None,
        }
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {
                "$set": {
                    "adopted_candidate_id": int(candidate_id),
                    "adopted_strategy_id": str(updated_strategy.get("id") or ""),
                    "derived_strategy_created": bool(materialized.get("created")),
                    "adopted_at": adopted_at,
                    "adopted_by": (actor_email or "").strip().lower() or None,
                    "updated_at": adopted_at,
                },
                "$push": {"adoption_history": bson_value(adoption_entry)},
            },
        )
        return {
            "strategy": updated_strategy,
            "candidate_id": int(candidate_id),
            "derived_strategy_created": bool(materialized.get("created")),
            "source_strategy_preserved": True,
            "ready_for_backtest": False,
            "ready_for_model_tuning": True,
            "recommended_tuning_target": TEMPORAL_POLICY_TUNING_SCOPE,
            "auto_candidate_after_backtest": False,
        }
    if str(document.get("tuning_scope") or "") == TEMPORAL_POLICY_TUNING_SCOPE:
        adoption_note = (reason or f"Use Temporal Policy tuning candidate #{int(candidate_id)} from {run_id}.").strip()
        method_tag = ("LHS-CARO" if str(document.get("method") or "") == PIPELINE_METHOD else ("CARO" if str(document.get("method") or "") == PROBABILITY_METHOD else "LHS"))
        suffix = f" Temporal {method_tag} C{int(candidate_id)} {str(run_id)[-8:]}"
        source_name = str(source_strategy.get("name") or "TEMPORAL Strategy")
        derived_name = f"{source_name[:max(3, 120 - len(suffix))]}{suffix}"
        metrics = dict(candidate.get("metrics") or {})
        policy_snapshot = derived_temporal_policy_snapshot(
            source_strategy,
            tuning_run_id=run_id,
            candidate_id=int(candidate_id),
            settings=dict(candidate.get("settings") or {}),
            metrics=metrics,
        )
        description = (
            f"Derived from TEMPORAL Strategy {source_strategy.get('id')} by Model Tuning {run_id}, "
            f"candidate #{int(candidate_id)}. {adoption_note}"
        )
        updated_strategy = create_tuned_temporal_strategy(
            db,
            str(source_strategy["id"]),
            name=derived_name,
            description=description,
            policy_snapshot=policy_snapshot,
            tuning_run_id=run_id,
            tuning_candidate_id=int(candidate_id),
            tuning_metrics=metrics,
            actor_email=actor_email,
        )
        control = get_strategy_control(db)
        select_model_tuning_strategy(
            db,
            str(updated_strategy["id"]),
            expected_control_revision=int(control["revision"]),
            note=adoption_note,
            actor_email=actor_email,
        )
        updated_strategy = get_strategy(db, str(updated_strategy["id"]))
        adopted_at = utc_now()
        adoption_entry = {
            "candidate_id": int(candidate_id),
            "strategy_id": str(updated_strategy.get("id") or ""),
            "strategy_name": str(updated_strategy.get("name") or derived_name),
            "at": adopted_at,
            "by": (actor_email or "").strip().lower() or None,
        }
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {
                "$set": {
                    "adopted_candidate_id": int(candidate_id),
                    "adopted_strategy_id": str(updated_strategy.get("id") or ""),
                    "derived_strategy_created": True,
                    "adopted_at": adopted_at,
                    "adopted_by": (actor_email or "").strip().lower() or None,
                    "updated_at": adopted_at,
                },
                "$push": {"adoption_history": bson_value(adoption_entry)},
            },
        )
        return {
            "strategy": updated_strategy,
            "candidate_id": int(candidate_id),
            "derived_strategy_created": True,
            "source_strategy_preserved": True,
            "ready_for_backtest": False,
            "ready_for_model_tuning": True,
            "auto_candidate_after_backtest": False,
        }
    frozen_request = document.get("execution_request_snapshot") if isinstance(document.get("execution_request_snapshot"), dict) else {}
    frozen_configuration = None
    if frozen_request:
        configuration_payload = {
            name: deepcopy(frozen_request[name])
            for name in BacktestRequest.model_fields
            if name in frozen_request
        }
        try:
            frozen_configuration = BacktestRequest.model_validate(configuration_payload)
        except Exception as exc:
            if int(source_strategy.get("revision") or 0) != int(document.get("strategy_profile_revision") or -1):
                raise ModelTuningConflict(
                    "The historical tuning source changed and its frozen Strategy configuration cannot be reconstructed safely."
                ) from exc
    elif int(source_strategy.get("revision") or 0) != int(document.get("strategy_profile_revision") or -1):
        raise ModelTuningConflict(
            "The historical tuning source changed and this legacy campaign has no frozen Strategy configuration snapshot."
        )

    adoption_note = (reason or f"Use tuning candidate #{int(candidate_id)} from {run_id} in Backtest.").strip()
    method_tag = ("LHS-CARO" if str(document.get("method") or "") == PIPELINE_METHOD else ("CARO" if str(document.get("method") or "") == PROBABILITY_METHOD else "LHS"))
    suffix = f" {method_tag} C{int(candidate_id)} {str(run_id)[-8:]}"
    source_name = str(source_strategy.get("name") or "Strategy")
    derived_name = f"{source_name[:max(3, 120 - len(suffix))]}{suffix}"
    description_source = {
        **source_strategy,
        "name": document.get("strategy_profile_name") or source_strategy.get("name"),
        "revision": int(document.get("strategy_profile_revision") or source_strategy.get("revision") or 1),
    }
    description = _format_adopted_strategy_description(document, candidate, description_source)
    created = create_strategy(
        db,
        name=derived_name,
        description=description,
        clone_from_strategy_id=str(source_strategy["id"]),
        actor_email=actor_email,
    )
    working_strategy = created
    if frozen_configuration is not None:
        working_strategy = update_strategy(
            db,
            str(created["id"]),
            configuration=frozen_configuration,
            name=derived_name,
            description=description,
            note=f"Restore frozen Strategy configuration from tuning campaign {run_id}.",
            expected_revision=int(created["revision"]),
            actor_email=actor_email,
        )
    tuning_scope = str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE)
    if tuning_scope in {ABSOLUTE_UTILITY_TUNING_SCOPE, LEGACY_ABSOLUTE_UTILITY_TUNING_SCOPE}:
        candidate_settings = dict(candidate["settings"])
        strategy_overrides = {
            name: candidate_settings[name]
            for name in _ABSOLUTE_UTILITY_STRATEGY_PARAMETER_NAMES
            if name in candidate_settings
        }
        base_configuration = frozen_configuration
        if base_configuration is None:
            raw_configuration = working_strategy.get("configuration") if isinstance(working_strategy.get("configuration"), dict) else {}
            base_configuration = BacktestRequest.model_validate(raw_configuration)
        candidate_configuration = base_configuration.model_copy(update=strategy_overrides)
        updated_strategy = update_strategy(
            db,
            str(working_strategy["id"]),
            configuration=candidate_configuration,
            name=derived_name,
            description=description,
            note=adoption_note,
            expected_revision=int(working_strategy["revision"]),
            actor_email=actor_email,
        )
        if tuning_scope == ABSOLUTE_UTILITY_TUNING_SCOPE:
            model_values = deepcopy(document.get("base_model_values") or {})
            for name in _TUNED_NAMES:
                if name in candidate_settings:
                    model_values[name] = candidate_settings[name]
            updated_strategy = update_strategy_model(
                db,
                str(updated_strategy["id"]),
                model_family=TUNING_MODEL_FAMILY,
                values=model_values,
                note=adoption_note,
                expected_strategy_revision=int(updated_strategy["revision"]),
                actor_email=actor_email,
            )
    else:
        updated_strategy = update_strategy_model(
            db,
            str(working_strategy["id"]),
            model_family=TUNING_MODEL_FAMILY,
            values=dict(candidate["settings"]),
            note=adoption_note,
            expected_strategy_revision=int(working_strategy["revision"]),
            actor_email=actor_email,
        )
    updated_strategy = prepare_strategy_for_backtest_candidate(
        db,
        str(updated_strategy["id"]),
        expected_strategy_revision=int(updated_strategy["revision"]),
        tuning_run_id=run_id,
        tuning_candidate_id=int(candidate_id),
        tuning_metrics=dict(candidate.get("metrics") or {}),
        actor_email=actor_email,
    )
    control = get_strategy_control(db)
    select_research_strategy(
        db,
        str(updated_strategy["id"]),
        expected_control_revision=int(control["revision"]),
        note=adoption_note,
        actor_email=actor_email,
    )
    updated_strategy = get_strategy(db, str(updated_strategy["id"]))

    adopted_at = utc_now()
    adoption_entry = {
        "candidate_id": int(candidate_id),
        "strategy_id": str(updated_strategy.get("id") or ""),
        "strategy_name": str(updated_strategy.get("name") or derived_name),
        "at": adopted_at,
        "by": (actor_email or "").strip().lower() or None,
    }
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {
            "$set": {
                "adopted_candidate_id": int(candidate_id),
                "adopted_strategy_id": str(updated_strategy.get("id") or ""),
                "derived_strategy_created": True,
                "adopted_at": adopted_at,
                "adopted_by": (actor_email or "").strip().lower() or None,
                "updated_at": adopted_at,
            },
            "$push": {"adoption_history": bson_value(adoption_entry)},
        },
    )
    return {
        "strategy": updated_strategy,
        "candidate_id": int(candidate_id),
        "derived_strategy_created": True,
        "source_strategy_preserved": True,
        "ready_for_backtest": True,
        "auto_candidate_after_backtest": True,
    }

def _public_candidate(
    candidate: dict[str, Any],
    current_jobs: dict[str, dict[str, Any]] | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": int(candidate.get("candidate_id") or 0),
        "kind": str(candidate.get("kind") or "latin_hypercube"),
        "is_control": bool(candidate.get("is_control")),
        "settings": deepcopy(candidate.get("settings") or {}),
        "settings_hash": str(candidate.get("settings_hash") or ""),
        "status": str(candidate.get("status") or "pending"),
        "rank": candidate.get("rank"),
        "metrics": deepcopy(candidate.get("metrics") or None),
        "champion_gate_passed": candidate.get("champion_gate_passed"),
        "champion_gate": deepcopy(candidate.get("champion_gate") or None),
        "proposal": deepcopy(candidate.get("proposal") or None),
        "job_id": candidate.get("job_id"),
        "source_job_id": candidate.get("source_job_id"),
        "source_temporal_run_id": candidate.get("source_temporal_run_id"),
        "baseline_reused": bool(candidate.get("baseline_reused")),
        "worker_id": candidate.get("worker_id"),
        "worker_cpu_count": candidate.get("worker_cpu_count"),
        "worker_concurrency": candidate.get("worker_concurrency"),
        "runtime_thread_limit": candidate.get("runtime_thread_limit"),
        "retry_count": int(candidate.get("retry_count") or 0),
        "started_at": candidate.get("started_at"),
        "finished_at": candidate.get("finished_at"),
        "error": candidate.get("error"),
        "failure_type": candidate.get("failure_type"),
        "failure_message": candidate.get("failure_message"),
        "has_diagnostic_log": bool(candidate.get("diagnostic_log") or candidate.get("job_id")),
        "validation": ({
            "processing_id": validation.get("id"),
            "fold_count": validation.get("validation_fold_count"),
            "passed": validation.get("validation_passed"),
            "gate": deepcopy(validation.get("validation_gate") or None),
            "completed_at": validation.get("validation_completed_at"),
            "strategy_profile_id": validation.get("strategy_profile_id"),
        } if validation else None),
        "certification": ({
            "processing_id": validation.get("certification_processing_id"),
            "fold_count": validation.get("certification_fold_count"),
            "passed": validation.get("certification_passed"),
            "gate": deepcopy(validation.get("certification_gate") or None),
            "completed_at": validation.get("certification_completed_at"),
        } if validation and validation.get("certification_completed_at") else None),
    }
    job_id = str(candidate.get("job_id") or "")
    current_job = (current_jobs or {}).get(job_id) if job_id else None
    if current_job is not None:
        payload["job_progress"] = float(current_job.get("progress") or 0.0)
        payload["job_stage"] = str(current_job.get("stage") or "Running analysis")
    return bson_value(payload)


def public_model_tuning_run(db: Any, document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    raw_candidates = _rank_candidates(list(document.get("candidates") or []))
    ranked_public_candidates = sorted((item for item in raw_candidates if item.get("rank") is not None), key=lambda item: int(item["rank"]))
    public_best = ranked_public_candidates[0] if ranked_public_candidates else None
    public_best_exploratory = next((item for item in ranked_public_candidates if not bool(item.get("is_control"))), None)
    public_best_champion = next((item for item in ranked_public_candidates if item.get("champion_gate_passed") is True), None)
    active_job_ids = [str(item.get("job_id") or "") for item in raw_candidates if item.get("status") == "running" and item.get("job_id")]
    current_jobs: dict[str, dict[str, Any]] = {}
    if active_job_ids:
        for job in db[JOBS_COLLECTION].find(
            {"id": {"$in": active_job_ids}},
            {"_id": 0, "id": 1, "status": 1, "stage": 1, "progress": 1},
        ):
            current_jobs[str(job.get("id") or "")] = job
    validation_map: dict[int, dict[str, Any]] = {}
    run_id = str(document.get("id") or "")
    if run_id:
        for row in db[MODEL_TUNING_VALIDATIONS_COLLECTION].find(
            {"tuning_run_id": run_id},
            {"_id": 0, "candidate_id": 1, "id": 1, "validation_fold_count": 1, "validation_passed": 1,
             "validation_gate": 1, "validation_completed_at": 1, "strategy_profile_id": 1,
             "certification_processing_id": 1, "certification_fold_count": 1, "certification_passed": 1,
             "certification_gate": 1, "certification_completed_at": 1},
        ):
            if row.get("candidate_id") is not None:
                validation_map[int(row["candidate_id"])] = row
    candidates = [
        _public_candidate(item, current_jobs, validation_map.get(int(item.get("candidate_id") or 0)))
        for item in raw_candidates
    ]
    total = max(1, int(document.get("total_candidates") or len(candidates) or 1))
    reused_control_count = sum(
        1 for item in raw_candidates if bool(item.get("is_control")) and bool(item.get("baseline_reused"))
    )
    research_total = max(1, total - reused_control_count)
    research_completed = max(
        0, int(document.get("completed_candidates") or 0) - reused_control_count,
    )
    completed = (
        research_completed
        + int(document.get("failed_candidates") or 0)
        + int(document.get("cancelled_candidates") or 0)
    )
    fractional_active = sum(float(job.get("progress") or 0.0) / 100.0 for job in current_jobs.values())
    if not current_jobs and str(document.get("tuning_scope") or "") == TEMPORAL_MODEL_TUNING_SCOPE and document.get("current_candidate_id") is not None:
        fractional_active = max(fractional_active, max(0.0, min(1.0, float(document.get("current_candidate_progress") or 0.0) / 100.0)))
    progress = min(100.0, 100.0 * (completed + fractional_active) / research_total)
    if current_jobs and str(document.get("status") or "") in _ACTIVE_STATUSES:
        progress = min(99.9, progress)
    active_candidate_ids = [int(item.get("candidate_id") or 0) for item in raw_candidates if item.get("status") == "running"]
    return bson_value({
        "id": document.get("id"),
        "schema_version": int(document.get("schema_version") or TUNING_SCHEMA_VERSION),
        "status": str(document.get("status") or "queued"),
        "phase": str(document.get("phase") or "queued"),
        "method": str(document.get("method") or TUNING_METHOD),
        "tuning_scope": str(document.get("tuning_scope") or MODEL_PARAMETER_TUNING_SCOPE),
        "tuning_scope_label": str(document.get("tuning_scope_label") or "LightGBM model parameters"),
        "tuning_scope_description": str(document.get("tuning_scope_description") or ""),
        "execution_mode": str(document.get("execution_mode") or "integrated_api_worker"),
        "model_family": str(document.get("model_family") or TUNING_MODEL_FAMILY),
        "model_label": str(document.get("model_label") or "LightGBM Utility"),
        "candidate_count": int(document.get("candidate_count") or 0),
        "caro_candidate_count": (int(document.get("caro_candidate_count")) if document.get("caro_candidate_count") is not None else None),
        "pipeline_mode": document.get("pipeline_mode"),
        "pipeline_handoff_completed": bool(document.get("pipeline_handoff_completed")),
        "total_candidates": total,
        "research_total_candidates": research_total,
        "research_completed_candidates": research_completed,
        "generated_candidates": int(document.get("generated_candidates") or len(candidates)),
        "completed_candidates": int(document.get("completed_candidates") or 0),
        "failed_candidates": int(document.get("failed_candidates") or 0),
        "cancelled_candidates": int(document.get("cancelled_candidates") or 0),
        "progress": progress,
        "seed": int(document.get("seed") or DEFAULT_SEED),
        "fold_protocol": deepcopy(document.get("fold_protocol") or {
            "research_folds": DEFAULT_RESEARCH_FOLDS,
            "validation_folds": DEFAULT_VALIDATION_FOLDS,
            "certification_folds": DEFAULT_CERTIFICATION_FOLDS,
        }),
        "search_space": deepcopy(document.get("search_space") or []),
        "tuned_parameters": deepcopy(document.get("tuned_parameters") or []),
        "tuned_model_parameters": deepcopy(document.get("tuned_model_parameters") or []),
        "tuned_strategy_parameters": deepcopy(document.get("tuned_strategy_parameters") or []),
        "probability_config": deepcopy(document.get("probability_config") or {}),
        "probability_anchor": deepcopy(document.get("probability_anchor") or None),
        "starting_probability_anchor": deepcopy(document.get("starting_probability_anchor") or None),
        "probability_state": deepcopy(document.get("probability_state") or None),
        "probability_champion_history": deepcopy(document.get("probability_champion_history") or []),
        "source_tuning_run_id": document.get("source_tuning_run_id"),
        "source_temporal_run_id": document.get("source_temporal_run_id"),
        "imported_observation_count": int(document.get("imported_observation_count") or 0),
        "market_data_cutoff_date": document.get("market_data_cutoff_date"),
        "research_snapshot_cutoff": document.get("market_data_cutoff_date"),
        "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
        "market_data_snapshot_id": document.get("market_data_snapshot_id"),
        "execution_context_hash": document.get("execution_context_hash"),
        "adoption_context_compatible": bool(document.get("adoption_context_compatible", True)),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": int(document.get("strategy_profile_revision") or 0),
        "strategy_profile_status": document.get("strategy_profile_status"),
        "tuning_target_source": document.get("tuning_target_source"),
        "created_at": document.get("created_at"),
        "created_by": document.get("created_by"),
        "explicit_start_confirmation": bool(document.get("explicit_start_confirmation")),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
        "failure_type": document.get("failure_type"),
        "failure_message": document.get("failure_message"),
        "has_campaign_log": bool(document.get("event_log") or document.get("failure_message")),
        "stop_requested": bool(document.get("stop_requested")),
        "active_candidate_ids": active_candidate_ids,
        "active_job_ids": active_job_ids,
        "current_candidate_id": (active_candidate_ids[0] if active_candidate_ids else document.get("current_candidate_id")),
        "current_job_id": (active_job_ids[0] if active_job_ids else document.get("current_job_id")),
        "current_candidate_progress": document.get("current_candidate_progress"),
        "current_candidate_stage": document.get("current_candidate_stage"),
        "temporal_model_champion_cache_run_id": document.get("temporal_model_champion_cache_run_id"),
        "temporal_model_champion_candidate_id": document.get("temporal_model_champion_candidate_id"),
        "best_candidate_id": int(public_best["candidate_id"]) if public_best is not None else None,
        "best_exploratory_candidate_id": int(public_best_exploratory["candidate_id"]) if public_best_exploratory is not None else None,
        "best_champion_beating_candidate_id": int(public_best_champion["candidate_id"]) if public_best_champion is not None else None,
        "control_candidate_id": int(document["control_candidate_id"]) if document.get("control_candidate_id") is not None else None,
        "validated_candidate_id": document.get("validated_candidate_id"),
        "validation_processing_id": document.get("validation_processing_id"),
        "validation_strategy_id": document.get("validation_strategy_id"),
        "certified_candidate_id": document.get("certified_candidate_id"),
        "certification_processing_id": document.get("certification_processing_id"),
        "adopted_candidate_id": document.get("adopted_candidate_id"),
        "adopted_strategy_id": document.get("adopted_strategy_id"),
        "adoption_history": deepcopy(document.get("adoption_history") or []),
        "baseline_execution": deepcopy(document.get("baseline_execution") or None),
        "candidates": candidates,
    })


def get_model_tuning_run(db: Any, run_id: str) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    return public_model_tuning_run(db, document) or {}


def get_latest_model_tuning_run(db: Any) -> dict[str, Any] | None:
    try:
        strategy, _, _ = _tuning_target_strategy(db)
        strategy_id = str(strategy["id"])
    except Exception:
        return None
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"strategy_profile_id": strategy_id},
        sort=[("created_at", -1)],
    )
    return public_model_tuning_run(db, document)
