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
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
    bson_value,
    utc_now,
)
from .jobs import _redact_sensitive_text, run_job
from .model_research import model_values_from_snapshot
from .model_tuning_probability import PROBABILITY_MODEL, champion_gate_evaluation, propose_champion_probability_candidate
from .strategy_lab import (
    get_research_strategy_context,
    get_research_strategy_model_snapshot,
    update_strategy_model,
)

TUNING_METHOD = "latin_hypercube"
PROBABILITY_METHOD = "champion_probability"
TUNING_MODEL_FAMILY = "lightgbm_utility"
TUNING_SCHEMA_VERSION = 2
DEFAULT_CANDIDATE_COUNT = 20
DEFAULT_SEED = 42

# Search ranges are server-owned research metadata. They are returned only through
# the admin tuning API and are never embedded in the public frontend source.
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


def tuning_catalog() -> dict[str, Any]:
    return {
        "schema_version": TUNING_SCHEMA_VERSION,
        "method": TUNING_METHOD,
        "methods": [
            {
                "id": TUNING_METHOD,
                "label": "Latin Hypercube",
                "description": "Static space-filling exploration. Candidates are evaluated against one immutable historical execution snapshot.",
            },
            {
                "id": PROBABILITY_METHOD,
                "label": "CARO Probability",
                "description": "Optional champion-anchored probabilistic search. It can start independently or reuse a completed Latin Hypercube campaign as prior observations, then proposes adaptive candidates one by one.",
            },
        ],
        "model_family": TUNING_MODEL_FAMILY,
        "model_label": "LightGBM Utility",
        "default_candidate_count": DEFAULT_CANDIDATE_COUNT,
        "candidate_count_min": 4,
        "candidate_count_max": 60,
        "default_seed": DEFAULT_SEED,
        "control_candidate_included": True,
        "control_execution_mode": "fresh_rerun",
        "baseline_execution_required": True,
        "campaign_export_available": True,
        "validation": "chronological_walk_forward",
        "selection_metric": "risk_adjusted_compound_score",
        "eligibility_gate": "all_walk_forward_folds_positive",
        "search_space": [dict(item) for item in _SEARCH_SPACE],
        "raw_artifacts": "summary_only",
        "adoption_requires_final_backtest": True,
        "dedicated_worker": False,
        "execution_mode": "integrated_api_worker",
        "market_data_access": "database_only",
        "prior_campaign_reuse": True,
        "reproducibility_guard": "frozen_execution_snapshot_and_market_data_signature",
        "restart_recovery": "rerun_current_candidate_from_frozen_snapshot",
        "probability": {
            "label": "CARO Probability",
            "probability_model": PROBABILITY_MODEL,
            "default_startup_trials": 8,
            "default_min_capital_improvement": 0.03,
            "default_sharpe_tolerance": 0.05,
            "default_drawdown_tolerance": 0.03,
            "default_min_worst_fold_return": 0.0,
            "default_candidate_pool_size": 2048,
            "default_exploration_weight": 0.15,
            "interpretation": "Estimated probability of outperforming the research Champion under the validation protocol; not a probability of future market profit.",
        },
    }


def _settings_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_value(spec: dict[str, Any], unit_value: float) -> Any:
    low = float(spec["min"])
    high = float(spec["max"])
    value = low + float(unit_value) * (high - low)
    if spec["type"] == "integer":
        return int(round(value))
    return round(value, int(spec.get("precision") or 8))


def generate_latin_hypercube_candidates(
    base_values: dict[str, Any],
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Return control + unique LHS configurations without mutating the Strategy."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    if not base_values:
        raise ValueError("A LightGBM Strategy model snapshot is required.")

    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": 0,
            "kind": "control",
            "is_control": True,
            "settings": deepcopy(base_values),
            "settings_hash": _settings_hash(base_values),
            "status": "pending",
        }
    ]
    seen = {candidates[0]["settings_hash"]}
    def add_points(points: Any) -> None:
        for point in points:
            if len(candidates) >= candidate_count + 1:
                return
            values = deepcopy(base_values)
            for spec, unit_value in zip(_SEARCH_SPACE, point, strict=True):
                values[spec["name"]] = _sample_value(spec, float(unit_value))

            depth = int(values["max_depth"])
            if depth > 0:
                values["num_leaves"] = min(int(values["num_leaves"]), 2 ** depth)
            values["num_leaves"] = max(2, int(values["num_leaves"]))

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

    # The primary design contains exactly candidate_count points so each tuned
    # dimension is stratified into candidate_count equally probable intervals.
    add_points(qmc.LatinHypercube(d=len(_SEARCH_SPACE), seed=seed).random(n=candidate_count))

    # Integer rounding plus the depth/leaves structural constraint can, rarely,
    # collapse two points into the same full configuration. Fill only those gaps
    # with deterministic auxiliary LHS batches while preserving uniqueness.
    attempt = 1
    while len(candidates) < candidate_count + 1 and attempt <= 8:
        missing = candidate_count + 1 - len(candidates)
        auxiliary = qmc.LatinHypercube(d=len(_SEARCH_SPACE), seed=seed + attempt)
        add_points(auxiliary.random(n=max(4, missing)))
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
        "benchmark_ending_capital": float(metrics.get("buy_hold_ending_capital") or 0.0),
        "market_data_signature_sha256": metrics.get("market_data_signature_sha256"),
        "market_data_last_timestamp": market_data_last_timestamp,
        "folds": fold_rows,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": eligible,
    }


def _cleanup_job_artifacts(db: Any, job_id: str) -> None:
    """Keep tuning summaries compact; candidate jobs are not certification runs."""
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
    eligible.sort(
        key=lambda item: (
            float(item["metrics"].get("risk_adjusted_compound_score") or -math.inf),
            float(item["metrics"].get("ending_capital") or -math.inf),
            float(item["metrics"].get("sharpe") or -math.inf),
        ),
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
    """Return completed normal backtests compatible with the currently saved LightGBM Strategy snapshot."""
    _, strategy = get_research_strategy_context(db)
    model_snapshot = get_research_strategy_model_snapshot(db)
    if str(model_snapshot.get("family") or "") != TUNING_MODEL_FAMILY:
        return []

    query = {
        "status": "completed",
        "internal_job": {"$ne": True},
        "strategy_profile_id": str(strategy["id"]),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "research_model_family": TUNING_MODEL_FAMILY,
        "research_model_settings_hash": model_snapshot.get("settings_hash"),
    }
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
                }
            )
        )
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



def _execution_request_context_hash(request_payload: dict[str, Any]) -> str:
    """Hash the immutable research context while ignoring the model hyperparameter payload."""
    context = deepcopy(request_payload)
    context.pop("research_model_settings", None)
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
    request_snapshot = deepcopy(job["request"])
    last_timestamp = summary.get("market_data_last_timestamp")
    cutoff_date = str(last_timestamp)[:10] if last_timestamp else None
    if cutoff_date:
        # A tuning campaign must not move its right-hand time boundary while it runs.
        # Freezing both fields makes all candidates consume the same historical sessions.
        request_snapshot["end_date"] = cutoff_date
        request_snapshot["analysis_end_date"] = cutoff_date
    # Optimization is a pure replay over MongoDB. It must never download, refresh
    # or backfill market data, even when the baseline was a normal backtest that
    # was allowed to bootstrap a completely missing asset before analysis.
    request_snapshot["research_market_data_mode"] = "database_only"
    return {
        "job_id": str(job_id),
        "request": bson_value(request_snapshot),
        "context_hash": _execution_request_context_hash(request_snapshot),
        "market_data_signature_sha256": summary.get("market_data_signature_sha256"),
        "market_data_cutoff_date": cutoff_date,
        "strategy_profile_id": job.get("strategy_profile_id"),
        "strategy_profile_name": job.get("strategy_profile_name"),
        "strategy_profile_revision": job.get("strategy_profile_revision"),
        "strategy_configuration_hash": job.get("strategy_configuration_hash"),
        "model_family": job.get("research_model_family"),
        "model_label": job.get("research_model_label") or "LightGBM Utility",
    }


def list_model_tuning_sources(db: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return completed Latin Hypercube campaigns that can seed CARO without rerunning them."""
    documents = list(
        db[MODEL_TUNING_RUNS_COLLECTION]
        .find(
            {"status": "completed", "method": TUNING_METHOD},
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
            },
        )
        .sort("finished_at", -1)
        .limit(max(1, min(int(limit), 100)))
    )
    result: list[dict[str, Any]] = []
    for document in documents:
        observations = [
            item for item in document.get("candidates") or []
            if item.get("status") == "completed"
            and isinstance(item.get("settings"), dict)
            and isinstance(item.get("metrics"), dict)
        ]
        if len(observations) < 4 or list(document.get("search_space") or []) != [dict(item) for item in _SEARCH_SPACE]:
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
        # Only expose sources whose immutable baseline request is still present.
        try:
            context = _frozen_execution_context_from_job(db, baseline_job_id)
        except ModelTuningConflict:
            continue
        result.append(
            bson_value({
                "run_id": document.get("id"),
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
                "market_data_cutoff_date": context.get("market_data_cutoff_date"),
            })
        )
    return result


def _source_campaign(db: Any, run_id: str) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise ModelTuningNotFound("Source tuning campaign not found.")
    if str(document.get("status") or "") != "completed" or str(document.get("method") or "") != TUNING_METHOD:
        raise ModelTuningConflict("CARO can import observations only from a completed Latin Hypercube campaign.")
    if list(document.get("search_space") or []) != [dict(item) for item in _SEARCH_SPACE]:
        raise ModelTuningConflict("The source tuning campaign uses a different LightGBM search space.")
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


def _source_anchor(document: dict[str, Any], anchor_candidate_id: int | None) -> dict[str, Any]:
    candidate_id = int(anchor_candidate_id) if anchor_candidate_id is not None else int(document.get("best_candidate_id") or -1)
    candidate = next(
        (item for item in document.get("candidates") or [] if int(item.get("candidate_id") or 0) == candidate_id),
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
    for candidate in document.get("candidates") or []:
        settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else {}
        for name in settings:
            if name not in all_setting_names:
                all_setting_names.append(name)
    for candidate in document.get("candidates") or []:
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

    candidate_rows = _candidate_export_rows(document)
    prior_rows = _candidate_export_rows({
        "tuned_parameters": document.get("tuned_parameters") or list(_TUNED_NAMES),
        "candidates": document.get("prior_observations") or [],
    })
    ranked = [item for item in candidate_rows if item.get("rank") is not None]
    ranked.sort(key=lambda item: int(item["rank"]))
    baseline = deepcopy(document.get("baseline_execution") or {})
    summary_row = {
        "run_id": document.get("id"),
        "status": document.get("status"),
        "method": document.get("method"),
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
        "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
        "control_candidate_id": document.get("control_candidate_id"),
        "best_candidate_id": document.get("best_candidate_id"),
        "best_exploratory_candidate_id": document.get("best_exploratory_candidate_id"),
        "best_champion_beating_candidate_id": document.get("best_champion_beating_candidate_id"),
        "best_ending_capital": ranked[0].get("ending_capital") if ranked else None,
        "best_sharpe": ranked[0].get("sharpe") if ranked else None,
        "best_maximum_drawdown": ranked[0].get("maximum_drawdown") if ranked else None,
        "created_at": document.get("created_at"),
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
            "execution_mode": document.get("execution_mode"),
            "generated_candidates": document.get("generated_candidates"),
            "probability_config": deepcopy(document.get("probability_config") or {}),
            "validation": "chronological_walk_forward",
            "selection_metric": "risk_adjusted_compound_score",
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
            "strategy_profile_id": document.get("strategy_profile_id"),
            "strategy_profile_name": document.get("strategy_profile_name"),
            "strategy_profile_revision": document.get("strategy_profile_revision"),
            "strategy_configuration_hash": document.get("strategy_configuration_hash"),
            "base_model_settings_hash": document.get("base_model_settings_hash"),
            "base_model_settings_revision": document.get("base_model_settings_revision"),
            "baseline_execution": baseline,
            "source_tuning_run_id": document.get("source_tuning_run_id"),
            "source_strategy_profile_id": document.get("source_strategy_profile_id"),
            "source_strategy_profile_revision": document.get("source_strategy_profile_revision"),
            "prior_observations": deepcopy(document.get("prior_observations") or []),
            "imported_observation_count": document.get("imported_observation_count"),
            "probability_anchor": deepcopy(document.get("probability_anchor") or None),
            "execution_context_hash": document.get("execution_context_hash"),
            "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
            "market_data_cutoff_date": document.get("market_data_cutoff_date"),
            "adoption_context_compatible": document.get("adoption_context_compatible"),
            "control_candidate_id": document.get("control_candidate_id"),
            "best_candidate_id": document.get("best_candidate_id"),
            "best_exploratory_candidate_id": document.get("best_exploratory_candidate_id"),
            "best_champion_beating_candidate_id": document.get("best_champion_beating_candidate_id"),
            "adopted_candidate_id": document.get("adopted_candidate_id"),
            "created_at": document.get("created_at"),
            "started_at": document.get("started_at"),
            "finished_at": document.get("finished_at"),
            "candidates": deepcopy(document.get("candidates") or []),
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


def start_model_tuning(
    db: Any,
    *,
    method: str = TUNING_METHOD,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
    baseline_job_id: str | None = None,
    source_tuning_run_id: str | None = None,
    anchor_candidate_id: int | None = None,
    probability_config: dict[str, Any] | None = None,
    actor_email: str | None = None,
) -> dict[str, Any]:
    normalized_method = str(method or TUNING_METHOD).strip().lower()
    if normalized_method not in {TUNING_METHOD, PROBABILITY_METHOD}:
        raise ModelTuningConflict("Unsupported model tuning method.")

    active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"status": {"$in": list(_ACTIVE_STATUSES)}}, {"_id": 0, "id": 1}
    )
    if active_tuning is not None:
        raise ModelTuningConflict(f"Model tuning {active_tuning.get('id', 'unknown')} is already active.")

    _, strategy = get_research_strategy_context(db)
    model_snapshot = get_research_strategy_model_snapshot(db)
    if str(model_snapshot.get("family") or "") != TUNING_MODEL_FAMILY:
        raise ModelTuningConflict("Save LightGBM on the selected Strategy before starting model tuning.")
    if bool(strategy.get("locked")):
        raise ModelTuningConflict("Clone the protected Strategy before starting model tuning.")

    probability = dict(probability_config or {})
    prior_observations: list[dict[str, Any]] = []
    probability_anchor: dict[str, Any] | None = None
    source_run_id: str | None = None
    source_strategy_profile_id: str | None = None
    source_strategy_profile_revision: int | None = None
    adoption_context_compatible = True

    if normalized_method == PROBABILITY_METHOD and source_tuning_run_id:
        source = _source_campaign(db, source_tuning_run_id)
        prior_observations = _source_observations(source)
        anchor = _source_anchor(source, anchor_candidate_id)
        source_run_id = str(source.get("id") or source_tuning_run_id)
        source_strategy_profile_id = str(source.get("strategy_profile_id") or "") or None
        source_strategy_profile_revision = int(source.get("strategy_profile_revision") or 0)
        baseline = deepcopy(source.get("baseline_execution") or {})
        reference_job_id = str(baseline.get("job_id") or "")
        if not reference_job_id:
            raise ModelTuningConflict("The source campaign no longer references its baseline Backtest.")
        execution_context = _frozen_execution_context_from_job(db, reference_job_id)
        base_values = deepcopy(anchor.get("settings") or {})
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
            "probability_model": PROBABILITY_MODEL,
            "source_mode": "prior_campaign",
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
        base_values = model_values_from_snapshot(model_snapshot)

        if normalized_method == PROBABILITY_METHOD:
            startup_trials = max(4, int(probability.get("startup_trials") or 8))
            if startup_trials >= int(candidate_count):
                raise ModelTuningConflict("Probabilistic startup trials must be smaller than the total candidate count.")
            candidates = generate_latin_hypercube_candidates(base_values, candidate_count=startup_trials, seed=seed)
            for candidate in candidates:
                if not candidate.get("is_control"):
                    candidate["kind"] = "probability_startup"
            total_candidates = int(candidate_count) + 1
            probability_anchor = {
                "source": "baseline_backtest",
                "job_id": reference_job_id,
                "settings_hash": str(baseline.get("model_settings_hash") or ""),
                "settings": deepcopy(base_values),
                "metrics": deepcopy(baseline.get("metrics") or {}),
            }
            probability = {
                "startup_trials": startup_trials,
                "imported_observation_count": 0,
                "min_capital_improvement": float(probability.get("min_capital_improvement", 0.03)),
                "sharpe_tolerance": float(probability.get("sharpe_tolerance", 0.05)),
                "drawdown_tolerance": float(probability.get("drawdown_tolerance", 0.03)),
                "min_worst_fold_return": float(probability.get("min_worst_fold_return", 0.0)),
                "candidate_pool_size": int(probability.get("candidate_pool_size", 2048)),
                "exploration_weight": float(probability.get("exploration_weight", 0.15)),
                "probability_model": PROBABILITY_MODEL,
                "source_mode": "standalone",
            }
        else:
            candidates = generate_latin_hypercube_candidates(base_values, candidate_count=candidate_count, seed=seed)
            total_candidates = int(candidate_count) + 1
            probability = {}

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
        "candidate_count": int(candidate_count),
        "total_candidates": int(total_candidates),
        "generated_candidates": len(candidates),
        "completed_candidates": 0,
        "failed_candidates": 0,
        "seed": int(seed),
        "search_space": [dict(item) for item in _SEARCH_SPACE],
        "tuned_parameters": list(_TUNED_NAMES),
        "strategy_profile_id": strategy["id"],
        "strategy_profile_name": strategy["name"],
        "strategy_profile_revision": int(strategy["revision"]),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "base_model_settings_hash": model_snapshot.get("settings_hash"),
        "base_model_settings_revision": int(model_snapshot.get("settings_revision") or 0),
        "base_model_values": bson_value(base_values),
        "baseline_execution": bson_value(baseline),
        "probability_config": bson_value(probability),
        "probability_anchor": bson_value(probability_anchor) if probability_anchor else None,
        "source_tuning_run_id": source_run_id,
        "source_strategy_profile_id": source_strategy_profile_id,
        "source_strategy_profile_revision": source_strategy_profile_revision,
        "prior_observations": bson_value(prior_observations),
        "imported_observation_count": len(prior_observations),
        "execution_mode": "integrated_api_worker",
        "execution_request_snapshot": bson_value(execution_context["request"]),
        "execution_context_hash": execution_context.get("context_hash"),
        "expected_market_data_signature_sha256": execution_context.get("market_data_signature_sha256"),
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
                    f"Campaign created. method={normalized_method}; total_candidates={int(total_candidates)}; "
                    f"imported_observations={len(prior_observations)}; source_campaign={source_run_id or 'none'}."
                ),
                "candidate_id": None,
                "job_id": None,
            }
        ]),
        "candidates": bson_value(candidates),
    }
    db[MODEL_TUNING_RUNS_COLLECTION].insert_one(document)
    threading.Thread(target=run_model_tuning, args=(run_id,), daemon=True).start()
    return public_model_tuning_run(db, document)


def run_model_tuning(run_id: str) -> None:
    db = database()
    from ..api.routers.jobs import queue_backtest_job  # Lazy import avoids router/service cycles.

    try:
        document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
        if document is None:
            return
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {"$set": {"status": "running", "phase": "running", "started_at": utc_now(), "updated_at": utc_now()}},
        )
        _append_campaign_event(db, run_id, message="Integrated tuning worker started.", stage="running")

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
            if pending is None and str(document.get("method") or "") == PROBABILITY_METHOD and len(candidates) < int(document.get("total_candidates") or 0):
                candidate = propose_champion_probability_candidate(document)
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {
                        "$push": {"candidates": bson_value(candidate)},
                        "$set": {
                            "phase": "probabilistic_refinement",
                            "generated_candidates": len(candidates) + 1,
                            "updated_at": utc_now(),
                        },
                    },
                )
                _append_campaign_event(
                    db, run_id,
                    message=f"CARO proposed candidate #{int(candidate.get('candidate_id') or 0)} using {int((candidate.get('proposal') or {}).get('observation_count') or 0)} observations.",
                    stage="probabilistic_refinement",
                    candidate_id=int(candidate.get("candidate_id") or 0),
                )
                continue

            if pending is None:
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id},
                    {"$set": {"status": "completed", "phase": "completed", "finished_at": utc_now(), "updated_at": utc_now(), "current_candidate_id": None, "current_job_id": None}},
                )
                _append_campaign_event(db, run_id, message="Campaign completed.", stage="completed")
                return

            candidate_id = int(pending["candidate_id"])
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id, "candidates.candidate_id": candidate_id},
                {
                    "$set": {
                        "current_candidate_id": candidate_id,
                        "phase": "running_candidate",
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
            job_id: str | None = None
            try:
                execution_metadata = {
                    "strategy_profile_id": (document.get("source_strategy_profile_id") or document.get("strategy_profile_id")),
                    "strategy_profile_name": (document.get("baseline_execution") or {}).get("strategy_profile_name") or document.get("strategy_profile_name"),
                    "strategy_profile_revision": (document.get("source_strategy_profile_revision") or document.get("strategy_profile_revision")),
                    "strategy_configuration_hash": (document.get("baseline_execution") or {}).get("strategy_configuration_hash") or document.get("strategy_configuration_hash"),
                }
                queued = queue_backtest_job(
                    model_values_override=dict(pending["settings"]),
                    start_thread=False,
                    certify_strategy=False,
                    tuning_run_id=run_id,
                    tuning_candidate_id=candidate_id,
                    execution_request_override=deepcopy(document.get("execution_request_snapshot") or {}),
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
                run_job(job_id)
                job = db[JOBS_COLLECTION].find_one({"id": job_id}) or {}
                if job.get("status") != "completed":
                    raise RuntimeError("The candidate backtest did not complete successfully.")
                metrics = _find_portfolio_metrics(db, job_id)
                if metrics is None:
                    raise RuntimeError("Portfolio metrics are missing for the tuning candidate.")
                summary = _metric_summary(metrics)
                champion_gate = (
                    champion_gate_evaluation(document, summary)
                    if str(document.get("method") or "") == PROBABILITY_METHOD
                    else None
                )
                expected_signature = str(document.get("expected_market_data_signature_sha256") or "")
                actual_signature = str(summary.get("market_data_signature_sha256") or "")
                if expected_signature and actual_signature != expected_signature:
                    raise RuntimeError(
                        f"MarketDataSignatureMismatch: expected {expected_signature}, got {actual_signature or 'missing'}"
                    )
                db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                    {"id": run_id, "candidates.candidate_id": candidate_id},
                    {
                        "$set": {
                            "candidates.$.status": "completed",
                            "candidates.$.metrics": bson_value(summary),
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
                if "MarketDataSignatureMismatch" in str(exc):
                    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                        {"id": run_id},
                        {"$set": {"status": "failed", "phase": "reproducibility_guard_failed", "failure_type": "MarketDataSignatureMismatch", "failure_message": _sanitize_tuning_log_line(str(exc))[:500], "finished_at": utc_now(), "updated_at": utc_now()}},
                    )
                    _append_campaign_event(
                        db, run_id, message=_sanitize_tuning_log_line(str(exc)),
                        level="error", stage="reproducibility_guard_failed", candidate_id=candidate_id, job_id=job_id,
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
    """Recover active in-process tuning campaigns after an API/container restart.

    A candidate that was running when the process disappeared is reset to pending and
    rerun from the campaign's immutable execution snapshot. Completed candidate
    summaries are preserved. This favors reproducibility over attempting to reuse
    partially written raw artifacts.
    """
    documents = list(
        db[MODEL_TUNING_RUNS_COLLECTION].find(
            {
                "status": {"$in": list(_ACTIVE_STATUSES)},
                "$or": [
                    {"execution_mode": "integrated_api_worker"},
                    {"execution_mode": {"$exists": False}},
                ],
            }
        )
    )
    recovered = 0
    for document in documents:
        run_id = str(document.get("id") or "")
        if not run_id:
            continue
        candidates = deepcopy(document.get("candidates") or [])
        reset_jobs: list[str] = []
        for candidate in candidates:
            if candidate.get("status") != "running":
                continue
            job_id = str(candidate.get("job_id") or "")
            if job_id:
                reset_jobs.append(job_id)
            candidate["status"] = "pending"
            candidate["job_id"] = None
            candidate["started_at"] = None
            candidate["finished_at"] = None
            candidate["error"] = None
            candidate["retry_count"] = int(candidate.get("retry_count") or 0) + 1

        for job_id in reset_jobs:
            _cleanup_job_artifacts(db, job_id)

        if bool(document.get("stop_requested")):
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": run_id},
                {
                    "$set": {
                        "candidates": bson_value(candidates),
                        "status": "stopped",
                        "phase": "stopped_after_restart",
                        "current_candidate_id": None,
                        "current_job_id": None,
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                },
            )
            recovered += 1
            continue

        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {
                "$set": {
                    "candidates": bson_value(candidates),
                    "status": "queued",
                    "phase": "recovered_after_restart",
                    "current_candidate_id": None,
                    "current_job_id": None,
                    "updated_at": utc_now(),
                },
                "$inc": {"restart_recovery_count": 1},
            },
        )
        threading.Thread(target=run_model_tuning, args=(run_id,), daemon=True).start()
        recovered += 1
    return recovered



def request_model_tuning_stop(db: Any, run_id: str) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    if str(document.get("status") or "") not in _ACTIVE_STATUSES:
        return public_model_tuning_run(db, document)
    running = any(item.get("status") == "running" for item in document.get("candidates") or [])
    now = utc_now()
    if not running:
        updated = db[MODEL_TUNING_RUNS_COLLECTION].find_one_and_update(
            {"id": run_id},
            {"$set": {"stop_requested": True, "status": "stopped", "phase": "stopped", "finished_at": now, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
    else:
        updated = db[MODEL_TUNING_RUNS_COLLECTION].find_one_and_update(
            {"id": run_id},
            {"$set": {"stop_requested": True, "status": "stop_requested", "phase": "finishing_active_candidates", "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
    return public_model_tuning_run(db, updated or document)


def adopt_model_tuning_candidate(
    db: Any,
    run_id: str,
    candidate_id: int,
    *,
    reason: str,
    actor_email: str | None,
) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": run_id})
    if document is None:
        raise ModelTuningNotFound("Model tuning run not found.")
    if str(document.get("status") or "") not in {"completed", "stopped"}:
        raise ModelTuningConflict("Wait for the tuning campaign to finish before adopting a candidate.")
    candidate = next(
        (item for item in document.get("candidates") or [] if int(item.get("candidate_id") if item.get("candidate_id") is not None else -1) == int(candidate_id)),
        None,
    )
    if candidate is None or candidate.get("status") != "completed":
        raise ModelTuningConflict("Only a completed tuning candidate can be adopted.")
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    if not bool(metrics.get("eligible")):
        raise ModelTuningConflict("This candidate failed the positive-fold robustness gate and cannot be adopted.")
    if str(document.get("method") or "") == PROBABILITY_METHOD:
        gate = champion_gate_evaluation(document, metrics)
        if not bool(gate.get("passed")):
            raise ModelTuningConflict("This CARO candidate did not beat the configured Champion robustness gate and cannot be adopted.")
    _, current_strategy = get_research_strategy_context(db)
    if str(current_strategy.get("id")) != str(document.get("strategy_profile_id")):
        raise ModelTuningConflict("The selected Strategy changed after this tuning campaign started.")
    if int(current_strategy.get("revision") or 0) != int(document.get("strategy_profile_revision") or -1):
        raise ModelTuningConflict("The selected Strategy revision changed after this tuning campaign started.")

    updated_strategy = update_strategy_model(
        db,
        str(document["strategy_profile_id"]),
        model_family=TUNING_MODEL_FAMILY,
        values=dict(candidate["settings"]),
        note=reason,
        expected_strategy_revision=int(document["strategy_profile_revision"]),
        actor_email=actor_email,
    )
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": run_id},
        {"$set": {"adopted_candidate_id": int(candidate_id), "adopted_at": utc_now(), "adopted_by": (actor_email or "").strip().lower() or None, "updated_at": utc_now()}},
    )
    return {
        "strategy": updated_strategy,
        "candidate_id": int(candidate_id),
        "final_backtest_required": True,
        "source_context_confirmation_required": not bool(document.get("adoption_context_compatible", True)),
    }


def _public_candidate(candidate: dict[str, Any], current_jobs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
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
    raw_candidates = list(document.get("candidates") or [])
    active_job_ids = [str(item.get("job_id") or "") for item in raw_candidates if item.get("status") == "running" and item.get("job_id")]
    current_jobs: dict[str, dict[str, Any]] = {}
    if active_job_ids:
        for job in db[JOBS_COLLECTION].find(
            {"id": {"$in": active_job_ids}},
            {"_id": 0, "id": 1, "status": 1, "stage": 1, "progress": 1},
        ):
            current_jobs[str(job.get("id") or "")] = job
    candidates = [_public_candidate(item, current_jobs) for item in raw_candidates]
    total = max(1, int(document.get("total_candidates") or len(candidates) or 1))
    completed = int(document.get("completed_candidates") or 0) + int(document.get("failed_candidates") or 0)
    fractional_active = sum(float(job.get("progress") or 0.0) / 100.0 for job in current_jobs.values())
    progress = min(100.0, 100.0 * (completed + fractional_active) / total)
    if current_jobs and str(document.get("status") or "") in _ACTIVE_STATUSES:
        progress = min(99.9, progress)
    active_candidate_ids = [int(item.get("candidate_id") or 0) for item in raw_candidates if item.get("status") == "running"]
    return bson_value({
        "id": document.get("id"),
        "schema_version": int(document.get("schema_version") or TUNING_SCHEMA_VERSION),
        "status": str(document.get("status") or "queued"),
        "phase": str(document.get("phase") or "queued"),
        "method": str(document.get("method") or TUNING_METHOD),
        "execution_mode": str(document.get("execution_mode") or "integrated_api_worker"),
        "model_family": str(document.get("model_family") or TUNING_MODEL_FAMILY),
        "model_label": str(document.get("model_label") or "LightGBM Utility"),
        "candidate_count": int(document.get("candidate_count") or 0),
        "total_candidates": total,
        "generated_candidates": int(document.get("generated_candidates") or len(candidates)),
        "completed_candidates": int(document.get("completed_candidates") or 0),
        "failed_candidates": int(document.get("failed_candidates") or 0),
        "progress": progress,
        "seed": int(document.get("seed") or DEFAULT_SEED),
        "search_space": deepcopy(document.get("search_space") or []),
        "probability_config": deepcopy(document.get("probability_config") or {}),
        "probability_anchor": deepcopy(document.get("probability_anchor") or None),
        "source_tuning_run_id": document.get("source_tuning_run_id"),
        "imported_observation_count": int(document.get("imported_observation_count") or 0),
        "market_data_cutoff_date": document.get("market_data_cutoff_date"),
        "expected_market_data_signature_sha256": document.get("expected_market_data_signature_sha256"),
        "execution_context_hash": document.get("execution_context_hash"),
        "adoption_context_compatible": bool(document.get("adoption_context_compatible", True)),
        "strategy_profile_id": document.get("strategy_profile_id"),
        "strategy_profile_name": document.get("strategy_profile_name"),
        "strategy_profile_revision": int(document.get("strategy_profile_revision") or 0),
        "created_at": document.get("created_at"),
        "started_at": document.get("started_at"),
        "finished_at": document.get("finished_at"),
        "failure_type": document.get("failure_type"),
        "failure_message": document.get("failure_message"),
        "has_campaign_log": bool(document.get("event_log") or document.get("failure_message")),
        "stop_requested": bool(document.get("stop_requested")),
        "active_candidate_ids": active_candidate_ids,
        "active_job_ids": active_job_ids,
        "current_candidate_id": active_candidate_ids[0] if active_candidate_ids else None,
        "current_job_id": active_job_ids[0] if active_job_ids else None,
        "best_candidate_id": document.get("best_candidate_id"),
        "best_exploratory_candidate_id": document.get("best_exploratory_candidate_id"),
        "best_champion_beating_candidate_id": document.get("best_champion_beating_candidate_id"),
        "control_candidate_id": int(document["control_candidate_id"]) if document.get("control_candidate_id") is not None else None,
        "adopted_candidate_id": document.get("adopted_candidate_id"),
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
        _, strategy = get_research_strategy_context(db)
        strategy_id = str(strategy["id"])
    except Exception:
        return None
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"strategy_profile_id": strategy_id},
        sort=[("created_at", -1)],
    )
    return public_model_tuning_run(db, document)
