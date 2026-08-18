from __future__ import annotations

import io
import json
import math
import threading
import uuid
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..schemas.temporal_rotation_quality import TemporalRotationQualityDiagnosticRequest
from .temporal_rotation_quality import (
    TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION,
    TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION,
    ReplayInputs,
    TemporalRotationQualityConflict,
    TemporalRotationQualityNotFound,
    _replay,
    _validation_replay_inputs,
)

TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION = "temporal_rotation_quality_diagnostics"

DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "entry_rank_score",
    "risk_adjusted_asset_rank_score",
    "entry_rank_percentile",
    "opportunity_gate_score",
    "risk_adjusted_entry_score",
    "hold_score",
    "incumbent_persistence_raw",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "short_profit_consensus",
    "short_risk_safety",
    "short_bottom_support",
    "short_horizon_agreement",
    "long_profit_confirmation",
    "long_risk_safety",
    "long_trend_support",
    "long_horizon_agreement",
    "cross_horizon_agreement",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "entry_separation_strength",
    "entry_top_gap_strength",
    "short_profit_quality",
)

DEFAULT_DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "entry_rank_score",
    "hold_score",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "short_profit_consensus",
    "short_risk_safety",
    "long_profit_confirmation",
    "long_risk_safety",
    "long_trend_support",
    "horizon_agreement",
    "predicted_drawdown",
)


class TemporalRotationQualityDiagnosticCancelled(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _public_document(document: dict[str, Any] | None, *, include_events: bool = False) -> dict[str, Any] | None:
    if not document:
        return None
    payload = {key: value for key, value in document.items() if key != "_id"}
    if not include_events:
        payload.pop("events", None)
    return payload


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _feature_history(
    daily_assets: pd.DataFrame,
    *,
    feature_names: list[str],
) -> dict[tuple[int, str], pd.DataFrame]:
    columns = ["fold_id", "timestamp", "symbol", *feature_names]
    frame = daily_assets[[column for column in columns if column in daily_assets.columns]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    history: dict[tuple[int, str], pd.DataFrame] = {}
    for (fold_id, symbol), group in frame.groupby(["fold_id", "symbol"], sort=False):
        history[(int(fold_id), str(symbol))] = group.sort_values("timestamp").reset_index(drop=True)
    return history


def _point_and_history_metrics(
    history: dict[tuple[int, str], pd.DataFrame],
    *,
    fold_id: int,
    symbol: str,
    timestamp: pd.Timestamp,
    feature: str,
    lookback_sessions: int,
) -> dict[str, float | None]:
    frame = history.get((int(fold_id), str(symbol)))
    if frame is None or feature not in frame.columns:
        return {"current": None, "lookback": None, "delta": None, "slope": None, "samples": 0}
    eligible = frame.loc[frame["timestamp"] <= timestamp, ["timestamp", feature]].copy()
    eligible[feature] = pd.to_numeric(eligible[feature], errors="coerce")
    eligible = eligible.dropna(subset=[feature])
    if eligible.empty:
        return {"current": None, "lookback": None, "delta": None, "slope": None, "samples": 0}
    window = eligible.tail(int(lookback_sessions) + 1)
    values = window[feature].astype(float).to_numpy()
    current = _finite(values[-1])
    lookback = _finite(values[0]) if len(values) > 1 else None
    delta = None if current is None or lookback is None else current - lookback
    slope = None
    if len(values) >= 2 and np.isfinite(values).all():
        x = np.arange(len(values), dtype="float64")
        slope = _finite(np.polyfit(x, values, 1)[0])
    return {
        "current": current,
        "lookback": lookback,
        "delta": delta,
        "slope": slope,
        "samples": int(len(values)),
    }


def build_rotation_quality_diagnostic(
    inputs: ReplayInputs,
    *,
    candidate_id: str,
    drawdown_trigger: float,
    rotation_score_tolerance: float,
    request: TemporalRotationQualityDiagnosticRequest,
) -> dict[str, Any]:
    missing = [feature for feature in request.feature_names if feature not in inputs.daily_assets.columns]
    if missing:
        raise TemporalRotationQualityConflict(
            "Diagnostic feature columns are missing from the retrained Temporal observations: " + ", ".join(missing)
        )

    replay = _replay(
        inputs,
        candidate_id=str(candidate_id),
        drawdown_trigger=float(drawdown_trigger),
        rotation_score_tolerance=float(rotation_score_tolerance),
    )
    history = _feature_history(inputs.daily_assets, feature_names=list(request.feature_names))
    events: list[dict[str, Any]] = []
    band = float(request.outcome_neutral_band)

    for blocked in replay.blocked_rows:
        fold_id = int(blocked["fold_id"])
        timestamp = pd.Timestamp(blocked["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        incumbent = str(blocked["simulated_incumbent"])
        challenger = str(blocked["original_target"])
        incremental_return = float(blocked["incremental_interval_return"])
        if incremental_return > band:
            outcome_class = "helpful"
        elif incremental_return < -band:
            outcome_class = "harmful"
        else:
            outcome_class = "neutral"

        event: dict[str, Any] = {
            **deepcopy(blocked),
            "outcome_class": outcome_class,
            "lookback_sessions": int(request.lookback_sessions),
        }
        for feature in request.feature_names:
            incumbent_metrics = _point_and_history_metrics(
                history,
                fold_id=fold_id,
                symbol=incumbent,
                timestamp=timestamp,
                feature=feature,
                lookback_sessions=int(request.lookback_sessions),
            )
            challenger_metrics = _point_and_history_metrics(
                history,
                fold_id=fold_id,
                symbol=challenger,
                timestamp=timestamp,
                feature=feature,
                lookback_sessions=int(request.lookback_sessions),
            )
            prefix_i = f"incumbent_{feature}"
            prefix_c = f"challenger_{feature}"
            for suffix in ("current", "lookback", "delta", "slope", "samples"):
                event[f"{prefix_i}_{suffix}"] = incumbent_metrics[suffix]
                event[f"{prefix_c}_{suffix}"] = challenger_metrics[suffix]

            current_i = incumbent_metrics["current"]
            current_c = challenger_metrics["current"]
            delta_i = incumbent_metrics["delta"]
            delta_c = challenger_metrics["delta"]
            slope_i = incumbent_metrics["slope"]
            slope_c = challenger_metrics["slope"]
            event[f"{feature}_gap_current"] = None if current_i is None or current_c is None else current_c - current_i
            event[f"{feature}_gap_delta"] = None if delta_i is None or delta_c is None else delta_c - delta_i
            event[f"{feature}_gap_slope"] = None if slope_i is None or slope_c is None else slope_c - slope_i
        events.append(event)

    fold_summary: list[dict[str, Any]] = []
    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        for fold_id, group in event_frame.groupby("fold_id", sort=True):
            fold_summary.append(
                {
                    "fold_id": int(fold_id),
                    "blocked_rotations": int(len(group)),
                    "helpful": int((group["outcome_class"] == "helpful").sum()),
                    "harmful": int((group["outcome_class"] == "harmful").sum()),
                    "neutral": int((group["outcome_class"] == "neutral").sum()),
                    "immediate_net_rotation_benefit_dollars": float(group["immediate_incremental_dollars"].sum()),
                    "mean_incremental_interval_return": float(group["incremental_interval_return"].mean()),
                }
            )

    engineered_columns: list[tuple[str, str]] = []
    for feature in request.feature_names:
        engineered_columns.extend(
            [
                (feature, f"incumbent_{feature}_current"),
                (feature, f"challenger_{feature}_current"),
                (feature, f"{feature}_gap_current"),
                (feature, f"incumbent_{feature}_delta"),
                (feature, f"challenger_{feature}_delta"),
                (feature, f"{feature}_gap_delta"),
                (feature, f"incumbent_{feature}_slope"),
                (feature, f"challenger_{feature}_slope"),
                (feature, f"{feature}_gap_slope"),
            ]
        )

    separation: list[dict[str, Any]] = []
    if not event_frame.empty:
        helpful = event_frame[event_frame["outcome_class"] == "helpful"]
        harmful = event_frame[event_frame["outcome_class"] == "harmful"]
        for base_feature, column in engineered_columns:
            if column not in event_frame.columns:
                continue
            helpful_values = pd.to_numeric(helpful[column], errors="coerce").dropna().astype(float)
            harmful_values = pd.to_numeric(harmful[column], errors="coerce").dropna().astype(float)
            if len(helpful_values) < request.minimum_group_samples or len(harmful_values) < request.minimum_group_samples:
                continue
            helpful_mean = float(helpful_values.mean())
            harmful_mean = float(harmful_values.mean())
            helpful_std = float(helpful_values.std(ddof=1)) if len(helpful_values) > 1 else 0.0
            harmful_std = float(harmful_values.std(ddof=1)) if len(harmful_values) > 1 else 0.0
            pooled_denom = max(1, len(helpful_values) + len(harmful_values) - 2)
            pooled_variance = (
                max(0, len(helpful_values) - 1) * helpful_std**2
                + max(0, len(harmful_values) - 1) * harmful_std**2
            ) / pooled_denom
            pooled_std = math.sqrt(max(0.0, pooled_variance))
            standardized = (helpful_mean - harmful_mean) / pooled_std if pooled_std > 1e-15 else 0.0
            separation.append(
                {
                    "feature": base_feature,
                    "engineered_metric": column,
                    "helpful_samples": int(len(helpful_values)),
                    "harmful_samples": int(len(harmful_values)),
                    "helpful_mean": helpful_mean,
                    "harmful_mean": harmful_mean,
                    "helpful_median": float(helpful_values.median()),
                    "harmful_median": float(harmful_values.median()),
                    "mean_difference": helpful_mean - harmful_mean,
                    "standardized_separation": float(standardized),
                    "absolute_standardized_separation": float(abs(standardized)),
                    "helpful_direction": "higher" if standardized > 0 else "lower" if standardized < 0 else "flat",
                }
            )
    separation.sort(key=lambda item: float(item["absolute_standardized_separation"]), reverse=True)

    helpful_count = sum(1 for item in events if item["outcome_class"] == "helpful")
    harmful_count = sum(1 for item in events if item["outcome_class"] == "harmful")
    neutral_count = sum(1 for item in events if item["outcome_class"] == "neutral")
    immediate_net = float(sum(float(item["immediate_incremental_dollars"]) for item in events))
    return {
        "candidate_id": str(candidate_id),
        "drawdown_trigger": float(drawdown_trigger),
        "rotation_score_tolerance": float(rotation_score_tolerance),
        "lookback_sessions": int(request.lookback_sessions),
        "outcome_neutral_band": float(request.outcome_neutral_band),
        "feature_names": list(request.feature_names),
        "blocked_rotations": int(len(events)),
        "helpful_blocks": int(helpful_count),
        "harmful_blocks": int(harmful_count),
        "neutral_blocks": int(neutral_count),
        "helpful_rate_excluding_neutral": float(helpful_count / max(1, helpful_count + harmful_count)),
        "immediate_net_rotation_benefit_dollars": immediate_net,
        "fold_summary": fold_summary,
        "top_feature_separation": separation[: int(request.top_feature_count)],
        "feature_separation": separation,
        "events": events,
        "diagnostic_policy": {
            "decision_time_features_only": True,
            "future_information_used_for_decision": False,
            "future_outcome_used_as_diagnostic_label_only": True,
            "outcome_label": "chosen incumbent interval return minus original Temporal target interval return",
            "lookback_sessions": int(request.lookback_sessions),
            "minimum_group_samples": int(request.minimum_group_samples),
            "outcome_neutral_band": float(request.outcome_neutral_band),
        },
    }


def _ensure_not_cancelled(db: Any, diagnostic_id: str) -> None:
    document = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one(
        {"id": str(diagnostic_id)}, {"_id": 0, "status": 1}
    )
    if document and str(document.get("status") or "") == "stop_requested":
        raise TemporalRotationQualityDiagnosticCancelled("Diagnostic stop requested.")


def _progress(db: Any, diagnostic_id: str, percent: float, stage: str) -> None:
    _ensure_not_cancelled(db, diagnostic_id)
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
        {"id": str(diagnostic_id)},
        {
            "$set": {
                "progress": max(0.0, min(100.0, float(percent))),
                "stage": str(stage),
                "updated_at": _utc_now(),
            }
        },
    )


def _run_diagnostic(db: Any, diagnostic_id: str) -> None:
    document = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one({"id": str(diagnostic_id)})
    if not document:
        return
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
        {"id": str(diagnostic_id)},
        {"$set": {"status": "running", "stage": "Rebuilding Temporal diagnostic surface", "progress": 1.0, "started_at": _utc_now(), "updated_at": _utc_now()}},
    )
    try:
        request = TemporalRotationQualityDiagnosticRequest.model_validate(document.get("request") or {})
        fold_count = int(document["fold_count"])
        inputs, _ = _validation_replay_inputs(
            db,
            source_run_id=str(document["source_run_id"]),
            fold_count=fold_count,
            progress_callback=lambda percent, stage: _progress(
                db,
                diagnostic_id,
                min(90.0, max(2.0, float(percent) * 0.88)),
                stage,
            ),
            cancel_callback=lambda: _ensure_not_cancelled(db, diagnostic_id),
        )
        _progress(db, diagnostic_id, 92.0, "Analyzing blocked Rotation Quality decisions")
        result = build_rotation_quality_diagnostic(
            inputs,
            candidate_id=str(document["candidate_id"]),
            drawdown_trigger=float(document["drawdown_trigger"]),
            rotation_score_tolerance=float(document["rotation_score_tolerance"]),
            request=request,
        )
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
            {"id": str(diagnostic_id)},
            {
                "$set": {
                    "status": "completed",
                    "stage": "Completed",
                    "progress": 100.0,
                    "updated_at": now,
                    "finished_at": now,
                    **result,
                    "failure_message": None,
                }
            },
        )
    except TemporalRotationQualityDiagnosticCancelled:
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
            {"id": str(diagnostic_id)},
            {"$set": {"status": "stopped", "stage": "Stopped", "updated_at": now, "finished_at": now, "failure_message": None}},
        )
    except Exception as exc:
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
            {"id": str(diagnostic_id)},
            {"$set": {"status": "failed", "stage": "Failed", "updated_at": now, "finished_at": now, "failure_message": str(exc)}},
        )


def start_temporal_rotation_quality_diagnostic(
    db: Any,
    research_id: str,
    validation_id: str,
    request: TemporalRotationQualityDiagnosticRequest,
    *,
    actor_email: str | None = None,
) -> dict[str, Any]:
    research = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one({"id": str(research_id)}, {"_id": 0, "id": 1})
    if not research:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    validation = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"id": str(validation_id), "research_id": str(research_id)}
    )
    if not validation:
        raise TemporalRotationQualityNotFound(
            f"Temporal Rotation Quality evidence run {validation_id} was not found for research {research_id}."
        )
    if str(validation.get("status") or "") != "completed":
        raise TemporalRotationQualityConflict("Rotation Quality diagnostics require a completed validation/certification execution.")

    candidate = next(
        (item for item in validation.get("candidates") or [] if str(item.get("candidate_id") or "") == request.candidate_id),
        None,
    )
    if not candidate:
        raise TemporalRotationQualityNotFound(
            f"Candidate {request.candidate_id} was not evaluated by evidence run {validation_id}."
        )
    frozen = next(
        (item for item in validation.get("frozen_candidates") or [] if str(item.get("candidate_id") or "") == request.candidate_id),
        None,
    )
    if not frozen:
        raise TemporalRotationQualityConflict(
            f"Evidence run {validation_id} does not contain frozen parameters for {request.candidate_id}."
        )

    from ..infrastructure.persistence.mongo_repository import (
        JOBS_COLLECTION,
        MODEL_TUNING_RUNS_COLLECTION,
        TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    )
    from .system_settings import get_system_settings

    runtime_settings = get_system_settings(db)
    if not bool(runtime_settings["training"]["enabled"]):
        raise TemporalRotationQualityConflict("Model training is disabled in System Settings.")

    active_temporal = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active_temporal:
        raise TemporalRotationQualityConflict(
            f"Wait for Temporal Intelligence {active_temporal.get('id', 'unknown')} to finish before starting diagnostics."
        )
    active_backtest = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active_backtest:
        raise TemporalRotationQualityConflict("Wait for the active Simulation Backtest to finish before starting diagnostics.")
    active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active_tuning:
        raise TemporalRotationQualityConflict("Wait for the active Model Tuning campaign to finish before starting diagnostics.")
    active_research = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active_research:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality research {active_research.get('id', 'unknown')} to finish before starting diagnostics."
        )
    active_evidence = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active_evidence:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality evidence run {active_evidence.get('id', 'unknown')} to finish before starting diagnostics."
        )

    unknown_features = sorted(set(request.feature_names) - set(DIAGNOSTIC_FEATURES))
    if unknown_features:
        raise ValueError("Unsupported diagnostic features: " + ", ".join(unknown_features))

    active = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}
    )
    if active:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality diagnostic {active.get('id', 'unknown')} to finish before starting another diagnostic."
        )

    diagnostic_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-rq-diagnostic-" + uuid.uuid4().hex[:8]
    now = _utc_now()
    document = {
        "id": diagnostic_id,
        "research_id": str(research_id),
        "validation_id": str(validation_id),
        "source_run_id": str(validation.get("source_run_id") or ""),
        "evidence_kind": str(validation.get("kind") or "validation"),
        "fold_count": int(validation.get("fold_count") or 0),
        "candidate_id": str(request.candidate_id),
        "drawdown_trigger": float(frozen["drawdown_trigger"]),
        "rotation_score_tolerance": float(frozen["rotation_score_tolerance"]),
        "request": request.model_dump(),
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "actor_email": actor_email,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "blocked_rotations": 0,
        "helpful_blocks": 0,
        "harmful_blocks": 0,
        "neutral_blocks": 0,
        "fold_summary": [],
        "top_feature_separation": [],
        "feature_separation": [],
        "events": [],
        "diagnostic_policy": None,
        "failure_message": None,
    }
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].insert_one(deepcopy(document))
    threading.Thread(target=_run_diagnostic, args=(db, diagnostic_id), daemon=True).start()
    return _public_document(document) or {}


def request_temporal_rotation_quality_diagnostic_stop(
    db: Any,
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one(
        {"id": str(diagnostic_id), "research_id": str(research_id), "validation_id": str(validation_id)}
    )
    if not document:
        raise TemporalRotationQualityNotFound(f"Rotation Quality diagnostic {diagnostic_id} was not found.")
    status = str(document.get("status") or "")
    if status not in {"queued", "running", "stop_requested"}:
        return _public_document(document) or {}
    now = _utc_now()
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].update_one(
        {"id": str(diagnostic_id)},
        {"$set": {"status": "stop_requested", "stage": "Stop requested", "updated_at": now}},
    )
    updated = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one({"id": str(diagnostic_id)})
    return _public_document(updated) or {}


def get_temporal_rotation_quality_diagnostic(
    db: Any,
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one(
        {"id": str(diagnostic_id), "research_id": str(research_id), "validation_id": str(validation_id)}
    )
    if not document:
        raise TemporalRotationQualityNotFound(f"Rotation Quality diagnostic {diagnostic_id} was not found.")
    payload = _public_document(document) or {}
    events = document.get("events") or []
    payload["event_preview"] = deepcopy(events[:50])
    payload.pop("events", None)
    payload.pop("feature_separation", None)
    return payload


def list_temporal_rotation_quality_diagnostics(
    db: Any,
    research_id: str,
    validation_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    cursor = (
        db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION]
        .find({"research_id": str(research_id), "validation_id": str(validation_id)})
        .sort("created_at", -1)
        .limit(int(limit))
    )
    return [_public_document(item) or {} for item in cursor]


def _write_csv(archive: zipfile.ZipFile, name: str, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    archive.writestr(name, frame.to_csv(index=False).encode("utf-8"))


def build_temporal_rotation_quality_diagnostic_export(
    db: Any,
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
) -> bytes:
    document = db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].find_one(
        {"id": str(diagnostic_id), "research_id": str(research_id), "validation_id": str(validation_id)}
    )
    if not document:
        raise TemporalRotationQualityNotFound(f"Rotation Quality diagnostic {diagnostic_id} was not found.")
    if str(document.get("status") or "") != "completed":
        raise TemporalRotationQualityConflict("Rotation Quality diagnostic export requires a completed execution.")

    summary = {
        key: value
        for key, value in document.items()
        if key not in {"_id", "events", "feature_separation", "fold_summary", "top_feature_separation"}
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        archive.writestr(
            "diagnostic_policy.json",
            json.dumps(document.get("diagnostic_policy") or {}, ensure_ascii=False, indent=2, default=str),
        )
        _write_csv(archive, "blocked_rotation_diagnostics.csv", list(document.get("events") or []))
        _write_csv(archive, "feature_separation.csv", list(document.get("feature_separation") or []))
        _write_csv(archive, "fold_summary.csv", list(document.get("fold_summary") or []))
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "research_id": document.get("research_id"),
                    "validation_id": document.get("validation_id"),
                    "diagnostic_id": document.get("id"),
                    "source_run_id": document.get("source_run_id"),
                    "candidate_id": document.get("candidate_id"),
                    "evidence_kind": document.get("evidence_kind"),
                    "fold_count": document.get("fold_count"),
                    "created_at": document.get("created_at"),
                    "finished_at": document.get("finished_at"),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
    return buffer.getvalue()
