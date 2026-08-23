from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
import uuid
import zipfile
import zlib
from typing import Any

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION,
    bson_value,
)
from .analysis import build_analysis
from .config import ANALYSIS_VERSION, SCHEMA_VERSION


def _artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        current = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            current = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        rows.extend(dict(row) for row in current if isinstance(row, dict))
    return rows


def _observation_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)},
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        current = document.get("rows") or []
        if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
            current = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
        for row in current:
            if not isinstance(row, dict):
                continue
            value = dict(row)
            value.setdefault("timestamp", document.get("timestamp"))
            rows.append(value)
    return rows


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION]
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_decision_science_id")
    collection.create_index(
        [("run_id", ASCENDING), ("analysis_version", ASCENDING), ("created_at", DESCENDING)],
        name="ix_decision_science_run_version_created",
    )




def _monthly_return_rows(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in values or []:
        if not isinstance(row, dict):
            continue
        month = str(row.get("month") or "").strip()
        try:
            value = float(row.get("return"))
        except (TypeError, ValueError):
            continue
        if month:
            result[month] = {"month": month, "return": value}
    return result


def _strategy_monthly_performance(db: Any, document: dict[str, Any]) -> dict[str, Any] | None:
    run_id = str(document.get("run_id") or "")
    processing_id = str(document.get("processing_id") or "")
    if not run_id:
        return None
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 0})
    if run is None:
        return None
    validation = run.get("strategy_research_final_validation") if isinstance(run.get("strategy_research_final_validation"), dict) else {}
    stateful_info = validation.get("stateful") if isinstance(validation.get("stateful"), dict) else {}
    stateful_id = str(stateful_info.get("id") or "")
    query: dict[str, Any] = {"run_id": run_id, "status": "completed"}
    if stateful_id:
        query = {"id": stateful_id}
    elif processing_id:
        query["processing_id"] = processing_id
    stateful = db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].find_one(
        query, {"_id": 0}, sort=[("created_at", DESCENDING)]
    )
    if stateful is None:
        return None
    candidate = stateful.get("candidate_a") if isinstance(stateful.get("candidate_a"), dict) else {}
    candidate_analytics = candidate.get("analytics") if isinstance(candidate.get("analytics"), dict) else {}
    candidate_metrics = candidate_analytics.get("metrics") if isinstance(candidate_analytics.get("metrics"), dict) else {}
    control_replay = stateful.get("control_replay") if isinstance(stateful.get("control_replay"), dict) else {}
    control_analytics = control_replay.get("analytics") if isinstance(control_replay.get("analytics"), dict) else {}
    control_metrics = control_analytics.get("metrics") if isinstance(control_analytics.get("metrics"), dict) else {}
    strategy_months = _monthly_return_rows(candidate_metrics.get("monthly_returns") or [])
    control_months = _monthly_return_rows(control_metrics.get("monthly_returns") or [])
    months = sorted(set(strategy_months).intersection(control_months))
    rows = [
        {
            "month": month,
            "simulation_return": strategy_months[month]["return"],
            "reference_return": control_months[month]["return"],
        }
        for month in months
    ]
    if not rows:
        return None
    return bson_value({
        "source": "strategy_research_final_validation",
        "stateful_replay_id": stateful.get("id"),
        "strategy_label": candidate.get("label") or "Strategy",
        "reference_label": "Control",
        "rows": rows,
    })


def _enrich_for_view(db: Any, document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = bson_value(dict(document))
    monthly = _strategy_monthly_performance(db, result)
    if monthly is not None:
        result["strategy_monthly_performance"] = monthly
    return result


def history(db: Any, *, limit: int = 30) -> dict[str, Any]:
    safe_limit = max(1, min(100, int(limit)))
    _ensure_indexes(db)
    rows = list(
        db[TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION].find(
            {"status": "completed"},
            {"_id": 0},
        ).sort("created_at", DESCENDING).limit(safe_limit)
    )
    return {"items": [_enrich_for_view(db, row) for row in rows]}

def latest(db: Any, *, run_id: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"status": "completed"}
    if run_id:
        query["run_id"] = str(run_id)
    document = db[TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION].find_one(
        query, {"_id": 0}, sort=[("created_at", DESCENDING)]
    )
    return _enrich_for_view(db, document)



def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for source in rows:
        row: dict[str, Any] = {}
        for key in fieldnames:
            value = source.get(key)
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(bson_value(value), ensure_ascii=False, default=str)
            else:
                row[key] = bson_value(value)
        writer.writerow(row)
    return buffer.getvalue()


def _model_metric_columns(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    values = metrics if isinstance(metrics, dict) else {}
    return {
        f"{prefix}_status": values.get("status") or "available",
        f"{prefix}_rows": values.get("rows"),
        f"{prefix}_positive_rate": values.get("positive_rate"),
        f"{prefix}_auc": values.get("auc"),
        f"{prefix}_brier": values.get("brier"),
        f"{prefix}_log_loss": values.get("log_loss"),
        f"{prefix}_calibration_error": values.get("calibration_error"),
        f"{prefix}_average_probability": values.get("average_probability"),
        f"{prefix}_message": values.get("message"),
    }


def _opportunity_fold_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    opportunity = document.get("absolute_opportunity") if isinstance(document.get("absolute_opportunity"), dict) else {}
    walk_forward = opportunity.get("walk_forward") if isinstance(opportunity.get("walk_forward"), dict) else {}
    rows: list[dict[str, Any]] = []
    for fold in walk_forward.get("folds") or []:
        if not isinstance(fold, dict):
            continue
        selection = fold.get("selection") if isinstance(fold.get("selection"), dict) else {}
        models = fold.get("models") if isinstance(fold.get("models"), dict) else {}
        row = {
            "fold_id": fold.get("fold_id"),
            "train_rows": fold.get("train_rows"),
            "test_rows": fold.get("test_rows"),
            "selected_model": fold.get("selected_model"),
            "selection_threshold": selection.get("threshold"),
        }
        row.update(_model_metric_columns("logistic_regression", models.get("logistic_regression")))
        row.update(_model_metric_columns("lightgbm", models.get("lightgbm")))
        rows.append(bson_value(row))
    return rows


def _opportunity_selection_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    opportunity = document.get("absolute_opportunity") if isinstance(document.get("absolute_opportunity"), dict) else {}
    walk_forward = opportunity.get("walk_forward") if isinstance(opportunity.get("walk_forward"), dict) else {}
    rows: list[dict[str, Any]] = []
    for fold in walk_forward.get("folds") or []:
        if not isinstance(fold, dict):
            continue
        selection = fold.get("selection") if isinstance(fold.get("selection"), dict) else {}
        selected = selection.get("selected")
        for candidate in selection.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            validation = candidate.get("validation") if isinstance(candidate.get("validation"), dict) else {}
            rows.append(bson_value({
                "fold_id": fold.get("fold_id"),
                "model": candidate.get("model"),
                "selected": candidate.get("model") == selected,
                "status": candidate.get("status") or "available",
                "threshold": candidate.get("threshold"),
                "threshold_balanced_accuracy": candidate.get("threshold_balanced_accuracy"),
                "validation_rows": validation.get("rows"),
                "validation_positive_rate": validation.get("positive_rate"),
                "validation_auc": validation.get("auc"),
                "validation_brier": validation.get("brier"),
                "validation_log_loss": validation.get("log_loss"),
                "validation_calibration_error": validation.get("calibration_error"),
                "validation_average_probability": validation.get("average_probability"),
                "message": candidate.get("message"),
            }))
    return rows


def _probability_bin_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    opportunity = document.get("absolute_opportunity") if isinstance(document.get("absolute_opportunity"), dict) else {}
    walk_forward = opportunity.get("walk_forward") if isinstance(opportunity.get("walk_forward"), dict) else {}
    rows: list[dict[str, Any]] = []
    for fold in walk_forward.get("folds") or []:
        if not isinstance(fold, dict):
            continue
        for item in fold.get("selected_model_probability_bins") or []:
            if not isinstance(item, dict):
                continue
            rows.append(bson_value({
                "fold_id": fold.get("fold_id"),
                "selected_model": fold.get("selected_model"),
                **item,
            }))
    return rows


def _transition_fold_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    transition = document.get("leader_transition") if isinstance(document.get("leader_transition"), dict) else {}
    rows: list[dict[str, Any]] = []
    for fold in transition.get("walk_forward_folds") or []:
        if not isinstance(fold, dict):
            continue
        selection = fold.get("selection") if isinstance(fold.get("selection"), dict) else {}
        models = fold.get("models") if isinstance(fold.get("models"), dict) else {}
        row = {
            "fold_id": fold.get("fold_id"),
            "train_rows": fold.get("train_rows"),
            "test_rows": fold.get("test_rows"),
            "positive_rate": fold.get("positive_rate"),
            "selected_model": selection.get("selected"),
            "selection_threshold": selection.get("threshold"),
        }
        row.update(_model_metric_columns("logistic_regression", models.get("logistic_regression")))
        row.update(_model_metric_columns("lightgbm", models.get("lightgbm")))
        rows.append(bson_value(row))
    return rows


def build_export(db: Any, analysis_id: str) -> bytes:
    document = db[TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION].find_one(
        {"id": str(analysis_id)}, {"_id": 0}
    )
    if document is None:
        raise ValueError("Decision Science analysis not found.")
    result = bson_value(document)
    run_id = str(result.get("run_id") or "")
    source_run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 0}) if run_id else None
    source_run_context = None
    if source_run is not None:
        source_run_context = bson_value({
            "id": source_run.get("id"),
            "status": source_run.get("status"),
            "analysis_end_date": source_run.get("analysis_end_date"),
            "research_processing_id": source_run.get("research_processing_id"),
            "market_data_snapshot_id": source_run.get("market_data_snapshot_id"),
            "strategy_profile_id": source_run.get("strategy_profile_id"),
            "strategy_profile_name": source_run.get("strategy_profile_name"),
            "strategy_profile_revision": source_run.get("strategy_profile_revision"),
            "strategy_configuration_hash": source_run.get("strategy_configuration_hash"),
            "strategy_kind": source_run.get("strategy_kind"),
            "temporal_strategy_variant": source_run.get("temporal_strategy_variant"),
            "strategy_research_pipeline": source_run.get("strategy_research_pipeline"),
            "strategy_research_final_validation": source_run.get("strategy_research_final_validation"),
        })
    observations = [bson_value(row) for row in _observation_rows(db, run_id)] if run_id else []
    winner_reference = [bson_value(row) for row in _artifact_rows(db, run_id, "winner_reference_daily")] if run_id else []
    opportunity = result.get("absolute_opportunity") if isinstance(result.get("absolute_opportunity"), dict) else {}
    shadow = opportunity.get("shadow_cash") if isinstance(opportunity.get("shadow_cash"), dict) else {}
    coefficients = [bson_value(row) for row in opportunity.get("logistic_interpretability") or [] if isinstance(row, dict)]
    monthly = [bson_value(row) for row in shadow.get("monthly") or [] if isinstance(row, dict)]
    recent_sessions = [bson_value(row) for row in shadow.get("recent_sessions") or [] if isinstance(row, dict)]
    transition = result.get("leader_transition") if isinstance(result.get("leader_transition"), dict) else {}
    monthly_performance = _strategy_monthly_performance(db, result) or {}
    monthly_performance_rows = [bson_value(row) for row in monthly_performance.get("rows") or [] if isinstance(row, dict)]

    files = [
        "decision_science_manifest.json",
        "decision_science_analysis.json",
        "decision_science_source_run_context.json",
        "decision_science_opportunity_folds.csv",
        "decision_science_model_selection.csv",
        "decision_science_probability_calibration.csv",
        "decision_science_cash_shadow_monthly.csv",
        "decision_science_cash_shadow_recent_sessions.csv",
        "decision_science_strategy_monthly_performance.csv",
        "decision_science_logistic_coefficients.csv",
        "decision_science_leader_transition_folds.csv",
        "decision_science_source_observations.csv",
        "decision_science_source_winner_reference_daily.csv",
    ]
    manifest = bson_value({
        "schema_version": 1,
        "analysis_id": result.get("id"),
        "analysis_version": result.get("analysis_version"),
        "analysis_schema_version": result.get("schema_version"),
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "processing_id": result.get("processing_id"),
        "market_data_snapshot_id": result.get("market_data_snapshot_id"),
        "analysis_end_date": result.get("analysis_end_date"),
        "strategy_profile_id": result.get("strategy_profile_id"),
        "strategy_profile_name": result.get("strategy_profile_name"),
        "strategy_profile_revision": result.get("strategy_profile_revision"),
        "strategy_configuration_hash": result.get("strategy_configuration_hash"),
        "created_at": result.get("created_at"),
        "method": result.get("method"),
        "absolute_opportunity_target": opportunity.get("target"),
        "leader_transition_status": transition.get("status"),
        "source_observation_rows": len(observations),
        "source_winner_reference_rows": len(winner_reference),
        "files": files,
    })

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("decision_science_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
        archive.writestr("decision_science_analysis.json", json.dumps(result, indent=2, ensure_ascii=False, default=str))
        archive.writestr("decision_science_source_run_context.json", json.dumps(source_run_context or {}, indent=2, ensure_ascii=False, default=str))
        archive.writestr("decision_science_opportunity_folds.csv", _csv_text(_opportunity_fold_rows(result)))
        archive.writestr("decision_science_model_selection.csv", _csv_text(_opportunity_selection_rows(result)))
        archive.writestr("decision_science_probability_calibration.csv", _csv_text(_probability_bin_rows(result)))
        archive.writestr("decision_science_cash_shadow_monthly.csv", _csv_text(monthly))
        archive.writestr("decision_science_cash_shadow_recent_sessions.csv", _csv_text(recent_sessions))
        archive.writestr("decision_science_strategy_monthly_performance.csv", _csv_text(monthly_performance_rows))
        archive.writestr("decision_science_logistic_coefficients.csv", _csv_text(coefficients))
        archive.writestr("decision_science_leader_transition_folds.csv", _csv_text(_transition_fold_rows(result)))
        archive.writestr("decision_science_source_observations.csv", _csv_text(observations))
        archive.writestr("decision_science_source_winner_reference_daily.csv", _csv_text(winner_reference))
    return buffer.getvalue()

def run_analysis(db: Any, run_id: str) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0})
    if run is None:
        raise ValueError("Temporal Intelligence run not found.")
    if str(run.get("status") or "").lower() != "completed":
        raise ValueError("Decision Science requires a completed Temporal Intelligence run.")
    pipeline = run.get("strategy_research_pipeline") if isinstance(run.get("strategy_research_pipeline"), dict) else {}
    if pipeline and str(pipeline.get("status") or "").lower() != "completed":
        raise ValueError("Decision Science requires a completed Strategy Research pipeline.")
    observations = _observation_rows(db, str(run_id))
    winner_rows = _artifact_rows(db, str(run_id), "winner_reference_daily")
    if not observations:
        raise ValueError("Temporal Intelligence observations are unavailable for this run.")
    if not winner_rows:
        raise ValueError("winner_reference_daily is unavailable for this run.")
    result = build_analysis(observations, winner_rows, run=run)
    now = datetime.now(timezone.utc)
    result.update({
        "id": str(uuid.uuid4()),
        "analysis_version": ANALYSIS_VERSION,
        "created_at": now,
        "updated_at": now,
    })
    _ensure_indexes(db)
    db[TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION].insert_one(bson_value(dict(result)))
    return _enrich_for_view(db, result)
