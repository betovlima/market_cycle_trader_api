from __future__ import annotations

import json
import zlib
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
)
from ..schemas.requests import BacktestExecutionRequest
from ..services.temporal_model.inputs import load_frozen_bars
from ..services.temporal_model.preprocessing import prepare_training_context
from .errors import RocDecisionPolicyConflict, RocDecisionPolicyNotFound


def load_source_run(db: Any, run_id: str, processing_id: str) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if run is None:
        raise RocDecisionPolicyNotFound("Temporal Intelligence run not found.")
    if str(run.get("status") or "").lower() != "completed":
        raise RocDecisionPolicyConflict("ROC Decision Policy requires a completed Temporal Intelligence run.")
    expected = str(run.get("research_processing_id") or "").strip()
    if not expected or expected != str(processing_id or "").strip():
        raise RocDecisionPolicyConflict("ROC Decision Policy must use the processing bound to the Temporal Intelligence run.")
    if not isinstance(run.get("request"), dict):
        raise RocDecisionPolicyConflict("Temporal Intelligence run does not contain its immutable execution request.")
    return run


def _decoded_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


def load_observations(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)},
        {"_id": 0, "timestamp": 1, "rows": 1, "encoding": 1, "payload": 1},
    ).sort("timestamp", 1)
    for item in cursor:
        for row in _decoded_rows(item):
            rows.append(bson_value({"timestamp": item.get("timestamp"), **row}))
    if not rows:
        raise RocDecisionPolicyConflict("Temporal Intelligence observations are unavailable for ROC Decision Policy.")
    return rows


def load_artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        rows.extend(bson_value(row) for row in _decoded_rows(item))
    return rows


def load_embedded_artifact_rows(db: Any, run_id: str, artifact_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": "decision_diagnostics"},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        for row in _decoded_rows(item):
            if str(row.get("artifact_kind") or "") == str(artifact_kind):
                rows.append(bson_value(row))
    return rows


def prepare_inputs(db: Any, run: dict[str, Any], processing_id: str) -> dict[str, Any]:
    request = BacktestExecutionRequest.model_validate(run["request"])
    bars = load_frozen_bars(request)
    training = prepare_training_context(bars, request)
    run_id = str(run.get("id") or "")
    winner_daily = load_artifact_rows(db, run_id, "winner_reference_daily")
    if not winner_daily:
        raise RocDecisionPolicyConflict("Immutable Winner/reference daily rows are unavailable for ROC Decision Policy.")
    temporal_curve = load_embedded_artifact_rows(db, run_id, "multi_horizon_equity_curve")
    if not temporal_curve:
        raise RocDecisionPolicyConflict("Frozen Temporal economic replay rows are unavailable for ROC Decision Policy.")
    return {
        "request": request,
        "training": training,
        "observations": load_observations(db, run_id),
        "winner_daily": winner_daily,
        "temporal_curve": temporal_curve,
    }
