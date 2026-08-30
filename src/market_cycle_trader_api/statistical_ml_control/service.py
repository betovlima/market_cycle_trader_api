from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
import uuid
import zlib

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_STATISTICAL_ML_CONTROL_COLLECTION,
    bson_value,
)
from ..services.temporal_research_settings import temporal_research_settings_snapshot
from .analysis import build_analysis
from .config import SCHEMA_VERSION


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_STATISTICAL_ML_CONTROL_COLLECTION]
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_statistical_ml_control_id")
    collection.create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_statistical_ml_control_scope",
    )


def _decode_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


def _winner_reference_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": "winner_reference_daily"},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for document in cursor:
        rows.extend(_decode_rows(document))
    return rows


def _observation_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)},
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        timestamp = document.get("timestamp")
        for row in _decode_rows(document):
            rows.append({"timestamp": timestamp, **row})
    return rows


def get_persisted(
    db: Any,
    run_id: str,
    *,
    processing_id: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id), "schema_version": {"$gte": SCHEMA_VERSION}}
    if processing_id:
        query["processing_id"] = str(processing_id)
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    row = db[TEMPORAL_STATISTICAL_ML_CONTROL_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def build_and_persist(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    settings_snapshot = temporal_research_settings_snapshot(db)
    settings = ((settings_snapshot.get("settings") or {}).get("statistical_ml_control") or {})
    result = build_analysis(
        reference_rows=_winner_reference_rows(db, run_id),
        observation_rows=_observation_rows(db, run_id),
        settings=settings,
        run_id=run_id,
        processing_id=processing_id,
        period_start=start_month,
        period_end=end_month,
    )
    now = datetime.now(timezone.utc)
    result.update({
        "id": str(uuid.uuid4()),
        "research_settings": settings_snapshot,
        "created_at": now,
        "updated_at": now,
    })
    _ensure_indexes(db)
    db[TEMPORAL_STATISTICAL_ML_CONTROL_COLLECTION].insert_one(bson_value(dict(result)))
    return bson_value(result)


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("_id", None)
    payload.pop("predictions", None)
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[TEMPORAL_STATISTICAL_ML_CONTROL_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
