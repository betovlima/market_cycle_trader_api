from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
import uuid
import zlib

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    bson_value,
)
from .analysis import build_analysis
from .config import SCHEMA_VERSION


def _artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        values = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            values = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        rows.extend(dict(value) for value in values if isinstance(value, dict))
    return rows


def _observation_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)}, {"_id": 0, "timestamp": 1, "rows": 1, "encoding": 1, "payload": 1}
    ).sort("timestamp", 1)
    for item in cursor:
        timestamp = item.get("timestamp")
        values = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            values = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        for value in values:
            if isinstance(value, dict):
                rows.append({"timestamp": timestamp, **dict(value)})
    return rows


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION]
    collection.create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_emerging_trend_scope_created",
    )
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_emerging_trend_id")


def get_persisted(db: Any, run_id: str, *, processing_id: str | None = None, start_month: str | None = None, end_month: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id), "schema_version": {"$gte": SCHEMA_VERSION}}
    if processing_id:
        query["processing_id"] = str(processing_id)
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    row = db[TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def _save(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    _ensure_indexes(db)
    db[TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION].insert_one(dict(document))
    return document


def build_and_persist(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    winner_rows = _artifact_rows(db, run_id, "winner_reference_daily")
    observation_rows = _observation_rows(db, run_id)
    result = build_analysis(
        winner_rows,
        observation_rows,
        run_id=run_id,
        processing_id=processing_id,
        period_start=start_month,
        period_end=end_month,
    )
    now = datetime.now(timezone.utc)
    result.update({"id": str(uuid.uuid4()), "created_at": now, "updated_at": now})
    return bson_value(_save(db, bson_value(result)))


def unavailable(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str, message: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    payload = bson_value({
        "id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(start_month),
        "period_end": str(end_month),
        "failure_message": str(message),
        "created_at": now,
        "updated_at": now,
    })
    return bson_value(_save(db, payload))


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("_id", None)
    payload.pop("sessions", None)
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
