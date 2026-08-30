from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
import uuid
import zlib

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    bson_value,
)
from ..services.temporal_research_settings import temporal_research_settings_snapshot
from .analysis import build_analysis
from .config import SCHEMA_VERSION

TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION = "temporal_asset_state_clustering"
TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION = "temporal_asset_state_clustering_points"


def _ensure_indexes(db: Any) -> None:
    db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].create_index([("id", ASCENDING)], unique=True, name="uq_asset_state_clustering_id")
    db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_asset_state_clustering_scope",
    )
    db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].create_index(
        [("analysis_id", ASCENDING), ("symbol", ASCENDING)], unique=True, name="uq_asset_state_points_symbol"
    )
    db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].create_index([("run_id", ASCENDING)], name="ix_asset_state_points_run")


def _decode_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


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
    row = db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def build_and_persist(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    settings_snapshot = temporal_research_settings_snapshot(db)
    settings = ((settings_snapshot.get("settings") or {}).get("asset_state_clustering") or {})
    result = build_analysis(
        observation_rows=_observation_rows(db, run_id),
        settings=settings,
        run_id=run_id,
        processing_id=processing_id,
        period_start=start_month,
        period_end=end_month,
    )
    daily_states_by_symbol = dict(result.pop("daily_states_by_symbol", {}) or {})
    now = datetime.now(timezone.utc)
    analysis_id = str(uuid.uuid4())
    result.update({
        "id": analysis_id,
        "research_settings": settings_snapshot,
        "created_at": now,
        "updated_at": now,
    })
    _ensure_indexes(db)
    db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].insert_one(bson_value(dict(result)))
    point_docs = []
    for symbol, rows in daily_states_by_symbol.items():
        raw = json.dumps(bson_value(rows), separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
        point_docs.append({
            "analysis_id": analysis_id,
            "run_id": str(run_id),
            "symbol": str(symbol),
            "encoding": "zlib-json-v1",
            "payload": zlib.compress(raw, level=6),
            "rows_count": len(rows),
            "created_at": now,
        })
    if point_docs:
        db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].insert_many(point_docs, ordered=False)
    return bson_value(result)


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("_id", None)
    return bson_value(payload)


def get_symbol_states(db: Any, analysis_id: str, symbol: str) -> list[dict[str, Any]]:
    row = db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].find_one(
        {"analysis_id": str(analysis_id), "symbol": str(symbol).strip().upper()}, {"_id": 0}
    )
    return _decode_rows(row or {})


def iter_all_states(db: Any, analysis_id: str):
    cursor = db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].find({"analysis_id": str(analysis_id)}, {"_id": 0}).sort("symbol", 1)
    for row in cursor:
        for item in _decode_rows(row):
            yield item


def delete_run_results(db: Any, run_id: str) -> int:
    analyses = list(db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].find({"run_id": str(run_id)}, {"id": 1, "_id": 0}))
    ids = [str(item.get("id")) for item in analyses if item.get("id")]
    deleted = int(db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
    if ids:
        db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].delete_many({"analysis_id": {"$in": ids}})
    else:
        db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].delete_many({"run_id": str(run_id)})
    return deleted
