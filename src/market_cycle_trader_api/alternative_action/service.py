from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
import zlib
from typing import Any

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    bson_value,
)
from .analysis import build_analysis
from .config import SCHEMA_VERSION


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION]
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_risk_aware_alternative_action_id")
    collection.create_index([("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)], name="ix_risk_aware_alternative_action_scope")


def _market_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    columns = ("timestamp", "fold_id", "symbol", "execution_date", "execution_open")
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find({"run_id": str(run_id)}, {"_id": 0, "timestamp": 1, "rows": 1, "encoding": 1, "payload": 1}).sort("timestamp", 1)
    for document in cursor:
        observation_rows = document.get("rows") or []
        if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
            observation_rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
        timestamp = document.get("timestamp")
        for item in observation_rows:
            if not isinstance(item, dict):
                continue
            if item.get("execution_open") is None or item.get("execution_date") is None:
                continue
            rows.append({key: (timestamp if key == "timestamp" else item.get(key)) for key in columns})
    return rows


def get_persisted(db: Any, run_id: str, *, processing_id: str | None = None, start_month: str | None = None, end_month: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id), "schema_version": {"$gte": SCHEMA_VERSION}}
    if processing_id: query["processing_id"] = str(processing_id)
    if start_month: query["period_start"] = str(start_month)
    if end_month: query["period_end"] = str(end_month)
    row = db[TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def build_and_persist(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one({"run_id": str(run_id), "processing_id": str(processing_id), "period_start": str(start_month), "period_end": str(end_month), "status": "completed"}, {"_id": 0}, sort=[("created_at", DESCENDING)])
    if not risk:
        raise ValueError("Risk-Aware Alternative Action requires a completed OOS risk detector result.")
    result = build_analysis(risk=bson_value(risk), market_rows=_market_rows(db, run_id), run_id=run_id, processing_id=processing_id, period_start=start_month, period_end=end_month)
    now = datetime.now(timezone.utc)
    result.update({"id": str(uuid.uuid4()), "created_at": now, "updated_at": now})
    _ensure_indexes(db)
    db[TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION].insert_one(bson_value(dict(result)))
    return bson_value(result)


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict): return None
    payload = dict(document); payload.pop("_id", None); return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
