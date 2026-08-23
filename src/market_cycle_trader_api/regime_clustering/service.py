from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION, bson_value
from ..leadership_regime.service import get_persisted as get_leadership_regime
from ..services.analytics import processing_analytics
from .analysis import build_analysis
from .config import SCHEMA_VERSION


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION]
    collection.create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_regime_clustering_scope_created",
    )
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_regime_clustering_id")


def get_persisted(db: Any, run_id: str, *, processing_id: str | None = None, start_month: str | None = None, end_month: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id), "schema_version": {"$gte": SCHEMA_VERSION}}
    if processing_id:
        query["processing_id"] = str(processing_id)
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    row = db[TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def _save(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    _ensure_indexes(db)
    db[TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION].insert_one(dict(document))
    return document


def build_and_persist(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    leadership = get_leadership_regime(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if not leadership or str(leadership.get("status") or "").lower() != "completed":
        raise ValueError("Regime Clustering requires completed Leadership Regime diagnostics.")
    analytics = processing_analytics(db, processing_id)
    monthly_returns = analytics.get("monthly_returns") or ((analytics.get("metrics") or {}).get("monthly_returns")) or []
    result = build_analysis(leadership, monthly_returns, run_id=run_id, processing_id=processing_id, period_start=start_month, period_end=end_month)
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
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
