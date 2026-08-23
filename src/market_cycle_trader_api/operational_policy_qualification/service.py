from __future__ import annotations

from typing import Any
import uuid

from pymongo import ASCENDING, DESCENDING

from ..alternative_action.service import _market_rows
from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    bson_value,
    utc_now,
)
from ..services.analytics import processing_analytics
from ..services.temporal_winner_transition_attribution import get_winner_transition_attribution
from .analysis import build_analysis
from .config import SCHEMA_VERSION


def _ensure_indexes(db: Any) -> None:
    collection = db[TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION]
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_operational_policy_qualification_id")
    collection.create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_operational_policy_qualification_scope",
    )


def get_persisted(db: Any, run_id: str, *, processing_id: str | None = None, start_month: str | None = None, end_month: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id), "schema_version": {"$gte": SCHEMA_VERSION}}
    if processing_id:
        query["processing_id"] = str(processing_id)
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    row = db[TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def build_and_persist(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing
    risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one(
        {"run_id": str(run_id), "processing_id": str(processing_id), "period_start": str(start_month), "period_end": str(end_month), "status": "completed"},
        {"_id": 0}, sort=[("created_at", DESCENDING)],
    )
    if not risk:
        raise ValueError("Operational Policy Qualification requires completed OOS transition risk research.")
    attribution = get_winner_transition_attribution(db, run_id, start_month=start_month, end_month=end_month)
    if not attribution or not (attribution.get("items") or []):
        raise ValueError("Operational Policy Qualification requires transition attribution history.")
    result = build_analysis(
        transition_attribution=bson_value(attribution),
        risk=bson_value(risk),
        market_rows=_market_rows(db, run_id),
        analytics=processing_analytics(db, processing_id),
        run_id=run_id,
        processing_id=processing_id,
        period_start=start_month,
        period_end=end_month,
    )
    now = utc_now()
    result.update({"id": str(uuid.uuid4()), "created_at": now, "updated_at": now})
    _ensure_indexes(db)
    db[TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION].insert_one(bson_value(dict(result)))
    return bson_value(result)


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("_id", None)
    replay = payload.get("replay") if isinstance(payload.get("replay"), dict) else {}
    if replay:
        compact = dict(replay)
        compact.pop("equity", None)
        payload["replay"] = compact
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
