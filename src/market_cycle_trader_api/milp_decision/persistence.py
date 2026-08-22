from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import bson_value
from .config import COLLECTION


def ensure_indexes(db: Any) -> None:
    collection = db[COLLECTION]
    collection.create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_milp_decision_scope_created",
    )
    collection.create_index([("id", ASCENDING)], unique=True, name="uq_milp_decision_id")


def save(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    ensure_indexes(db)
    db[COLLECTION].insert_one(dict(document))
    return document


def latest_raw(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any] | None:
    row = db[COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
            "schema_version": {"$gte": 3},
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    return bson_value(row) if row is not None else None


def get_completed(db: Any, run_id: str, optimization_id: str) -> dict[str, Any] | None:
    row = db[COLLECTION].find_one({
        "id": str(optimization_id),
        "run_id": str(run_id),
        "status": "completed",
        "schema_version": {"$gte": 3},
    })
    return bson_value(row) if row is not None else None


def public_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("_id", None)
    payload.pop("decisions", None)
    analytics = payload.get("analytics") if isinstance(payload.get("analytics"), dict) else None
    if analytics is not None:
        compact = dict(analytics)
        compact.pop("equity", None)
        compact.pop("rotations", None)
        payload["analytics"] = compact
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)
