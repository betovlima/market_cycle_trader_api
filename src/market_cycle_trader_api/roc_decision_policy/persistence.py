from __future__ import annotations

from typing import Any

from pymongo import DESCENDING

from ..infrastructure.persistence.mongo_repository import ROC_DECISION_POLICY_RESULTS_COLLECTION as RESULTS_COLLECTION, bson_value, utc_now


def latest_raw(db: Any, run_id: str, *, processing_id: str | None = None, start_month: str | None = None, end_month: str | None = None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id)}
    if processing_id:
        query["processing_id"] = str(processing_id)
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    row = db[RESULTS_COLLECTION].find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return bson_value(row) if row is not None else None


def persist(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    payload = bson_value({**document, "created_at": document.get("created_at") or now, "updated_at": now})
    db[RESULTS_COLLECTION].replace_one({"id": payload["id"]}, payload, upsert=True)
    return bson_value(payload)


def delete_run_results(db: Any, run_id: str) -> int:
    return int(db[RESULTS_COLLECTION].delete_many({"run_id": str(run_id)}).deleted_count or 0)


def public_summary(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    payload = dict(document)
    payload.pop("technical_error", None)
    payload.pop("decision_diagnostics", None)
    payload.pop("equity", None)
    return bson_value(payload)
