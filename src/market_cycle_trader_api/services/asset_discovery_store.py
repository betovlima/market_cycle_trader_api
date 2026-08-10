from __future__ import annotations

from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    ASSET_DISCOVERY_CANDIDATES_COLLECTION,
    ASSET_DISCOVERY_RUNS_COLLECTION,
    bson_value,
    utc_now,
)

ACTIVE_KEY = "asset-discovery"


def public_run(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    allowed = (
        "run_id", "source", "api_version", "evaluation_policy_version", "status", "phase", "created_at", "started_at", "updated_at",
        "finished_at", "requested_by", "batch_size", "universe_size", "processed_count",
        "attempted_count", "candidate_count", "watchlist_count", "rejected_count", "failed_count",
        "skipped_count", "scan_limit", "current_symbol",
        "last_message", "cancel_requested",
    )
    return {
        key: bson_value(document.get(key))
        for key in allowed
        if document.get(key) is not None
    }


def append_run_update(
    db: Any,
    run_id: str,
    *,
    message: str,
    changes: dict[str, Any] | None = None,
) -> None:
    now = utc_now()
    db[ASSET_DISCOVERY_RUNS_COLLECTION].update_one(
        {"run_id": run_id},
        {
            "$set": bson_value({"updated_at": now, "last_message": message, **(changes or {})}),
            "$push": {"logs": {"$each": [f"{now.isoformat()} — {message}"], "$slice": -100}},
        },
    )


def finish_run(db: Any, run_id: str, *, status: str, message: str) -> None:
    now = utc_now()
    db[ASSET_DISCOVERY_RUNS_COLLECTION].update_one(
        {"run_id": run_id},
        {
            "$set": {
                "status": status,
                "phase": status,
                "updated_at": now,
                "finished_at": now,
                "last_message": message,
                "current_symbol": None,
            },
            "$unset": {"active_key": ""},
            "$push": {"logs": {"$each": [f"{now.isoformat()} — {message}"], "$slice": -100}},
        },
    )


def public_candidate(document: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "symbol", "status", "discovered_at", "discovered_api_version", "last_evaluated_at",
        "last_evaluated_api_version", "evaluation_policy_version", "next_evaluation_at",
        "historical_cache_ready", "history_profile", "model_ready", "history_start", "history_end",
        "history_sessions", "latest_close", "median_dollar_volume_63d", "nonzero_volume_ratio", "reason_codes",
        "evaluation_count", "last_error",
    )
    return {
        key: bson_value(document.get(key))
        for key in allowed
        if document.get(key) is not None
    }


def list_candidates(
    db: Any,
    *,
    status: str | None = None,
    query: str | None = None,
    limit: int = 250,
) -> list[dict[str, Any]]:
    mongo_query: dict[str, Any] = {}
    normalized_status = str(status or "").strip().lower()
    if normalized_status and normalized_status != "all":
        mongo_query["status"] = normalized_status
    normalized_query = str(query or "").strip().upper()
    if normalized_query:
        mongo_query["symbol"] = {"$regex": normalized_query, "$options": "i"}
    cursor = (
        db[ASSET_DISCOVERY_CANDIDATES_COLLECTION]
        .find(mongo_query)
        .sort([("status", 1), ("last_evaluated_at", -1), ("symbol", 1)])
        .limit(max(1, min(int(limit), 1000)))
    )
    return [public_candidate(item) for item in cursor]


def list_runs(db: Any, *, limit: int = 30) -> list[dict[str, Any]]:
    cursor = (
        db[ASSET_DISCOVERY_RUNS_COLLECTION]
        .find({})
        .sort("created_at", -1)
        .limit(max(1, min(int(limit), 100)))
    )
    return [public_run(item) or {} for item in cursor]


def candidate_counts(db: Any) -> dict[str, int]:
    counts = {"candidate": 0, "watchlist": 0, "rejected": 0, "evaluating": 0, "skipped": 0, "failed": 0}
    for item in db[ASSET_DISCOVERY_CANDIDATES_COLLECTION].aggregate(
        [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    ):
        key = str(item.get("_id") or "")
        if key in counts:
            counts[key] = int(item.get("count") or 0)
    return counts
