from __future__ import annotations

from typing import Any

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    ASSET_DISCOVERY_CANDIDATES_COLLECTION,
    ASSET_DISCOVERY_RUNS_COLLECTION,
    bson_value,
    utc_now,
)

EXPORT_SCHEMA_VERSION = 2
EXPORT_STATUSES = ("candidate", "watchlist", "rejected")


def _export_candidate(document: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "symbol",
        "status",
        "discovered_at",
        "discovered_api_version",
        "last_evaluated_at",
        "last_evaluated_api_version",
        "evaluation_policy_version",
        "next_evaluation_at",
        "historical_cache_ready",
        "history_profile",
        "model_ready",
        "history_start",
        "history_end",
        "history_sessions",
        "latest_close",
        "median_dollar_volume_63d",
        "nonzero_volume_ratio",
        "behavior_profile",
        "reason_codes",
        "evaluation_count",
    )
    return {
        key: bson_value(document.get(key))
        for key in allowed
        if document.get(key) is not None
    }


def _candidate_items(db: Any) -> list[dict[str, Any]]:
    status_order = {"candidate": 0, "watchlist": 1, "rejected": 2}
    documents = db[ASSET_DISCOVERY_CANDIDATES_COLLECTION].find(
        {"status": {"$in": list(EXPORT_STATUSES)}}
    )
    items = [_export_candidate(document) for document in documents]
    return sorted(
        items,
        key=lambda item: (
            status_order.get(str(item.get("status") or ""), 99),
            str(item.get("symbol") or ""),
        ),
    )


def _export_run(document: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "run_id",
        "source",
        "api_version",
        "evaluation_policy_version",
        "status",
        "phase",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
        "batch_size",
        "universe_size",
        "processed_count",
        "attempted_count",
        "candidate_count",
        "watchlist_count",
        "rejected_count",
        "failed_count",
        "skipped_count",
        "scan_limit",
        "cancel_requested",
    )
    return {
        key: bson_value(document.get(key))
        for key in allowed
        if document.get(key) is not None
    }


def _run_items(db: Any) -> list[dict[str, Any]]:
    cursor = db[ASSET_DISCOVERY_RUNS_COLLECTION].find({}).sort("created_at", -1)
    return [_export_run(document) for document in cursor]


def _run_totals(runs: list[dict[str, Any]]) -> dict[str, int]:
    fields = {
        "attempted": "attempted_count",
        "processed": "processed_count",
        "candidate": "candidate_count",
        "watchlist": "watchlist_count",
        "rejected": "rejected_count",
        "skipped": "skipped_count",
        "failed": "failed_count",
    }
    totals = {"runs": len(runs)}
    for output_key, source_key in fields.items():
        totals[output_key] = sum(int(run.get(source_key) or 0) for run in runs)
    return totals


def build_asset_discovery_export(
    db: Any,
    *,
    front_version: str | None = None,
) -> dict[str, Any]:
    candidates = _candidate_items(db)
    runs = _run_items(db)
    status_counts = {status: 0 for status in EXPORT_STATUSES}
    for candidate in candidates:
        status = str(candidate.get("status") or "")
        if status in status_counts:
            status_counts[status] += 1

    generated_at = utc_now()
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": bson_value(generated_at),
        "api_version": API_VERSION,
        "front_version": str(front_version or "").strip() or None,
        "summary": {
            **status_counts,
            "total_analytical_records": len(candidates),
            "run_totals": _run_totals(runs),
        },
        "candidates": candidates,
        "runs": runs,
    }
