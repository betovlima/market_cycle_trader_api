from __future__ import annotations

from datetime import datetime, timezone
import json
import threading
from typing import Any, Callable
import uuid
import zlib

from pymongo import ASCENDING, DESCENDING

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    bson_value,
)
from ..services.temporal_research_settings import temporal_research_settings_snapshot
from .analysis import AssetStateClusteringCancelled, build_analysis
from .config import ANALYSIS_VERSION, SCHEMA_VERSION

TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION = "temporal_asset_state_clustering"
TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION = "temporal_asset_state_clustering_points"
HEARTBEAT_INTERVAL_SECONDS = 2.0


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


def _checkpoint_results(db: Any, analysis_id: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    cursor = db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].find(
        {"analysis_id": str(analysis_id), "checkpoint_status": "completed"},
        {
            "_id": 0,
            "symbol": 1,
            "encoding": 1,
            "payload": 1,
            "rows": 1,
            "asset_summary": 1,
            "latest_map": 1,
        },
    )
    for document in cursor:
        symbol = str(document.get("symbol") or "").strip().upper()
        if not symbol or not isinstance(document.get("asset_summary"), dict):
            continue
        results[symbol] = {
            "symbol": symbol,
            "states": _decode_rows(document),
            "asset_summary": dict(document.get("asset_summary") or {}),
            "latest_map": (
                dict(document["latest_map"])
                if isinstance(document.get("latest_map"), dict)
                else None
            ),
        }
    return results


def _progress_payload(
    *,
    completed_assets: int,
    total_assets: int,
    last_completed_symbol: str | None,
    heartbeat_at: datetime,
) -> dict[str, Any]:
    percent = 100.0 if total_assets == 0 else 100.0 * completed_assets / total_assets
    return {
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "completed_assets": int(completed_assets),
        "total_assets": int(total_assets),
        "last_completed_symbol": last_completed_symbol,
        "heartbeat_at": heartbeat_at,
    }


def _persist_checkpoint(
    db: Any,
    *,
    analysis_id: str,
    run_id: str,
    result: dict[str, Any],
) -> None:
    symbol = str(result.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Asset State Clustering checkpoint is missing its symbol.")
    rows = list(result.get("states") or [])
    raw = json.dumps(
        bson_value(rows),
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    now = datetime.now(timezone.utc)
    db[TEMPORAL_ASSET_STATE_CLUSTERING_POINTS_COLLECTION].update_one(
        {"analysis_id": str(analysis_id), "symbol": symbol},
        {
            "$set": {
                "analysis_id": str(analysis_id),
                "run_id": str(run_id),
                "symbol": symbol,
                "checkpoint_status": "completed",
                "encoding": "zlib-json-v1",
                "payload": zlib.compress(raw, level=6),
                "rows_count": len(rows),
                "asset_summary": bson_value(result.get("asset_summary") or {}),
                "latest_map": bson_value(result.get("latest_map")),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def build_and_persist(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _ensure_indexes(db)
    existing = get_persisted(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if existing and str(existing.get("status") or "").lower() == "completed":
        return existing

    settings_snapshot = temporal_research_settings_snapshot(db)
    settings = ((settings_snapshot.get("settings") or {}).get("asset_state_clustering") or {})
    existing_settings = (existing or {}).get("research_settings") or {}
    resume_compatible = bool(
        existing
        and existing.get("id")
        and str(existing.get("analysis_version") or "") == ANALYSIS_VERSION
        and str(existing_settings.get("settings_hash") or "")
        == str(settings_snapshot.get("settings_hash") or "")
    )
    if existing and not resume_compatible:
        existing = None
    observation_rows = _observation_rows(db, run_id)
    observed_symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in observation_rows
        if str(row.get("symbol") or "").strip()
    }
    now = datetime.now(timezone.utc)
    analysis_id = str((existing or {}).get("id") or uuid.uuid4())
    created_at = (existing or {}).get("created_at") or now
    base_document = {
        "id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "running",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(start_month),
        "period_end": str(end_month),
        "shadow_only": True,
        "decision_effect": "none",
        "research_settings": settings_snapshot,
        "created_at": created_at,
        "started_at": (existing or {}).get("started_at") or now,
        "heartbeat_at": now,
        "updated_at": now,
    }
    if existing:
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].update_one(
            {"id": analysis_id},
            {
                "$set": bson_value(base_document),
                "$unset": {
                    "failure_message": "",
                    "failed_at": "",
                    "stopped_at": "",
                    "finished_at": "",
                },
            },
        )
    else:
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].insert_one(
            bson_value(base_document)
        )

    completed_results = _checkpoint_results(db, analysis_id)
    progress_lock = threading.Lock()
    progress_state: dict[str, Any] = {
        "completed_assets": len(completed_results),
        "total_assets": len(observed_symbols),
        "last_completed_symbol": ((existing or {}).get("progress") or {}).get("last_completed_symbol"),
    }
    cancel_event = threading.Event()
    heartbeat_stop = threading.Event()

    def publish_progress(*, heartbeat_only: bool = False) -> None:
        heartbeat_at = datetime.now(timezone.utc)
        with progress_lock:
            progress = _progress_payload(
                completed_assets=int(progress_state["completed_assets"]),
                total_assets=int(progress_state["total_assets"]),
                last_completed_symbol=progress_state.get("last_completed_symbol"),
                heartbeat_at=heartbeat_at,
            )
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].update_one(
            {"id": analysis_id, "status": "running"},
            {
                "$set": {
                    "progress": bson_value(progress),
                    "heartbeat_at": heartbeat_at,
                    "updated_at": heartbeat_at,
                }
            },
        )
        if progress_callback is not None:
            try:
                progress_callback({**progress, "heartbeat_only": bool(heartbeat_only)})
            except Exception:
                pass

    def heartbeat_worker() -> None:
        while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                if cancel_check is not None and bool(cancel_check()):
                    cancel_event.set()
                publish_progress(heartbeat_only=True)
            except Exception:
                continue

    if cancel_check is not None and bool(cancel_check()):
        cancel_event.set()
    publish_progress()
    heartbeat_thread = threading.Thread(
        target=heartbeat_worker,
        name="asset-state-clustering-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    def checkpoint(
        asset_result: dict[str, Any], completed_assets: int, total_assets: int
    ) -> None:
        _persist_checkpoint(
            db,
            analysis_id=analysis_id,
            run_id=run_id,
            result=asset_result,
        )
        with progress_lock:
            progress_state["completed_assets"] = int(completed_assets)
            progress_state["total_assets"] = int(total_assets)
            progress_state["last_completed_symbol"] = str(
                asset_result.get("symbol") or ""
            )
        publish_progress()

    try:
        result = build_analysis(
            observation_rows=observation_rows,
            settings=settings,
            run_id=run_id,
            processing_id=processing_id,
            period_start=start_month,
            period_end=end_month,
            checkpoint_callback=checkpoint,
            cancel_check=cancel_event.is_set,
            completed_asset_results=completed_results,
        )
        result.pop("daily_states_by_symbol", None)

        finished_at = datetime.now(timezone.utc)
        total_assets = int((result.get("summary") or {}).get("asset_count") or 0)
        final_progress = _progress_payload(
            completed_assets=total_assets,
            total_assets=total_assets,
            last_completed_symbol=progress_state.get("last_completed_symbol"),
            heartbeat_at=finished_at,
        )
        result.update({
            "id": analysis_id,
            "research_settings": settings_snapshot,
            "created_at": created_at,
            "started_at": base_document["started_at"],
            "finished_at": finished_at,
            "heartbeat_at": finished_at,
            "progress": final_progress,
            "updated_at": finished_at,
        })
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].replace_one(
            {"id": analysis_id}, bson_value(dict(result)), upsert=True
        )
        if progress_callback is not None:
            try:
                progress_callback(final_progress)
            except Exception:
                pass
        return bson_value(result)
    except AssetStateClusteringCancelled:
        stopped_at = datetime.now(timezone.utc)
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].update_one(
            {"id": analysis_id},
            {
                "$set": {
                    "status": "stopped",
                    "stopped_at": stopped_at,
                    "heartbeat_at": stopped_at,
                    "updated_at": stopped_at,
                }
            },
        )
        raise
    except Exception as exc:
        failed_at = datetime.now(timezone.utc)
        db[TEMPORAL_ASSET_STATE_CLUSTERING_COLLECTION].update_one(
            {"id": analysis_id},
            {
                "$set": {
                    "status": "failed",
                    "failure_message": str(exc),
                    "failed_at": failed_at,
                    "heartbeat_at": failed_at,
                    "updated_at": failed_at,
                }
            },
        )
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


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
