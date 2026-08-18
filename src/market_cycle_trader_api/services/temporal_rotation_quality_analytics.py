from __future__ import annotations

import json
import zlib
from collections import OrderedDict
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION,
    bson_value,
)
from .analytics import _rotation_period_from_data, analytics_from_equity_rotations


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _stitch_equity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
    for raw in rows or []:
        row = dict(raw)
        try:
            fold_id = int(row.get("fold_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(fold_id, []).append(row)

    stitched: list[dict[str, Any]] = []
    compound_base = 10_000.0
    for fold_id in sorted(grouped):
        fold_rows = sorted(grouped[fold_id], key=lambda item: str(item.get("decision_timestamp") or item.get("timestamp") or ""))
        if not fold_rows:
            continue
        scale = compound_base / 10_000.0
        for row in fold_rows:
            raw_equity = _as_float(row.get("strategy_equity"))
            if raw_equity is None:
                continue
            stitched.append({
                "fold_id": fold_id,
                "timestamp": str(row.get("decision_timestamp") or row.get("timestamp") or ""),
                "value": raw_equity * scale,
            })
        if stitched:
            compound_base = stitched[-1]["value"]
    return stitched


def _rotations_from_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rotations: list[dict[str, Any]] = []
    for row in sorted(rows or [], key=lambda item: (int(item.get("fold_id") or 0), str(item.get("decision_timestamp") or ""))):
        from_asset = str(row.get("simulated_current_symbol") or "CASH").upper()
        to_asset = str(row.get("chosen_target_symbol") or from_asset).upper()
        if from_asset == to_asset:
            continue
        rotations.append({
            "executed_at": str(row.get("decision_timestamp") or ""),
            "fold_id": int(row.get("fold_id") or 0),
            "from_asset": from_asset,
            "to_asset": to_asset,
            "rotation_blocked": bool(row.get("rotation_blocked")),
            "strong_challenger_override": bool(row.get("strong_challenger_override")),
            "drawdown_before": row.get("drawdown_before"),
            "incumbent_entry_rank_score": row.get("incumbent_entry_rank_score"),
            "challenger_entry_rank_score": row.get("challenger_entry_rank_score"),
            "challenger_minus_incumbent_score": row.get("challenger_minus_incumbent_score"),
            "challenger_quality_floor": row.get("challenger_quality_floor"),
            "transaction_fees": 0.0,
        })
    return rotations


def _combined_equity(candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate = _stitch_equity_rows(candidate_rows)
    control = _stitch_equity_rows(control_rows)
    control_map = {(int(row["fold_id"]), str(row["timestamp"])): float(row["value"]) for row in control}
    output: list[dict[str, Any]] = []
    for row in candidate:
        key = (int(row["fold_id"]), str(row["timestamp"]))
        reference = control_map.get(key)
        if reference is None:
            continue
        output.append({
            "timestamp": str(row["timestamp"]),
            "fold_id": int(row["fold_id"]),
            "simulation_equity": float(row["value"]),
            "reference_equity": float(reference),
        })
    return output


def _pack(payload: dict[str, Any]) -> bytes:
    clean = bson_value(payload)
    return zlib.compress(json.dumps(clean, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"), level=9)


def _unpack(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    return json.loads(zlib.decompress(bytes(payload)).decode("utf-8"))


def persist_rotation_quality_analytics(
    db: Any,
    *,
    processing_id: str,
    research_id: str,
    processing_kind: str,
    source_run_id: str,
    candidate_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    candidate_equity_rows: list[dict[str, Any]],
    control_equity_rows: list[dict[str, Any]],
    created_at: Any = None,
    finished_at: Any = None,
) -> None:
    candidate_id = str(candidate_metrics.get("candidate_id") or "").strip()
    if not candidate_id or candidate_id == "CONTROL":
        return
    equity = _combined_equity(candidate_equity_rows, control_equity_rows)
    rotations = _rotations_from_replay_rows(candidate_equity_rows)
    generic_metrics = {
        "ending_capital": candidate_metrics.get("ending_capital"),
        "simulation_return": candidate_metrics.get("total_return"),
        "sharpe": candidate_metrics.get("sharpe"),
        "maximum_drawdown": candidate_metrics.get("max_drawdown"),
        "position_changes": candidate_metrics.get("switch_count"),
    }
    analytics = analytics_from_equity_rotations(
        processing_id=str(processing_id),
        equity=equity,
        rotations=rotations,
        metrics=generic_metrics,
        created_at=created_at,
        finished_at=finished_at,
        processing_kind=f"rotation_quality_{processing_kind}",
        processing_label=f"Rotation Quality {processing_kind.title()} · {candidate_id}",
        reference_label="Control",
    )
    analytics["candidate_id"] = candidate_id
    analytics["candidate_metrics"] = dict(candidate_metrics)
    analytics["control_metrics"] = dict(control_metrics)
    analytics["research_id"] = str(research_id)
    analytics["source_run_id"] = str(source_run_id)
    analytics["rotation_quality_kind"] = str(processing_kind)

    document = {
        "processing_id": str(processing_id),
        "research_id": str(research_id),
        "source_run_id": str(source_run_id),
        "processing_kind": str(processing_kind),
        "candidate_id": candidate_id,
        "created_at": created_at,
        "finished_at": finished_at,
        "candidate_metrics": bson_value(candidate_metrics),
        "control_metrics": bson_value(control_metrics),
        "encoding": "zlib-json-v1",
        "payload": _pack(analytics),
    }
    db[TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION].replace_one(
        {"processing_id": str(processing_id), "candidate_id": candidate_id},
        document,
        upsert=True,
    )


def list_rotation_quality_analytics_processings(db: Any, *, limit: int = 50) -> dict[str, Any]:
    safe_limit = max(1, min(200, int(limit)))
    cursor = db[TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION].find(
        {},
        {
            "_id": 0,
            "processing_id": 1,
            "research_id": 1,
            "source_run_id": 1,
            "processing_kind": 1,
            "candidate_id": 1,
            "created_at": 1,
            "finished_at": 1,
            "candidate_metrics": 1,
            "control_metrics": 1,
        },
    ).sort("finished_at", -1)

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in cursor:
        processing_id = str(row.get("processing_id") or "")
        if not processing_id:
            continue
        item = grouped.setdefault(processing_id, {
            "id": processing_id,
            "research_id": str(row.get("research_id") or ""),
            "source_run_id": str(row.get("source_run_id") or ""),
            "kind": str(row.get("processing_kind") or "research"),
            "created_at": row.get("created_at"),
            "finished_at": row.get("finished_at"),
            "candidates": [],
        })
        item["candidates"].append({
            "candidate_id": str(row.get("candidate_id") or ""),
            "candidate_metrics": dict(row.get("candidate_metrics") or {}),
            "control_metrics": dict(row.get("control_metrics") or {}),
        })
    items = list(grouped.values())[:safe_limit]
    return {"items": bson_value(items), "count": len(items)}


def get_rotation_quality_analytics(db: Any, processing_id: str, candidate_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION].find_one(
        {"processing_id": str(processing_id), "candidate_id": str(candidate_id)},
        {"_id": 0, "payload": 1},
    )
    if not document:
        raise HTTPException(status_code=404, detail="Rotation Quality analytics are not available for this execution/candidate.")
    return _unpack(document.get("payload"))


def get_rotation_quality_rotation_period(
    db: Any,
    processing_id: str,
    candidate_id: str,
    *,
    year: int,
    month: int,
) -> dict[str, Any]:
    analytics = get_rotation_quality_analytics(db, processing_id, candidate_id)
    result = _rotation_period_from_data(
        db,
        str(processing_id),
        equity=list(analytics.get("equity") or []),
        rotations=list(analytics.get("rotations") or []),
        year=int(year),
        month=int(month),
    )
    reference_rows: list[dict[str, Any]] = []
    prefix = f"{int(year):04d}-{int(month):02d}"
    for row in analytics.get("equity") or []:
        timestamp = str(row.get("timestamp") or "")
        reference = _as_float(row.get("reference_equity"))
        if timestamp.startswith(prefix) and reference is not None:
            reference_rows.append({"timestamp": timestamp, "value": reference})
    result["control_equity"] = reference_rows
    if len(reference_rows) >= 2:
        first = _as_float(reference_rows[0].get("value"))
        last = _as_float(reference_rows[-1].get("value"))
        result["control_return"] = last / first - 1.0 if first not in {None, 0} and last is not None else None
    else:
        result["control_return"] = None
    result["candidate_id"] = str(candidate_id)
    result["research_id"] = str(analytics.get("research_id") or "")
    result["rotation_quality_kind"] = str(analytics.get("rotation_quality_kind") or "")
    return bson_value(result)
