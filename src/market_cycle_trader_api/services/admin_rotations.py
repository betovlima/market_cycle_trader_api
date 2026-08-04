from __future__ import annotations

from datetime import datetime, timezone
from statistics import fmean
from typing import Any

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
)
from .dashboard import _selected_internal_row
from .serialization import iso_value


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _selected_backend(db: Any, job_id: str) -> str | None:
    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "results": 1},
    )
    selected = _selected_internal_row(comparison)
    backend = str((selected or {}).get("backend") or "").strip()
    if backend:
        return backend

    run = db[RUNS_COLLECTION].find_one(
        {"job_id": job_id, "symbol": "PORTFOLIO"},
        {"_id": 0, "backend": 1},
    )
    fallback = str((run or {}).get("backend") or "").strip()
    return fallback or None


def _rotation_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    sell = next((row for row in rows if str(row.get("action") or "").upper() == "SELL"), None)
    buy = next((row for row in rows if str(row.get("action") or "").upper() == "BUY"), None)
    if sell is None and buy is None:
        return None

    source = buy or sell or {}
    timestamp = _as_utc(source.get("timestamp"))
    if timestamp is None:
        timestamp = _as_utc((sell or {}).get("timestamp"))

    fees = [
        value
        for row in rows
        if (value := _as_float(row.get("total_fee"))) is not None
    ]

    return {
        "executed_at": iso_value(timestamp),
        "from_asset": str(
            source.get("rotation_from_asset")
            or (sell or {}).get("asset")
            or "CASH"
        ),
        "to_asset": str(
            source.get("rotation_to_asset")
            or (buy or {}).get("asset")
            or "CASH"
        ),
        "holding_days": _as_float((sell or {}).get("holding_bars")),
        "position_return": _as_float((sell or {}).get("position_return")),
        "realized_pnl": _as_float((sell or {}).get("realized_pnl")),
        "transaction_fees": float(sum(fees)) if fees else 0.0,
    }


def admin_job_rotations(db: Any, job_id: str) -> dict[str, Any]:
    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {"_id": 0, "id": 1, "status": 1},
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    if str(job.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail="Rotations are available after the backtest completes.")

    backend = _selected_backend(db, job_id)
    if not backend:
        return {
            "job_id": job_id,
            "summary": {
                "total_rotations": 0,
                "profitable_rotations": 0,
                "losing_rotations": 0,
                "flat_rotations": 0,
                "total_realized_pnl": 0.0,
                "average_holding_days": None,
                "total_transaction_fees": 0.0,
                "first_rotation_at": None,
                "last_rotation_at": None,
            },
            "rotations": [],
        }

    trade_rows = list(
        db[TRADES_COLLECTION].find(
            {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend},
            {
                "_id": 0,
                "timestamp": 1,
                "sequence": 1,
                "action": 1,
                "asset": 1,
                "rotation_id": 1,
                "rotation_from_asset": 1,
                "rotation_to_asset": 1,
                "holding_bars": 1,
                "position_return": 1,
                "realized_pnl": 1,
                "total_fee": 1,
            },
        )
    )
    trade_rows.sort(
        key=lambda row: (
            _as_utc(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            int(row.get("sequence") or 0),
        )
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in trade_rows:
        rotation_id = str(row.get("rotation_id") or "").strip()
        if rotation_id:
            grouped.setdefault(rotation_id, []).append(row)

    rotations = [
        item
        for rows in grouped.values()
        if (item := _rotation_row(rows)) is not None
    ]
    rotations.sort(key=lambda item: str(item.get("executed_at") or ""))

    realized = [
        value
        for item in rotations
        if (value := _as_float(item.get("realized_pnl"))) is not None
    ]
    holdings = [
        value
        for item in rotations
        if (value := _as_float(item.get("holding_days"))) is not None
    ]
    fees = [
        value
        for item in rotations
        if (value := _as_float(item.get("transaction_fees"))) is not None
    ]

    return {
        "job_id": job_id,
        "summary": {
            "total_rotations": len(rotations),
            "profitable_rotations": sum(value > 0 for value in realized),
            "losing_rotations": sum(value < 0 for value in realized),
            "flat_rotations": sum(value == 0 for value in realized),
            "total_realized_pnl": float(sum(realized)) if realized else 0.0,
            "average_holding_days": float(fmean(holdings)) if holdings else None,
            "total_transaction_fees": float(sum(fees)) if fees else 0.0,
            "first_rotation_at": rotations[0]["executed_at"] if rotations else None,
            "last_rotation_at": rotations[-1]["executed_at"] if rotations else None,
        },
        "rotations": rotations,
    }
