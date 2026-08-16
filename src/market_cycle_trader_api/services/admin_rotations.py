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

    analytics_source = buy or {}
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
        "sell_execution_price": _as_float((sell or {}).get("execution_price")),
        "buy_execution_price": _as_float((buy or {}).get("execution_price")),
        "sell_reason": (sell or {}).get("reason"),
        "buy_reason": (buy or {}).get("reason"),
        "subsequent_holding_days": _as_float(analytics_source.get("subsequent_holding_days")),
        "subsequent_position_return": _as_float(analytics_source.get("subsequent_position_return")),
        "chosen_market_return": _as_float(analytics_source.get("chosen_market_return")),
        "counterfactual_previous_asset_return": _as_float(analytics_source.get("counterfactual_previous_asset_return")),
        "rotation_value_added": _as_float(analytics_source.get("rotation_value_added")),
        "rotation_regret": _as_float(analytics_source.get("rotation_regret")),
        "best_alternative_asset": analytics_source.get("best_alternative_asset"),
        "best_alternative_return": _as_float(analytics_source.get("best_alternative_return")),
        "opportunity_cost": _as_float(analytics_source.get("opportunity_cost")),
        "maximum_favorable_excursion": _as_float(analytics_source.get("maximum_favorable_excursion")),
        "maximum_adverse_excursion": _as_float(analytics_source.get("maximum_adverse_excursion")),
        "profit_capture_ratio": _as_float(analytics_source.get("profit_capture_ratio")),
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
                "asset_to_asset_rotations": 0,
                "market_to_cash_moves": 0,
                "cash_to_market_moves": 0,
                "profitable_rotations": 0,
                "losing_rotations": 0,
                "flat_rotations": 0,
                "total_realized_pnl": 0.0,
                "average_holding_days": None,
                "total_transaction_fees": 0.0,
                "first_rotation_at": None,
                "last_rotation_at": None,
                "diagnosed_rotations": 0,
                "positive_value_added_rotations": 0,
                "negative_value_added_rotations": 0,
                "average_rotation_value_added": None,
                "positive_value_added_rate": None,
                "average_opportunity_cost": None,
                "average_maximum_favorable_excursion": None,
                "average_maximum_adverse_excursion": None,
                "average_profit_capture_ratio": None,
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
                "reason": 1,
                "rotation_id": 1,
                "rotation_from_asset": 1,
                "rotation_to_asset": 1,
                "holding_bars": 1,
                "position_return": 1,
                "realized_pnl": 1,
                "total_fee": 1,
                "execution_price": 1,
                "subsequent_holding_days": 1,
                "subsequent_position_return": 1,
                "chosen_market_return": 1,
                "counterfactual_previous_asset_return": 1,
                "rotation_value_added": 1,
                "rotation_regret": 1,
                "best_alternative_asset": 1,
                "best_alternative_return": 1,
                "opportunity_cost": 1,
                "maximum_favorable_excursion": 1,
                "maximum_adverse_excursion": 1,
                "profit_capture_ratio": 1,
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
        if not rotation_id:
            # Backward-compatible reconstruction for historical jobs created
            # before CASH transitions received an explicit rotation_id.
            action = str(row.get("action") or "").upper()
            timestamp = _as_utc(row.get("timestamp"))
            asset = str(row.get("asset") or "").strip()
            if timestamp is not None and asset and action == "SELL":
                rotation_id = f"{timestamp.isoformat()}::{asset}->CASH"
                row = {
                    **row,
                    "rotation_id": rotation_id,
                    "rotation_from_asset": asset,
                    "rotation_to_asset": "CASH",
                }
            elif timestamp is not None and asset and action == "BUY":
                rotation_id = f"{timestamp.isoformat()}::CASH->{asset}"
                row = {
                    **row,
                    "rotation_id": rotation_id,
                    "rotation_from_asset": "CASH",
                    "rotation_to_asset": asset,
                }
        if rotation_id:
            grouped.setdefault(rotation_id, []).append(row)

    rotations = [
        item
        for rows in grouped.values()
        if (item := _rotation_row(rows)) is not None
    ]
    rotations.sort(key=lambda item: str(item.get("executed_at") or ""))

    asset_to_asset_rotations = sum(
        item.get("from_asset") != "CASH" and item.get("to_asset") != "CASH"
        for item in rotations
    )
    market_to_cash_moves = sum(
        item.get("from_asset") != "CASH" and item.get("to_asset") == "CASH"
        for item in rotations
    )
    cash_to_market_moves = sum(
        item.get("from_asset") == "CASH" and item.get("to_asset") != "CASH"
        for item in rotations
    )

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
    value_added = [
        value
        for item in rotations
        if (value := _as_float(item.get("rotation_value_added"))) is not None
    ]
    opportunity_costs = [
        value
        for item in rotations
        if (value := _as_float(item.get("opportunity_cost"))) is not None
    ]
    mfe_values = [
        value
        for item in rotations
        if (value := _as_float(item.get("maximum_favorable_excursion"))) is not None
    ]
    mae_values = [
        value
        for item in rotations
        if (value := _as_float(item.get("maximum_adverse_excursion"))) is not None
    ]
    capture_values = [
        value
        for item in rotations
        if (value := _as_float(item.get("profit_capture_ratio"))) is not None
    ]

    return {
        "job_id": job_id,
        "summary": {
            "total_rotations": len(rotations),
            "asset_to_asset_rotations": int(asset_to_asset_rotations),
            "market_to_cash_moves": int(market_to_cash_moves),
            "cash_to_market_moves": int(cash_to_market_moves),
            "profitable_rotations": sum(value > 0 for value in realized),
            "losing_rotations": sum(value < 0 for value in realized),
            "flat_rotations": sum(value == 0 for value in realized),
            "total_realized_pnl": float(sum(realized)) if realized else 0.0,
            "average_holding_days": float(fmean(holdings)) if holdings else None,
            "total_transaction_fees": float(sum(fees)) if fees else 0.0,
            "first_rotation_at": rotations[0]["executed_at"] if rotations else None,
            "last_rotation_at": rotations[-1]["executed_at"] if rotations else None,
            "diagnosed_rotations": len(value_added),
            "positive_value_added_rotations": sum(value > 0 for value in value_added),
            "negative_value_added_rotations": sum(value < 0 for value in value_added),
            "average_rotation_value_added": float(fmean(value_added)) if value_added else None,
            "positive_value_added_rate": (
                sum(value > 0 for value in value_added) / len(value_added)
                if value_added else None
            ),
            "average_opportunity_cost": float(fmean(opportunity_costs)) if opportunity_costs else None,
            "average_maximum_favorable_excursion": float(fmean(mfe_values)) if mfe_values else None,
            "average_maximum_adverse_excursion": float(fmean(mae_values)) if mae_values else None,
            "average_profit_capture_ratio": float(fmean(capture_values)) if capture_values else None,
        },
        "rotations": rotations,
    }
