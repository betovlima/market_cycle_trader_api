from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import bson_value


def iso_value(value: Any) -> Any:
    value = bson_value(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {key: iso_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [iso_value(item) for item in value]
    return value


def clean_mongo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: iso_value(value)
            for key, value in row.items()
            if key not in {"_id", "job_id", "symbol", "backend"}
        }
        for row in rows
    ]


def downsample_documents(rows: list[dict[str, Any]], maximum_points: int = 650) -> list[dict[str, Any]]:
    if len(rows) <= maximum_points:
        return rows
    important_indexes = {
        index
        for index, row in enumerate(rows)
        if str(row.get("trade_action", "")) in {"BUY", "SELL", "ROTATE", "FINAL_SELL"}
    }
    regular_count = max(2, maximum_points - len(important_indexes))
    step = max(1, len(rows) // regular_count)
    selected = set(range(0, len(rows), step))
    selected.update(important_indexes)
    selected.update({0, len(rows) - 1})
    return [rows[index] for index in sorted(selected)]
