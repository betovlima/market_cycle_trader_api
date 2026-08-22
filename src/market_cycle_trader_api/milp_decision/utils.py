from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def month_key(value: Any) -> str:
    parsed = as_datetime(value)
    return parsed.strftime("%Y-%m") if parsed else str(value or "")[:7]


def within_month_range(value: Any, start_month: str, end_month: str) -> bool:
    month = month_key(value)
    return bool(month and start_month <= month <= end_month)
