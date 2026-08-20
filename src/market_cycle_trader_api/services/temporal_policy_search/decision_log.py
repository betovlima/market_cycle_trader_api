from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def log_entry(stage: str, outcome: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "at": datetime.now(timezone.utc),
        "stage": str(stage),
        "outcome": str(outcome),
        "message": str(message),
        "details": dict(details or {}),
    }
