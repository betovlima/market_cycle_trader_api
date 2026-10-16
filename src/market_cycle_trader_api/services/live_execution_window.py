"""Independent, explicit Live trading window used by recalibration and execution.

Do not inherit Paper's potentially short open window. This window remains
bounded to protect against stale intraday Winner decisions. It has no effect on
Paper trading or research.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

_DEFAULT_LIVE_WINDOW_SECONDS = 1800
_MIN_LIVE_WINDOW_SECONDS = 60
_MAX_LIVE_WINDOW_SECONDS = 7200


def live_execution_window_seconds(settings: Any) -> int:
    """Return one consistent Live window for both the model and executor."""
    raw = str(os.getenv("ALPACA_LIVE_EXECUTION_WINDOW_SECONDS") or _DEFAULT_LIVE_WINDOW_SECONDS)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ALPACA_LIVE_EXECUTION_WINDOW_SECONDS must be an integer.") from exc
    delay = int(getattr(settings, "market_open_delay_seconds", 0))
    if not _MIN_LIVE_WINDOW_SECONDS <= value <= _MAX_LIVE_WINDOW_SECONDS:
        raise RuntimeError("ALPACA_LIVE_EXECUTION_WINDOW_SECONDS must be 60..7200.")
    if value <= delay:
        raise RuntimeError("ALPACA_LIVE_EXECUTION_WINDOW_SECONDS must exceed market_open_delay_seconds.")
    return value


def live_execution_window_deadline(expected_open: pd.Timestamp, settings: Any) -> tuple[pd.Timestamp, int]:
    duration = live_execution_window_seconds(settings)
    return expected_open + pd.Timedelta(seconds=duration), duration
