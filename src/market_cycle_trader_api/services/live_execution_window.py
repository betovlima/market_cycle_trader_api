"""Independent, explicit Live trading window for current-session execution.

Do not inherit Paper's potentially short open window. This window remains
bounded to protect against stale intraday Winner decisions. It has no effect on
Paper trading or research.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

_DEFAULT_LIVE_WINDOW_SECONDS = 5400
_MIN_LIVE_WINDOW_SECONDS = 60
_MAX_LIVE_WINDOW_SECONDS = 7200


def live_execution_window_seconds(settings: Any = None) -> int:
    """Return the bounded Live execution deadline and duration."""
    value = max(_DEFAULT_LIVE_WINDOW_SECONDS, int(getattr(settings, "market_execution_window_seconds", 0)))
    delay = int(getattr(settings, "market_open_delay_seconds", 0))
    if not _MIN_LIVE_WINDOW_SECONDS <= value <= _MAX_LIVE_WINDOW_SECONDS:
        raise RuntimeError("Live execution window outside the supported range.")
    if value <= delay:
        raise RuntimeError("Live execution window must exceed market_open_delay_seconds.")
    return value


def live_execution_window_deadline(expected_open: pd.Timestamp, settings: Any = None) -> tuple[pd.Timestamp, int]:
    duration = live_execution_window_seconds(settings)
    return expected_open + pd.Timedelta(seconds=duration), duration
