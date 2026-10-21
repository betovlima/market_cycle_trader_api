"""Retired Live intraday market-open recalibration entry point.

The Live strategy uses the prior completed XNYS daily session. Its persisted
after-close signal is executed at the next regular open after broker safety
validation; a missing signal may be prepared from the same prior daily data.
This legacy entry point cannot download 1Min bars, retrain or submit orders.
"""
from __future__ import annotations

from typing import Any


def refresh_live_open_winner_signal(db: Any, session: str) -> dict[str, Any]:
    raise RuntimeError(
        "LiveIntradayRecalibrationDisabled: no market-open 1Min Winner "
        "recalibration. Only incomplete prior-day daily preparation may "
        "be recovered by the scheduler."
    )
