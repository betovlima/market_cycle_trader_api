"""Live execution uses the actual XNYS regular-session close, not an arbitrary window.

NYSE sessions, DST changes, holidays and early closes are resolved from the
exchange calendar; real order time/open state comes from Alpaca Live's clock.
"""
from __future__ import annotations

from typing import Any

import exchange_calendars as xcals
import pandas as pd


def live_execution_window_seconds(settings: Any = None) -> int:
    """Legacy informational default: length of a full regular US stock session.

    Actual execution uses live_execution_window_deadline for each trading date.
    Historical paper settings must never constrain the Live schedule.
    """
    return 6 * 60 * 60 + 30 * 60


def live_execution_window_deadline(
    expected_open: pd.Timestamp, settings: Any = None, *, market_clock: Any = None,
) -> tuple[pd.Timestamp, int]:
    """Return the real XNYS session close (including early closes), in UTC."""
    opened = pd.Timestamp(expected_open)
    if opened.tzinfo is None:
        raise ValueError("Expected XNYS market open must be timezone-aware.")
    opened = opened.tz_convert("UTC")
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(opened.tz_convert("America/New_York").date())
    if not calendar.is_session(session):
        raise ValueError("Expected open is not an XNYS trading session.")
    market_open = calendar.session_open(session)
    market_close = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    # Prefer Alpaca Live's current-session next_close when the exchange is
    # already open. It incorporates provider-known early or exceptional closes.
    if market_clock is not None and bool(getattr(market_clock, "is_open", False)):
        broker_close = getattr(market_clock, "next_close", None)
        if broker_close is not None:
            close = pd.Timestamp(broker_close)
            if close.tzinfo is not None:
                close = close.tz_convert("UTC")
                if (
                    close > market_open
                    and close.tz_convert("America/New_York").date() == session.date()
                ):
                    market_close = close
    duration = int((market_close - market_open).total_seconds())
    return market_close, duration
