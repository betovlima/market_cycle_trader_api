"""Read-only historical Live performance from the SAME Alpaca broker account.

The MCT never stores or invents portfolio capital. Alpaca portfolio history
provides cashflow-aware P/L; if the requested Winner period cannot be verified,
the return stays unavailable. This measures the broker account while enrolled
in Live Winner, NOT attribution to the model alone (manual trades can exist).
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetPortfolioHistoryRequest

from .live_order_execution import ORDER_LEDGER
from .live_trader_state import LIVE_STATE
from .live_winner_session import LIVE_RUNS

_EASTERN = ZoneInfo("America/New_York")


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.astimezone(_EASTERN).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_live_winner_date(
    db: Any, *, account_id: str, joined_at: Any = None,
    canonical_account: bool = False,
) -> date | None:
    """Choose an auditable broker-specific start; never mix accounts.

    The primary account may include reconciled manual Live entries. For other
    users, OAuth authorization is the enrollment timestamp; earlier orders in
    the independent MCT Live broker ledger take precedence after a reconnect.
    """
    dates: list[date] = []
    joined = _date(joined_at)
    if joined is not None:
        dates.append(joined)

    filled = db[ORDER_LEDGER].find_one(
        {"broker_account_id": account_id, "status": "filled"},
        {"created_at": 1, "filled_at": 1, "session": 1},
        sort=[("created_at", 1)],
    )
    if filled:
        filled_date = _date(filled.get("session")) or _date(filled.get("filled_at"))
        if filled_date:
            dates.append(filled_date)

    if canonical_account:
        first_run = db[LIVE_RUNS].find_one(
            {"live_account_id": account_id,
             "status": {"$in": ["completed", "manual_reconciled"]}},
            {"execution_session": 1},
            sort=[("execution_session", 1)],
        )
        if first_run:
            day = _date(first_run.get("execution_session"))
            if day:
                dates.append(day)
        state = db[LIVE_STATE].find_one({"_id": account_id}, {"entry_session": 1})
        if state:
            entry = _date(state.get("entry_session"))
            if entry:
                dates.append(entry)
    return min(dates) if dates else None


def _unavailable(start_date: date | None, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable", "return_pct": None, "profit_loss": None,
        "start_date": start_date.isoformat() if start_date else None,
        "as_of": None, "reason": reason,
        "source": "alpaca_portfolio_history",
        "scope": "broker_account_during_winner",
    }


def live_winner_performance(broker: Any, *, start_date: date | None) -> dict[str, Any]:
    """Use Alpaca's adjusted historical profit/loss, not an equity delta.

    Deposits and withdrawals make (equity-now - initial-equity) incorrect.
    The provider is asked to include ALL cashflow categories in its official
    portfolio-history P/L calculation. Do not fall back to unadjusted balances.
    """
    if start_date is None:
        return _unavailable(None, "start_date_not_verified")

    try:
        history = broker.get_portfolio_history(
            GetPortfolioHistoryRequest(
                start=datetime.combine(start_date, time(9, 30), tzinfo=_EASTERN),
                timeframe="1D", cashflow_types="ALL",
            )
        )
    except Exception:
        # A history outage must not hide the live portfolio or change orders.
        return _unavailable(start_date, "alpaca_history_unavailable")

    points = getattr(history, "timestamp", None)
    equities = getattr(history, "equity", None)
    pnls = getattr(history, "profit_loss", None)
    returns = getattr(history, "profit_loss_pct", None)
    base = _finite(getattr(history, "base_value", None))
    if (not isinstance(points, (list, tuple)) or not points
        or not isinstance(equities, (list, tuple))
        or not isinstance(pnls, (list, tuple))
        or not isinstance(returns, (list, tuple))
        or not (len(points) == len(equities) == len(pnls) == len(returns))
        or base is None or base <= 0):
        return _unavailable(start_date, "incomplete_alpaca_history")

    last = next((
        i for i in range(len(points) - 1, -1, -1)
        if _finite(equities[i]) is not None
        and _finite(pnls[i]) is not None
        and _finite(returns[i]) is not None
    ), None)
    if last is None:
        return _unavailable(start_date, "incomplete_alpaca_history")
    try:
        first_day = datetime.fromtimestamp(int(points[0]), timezone.utc).astimezone(_EASTERN).date()
        final_day = datetime.fromtimestamp(int(points[last]), timezone.utc).astimezone(_EASTERN).date()
    except (ValueError, OverflowError, TypeError, OSError):
        return _unavailable(start_date, "invalid_alpaca_timestamps")

    if first_day > start_date or final_day < start_date:
        return _unavailable(start_date, "history_does_not_cover_start")

    # Alpaca may use the first available 1D equity bar as the base when no
    # prior closing value is available. Then the first day's P/L is unknown;
    # we must not call that return 'since the Winner began'.
    base_asof = _date(getattr(history, "base_value_asof", None))
    if base_asof is None or base_asof > start_date:
        return _unavailable(start_date, "prior_closing_baseline_missing")

    return {
        "status": "available",
        "return_pct": float(returns[last]),  # Fraction, like broker unrealized_plpc.
        "profit_loss": float(pnls[last]),
        "start_date": start_date.isoformat(),
        "as_of": final_day.isoformat(),
        "reason": None,
        "source": "alpaca_portfolio_history",
        "scope": "broker_account_during_winner",
        "includes_manual_trades": True,
    }
