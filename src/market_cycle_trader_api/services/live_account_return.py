"""Read-only cumulative Alpaca account performance (never an MCT cash ledger).

This is brokerage ACCOUNT performance, not verified Winner-only attribution.
The first available funded daily closing balance is the reference. We do not
manufacture a trading inception date or infer deposits from equity changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from alpaca.trading.requests import GetPortfolioHistoryRequest


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def alpaca_account_cumulative_return(broker: Any, *, current_equity: float) -> dict[str, Any]:
    """Return an observable account-equity comparison, or a reason it is unavailable.

    Use the same authenticated Alpaca broker used for the account snapshot.
    Never write an equity/balance reference into MCT or use another account.
    A capital transfer after the reference makes a naive equity/base ratio
    invalid; therefore display no percentage when transfer evidence exists.
    """
    unavailable: dict[str, Any] = {
        "status": "unavailable", "return_pct": None, "profit_loss": None,
        "start_date": None, "coverage": None,
        "scope": "alpaca_account", "source": "alpaca_portfolio_history",
        "reason": "history_unavailable",
    }
    try:
        history = broker.get_portfolio_history(GetPortfolioHistoryRequest(
            period="5A", timeframe="1D", cashflow_types="ALL",
        ))
    except Exception:
        return unavailable

    times = _field(history, "timestamp")
    equities = _field(history, "equity")
    basis = _finite(_field(history, "base_value"))
    if (not isinstance(times, (list, tuple))
        or not isinstance(equities, (list, tuple))
        or len(times) != len(equities)
        or basis is None or basis <= 0
        or len(times) < 2):
        return unavailable
    available_points = [
        (timestamp, value) for timestamp, raw in zip(times, equities)
        if (value := _finite(raw)) is not None and value > 0
    ]
    if len(available_points) < 2:
        return unavailable

    asof = _field(history, "base_value_asof")
    if asof is not None:
        first_date = str(asof)[:10]
    else:
        try:
            first_date = datetime.fromtimestamp(
                int(available_points[0][0]), tz=timezone.utc,
            ).date().isoformat()
        except (ValueError, TypeError, OverflowError):
            return unavailable

    try:
        # The Alpaca cashflow arrays are cumulative across daily windows.
        # Any nonzero cash movement (including dividends, journal entries or
        # other activity types we cannot classify) makes the raw equity ratio
        # unsuitable as a clean trading-performance percentage.
        cashflow = _field(history, "cashflow")
        if not isinstance(cashflow, dict):
            return {**unavailable, "start_date": first_date,
                    "reason": "cashflow_unverified"}
        for values in cashflow.values():
            if not isinstance(values, (list, tuple)) or len(values) != len(times):
                return {**unavailable, "start_date": first_date,
                        "reason": "cashflow_unverified"}
            if any(_finite(value) is None or abs(float(value)) > 1e-8
                   for value in values):
                return {**unavailable, "start_date": first_date,
                        "reason": "cashflows_present"}
    except (TypeError, ValueError, OverflowError):
        return {**unavailable, "start_date": first_date,
                "reason": "cashflow_unverified"}

    equity = _finite(current_equity)
    if equity is None or equity < 0:
        return unavailable
    return {
        "status": "available",
        "return_pct": equity / basis - 1.0,
        "profit_loss": equity - basis,
        "start_date": first_date,
        "coverage": "from_first_daily_close",
        "scope": "alpaca_account",
        "source": "alpaca_portfolio_history",
        "reason": None,
    }
