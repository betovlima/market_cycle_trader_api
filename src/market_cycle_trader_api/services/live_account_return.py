"""Read-only Alpaca ACCOUNT return, linked from observed daily equity.

The broker's 1D cashflow series is cumulative per activity type. We subtract
changes in known external transfers before linking daily returns. This is a
daily time-weighted approximation, NOT exact transaction-time TWR and NOT
Winner-only P/L. All figures remain read-only and account-scoped.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from alpaca.trading.requests import GetPortfolioHistoryRequest


# Only cash movements known to represent capital entering/leaving this account.
# Unknown nonzero activities remain unavailable rather than silently treating
# a stock transfer, journal or corporate action as investment performance.
_EXTERNAL_FLOW_TYPES = frozenset({"CSD", "CSW", "ACATC"})
# These types are financial results / operating costs already reflected in equity.
# In particular dividends must remain part of investment performance.
_INTERNAL_FLOW_TYPES = frozenset({
    "DIV", "DIVCGL", "DIVCGS", "DIVROC", "DIVTXEX", "DIVNRA",
    "DIVFEE", "DIVFT", "DIVTW", "CGD", "INT", "INTNRA", "INTTW",
    "FEE", "CFEE", "PTC", "PTR", "CIL", "FILL",
})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _activity_code(activity: Any) -> str:
    # Alpaca-py may expose ActivityType enum keys rather than plain strings.
    return str(getattr(activity, "value", activity)).rsplit(".", 1)[-1].upper()


def _date_from_timestamp(raw: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc).date().isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def alpaca_account_cumulative_return(broker: Any, *, current_equity: float) -> dict[str, Any]:
    """Daily-linked return over observed funded closes, net of external flows.

    The current intraday equity is *not* chained onto the 1D history: we
    cannot prove whether an unreported intraday transfer occurred. The display
    includes an explicit as_of_date. Cashflows are cumulative per time window,
    so differences of successive rows (not their totals) enter daily links.

    End-of-day external-flow convention makes this an approximation when
    money moves intraday; transaction-time TWR needs intraday valuations.
    """
    base: dict[str, Any] = {
        "status": "unavailable", "return_pct": None, "profit_loss": None,
        "start_date": None, "as_of_date": None, "coverage": None,
        "method": "daily_linked_external_flow_adjusted",
        "cashflow_adjusted": None,
        "external_cashflow_total": None,
        "scope": "alpaca_account", "source": "alpaca_portfolio_history",
        "reason": "history_unavailable",
    }

    def unavailable(reason: str, start_date: str | None = None) -> dict[str, Any]:
        return {**base, "reason": reason, "start_date": start_date}

    # Read-only from the same broker identity as the enclosing portfolio.
    # Keep the function signature compatible with existing snapshot callers.
    _ = current_equity
    try:
        history = broker.get_portfolio_history(GetPortfolioHistoryRequest(
            period="5A", timeframe="1D", cashflow_types="ALL",
        ))
    except Exception:
        return unavailable("history_unavailable")

    times = _field(history, "timestamp")
    equities = _field(history, "equity")
    cashflow = _field(history, "cashflow")
    if (not isinstance(times, (list, tuple))
        or not isinstance(equities, (list, tuple))
        or len(times) != len(equities)
        or len(times) < 2):
        return unavailable("history_unavailable")
    if not isinstance(cashflow, dict):
        return unavailable("cashflow_unverified")

    # Validate the entire time axis. Silently sorting or skipping an
    # intermediate close would change the economic time windows.
    dates: list[str] = []
    for idx, raw_time in enumerate(times):
        day = _date_from_timestamp(raw_time)
        if day is None or (idx > 0 and int(raw_time) <= int(times[idx - 1])):
            return unavailable("history_incomplete")
        dates.append(day)

    flows: dict[str, list[float]] = {}
    for activity, raw_values in cashflow.items():
        if not isinstance(raw_values, (list, tuple)) or len(raw_values) != len(times):
            return unavailable("cashflow_unverified")
        values = [_finite(value) for value in raw_values]
        if any(value is None for value in values):
            return unavailable("cashflow_unverified")
        code = _activity_code(activity)
        # Different enum/string representations of the same activity must not
        # be double-counted as two independent cumulative series.
        if code in flows:
            return unavailable("cashflow_unverified")
        flows[code] = values

    # An account with an initial unfunded (zero/null) interval begins at the
    # first positive observed *closing equity*. Do not claim the deposit that
    # funded that initial close as investment gain.
    observed = [_finite(item) for item in equities]
    start_idx = next((i for i, value in enumerate(observed)
                      if value is not None and value > 0), None)
    if start_idx is None or start_idx >= len(times) - 1:
        return unavailable("history_unavailable")
    start_date = dates[start_idx]
    base_equity = observed[start_idx]
    previous_equity = base_equity
    linked_growth = 1.0
    net_external = 0.0
    adjusted = False

    for i in range(start_idx + 1, len(times)):
        equity = observed[i]
        if equity is None or equity <= 0:
            return unavailable("history_incomplete", start_date)
        external = 0.0
        for code, values in flows.items():
            daily_change = values[i] - values[i - 1]
            if not math.isfinite(daily_change):
                return unavailable("cashflow_unverified", start_date)
            if code in _EXTERNAL_FLOW_TYPES:
                external += daily_change  # Alpaca CSD positive, CSW negative.
            elif code not in _INTERNAL_FLOW_TYPES and abs(daily_change) > 1e-8:
                return unavailable("cashflow_type_unverified", start_date)
        if not math.isfinite(external):
            return unavailable("cashflow_unverified", start_date)
        # TWR approximation: external transfers assumed to occur at each
        # daily close. Dividend / interest / fees remain part of performance.
        factor = (equity - external) / previous_equity
        if not math.isfinite(factor) or factor <= 0:
            return unavailable("history_incomplete", start_date)
        linked_growth *= factor
        if not math.isfinite(linked_growth):
            return unavailable("history_incomplete", start_date)
        net_external += external
        adjusted = adjusted or abs(external) > 1e-8
        previous_equity = equity

    # USD result is a separate capital-flow-adjusted net P/L, *not*
    # "TWR multiplied by invested capital" or a Winner-only attribution.
    net_profit = previous_equity - base_equity - net_external
    if not math.isfinite(net_profit):
        return unavailable("history_incomplete", start_date)
    return {
        **base, "status": "available",
        "return_pct": linked_growth - 1.0,
        "profit_loss": net_profit,
        "start_date": start_date,
        "as_of_date": dates[-1],
        "coverage": "from_first_daily_close",
        "cashflow_adjusted": adjusted,
        "external_cashflow_total": net_external,
        "reason": None,
    }
