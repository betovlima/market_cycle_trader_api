"""Read-only peak/giveback accounting for the current Paper position.

Only observed account snapshots are used. This is NOT an intraday high
estimate, a stop-loss, a limit order, or a trading decision. Old snapshots
have no stored position quantity; a historical price is inferred only when
the order ledger establishes that the position quantity was unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except ValueError:
        return None


def most_recent_open_buy(
    symbol: str,
    orders: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Locate opening fill of current contiguous position; count partial fills."""
    events = []
    for item in orders:
        if str(item.get("symbol") or "").upper() != symbol.upper():
            continue
        side = str(item.get("side") or "").lower()
        qty = _num(item.get("filled_quantity"))
        when = _timestamp(item.get("filled_at") or item.get("created_at"))
        if side not in {"buy", "sell"} or qty is None or qty <= 0 or when is None:
            continue
        events.append((when, side, qty))
    events.sort(key=lambda x: x[0])
    balance = 0.0
    entry: datetime | None = None
    subsequent_fills = 0
    for when, side, qty in events:
        if side == "buy":
            if balance <= 1e-9:
                entry = when
                subsequent_fills = 0
            else:
                subsequent_fills += 1
            balance += qty
        else:
            if balance > 1e-9:
                subsequent_fills += 1
            balance = max(0.0, balance - qty)
            if balance <= 1e-9:
                entry = None
                subsequent_fills = 0
    return (
        {"entry_filled_at": entry.isoformat(), "entry_timestamp": entry,
         "constant_quantity": subsequent_fills == 0,
         "filled_quantity_ledger": balance}
        if entry is not None else None
    )


def completed_sessions_after_entry(
    entry_at: Any,
    session_dates: list[str],
    *,
    as_of_session: str | None = None,
) -> int | None:
    entry = _timestamp(entry_at)
    if entry is None or not session_dates:
        return None
    entry_day = entry.astimezone(NY).date().isoformat()
    return sum(
        1 for session in set(session_dates)
        if entry_day < str(session) and
        (as_of_session is None or str(session) <= str(as_of_session))
    )


def position_excursion_audit(
    position: dict[str, Any] | None,
    orders: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    completed_sessions: list[str] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not position or not position.get("symbol"):
        return None
    symbol = str(position["symbol"]).upper()
    opening = most_recent_open_buy(symbol, orders)
    if opening is None:
        return {
            "symbol": symbol, "status": "entry_fill_unavailable",
            "measurement": "observed_snapshots_only",
            "observations": [],
        }
    entry = opening["entry_timestamp"]
    quantity = _num(position.get("quantity"))
    basis = _num(position.get("cost_basis"))
    if quantity is None or quantity <= 0 or basis is None or basis <= 0:
        return {"symbol": symbol, "status": "position_basis_unavailable", "observations": []}

    series = []
    for row in history:
        if str(row.get("managed_symbol") or "").upper() != symbol:
            continue
        when = _timestamp(row.get("recorded_at"))
        if when is None or when < entry:
            continue
        recorded_qty = _num(row.get("position_quantity"))
        recorded_price = _num(row.get("position_price"))
        recorded_basis = _num(row.get("position_cost_basis"))
        if recorded_qty is not None and recorded_qty > 0 and recorded_price is not None:
            observation = (recorded_price / (recorded_basis / recorded_qty) - 1.0) if recorded_basis and recorded_basis > 0 else (recorded_price / (basis / quantity) - 1.0)
            market_value = recorded_price * recorded_qty
            source = "captured_position_snapshot"
        elif opening["constant_quantity"] and (recorded_qty is None or abs(recorded_qty - quantity) <= 1e-6):
            market_value = _num(row.get("market_value"))
            if market_value is None:
                continue
            observation = market_value / basis - 1.0
            source = "inferred_unchanged_quantity"
        else:
            continue
        if not math.isfinite(observation):
            continue
        series.append({
            "recorded_at": when.isoformat(),
            "unrealized_return": observation,
            "market_value": market_value,
            "portfolio_value": _num(row.get("portfolio_value")),
            "source": source,
        })

    current_time = _timestamp(position.get("observed_at"))
    current_return = _num(position.get("unrealized_return"))
    current_value = _num(position.get("market_value"))
    if current_return is not None and current_value is not None and current_time is not None:
        if not series or current_time > _timestamp(series[-1]["recorded_at"]):
            series.append({
                "recorded_at": current_time.isoformat(),
                "unrealized_return": current_return,
                "market_value": current_value,
                "portfolio_value": _num(position.get("portfolio_value")),
                "source": "current_broker_position",
            })
    if not series:
        return {"symbol": symbol, "status": "no_position_observations",
                "entry_filled_at": opening["entry_filled_at"], "observations": []}
    series.sort(key=lambda item: item["recorded_at"])
    best = max(series, key=lambda item: item["unrealized_return"])
    worst = min(series, key=lambda item: item["unrealized_return"])
    last = series[-1]
    peak_return = float(best["unrealized_return"])
    last_return = float(last["unrealized_return"])
    peak_profit = basis * peak_return
    current_profit = basis * last_return
    giveback = max(0.0, peak_profit - current_profit)

    tracked_portfolio = [r for r in series if r.get("portfolio_value") is not None]
    account_peak = max(tracked_portfolio, key=lambda r: r["portfolio_value"]) if tracked_portfolio else None
    account_current = tracked_portfolio[-1] if tracked_portfolio else None
    reported_holding = _num(position.get("holding_sessions"))
    sessions = completed_sessions_after_entry(
        opening["entry_filled_at"], completed_sessions or []
    )
    decision_date = str((decision or {}).get("decision_date") or "")
    age_at_decision = completed_sessions_after_entry(
        opening["entry_filled_at"], completed_sessions or [],
        as_of_session=decision_date or None,
    )
    plan_holding = _num((decision or {}).get("holding_sessions_at_decision"))

    return {
        "symbol": symbol,
        "status": "observed",
        "measurement": "observed_snapshots_only",
        "entry_filled_at": opening["entry_filled_at"],
        "entry_price": _num(position.get("average_entry_price")),
        "current_price": _num(position.get("current_price")),
        "observation_count": len(series),
        "observed_peak_at": best["recorded_at"],
        "observed_peak_return": peak_return,
        "observed_trough_at": worst["recorded_at"],
        "observed_trough_return": worst["unrealized_return"],
        "current_return": last_return,
        "observed_peak_profit": round(peak_profit, 2),
        "current_unrealized_profit": round(current_profit, 2),
        "profit_given_back": round(giveback, 2),
        "fraction_of_peak_profit_given_back": (
            giveback / peak_profit if peak_profit > 0 else None
        ),
        "position_drawdown_from_observed_peak": (
            (1.0 + last_return) / (1.0 + peak_return) - 1.0
            if peak_return > -1.0 else None
        ),
        "account_peak_value_during_position": (
            account_peak["portfolio_value"] if account_peak else None
        ),
        "account_peak_at": account_peak["recorded_at"] if account_peak else None,
        "account_value_given_back": (
            round(max(0.0, account_peak["portfolio_value"] - account_current["portfolio_value"]), 2)
            if account_peak and account_current else None
        ),
        "recorded_holding_sessions": int(reported_holding) if reported_holding is not None else None,
        "observed_completed_sessions_since_entry": sessions,
        "holding_discrepancy": (
            int(reported_holding) != sessions
            if sessions is not None and reported_holding is not None else None
        ),
        "decision_date": decision_date or None,
        "plan_holding_sessions": int(plan_holding) if plan_holding is not None else None,
        "observed_sessions_at_decision": age_at_decision,
        "decision_holding_discrepancy": (
            int(plan_holding) != age_at_decision
            if plan_holding is not None and age_at_decision is not None else None
        ),
        "quantity_unchanged_in_order_ledger": opening["constant_quantity"],
        "observations": series,
        "limitations": (
            "The recorded high is the maximum observed in saved snapshots, not "
            "the intraday high; broker/market-data gaps may miss a larger move. "
            "Metrics are diagnostics and never place or modify orders."
        ),
    }
