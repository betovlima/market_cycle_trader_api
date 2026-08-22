from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from ..infrastructure.persistence.mongo_repository import bson_value
from .utils import as_datetime, month_key


def cost_sides(current_symbol: str, target_symbol: str) -> int:
    if current_symbol == target_symbol:
        return 0
    return 1 if "CASH" in {current_symbol, target_symbol} else 2


def action(current_symbol: str, target_symbol: str) -> str:
    if current_symbol == target_symbol:
        return "CASH" if target_symbol == "CASH" else "HOLD"
    if target_symbol == "CASH":
        return "CASH"
    if current_symbol == "CASH":
        return "BUY"
    return "ROTATE"


def net_interval_return(gross_return: float, sides: int, one_side_cost_rate: float) -> float:
    rate = max(0.0, float(one_side_cost_rate))
    return (1.0 + float(gross_return)) * ((1.0 - rate) ** max(0, int(sides))) - 1.0


def metrics(decisions: list[dict[str, Any]], initial_capital: float, one_side_cost_rate: float) -> dict[str, Any]:
    capital = float(initial_capital)
    peak = capital
    max_drawdown = 0.0
    interval_returns: list[float] = []
    equity: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    switches = cash_days = 0
    first_date = last_date = None
    for item in decisions:
        net = net_interval_return(float(item.get("gross_interval_return") or 0.0), int(item.get("cost_sides") or 0), one_side_cost_rate)
        capital *= 1.0 + net
        peak = max(peak, capital)
        drawdown = capital / peak - 1.0 if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        interval_returns.append(net)
        item_action = str(item.get("action") or "HOLD")
        action_counts[item_action] += 1
        switches += int(int(item.get("cost_sides") or 0) > 0)
        cash_days += int(str(item.get("target_symbol") or "CASH") == "CASH")
        stamp = as_datetime(item.get("execution_at") or item.get("decision_at"))
        if stamp is not None:
            first_date = first_date or stamp
            last_date = stamp
        equity.append({
            "timestamp": item.get("execution_at") or item.get("decision_at"),
            "simulation_equity": capital,
            "reference_equity": capital,
            "drawdown": drawdown,
            "selected_asset": item.get("target_symbol"),
            "trade_action": item_action,
            "objective": item.get("objective"),
        })
    total_return = capital / initial_capital - 1.0 if initial_capital > 0 else 0.0
    if first_date and last_date and last_date > first_date:
        years = max((last_date - first_date).total_seconds() / (365.25 * 86400.0), 1.0 / 252.0)
        cagr = (capital / initial_capital) ** (1.0 / years) - 1.0 if capital > 0 and initial_capital > 0 else -1.0
    else:
        cagr = (1.0 + total_return) ** (252.0 / max(1, len(decisions))) - 1.0 if total_return > -1 else -1.0
    sharpe = 0.0
    if len(interval_returns) > 1:
        std = statistics.stdev(interval_returns)
        if std > 0:
            sharpe = statistics.mean(interval_returns) / std * math.sqrt(252.0)
    tail_count = max(1, math.ceil(len(interval_returns) * 0.10)) if interval_returns else 0
    cvar = statistics.mean(sorted(interval_returns)[:tail_count]) if tail_count else 0.0
    return bson_value({
        "initial_capital": initial_capital,
        "ending_capital": capital,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "maximum_drawdown": max_drawdown,
        "market_exposure": (len(decisions) - cash_days) / len(decisions) if decisions else 0.0,
        "cash_days": cash_days,
        "decision_days": len(decisions),
        "capital_rotations": switches,
        "realized_cvar_10": cvar,
        "action_counts": dict(action_counts),
        "equity": equity,
    })


def fold_metrics(decisions: list[dict[str, Any]], initial_capital: float, one_side_cost_rate: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in decisions:
        grouped[str(item.get("fold_id") or "0")].append(item)
    rows: list[dict[str, Any]] = []
    for fold_id, items in sorted(grouped.items(), key=lambda pair: pair[0]):
        result = metrics(items, initial_capital, one_side_cost_rate)
        rows.append({
            "fold_id": int(fold_id) if fold_id.isdigit() else fold_id,
            "test_start": items[0].get("execution_at") if items else None,
            "test_end": items[-1].get("execution_at") if items else None,
            "metrics": {key: value for key, value in result.items() if key != "equity"},
        })
    return bson_value(rows)


def monthly_decision_map(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for item in decisions:
        grouped[month_key(item.get("execution_at") or item.get("decision_at"))][str(item.get("action") or "HOLD")] += 1
    rows: list[dict[str, Any]] = []
    for month, counts in sorted(grouped.items()):
        if not month:
            continue
        dominant = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0] if counts else "HOLD"
        rows.append({"month": month, "dominant_action": dominant, "counts": dict(counts), "decisions": int(sum(counts.values()))})
    return rows
