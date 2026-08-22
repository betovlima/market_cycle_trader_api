from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from ..infrastructure.persistence.mongo_repository import bson_value
from .utils import as_datetime, as_float, month_key


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


def _equity_values(
    decisions: list[dict[str, Any]],
    *,
    one_side_cost_rate: float,
) -> list[float]:
    values: list[float] = []
    stress_factor = 1.0
    requested_rate = max(0.0, float(one_side_cost_rate))
    for item in decisions:
        base_equity = as_float(item.get("candidate_equity"))
        if base_equity is None:
            raise ValueError("MILP decision replay is missing candidate_equity.")
        base_rate = max(0.0, float(as_float(item.get("one_side_cost_rate"), 0.0) or 0.0))
        sides = max(0, int(item.get("cost_sides") or 0))
        if sides:
            base_factor = max(1e-15, 1.0 - base_rate)
            requested_factor = max(0.0, 1.0 - requested_rate)
            stress_factor *= (requested_factor / base_factor) ** sides
        values.append(float(base_equity) * stress_factor)
    return values


def metrics(
    decisions: list[dict[str, Any]],
    initial_capital: float,
    one_side_cost_rate: float,
    *,
    count_cash_transitions_as_rotations: bool = True,
) -> dict[str, Any]:
    if not decisions:
        return bson_value({
            "initial_capital": initial_capital,
            "ending_capital": initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "maximum_drawdown": 0.0,
            "market_exposure": 0.0,
            "cash_days": 0,
            "decision_days": 0,
            "capital_rotations": 0,
            "realized_cvar_10": 0.0,
            "action_counts": {},
            "equity": [],
        })

    values = _equity_values(decisions, one_side_cost_rate=one_side_cost_rate)
    interval_returns: list[float] = []
    previous = float(initial_capital)
    peak = float(initial_capital)
    max_drawdown = 0.0
    equity: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    switches = cash_days = 0

    for item, capital in zip(decisions, values):
        interval_return = capital / previous - 1.0 if previous > 0 else 0.0
        interval_returns.append(interval_return)
        previous = capital
        peak = max(peak, capital)
        drawdown = capital / peak - 1.0 if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        item_action = str(item.get("action") or "HOLD")
        action_counts[item_action] += 1
        current_asset = str(item.get("current_symbol") or "CASH")
        target_asset = str(item.get("target_symbol") or "CASH")
        changed = current_asset != target_asset
        switches += int(changed and (count_cash_transitions_as_rotations or "CASH" not in {current_asset, target_asset}))
        cash_days += int(str(item.get("target_symbol") or "CASH") == "CASH")
        equity.append({
            "timestamp": item.get("execution_at") or item.get("decision_at"),
            "simulation_equity": capital,
            "reference_equity": item.get("reference_equity"),
            "drawdown": drawdown,
            "selected_asset": item.get("target_symbol"),
            "trade_action": item_action,
            "objective": item.get("objective"),
        })

    ending_capital = values[-1]
    total_return = ending_capital / initial_capital - 1.0 if initial_capital > 0 else 0.0
    years = max(len(decisions) / 252.0, 1.0 / 252.0)
    cagr = (ending_capital / initial_capital) ** (1.0 / years) - 1.0 if ending_capital > 0 and initial_capital > 0 else -1.0
    sharpe = 0.0
    if len(interval_returns) > 1:
        std = statistics.stdev(interval_returns)
        if std > 0:
            sharpe = statistics.mean(interval_returns) / std * math.sqrt(252.0)
    tail_count = max(1, math.ceil(len(interval_returns) * 0.10))
    cvar = statistics.mean(sorted(interval_returns)[:tail_count])
    return bson_value({
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "maximum_drawdown": max_drawdown,
        "market_exposure": (len(decisions) - cash_days) / len(decisions),
        "cash_days": cash_days,
        "decision_days": len(decisions),
        "capital_rotations": switches,
        "realized_cvar_10": cvar,
        "action_counts": dict(action_counts),
        "equity": equity,
    })


def _rebased_fold_decisions(items: list[dict[str, Any]], initial_capital: float) -> list[dict[str, Any]]:
    if not items:
        return []
    first_equity = as_float(items[0].get("candidate_equity"))
    if first_equity in {None, 0.0}:
        return [dict(item) for item in items]
    scale = float(initial_capital) / float(first_equity)
    rebased: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        value = as_float(row.get("candidate_equity"))
        reference = as_float(row.get("reference_equity"))
        if value is not None:
            row["candidate_equity"] = value * scale
        if reference is not None:
            row["reference_equity"] = reference * scale
        rebased.append(row)
    return rebased


def fold_metrics(
    decisions: list[dict[str, Any]],
    initial_capital: float,
    one_side_cost_rate: float,
    *,
    count_cash_transitions_as_rotations: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in decisions:
        grouped[str(item.get("fold_id") or "0")].append(item)
    rows: list[dict[str, Any]] = []
    for fold_id, items in sorted(grouped.items(), key=lambda pair: pair[0]):
        rebased = _rebased_fold_decisions(items, initial_capital)
        result = metrics(
            rebased, initial_capital, one_side_cost_rate,
            count_cash_transitions_as_rotations=count_cash_transitions_as_rotations,
        )
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
