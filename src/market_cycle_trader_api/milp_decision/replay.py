from __future__ import annotations

import time
from typing import Any, Callable

from ..infrastructure.persistence.mongo_repository import bson_value
from .errors import MilpDecisionError
from .metrics import action, cost_sides, net_interval_return
from .objective import objective_breakdown, rank_value
from .solver import solve_binary_one_hot
from .utils import as_datetime, as_float, within_month_range


def _candidate_symbols(
    diagnostic: dict[str, Any],
    observation_rows: list[dict[str, Any]],
    rows_by_symbol: dict[str, dict[str, Any]],
    current_symbol: str,
    rank_limit: int,
) -> tuple[str, list[str]]:
    anchor = str(diagnostic.get("winner_anchor_symbol") or diagnostic.get("winner_top1_symbol") or "").upper()
    ranked = sorted(observation_rows, key=rank_value, reverse=True)
    candidates: list[str] = []
    for symbol in [
        anchor,
        str(diagnostic.get("winner_top2_symbol") or "").upper(),
        *[str(row.get("symbol") or "").upper() for row in ranked[:rank_limit]],
        current_symbol,
    ]:
        if symbol and symbol != "CASH" and symbol in rows_by_symbol and symbol not in candidates:
            candidates.append(symbol)
    return anchor, candidates


def _alternatives(
    candidate_symbols: list[str],
    rows_by_symbol: dict[str, dict[str, Any]],
    *,
    current_symbol: str,
    anchor_symbol: str,
    configuration: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in candidate_symbols:
        source = rows_by_symbol[symbol]
        breakdown = objective_breakdown(
            source,
            current_symbol=current_symbol,
            anchor_symbol=anchor_symbol,
            configuration=configuration,
        )
        rows.append({
            "symbol": symbol,
            "eligible": True,
            "objective": breakdown["objective"],
            "breakdown": breakdown,
            "predicted_drawdown": as_float(source.get("predicted_drawdown"), 0.0),
            "risk_safety": as_float(source.get("all_horizon_risk_safety"), 0.5),
            "short_profit_consensus": as_float(source.get("short_profit_consensus"), 0.5),
            "rank_score": rank_value(source),
        })
    cash_objective = float(configuration["cash_objective"])
    rows.append({
        "symbol": "CASH",
        "eligible": True,
        "objective": cash_objective,
        "breakdown": {"cash": cash_objective, "objective": cash_objective},
        "predicted_drawdown": 0.0,
        "risk_safety": 1.0,
        "short_profit_consensus": 0.0,
        "rank_score": 0.0,
    })
    return rows


def _selected_interval_return(
    target_symbol: str,
    rows_by_symbol: dict[str, dict[str, Any]],
    economic: dict[str, Any],
    *,
    current_symbol: str,
    control_current: str,
    control_target: str,
    base_cost_rate: float,
) -> tuple[float | None, float | None, int]:
    sides = cost_sides(current_symbol, target_symbol)
    control_net = as_float(economic.get("net_interval_return"))
    if target_symbol == control_target and current_symbol == control_current and control_net is not None:
        return control_net, as_float(economic.get("gross_interval_return"), control_net), sides
    if target_symbol == "CASH":
        gross = 0.0
    else:
        row = rows_by_symbol.get(target_symbol)
        gross = as_float((row or {}).get("open_to_open_return")) if row else None
    if gross is None:
        return None, None, sides
    return net_interval_return(gross, sides, base_cost_rate), gross, sides


def build_decisions(
    *,
    diagnostics: list[dict[str, Any]],
    observations: dict[str, list[dict[str, Any]]],
    economics: dict[str, dict[str, Any]],
    configuration: dict[str, Any],
    start_month: str,
    end_month: str,
    base_cost_rate: float,
    should_stop: Callable[[], bool],
    force_control: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_symbol: str | None = None
    holding_days = 0
    decisions: list[dict[str, Any]] = []
    solver_nodes = solver_pruned = solver_decisions = forced_control_decisions = 0
    solve_ms_total = 0.0
    changed = better = worse = neutral = 0
    candidate_equity: float | None = None
    path_matches_control = True

    for index, diagnostic in enumerate(diagnostics):
        if index % 50 == 0 and should_stop():
            raise MilpDecisionError("MILP Decision Optimization stopped by user.")
        decision_at = as_datetime(diagnostic.get("timestamp"))
        if decision_at is None or not within_month_range(decision_at, start_month, end_month):
            continue

        economic = economics.get(decision_at.isoformat()) or {}
        reference_equity = as_float(economic.get("strategy_equity"))
        if path_matches_control and reference_equity is not None:
            candidate_equity = reference_equity
        if candidate_equity is None:
            raise MilpDecisionError("MILP replay is missing the reference equity anchor for the selected period.")

        control_target = str(diagnostic.get("target_symbol") or economic.get("target_symbol") or "CASH").upper() or "CASH"
        control_current = str(diagnostic.get("current_symbol") or economic.get("current_symbol") or "CASH").upper() or "CASH"
        if current_symbol is None:
            current_symbol = control_current

        observation_rows = observations.get(decision_at.isoformat()) or []
        rows_by_symbol = {
            str(row.get("symbol") or "").upper(): row
            for row in observation_rows
            if str(row.get("symbol") or "").strip()
        }
        must_follow_control = force_control or not observation_rows or not rows_by_symbol
        alternatives: list[dict[str, Any]] = []
        solver: dict[str, Any] | None = None
        selected: dict[str, Any]

        if must_follow_control:
            forced_control_decisions += 1
            selected = {
                "symbol": control_target,
                "objective": None,
                "breakdown": {"forced_control": 1.0},
            }
        else:
            anchor, candidates = _candidate_symbols(
                diagnostic,
                observation_rows,
                rows_by_symbol,
                current_symbol,
                int(configuration["candidate_rank_limit"]),
            )
            alternatives = _alternatives(
                candidates,
                rows_by_symbol,
                current_symbol=current_symbol,
                anchor_symbol=anchor,
                configuration=configuration,
            )
            started = time.perf_counter()
            selected, solver = solve_binary_one_hot(alternatives)
            solve_ms_total += (time.perf_counter() - started) * 1000.0
            solver_nodes += int(solver["nodes_explored"])
            solver_pruned += int(solver["nodes_pruned"])
            solver_decisions += 1

        target_symbol = str(selected["symbol"])
        decision_action = action(current_symbol, target_symbol)
        selected_net, gross_return, sides = _selected_interval_return(
            target_symbol,
            rows_by_symbol,
            economic,
            current_symbol=current_symbol,
            control_current=control_current,
            control_target=control_target,
            base_cost_rate=base_cost_rate,
        )
        control_net = as_float(economic.get("net_interval_return"))
        if control_net is None and control_target == "CASH":
            control_net = 0.0
        execution_at = economic.get("execution_date") or diagnostic.get("timestamp")
        next_execution_at = economic.get("next_execution_date")
        delta = selected_net - control_net if selected_net is not None and control_net is not None else None

        if target_symbol != control_target:
            changed += 1
            if delta is None or abs(delta) <= 1e-12:
                neutral += 1
            elif delta > 0:
                better += 1
            else:
                worse += 1

        control_path_match = current_symbol == control_current and target_symbol == control_target
        if not control_path_match:
            path_matches_control = False

        decisions.append(bson_value({
            "fold_id": diagnostic.get("fold_id"),
            "decision_at": diagnostic.get("timestamp"),
            "execution_at": execution_at,
            "next_execution_at": next_execution_at,
            "current_symbol": current_symbol,
            "target_symbol": target_symbol,
            "action": decision_action,
            "objective": selected.get("objective"),
            "objective_breakdown": selected.get("breakdown"),
            "gross_interval_return": gross_return,
            "effective_net_interval_return": selected_net,
            "cost_sides": sides,
            "one_side_cost_rate": base_cost_rate,
            "control_current_symbol": control_current,
            "control_target_symbol": control_target,
            "control_interval_return": control_net,
            "decision_value_added_vs_control": delta,
            "holding_days_before": holding_days,
            "candidate_equity": candidate_equity,
            "reference_equity": reference_equity,
            "forced_control": must_follow_control,
            "alternatives": [
                {
                    "symbol": item["symbol"],
                    "objective": item["objective"],
                    "eligible": item.get("eligible", True),
                    "breakdown": item.get("breakdown"),
                }
                for item in sorted(alternatives, key=lambda row: (-float(row["objective"]), str(row["symbol"])))
            ],
            "solver": solver,
        }))

        if selected_net is not None:
            candidate_equity *= 1.0 + selected_net
        if target_symbol == current_symbol:
            holding_days += 1
        else:
            current_symbol = target_symbol
            holding_days = 1 if target_symbol != "CASH" else 0

    if not decisions:
        raise MilpDecisionError("No causal decision rows were available for the selected period.")
    return decisions, {
        "nodes_explored": solver_nodes,
        "nodes_pruned": solver_pruned,
        "decisions_solved": solver_decisions,
        "forced_control_decisions": forced_control_decisions,
        "average_solve_ms": solve_ms_total / max(1, solver_decisions),
        "same_decision": len(decisions) - changed,
        "different_decision": changed,
        "milp_better": better,
        "control_better": worse,
        "neutral": neutral,
    }
