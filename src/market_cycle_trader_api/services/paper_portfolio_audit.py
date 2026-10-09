from __future__ import annotations

import math
from typing import Any

from ..infrastructure.persistence.mongo_repository import bson_value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def decision_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    utilities = plan.get("utilities") if isinstance(plan.get("utilities"), dict) else {}
    cash_edges = plan.get("cash_edges") if isinstance(plan.get("cash_edges"), dict) else {}
    current_asset = str(plan.get("current_asset") or "")
    target_asset = str(plan.get("target_asset") or "")
    raw_best_asset = str(plan.get("raw_best_asset") or "")

    current_utility = _finite_float(utilities.get(current_asset))
    rows: list[dict[str, Any]] = []
    for symbol, raw_value in utilities.items():
        symbol_text = str(symbol)
        if symbol_text == "CASH":
            continue
        utility = _finite_float(raw_value)
        if utility is None:
            continue
        cash_edge = _finite_float(cash_edges.get(symbol))
        rows.append(
            {
                "symbol": symbol_text,
                "utility": utility,
                "cash_edge": cash_edge,
                "is_target": symbol_text == target_asset,
                "is_current": symbol_text == current_asset,
                "is_raw_best": symbol_text == raw_best_asset,
                "utility_gap_vs_current": (
                    utility - current_utility
                    if current_utility is not None
                    else None
                ),
            }
        )

    rows.sort(key=lambda item: (-float(item["utility"]), str(item["symbol"])))
    best_utility = float(rows[0]["utility"]) if rows else None
    switch_margin = _finite_float(plan.get("effective_switch_margin"))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["utility_gap_vs_best"] = (
            float(row["utility"]) - best_utility
            if best_utility is not None
            else None
        )
        gap = row.get("utility_gap_vs_current")
        row["passes_switch_margin"] = bool(
            gap is not None
            and switch_margin is not None
            and float(gap) >= switch_margin
        )
    return rows


def decision_audit(
    document: dict[str, Any] | None,
    *,
    candidate_limit: int | None = 8,
) -> dict[str, Any] | None:
    if not document:
        return None

    utilities = document.get("utilities") if isinstance(document.get("utilities"), dict) else {}
    current_asset = str(document.get("current_asset") or "")
    target_asset = str(document.get("target_asset") or "")
    raw_best_asset = str(document.get("raw_best_asset") or "")
    current_utility = _finite_float(utilities.get(current_asset))
    target_utility = _finite_float(utilities.get(target_asset))
    raw_best_utility = _finite_float(utilities.get(raw_best_asset))
    raw_best_gap = (
        raw_best_utility - current_utility
        if raw_best_utility is not None and current_utility is not None
        else None
    )
    switch_margin = _finite_float(document.get("effective_switch_margin"))

    state_snapshot = (
        document.get("state_snapshot")
        if isinstance(document.get("state_snapshot"), dict)
        else {}
    )
    holding_sessions = document.get("holding_sessions_at_decision")
    if holding_sessions is None:
        holding_sessions = state_snapshot.get("holding_sessions")
    try:
        holding_sessions = int(holding_sessions) if holding_sessions is not None else None
    except (TypeError, ValueError):
        holding_sessions = None

    minimum_holding = document.get("minimum_holding_sessions")
    try:
        minimum_holding = int(minimum_holding) if minimum_holding is not None else None
    except (TypeError, ValueError):
        minimum_holding = None

    switch_margin_passed = bool(
        raw_best_gap is not None
        and switch_margin is not None
        and raw_best_gap >= switch_margin
    )
    holding_rule_satisfied = (
        None
        if holding_sessions is None or minimum_holding is None
        else holding_sessions >= minimum_holding
    )

    if bool(document.get("stateful_intervention")):
        selection_reason = "stateful_intervention"
    elif target_asset == "CASH":
        selection_reason = "cash_selected"
    elif target_asset and target_asset == current_asset:
        if raw_best_asset and raw_best_asset == current_asset:
            selection_reason = "current_asset_best"
        elif holding_rule_satisfied is False:
            selection_reason = "minimum_holding_not_reached"
        elif raw_best_gap is not None and switch_margin is not None and raw_best_gap < switch_margin:
            selection_reason = "switch_margin_not_reached"
        else:
            selection_reason = "hold_current"
    elif target_asset and target_asset == raw_best_asset:
        selection_reason = "raw_best_selected"
    else:
        selection_reason = "policy_selected_non_raw_best"

    explicit_origin = str(document.get("execution_origin") or "").strip().lower()
    if explicit_origin:
        execution_origin = explicit_origin
    elif bool(document.get("manual_current_session_recovery")) or str(
        document.get("plan_source") or ""
    ).strip().lower() == "manual_current_session_recovery":
        execution_origin = "manual_recovery"
    elif (
        document.get("contingency_completed_at") is not None
        or document.get("contingency_execution_started_at") is not None
    ):
        execution_origin = "manual_contingency"
    else:
        execution_origin = "historical_unknown"

    candidates = decision_candidates(document)
    visible_candidates = (
        candidates
        if candidate_limit is None
        else candidates[: max(0, int(candidate_limit))]
    )

    public_fields = (
        "plan_id",
        "status",
        "winner_strategy_id",
        "winner_strategy_name",
        "winner_strategy_revision",
        "winner_configuration_hash",
        "decision_date",
        "execution_session",
        "current_asset",
        "target_asset",
        "raw_best_asset",
        "action",
        "selected_utility",
        "effective_switch_margin",
        "calibrated_candidate_margin",
        "calibration_score",
        "training_end",
        "calibration_start",
        "calibration_end",
        "final_fit_end",
        "stateful_intervention",
        "stateful_control_target_asset",
        "stateful_risk_score",
        "stateful_risk_threshold",
        "stateful_confidence_margin",
        "stateful_confidence_threshold",
        "plan_source",
        "manual_current_session_recovery",
        "execution_trigger",
        "manual_execution_requested_at",
        "manual_execution_actor_email",
        "contingency_completed_at",
        "analysis_mode",
        "analysis_timestamp_utc",
        "live_session",
    )
    return {
        key: bson_value(document.get(key))
        for key in public_fields
        if document.get(key) is not None
    } | {
        "decision_origin": "model_generated_plan",
        "execution_origin": execution_origin,
        "selection_reason": selection_reason,
        "current_utility": current_utility,
        "target_utility": target_utility,
        "raw_best_utility": raw_best_utility,
        "target_vs_current_utility": (
            target_utility - current_utility
            if target_utility is not None and current_utility is not None
            else None
        ),
        "raw_best_vs_current_utility": raw_best_gap,
        "switch_margin_passed": switch_margin_passed,
        "holding_sessions_at_decision": holding_sessions,
        "minimum_holding_sessions": minimum_holding,
        "holding_rule_satisfied": holding_rule_satisfied,
        "candidate_count": len(candidates),
        "top_candidates": visible_candidates,
        "candidates": candidates if candidate_limit is None else None,
    }


def enrich_portfolio_history(
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    running_peak: float | None = None
    max_drawdown = 0.0

    for item in history:
        value = _finite_float(item.get("portfolio_value"))
        if value is None:
            continue
        running_peak = value if running_peak is None else max(running_peak, value)
        drawdown = value / running_peak - 1.0 if running_peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        enriched.append(
            {
                **{key: bson_value(value) for key, value in item.items()},
                "peak_portfolio_value": running_peak,
                "drawdown": drawdown,
            }
        )

    current_value = (
        _finite_float(enriched[-1].get("portfolio_value"))
        if enriched
        else None
    )
    peak_value = running_peak
    distance_to_peak = (
        current_value / peak_value - 1.0
        if current_value is not None and peak_value
        else None
    )
    return enriched, {
        "peak_portfolio_value": peak_value,
        "distance_to_peak": distance_to_peak,
        "maximum_drawdown": max_drawdown if enriched else None,
    }


def pnl_by_asset(
    orders: list[dict[str, Any]],
    *,
    current_position: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ledgers: dict[str, dict[str, float]] = {}

    def ledger(symbol: str) -> dict[str, float]:
        return ledgers.setdefault(
            symbol,
            {
                "quantity": 0.0,
                "average_cost": 0.0,
                "realized_pnl": 0.0,
                "buy_filled_quantity": 0.0,
                "sell_filled_quantity": 0.0,
                "economic_fill_count": 0.0,
                "unmatched_sell_quantity": 0.0,
            },
        )

    for order in orders:
        symbol = str(order.get("symbol") or "").strip().upper()
        side = str(order.get("side") or "").strip().lower()
        quantity = _finite_float(order.get("filled_quantity"))
        price = _finite_float(order.get("filled_average_price"))
        if not symbol or side not in {"buy", "sell"} or quantity is None or price is None:
            continue
        if quantity <= 0:
            continue

        item = ledger(symbol)
        item["economic_fill_count"] += 1.0
        if side == "buy":
            prior_quantity = item["quantity"]
            new_quantity = prior_quantity + quantity
            if new_quantity > 0:
                item["average_cost"] = (
                    prior_quantity * item["average_cost"] + quantity * price
                ) / new_quantity
            item["quantity"] = new_quantity
            item["buy_filled_quantity"] += quantity
            continue

        matched = min(quantity, max(0.0, item["quantity"]))
        if matched > 0:
            item["realized_pnl"] += matched * (price - item["average_cost"])
            item["quantity"] = max(0.0, item["quantity"] - matched)
            if item["quantity"] <= 1e-10:
                item["quantity"] = 0.0
                item["average_cost"] = 0.0
        unmatched = max(0.0, quantity - matched)
        item["unmatched_sell_quantity"] += unmatched
        item["sell_filled_quantity"] += quantity

    current_symbol = str((current_position or {}).get("symbol") or "").strip().upper()
    current_unrealized = _finite_float((current_position or {}).get("unrealized_pnl"))

    rows: list[dict[str, Any]] = []
    for symbol, item in ledgers.items():
        unrealized = current_unrealized if symbol == current_symbol and current_unrealized is not None else 0.0
        rows.append(
            {
                "symbol": symbol,
                "realized_pnl": item["realized_pnl"],
                "unrealized_pnl": unrealized,
                "total_pnl": item["realized_pnl"] + unrealized,
                "open_quantity": item["quantity"],
                "buy_filled_quantity": item["buy_filled_quantity"],
                "sell_filled_quantity": item["sell_filled_quantity"],
                "economic_fill_count": int(item["economic_fill_count"]),
                "unmatched_sell_quantity": item["unmatched_sell_quantity"],
            }
        )
    rows.sort(key=lambda item: (-float(item["total_pnl"]), str(item["symbol"])))
    return rows


def operation_rows(
    orders: list[dict[str, Any]],
    plan_map: dict[str, dict[str, Any]],
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order in reversed(orders[-max(0, int(limit)):]):
        filled_quantity = _finite_float(order.get("filled_quantity"))
        fill_price = _finite_float(order.get("filled_average_price"))
        economic_fill = bool(filled_quantity is not None and filled_quantity > 0)
        plan_id = str(order.get("plan_id") or "")
        audit = decision_audit(plan_map.get(plan_id), candidate_limit=8) if plan_id else None
        rows.append(
            {
                key: bson_value(order.get(key))
                for key in (
                    "client_order_id",
                    "plan_id",
                    "symbol",
                    "side",
                    "status",
                    "quantity",
                    "notional",
                    "filled_quantity",
                    "filled_average_price",
                    "submitted_at",
                    "filled_at",
                    "created_at",
                    "updated_at",
                )
                if order.get(key) is not None
            }
            | {
                "economic_fill": economic_fill,
                "filled_value": (
                    filled_quantity * fill_price
                    if economic_fill and fill_price is not None
                    else None
                ),
                "decision_available": audit is not None,
                "execution_origin": (
                    audit.get("execution_origin") if audit else "historical_unknown"
                ),
                "decision_audit": audit,
            }
        )
    return rows
