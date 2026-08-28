from __future__ import annotations

from typing import Any

from .paper_portfolio import paper_portfolio_snapshot


def _safe_order(order: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "symbol",
        "side",
        "status",
        "quantity",
        "filled_quantity",
        "filled_average_price",
        "submitted_at",
        "filled_at",
        "created_at",
        "updated_at",
    )
    output = {key: order.get(key) for key in allowed if order.get(key) is not None}
    decision_audit = order.get("decision_audit")
    if isinstance(decision_audit, dict):
        output["decision_audit"] = decision_audit
    return output


def public_paper_portfolio_snapshot(db: Any) -> dict[str, Any]:
    






    source = paper_portfolio_snapshot(db)
    allowed = (
        "status",
        "recorded_at",
        "initial_capital",
        "strategy_cash",
        "market_value",
        "portfolio_value",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "total_return",
        "position",
        "last_decision_date",
        "last_execution_session",
        "market_clock",
        "history",
    )
    output = {key: source.get(key) for key in allowed}
    output["recent_orders"] = [
        _safe_order(item) for item in source.get("recent_orders", [])
    ]
    return output
