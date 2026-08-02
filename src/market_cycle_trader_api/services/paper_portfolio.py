from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    bson_value,
    get_paper_trading_state,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    clock_snapshot,
    create_paper_trading_client,
    position_snapshots,
)
from ..schemas.paper_trading import PaperTradingState
from .paper_market_scheduler import latest_paper_market_run


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _public_order(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: bson_value(document.get(key))
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
        if document.get(key) is not None
    }


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _record_snapshot(db: Any, snapshot: dict[str, Any]) -> None:
    now = _as_utc(utc_now())
    if now is None:
        raise RuntimeError("Unable to determine the current UTC timestamp.")
    latest = db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].find_one(
        {}, sort=[("recorded_at", -1)]
    )
    if latest is not None:
        recorded_at = _as_utc(latest.get("recorded_at"))
        if recorded_at is not None and recorded_at >= now - timedelta(seconds=60):
            return

    db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].insert_one(
        {
            "recorded_at": now,
            "portfolio_value": snapshot["portfolio_value"],
            "strategy_cash": snapshot["strategy_cash"],
            "market_value": snapshot["market_value"],
            "total_pnl": snapshot["total_pnl"],
            "total_return": snapshot["total_return"],
            "managed_symbol": snapshot["position"]["symbol"] if snapshot["position"] else None,
        }
    )


def paper_portfolio_snapshot(db: Any) -> dict[str, Any]:
    state = PaperTradingState.model_validate(get_paper_trading_state(db))
    client = create_paper_trading_client(db)
    positions = position_snapshots(client)
    clock = clock_snapshot(client)

    position: dict[str, Any] | None = None
    market_value = 0.0
    unrealized_pnl = 0.0

    if state.managed_symbol:
        actual = positions.get(state.managed_symbol)
        if actual is None:
            raise RuntimeError(
                f"The strategy state owns {state.managed_symbol}, but Alpaca returned no matching position."
            )
        quantity = float(actual["quantity"])
        average_entry_price = float(actual["average_entry_price"])
        current_price = float(actual["current_price"])
        market_value = float(actual["market_value"])
        cost_basis = quantity * average_entry_price
        unrealized_pnl = market_value - cost_basis
        position = {
            "symbol": state.managed_symbol,
            "quantity": quantity,
            "average_entry_price": average_entry_price,
            "current_price": current_price,
            "market_value": _round_money(market_value),
            "cost_basis": _round_money(cost_basis),
            "unrealized_pnl": _round_money(unrealized_pnl),
            "unrealized_return": (unrealized_pnl / cost_basis) if cost_basis else 0.0,
            "holding_sessions": int(state.holding_sessions),
        }

    strategy_cash = float(state.strategy_cash)
    portfolio_value = strategy_cash + market_value
    total_pnl = portfolio_value - float(state.initial_capital)
    total_return = total_pnl / float(state.initial_capital)

    snapshot = {
        "status": "ready",
        "recorded_at": utc_now(),
        "initial_capital": _round_money(state.initial_capital),
        "strategy_cash": _round_money(strategy_cash),
        "market_value": _round_money(market_value),
        "portfolio_value": _round_money(portfolio_value),
        "realized_pnl": _round_money(state.realized_pnl),
        "unrealized_pnl": _round_money(unrealized_pnl),
        "total_pnl": _round_money(total_pnl),
        "total_return": total_return,
        "position": position,
        "last_decision_date": state.last_decision_date,
        "last_execution_session": state.last_execution_session,
        "market_clock": clock,
        "next_session_run": latest_paper_market_run(db),
    }

    _record_snapshot(db, snapshot)

    history = list(
        db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].find(
            {}, {"_id": 0}
        ).sort("recorded_at", -1).limit(500)
    )
    history.reverse()
    orders = list(
        db[PAPER_TRADE_ORDERS_COLLECTION].find({}).sort("created_at", -1).limit(20)
    )
    snapshot["history"] = [bson_value(item) for item in history]
    snapshot["recent_orders"] = [_public_order(item) for item in orders]
    return snapshot
