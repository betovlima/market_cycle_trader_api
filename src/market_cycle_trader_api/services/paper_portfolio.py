from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from ..infrastructure.persistence.mongo_repository import (
    PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    bson_value,
    get_paper_trading_state,
    replace_paper_trading_state,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    clock_snapshot,
    create_paper_trading_client,
    position_snapshots,
)
from ..schemas.paper_trading import PaperTradingState
from .paper_market_scheduler import latest_paper_market_run
from .paper_portfolio_audit import (
    decision_audit,
    enrich_portfolio_history,
    operation_rows,
    pnl_by_asset,
)


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



def _public_decision_audit(
    document: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return decision_audit(document, candidate_limit=8)


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
    raw_state = get_paper_trading_state(db)
    client = create_paper_trading_client(db)
    positions = position_snapshots(client)
    try:
        state = PaperTradingState.model_validate(raw_state)
    except ValidationError:
        symbol = str(raw_state.get("managed_symbol") or "").strip().upper() or None
        quantity = float(raw_state.get("managed_quantity") or 0.0)
        if symbol is None and quantity > 0.0 and not positions:
            repaired = {
                **raw_state,
                "managed_symbol": None,
                "managed_quantity": 0.0,
                "average_entry_price": None,
                "holding_sessions": 0,
            }
            state = PaperTradingState.model_validate(repaired)
            replace_paper_trading_state(db, state.model_dump(mode="python"))
        else:
            raise RuntimeError(
                "Paper portfolio state is inconsistent with the Alpaca account and requires reconciliation before display."
            )
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
    all_orders = list(
        db[PAPER_TRADE_ORDERS_COLLECTION].find({}).sort("created_at", 1)
    )
    plans = list(
        db[PAPER_TRADE_PLANS_COLLECTION].find({}).sort("created_at", 1)
    )
    plan_map = {
        str(plan.get("plan_id")): plan
        for plan in plans
        if plan.get("plan_id")
    }
    recent_orders = list(reversed(all_orders[-20:]))

    public_history = [bson_value(item) for item in history]
    audit_history, audit_summary = enrich_portfolio_history(public_history)
    latest_plan = plans[-1] if plans else None
    latest_decision = decision_audit(latest_plan, candidate_limit=None)

    snapshot["history"] = public_history
    snapshot["recent_orders"] = [
        {
            **_public_order(item),
            "decision_audit": _public_decision_audit(
                plan_map.get(str(item.get("plan_id")))
            )
            if item.get("plan_id")
            else None,
        }
        for item in recent_orders
    ]
    snapshot["audit"] = {
        "summary": {
            **audit_summary,
            "decision_plan_count": len(plans),
            "order_count": len(all_orders),
            "economic_fill_count": sum(
                1
                for item in all_orders
                if float(item.get("filled_quantity") or 0.0) > 0.0
            ),
        },
        "history": audit_history,
        "latest_decision": latest_decision,
        "operations": operation_rows(
            all_orders,
            plan_map,
            limit=100,
        ),
        "pnl_by_asset": pnl_by_asset(
            all_orders,
            current_position=position,
        ),
    }
    return snapshot
