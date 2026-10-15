"""Live portfolio from Live Alpaca only; no access to Paper transaction state."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone, timedelta
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .strategy_lab import get_trader_winner_summary

LIVE_PORTFOLIO_SNAPSHOTS_COLLECTION = "live_portfolio_snapshots"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _position(p: Any) -> dict[str, Any]:
    quantity = _float(getattr(p, "qty", 0))
    entry = _float(getattr(p, "avg_entry_price", 0))
    value = _float(getattr(p, "market_value", 0))
    return {
        "symbol": str(getattr(p, "symbol", "") or ""),
        "quantity": quantity,
        "average_entry_price": entry,
        "current_price": _float(getattr(p, "current_price", 0)),
        "market_value": value,
        "cost_basis": quantity * entry,
        "unrealized_pnl": _float(getattr(p, "unrealized_pl", 0)),
        "unrealized_return": (
            _float(getattr(p, "unrealized_plpc", 0))
            if quantity * entry else None
        ),
    }


def _order(order: Any) -> dict[str, Any]:
    quantity = _float(getattr(order, "qty", 0))
    filled = _float(getattr(order, "filled_qty", 0))
    avg = getattr(order, "filled_avg_price", None)
    return {
        "client_order_id": str(getattr(order, "client_order_id", "") or ""),
        "symbol": str(getattr(order, "symbol", "") or ""),
        "side": _label(getattr(order, "side", None)),
        "status": _label(getattr(order, "status", None)),
        "quantity": quantity,
        "filled_quantity": filled,
        "filled_average_price": _float(avg) if avg is not None else None,
        "filled_value": filled * _float(avg) if avg is not None else None,
        "submitted_at": _timestamp(getattr(order, "submitted_at", None)),
        "created_at": _timestamp(getattr(order, "created_at", None)),
        "filled_at": _timestamp(getattr(order, "filled_at", None)),
        "economic_fill": filled > 0,
        "execution_origin": "live_broker_history",
        "realized_pnl": None,  # Not derivable from a single fill.
        "decision_audit": None,  # No Live model decisions have been executed.
    }


def _store_observation(db: Any, equity: float, cash: float) -> list[dict[str, Any]]:
    collection = db[LIVE_PORTFOLIO_SNAPSHOTS_COLLECTION]
    now = datetime.now(timezone.utc)
    latest = collection.find_one({}, sort=[("recorded_at", -1)])
    recorded_at = latest.get("recorded_at") if latest else None
    if recorded_at is None or (
        now - (recorded_at.replace(tzinfo=timezone.utc) if recorded_at.tzinfo is None else recorded_at)
    ) >= timedelta(seconds=60):
        collection.insert_one({
            "recorded_at": now, "portfolio_value": equity,
            "strategy_cash": cash, "market": "live",
        })
    documents = list(collection.find(
        {}, {"_id": 0, "recorded_at": 1, "portfolio_value": 1, "strategy_cash": 1},
    ).sort("recorded_at", -1).limit(500))
    documents.reverse()
    return [{
        "recorded_at": _timestamp(item["recorded_at"]),
        "portfolio_value": _float(item["portfolio_value"]),
        "strategy_cash": _float(item.get("strategy_cash")),
        # Drawdown is deliberately omitted: deposits/withdrawals distort equity series.
        "peak_portfolio_value": None,
    } for item in documents]


def live_portfolio_snapshot(db: Any) -> dict[str, Any]:
    client = create_live_readonly_client()
    account = client.get_account()
    positions = [_position(p) for p in client.get_all_positions()]
    orders = [_order(o) for o in client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL, limit=100,
    ))]
    cash = _float(getattr(account, "cash", 0))
    equity = _float(getattr(account, "equity", 0))
    clock = client.get_clock()
    history = _store_observation(db, equity, cash)
    unrealized = sum(p["unrealized_pnl"] for p in positions)

    return {
        "status": "ready",
        "environment": "live",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "account_status": _label(getattr(account, "status", None)),
        "trading_blocked": bool(getattr(account, "trading_blocked", False)),
        "initial_capital": None,
        "strategy_cash": cash,
        "market_value": sum(p["market_value"] for p in positions),
        "portfolio_value": equity,
        "realized_pnl": None,
        "unrealized_pnl": unrealized,
        "total_pnl": None,
        "total_return": None,
        "position": positions[0] if len(positions) == 1 else None,
        "positions": positions,
        "last_decision_date": None,
        "last_execution_session": None,
        "market_clock": {
            "timestamp": _timestamp(getattr(clock, "timestamp", None)),
            "is_open": bool(getattr(clock, "is_open", False)),
            "next_open": _timestamp(getattr(clock, "next_open", None)),
            "next_close": _timestamp(getattr(clock, "next_close", None)),
        },
        "history": history,
        "recent_orders": orders,
        "audit": {
            "summary": {
                "distance_to_peak": None,
                "peak_portfolio_value": None,
                "maximum_drawdown": None,
            },
            "latest_decision": None,
            "position_excursion": None,
            "history": history,
            "pnl_by_asset": [{
                "symbol": item["symbol"], "realized_pnl": None,
                "unrealized_pnl": item["unrealized_pnl"],
            } for item in positions],
            "operations": orders,
        },
    }


def live_robot_status(db: Any) -> dict[str, Any]:
    """Live execution is deliberately disabled; this is not the Paper scheduler."""
    return {
        "enabled": False,
        "control_mode": "stopped",
        "status": "stopped",
        "phase": "live_readonly",
        "mode": "live_readonly",
        "scheduler_alive": False,
        "position_observer_alive": False,
        "execution_enabled": False,
        "next_execution_session": None,
        "active_run": None,
        "trader_winner": get_trader_winner_summary(db),
    }


def build_live_portfolio_export(db: Any) -> bytes:
    snapshot = live_portfolio_snapshot(db)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "live_portfolio.json",
            json.dumps(snapshot, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "README.txt",
            "LIVE account only. Broker orders and equity observations; "
            "no Paper orders, inferred model decisions or deposit-adjusted P/L. "
            "No order submission performed.\n",
        )
    return output.getvalue()
