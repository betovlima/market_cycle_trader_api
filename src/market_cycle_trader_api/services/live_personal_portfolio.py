"""Per-user Alpaca Live portfolio. All financial values come from Alpaca.

No MCT cash ledger, balance cache, portfolio Mongo writes or other account's
API-key fallback. Historical analytics may remain in the legacy admin view,
but the account owner gets fresh broker-reported values only.
"""
from __future__ import annotations

from datetime import timezone
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from .live_portfolio import _float, _label, _order, _position, _timestamp
from .live_account_return import alpaca_account_cumulative_return


def personal_live_portfolio_snapshot(broker: Any) -> dict[str, Any]:
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("Alpaca account identity is missing.")
    clock = broker.get_clock()
    timestamp = getattr(clock, "timestamp", None)
    if timestamp is None or timestamp.tzinfo is None:
        raise RuntimeError("Alpaca clock is missing an aware timestamp.")

    positions = [_position(item) for item in broker.get_all_positions()]
    orders = [
        _order(item)
        for item in broker.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=100,
        ))
    ]
    equity = _float(getattr(account, "equity", None))
    cash = _float(getattr(account, "cash", None))
    buying_power = getattr(account, "buying_power", None)
    non_marginable = getattr(account, "non_marginable_buying_power", None)
    return {
        "status": "ready",
        "environment": "live",
        "broker_authentication": "oauth",
        "account_id_suffix": account_id[-6:],
        "account_status": _label(getattr(account, "status", None)),
        "recorded_at": timestamp.astimezone(timezone.utc).isoformat(),
        "portfolio_value": equity,
        "strategy_cash": cash,
        "buying_power": _float(buying_power) if buying_power is not None else None,
        "non_marginable_buying_power": (
            _float(non_marginable) if non_marginable is not None else None
        ),
        "market_value": sum(p["market_value"] for p in positions),
        "unrealized_pnl": sum(p["unrealized_pnl"] for p in positions),
        "winner_performance": alpaca_account_cumulative_return(broker, current_equity=equity),
        "positions": positions,
        "recent_orders": orders,
        "market_clock": {
            "timestamp": _timestamp(timestamp),
            "is_open": bool(getattr(clock, "is_open", False)),
            "next_open": _timestamp(getattr(clock, "next_open", None)),
            "next_close": _timestamp(getattr(clock, "next_close", None)),
        },
    }


# Least-privilege response for invited Traders. The full Alpaca snapshot is
# still collected for the request, but only owner-specific display fields
# leave the API. The administrative Live API remains unchanged.
_TRADER_POSITION_FIELDS = frozenset({
    "symbol", "quantity", "average_entry_price", "current_price",
    "market_value", "unrealized_pnl", "unrealized_return",
})
_TRADER_ORDER_FIELDS = frozenset({
    "created_at", "symbol", "side", "status",
    "filled_quantity", "filled_average_price",
})
_TRADER_ACCOUNT_FIELDS = frozenset({
    "status", "environment", "account_id_suffix", "recorded_at",
    "portfolio_value", "strategy_cash", "unrealized_pnl",
})


def trader_personal_live_projection(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Only an OAuth owner's position, price, performance and order summary."""
    return {
        **{key: snapshot[key] for key in _TRADER_ACCOUNT_FIELDS if key in snapshot},
        "positions": [
            {key: value for key, value in position.items()
             if key in _TRADER_POSITION_FIELDS}
            for position in (snapshot.get("positions") or [])
            if isinstance(position, dict)
        ],
        "recent_orders": [
            {key: value for key, value in order.items()
             if key in _TRADER_ORDER_FIELDS}
            for order in (snapshot.get("recent_orders") or [])
            if isinstance(order, dict)
        ],
        "market_clock": {
            "is_open": bool((snapshot.get("market_clock") or {}).get("is_open", False)),
        },
    }
