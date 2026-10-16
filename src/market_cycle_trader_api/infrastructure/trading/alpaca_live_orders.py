"""Alpaca Live order primitives. No dependency on Paper state or broker clients."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


@dataclass(frozen=True)
class LiveOrderFill:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    filled_quantity: float
    filled_average_price: float | None
    submitted_at: datetime | None
    filled_at: datetime | None

    @property
    def filled_notional(self) -> float:
        return self.filled_quantity * float(self.filled_average_price or 0)


def asset_snapshot(client: Any, symbol: str) -> dict[str, Any]:
    asset = client.get_asset(symbol)
    return {
        "symbol": str(getattr(asset, "symbol", symbol) or symbol).upper(),
        "status": _value(getattr(asset, "status", "")),
        "tradable": bool(getattr(asset, "tradable", False)),
        "fractionable": bool(getattr(asset, "fractionable", False)),
    }


def submit_market_sell(
    client: Any, *, symbol: str, quantity: float, client_order_id: str,
) -> Any:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if float(quantity) <= 0:
        raise ValueError("Live sell quantity must be positive.")
    request = MarketOrderRequest(
        symbol=symbol, qty=float(quantity),
        side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        extended_hours=False, client_order_id=client_order_id,
    )
    return client.submit_order(order_data=request)


def submit_market_buy_notional(
    client: Any, *, symbol: str, notional: float, client_order_id: str,
) -> Any:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if float(notional) < 1:
        raise ValueError("Live buy notional must be at least USD 1.")
    request = MarketOrderRequest(
        symbol=symbol, notional=round(float(notional), 2),
        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        extended_hours=False, client_order_id=client_order_id,
    )
    return client.submit_order(order_data=request)


def wait_for_order(
    client: Any, *, order_id: str, timeout_seconds: int, poll_interval_seconds: float,
) -> LiveOrderFill:
    terminal = {
        "filled", "canceled", "cancelled", "expired", "rejected",
        "replaced", "stopped", "suspended", "calculated",
    }
    deadline = time.monotonic() + float(timeout_seconds)
    order = client.get_order_by_id(order_id)
    while _value(getattr(order, "status", "")) not in terminal:
        if time.monotonic() >= deadline:
            try:
                client.cancel_order_by_id(order_id)
            except Exception:
                pass
            time.sleep(min(1.0, float(poll_interval_seconds)))
            order = client.get_order_by_id(order_id)
            break
        time.sleep(float(poll_interval_seconds))
        order = client.get_order_by_id(order_id)

    avg = getattr(order, "filled_avg_price", None)
    return LiveOrderFill(
        order_id=str(getattr(order, "id", "") or ""),
        client_order_id=str(getattr(order, "client_order_id", "") or ""),
        symbol=str(getattr(order, "symbol", "") or "").upper(),
        side=_value(getattr(order, "side", "")),
        status=_value(getattr(order, "status", "")),
        filled_quantity=float(getattr(order, "filled_qty", 0) or 0),
        filled_average_price=float(avg) if avg is not None else None,
        submitted_at=getattr(order, "submitted_at", None),
        filled_at=getattr(order, "filled_at", None),
    )
