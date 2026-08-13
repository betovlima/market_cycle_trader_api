from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..persistence.mongo_repository import get_alpaca_credentials


@dataclass(frozen=True)
class PaperOrderFill:
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
        if self.filled_average_price is None:
            return 0.0
        return float(self.filled_quantity * self.filled_average_price)


def _require_trading_sdk():
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
    except ImportError as exc:
        raise RuntimeError(
            "alpaca-py is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return (
        TradingClient,
        OrderSide,
        QueryOrderStatus,
        TimeInForce,
        GetOrdersRequest,
        MarketOrderRequest,
    )


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_unverified_paper_trading_client():
    

    TradingClient, *_ = _require_trading_sdk()
    credentials = get_alpaca_credentials()
    return TradingClient(
        credentials["api_key_id"],
        credentials["secret_key"],
        paper=True,
    )


def create_paper_trading_client(db: Any):
    

    from ...schemas.paper_trading import PaperTradingSettings
    from ..persistence.mongo_repository import get_paper_trading_settings

    settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
    if not settings.enabled:
        raise RuntimeError("Paper trading is disabled in MongoDB.")
    expected_account = str(settings.paper_account_id or "").strip()
    if not expected_account:
        raise RuntimeError(
            "The Alpaca paper account has not been bound yet. "
            "Call POST /api/admin/setup/initialize."
        )

    client = create_unverified_paper_trading_client()
    account = client.get_account()
    actual_account = str(getattr(account, "id", "") or "").strip()
    if actual_account != expected_account:
        raise RuntimeError(
            "The configured Alpaca paper credentials belong to a different account: "
            f"expected={expected_account}, actual={actual_account or 'unknown'}."
        )
    return client

def account_snapshot(client: Any) -> dict[str, Any]:
    account = client.get_account()
    return {
        "id": str(getattr(account, "id", "") or ""),
        "account_number": str(getattr(account, "account_number", "") or ""),
        "status": _enum_text(getattr(account, "status", "")),
        "cash": _float_value(getattr(account, "cash", 0)),
        "equity": _float_value(getattr(account, "equity", 0)),
        "buying_power": _float_value(getattr(account, "buying_power", 0)),
        "trading_blocked": bool(getattr(account, "trading_blocked", False)),
        "account_blocked": bool(getattr(account, "account_blocked", False)),
        "trade_suspended_by_user": bool(
            getattr(account, "trade_suspended_by_user", False)
        ),
    }


def assert_account_can_trade(snapshot: dict[str, Any]) -> None:
    if snapshot.get("trading_blocked"):
        raise RuntimeError("The Alpaca paper account is trading-blocked.")
    if snapshot.get("account_blocked"):
        raise RuntimeError("The Alpaca paper account is blocked.")
    if snapshot.get("trade_suspended_by_user"):
        raise RuntimeError("Trading is suspended by the Alpaca paper account user.")
    status = str(snapshot.get("status") or "").lower()
    if status and status not in {"active"}:
        raise RuntimeError(f"The Alpaca paper account is not active: status={status}.")


def clock_snapshot(client: Any) -> dict[str, Any]:
    clock = client.get_clock()
    return {
        "timestamp": getattr(clock, "timestamp", None),
        "is_open": bool(getattr(clock, "is_open", False)),
        "next_open": getattr(clock, "next_open", None),
        "next_close": getattr(clock, "next_close", None),
    }


def asset_snapshot(client: Any, symbol: str) -> dict[str, Any]:
    asset = client.get_asset(symbol)
    return {
        "symbol": str(getattr(asset, "symbol", symbol) or symbol).upper(),
        "status": _enum_text(getattr(asset, "status", "")),
        "tradable": bool(getattr(asset, "tradable", False)),
        "fractionable": bool(getattr(asset, "fractionable", False)),
        "shortable": bool(getattr(asset, "shortable", False)),
    }


def position_snapshots(client: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for position in client.get_all_positions():
        symbol = str(getattr(position, "symbol", "") or "").upper()
        if not symbol:
            continue
        output[symbol] = {
            "symbol": symbol,
            "quantity": _float_value(getattr(position, "qty", 0)),
            "side": _enum_text(getattr(position, "side", "")),
            "market_value": _float_value(getattr(position, "market_value", 0)),
            "average_entry_price": _float_value(
                getattr(position, "avg_entry_price", 0)
            ),
            "current_price": _float_value(getattr(position, "current_price", 0)),
        }
    return output


def open_order_snapshots(client: Any, symbols: list[str] | None = None) -> list[dict[str, Any]]:
    _, _, QueryOrderStatus, _, GetOrdersRequest, _ = _require_trading_sdk()
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=symbols or None,
    )
    orders = client.get_orders(filter=request)
    return [order_snapshot(order) for order in orders]


def order_snapshot(order: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(order, "id", "") or ""),
        "client_order_id": str(getattr(order, "client_order_id", "") or ""),
        "symbol": str(getattr(order, "symbol", "") or "").upper(),
        "side": _enum_text(getattr(order, "side", "")),
        "status": _enum_text(getattr(order, "status", "")),
        "quantity": _float_value(getattr(order, "qty", 0)),
        "notional": _float_value(getattr(order, "notional", 0)),
        "filled_quantity": _float_value(getattr(order, "filled_qty", 0)),
        "filled_average_price": (
            _float_value(getattr(order, "filled_avg_price", 0))
            if getattr(order, "filled_avg_price", None) is not None
            else None
        ),
        "submitted_at": getattr(order, "submitted_at", None),
        "filled_at": getattr(order, "filled_at", None),
    }


def submit_market_sell(
    client: Any,
    *,
    symbol: str,
    quantity: float,
    client_order_id: str,
) -> Any:
    _, OrderSide, _, TimeInForce, _, MarketOrderRequest = _require_trading_sdk()
    if quantity <= 0:
        raise ValueError("Sell quantity must be positive.")
    request = MarketOrderRequest(
        symbol=symbol,
        qty=float(quantity),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        extended_hours=False,
        client_order_id=client_order_id,
    )
    return client.submit_order(order_data=request)


def submit_market_buy_notional(
    client: Any,
    *,
    symbol: str,
    notional: float,
    client_order_id: str,
) -> Any:
    _, OrderSide, _, TimeInForce, _, MarketOrderRequest = _require_trading_sdk()
    if notional < 1.0:
        raise ValueError("Buy notional must be at least $1.00.")
    request = MarketOrderRequest(
        symbol=symbol,
        notional=round(float(notional), 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        extended_hours=False,
        client_order_id=client_order_id,
    )
    return client.submit_order(order_data=request)


def wait_for_order(
    client: Any,
    *,
    order_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> PaperOrderFill:
    terminal_statuses = {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "replaced",
        "stopped",
        "suspended",
        "calculated",
    }
    deadline = time.monotonic() + float(timeout_seconds)
    latest = client.get_order_by_id(order_id)

    while _enum_text(getattr(latest, "status", "")) not in terminal_statuses:
        if time.monotonic() >= deadline:
            try:
                client.cancel_order_by_id(order_id)
            except Exception:
                pass
            time.sleep(min(1.0, float(poll_interval_seconds)))
            latest = client.get_order_by_id(order_id)
            break
        time.sleep(float(poll_interval_seconds))
        latest = client.get_order_by_id(order_id)

    snapshot = order_snapshot(latest)
    return PaperOrderFill(
        order_id=snapshot["id"],
        client_order_id=snapshot["client_order_id"],
        symbol=snapshot["symbol"],
        side=snapshot["side"],
        status=snapshot["status"],
        filled_quantity=float(snapshot["filled_quantity"]),
        filled_average_price=snapshot["filled_average_price"],
        submitted_at=snapshot["submitted_at"],
        filled_at=snapshot["filled_at"],
    )


def calendar_session_dates(
    client: Any,
    *,
    start_date: Any,
    end_date: Any,
) -> list[str]:
    try:
        from alpaca.trading.requests import GetCalendarRequest
    except ImportError as exc:
        raise RuntimeError(
            "alpaca-py is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    request = GetCalendarRequest(start=start_date, end=end_date)
    sessions = client.get_calendar(filters=request)
    output: list[str] = []
    for session in sessions:
        value = getattr(session, "date", None)
        if value is not None:
            output.append(str(value))
    return sorted(set(output))
