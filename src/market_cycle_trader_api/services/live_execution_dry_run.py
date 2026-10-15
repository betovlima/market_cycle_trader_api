"""Read-only Live order-intent simulation. Never constructs or submits an Alpaca order.

An operator-supplied target is a hypothetical scenario, NOT a Winner decision.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from ..services.live_control import live_control_status
from ..services.live_portfolio import live_portfolio_snapshot

def preview_live_intent(db: Any, *, target: str, max_allocation_fraction: float) -> dict[str, Any]:
    symbol = str(target).strip().upper()
    if symbol != "CASH" and (not symbol.isascii() or not symbol.isalnum() or len(symbol) > 8):
        raise ValueError("Target must be CASH or a simple ticker (up to eight ASCII letters/digits).")
    if not (0 < max_allocation_fraction <= 1):
        raise ValueError("max_allocation_fraction must be within (0, 1].")

    # Swagger placeholder is not a validated instrument.
    if symbol == "STRING":
        raise ValueError("STRING is a Swagger placeholder, not a validated trading target.")
    if symbol != "CASH":
        asset = create_live_readonly_client().get_asset(symbol)
        if str(getattr(asset, "symbol", "")).upper() != symbol or not bool(getattr(asset, "tradable", False)):
            raise ValueError(f"Asset {symbol} is unavailable or not tradable in Alpaca Live.")

    state = live_control_status(db)
    snapshot = live_portfolio_snapshot(db)
    equity = float(snapshot["portfolio_value"])
    cash = float(snapshot["strategy_cash"])
    positions = snapshot["positions"]
    market_open = bool(snapshot["market_clock"]["is_open"])
    blocked = bool(snapshot["trading_blocked"])
    existing = {str(p["symbol"]).upper(): float(p["market_value"]) for p in positions}
    current_target = existing.get(symbol, 0.0)
    desired = 0.0 if symbol == "CASH" else max(0.0, equity) * max_allocation_fraction
    delta = 0.0 if symbol == "CASH" else desired - current_target
    prospective_buy = max(0.0, delta)
    reduction_symbols = sorted(p["symbol"] for p in positions if symbol == "CASH" or p["symbol"].upper() != symbol)
    checks = {
        "broker_not_blocked": not blocked,
        "market_open": market_open,
        "valid_positive_equity": equity > 0,
        "cash_covers_target_increase_without_sales": prospective_buy <= max(0, cash),
        "control_active": state["control_mode"] == "active",
        "no_existing_open_orders": False,  # Requires separate broker reconciliation and locking.
        "strategy_signal_verified": False,  # No Winner decision is generated here.
        "risk_limits_verified": False,  # Must be implemented before execution.
    }
    return {
        "market": "live",
        "kind": "hypothetical_operator_preview",
        "target": symbol,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "control_mode": state["control_mode"],
        "order_submission_enabled": False,
        "would_submit_order": False,
        "ready_for_real_orders": False,
        "current_positions_count": len(positions),
        "cash": cash,
        "equity": equity,
        "current_target_market_value": current_target,
        "hypothetical_target_market_value": desired,
        "hypothetical_target_delta": delta,
        "positions_to_review_for_reduction": reduction_symbols,
        "checks": checks,
        "note": "Read-only snapshot preview, not an order plan. Quotes, fills, slippage, fees and Winner signal have not been validated. No orders are submitted.",
    }
