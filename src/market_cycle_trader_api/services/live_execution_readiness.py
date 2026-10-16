"""Informational Alpaca Live readiness checks; no additional activation switches."""
from __future__ import annotations

from typing import Any

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .strategy_lab import get_trader_winner_summary


def live_execution_readiness(db: Any) -> dict[str, Any]:
    """Describe actual Live broker/strategy prerequisites, without hardcoded failures."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    broker = None
    try:
        broker = create_live_readonly_client()
        account = broker.get_account()
        account_id = str(getattr(account, "id", "") or "").strip()
        add("live_account", bool(account_id), "Live broker account verified." if account_id else "Account ID unavailable.")
        blocked = bool(getattr(account, "trading_blocked", True)) or bool(getattr(account, "account_blocked", True))
        add("broker_restrictions", not blocked, "No trading restriction." if not blocked else "Broker restricted.")
    except Exception as exc:
        add("live_account", False, f"Broker connection failed: {type(exc).__name__}.")
        add("broker_restrictions", False, "Cannot verify broker restrictions.")

    control = live_control_status(db)
    mode = str(control.get("control_mode") or "stopped")
    add("live_control", mode in {"active", "exit_only"}, f"Operating mode: {mode}.")

    try:
        if broker is None:
            raise RuntimeError("Live broker unavailable.")
        orders = broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        add("broker_open_orders", not orders, "No open Live order." if not orders else "Open orders must be reconciled.")
    except Exception:
        add("broker_open_orders", False, "Cannot reconcile broker open orders.")

    try:
        winner = get_trader_winner_summary(db)
        selected = isinstance(winner, dict) and bool(winner.get("id")) and bool(winner.get("configuration_hash"))
        add("winner_selected", selected, "Winner identity available." if selected else "Winner selection unavailable.")
    except Exception:
        add("winner_selected", False, "Cannot read Winner selection.")

    ready = all(item["passed"] for item in checks)
    return {
        "market": "live",
        "ready_for_real_orders": ready,
        "order_submission_enabled": mode in {"active", "exit_only"},
        "control_mode": mode,
        "checks": checks,
        "message": "Live follows the existing operating mode. Each order independently validates session, Winner, broker positions and idempotency.",
    }
