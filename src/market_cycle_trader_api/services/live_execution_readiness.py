"""Preflight for future Live execution. Never sends broker orders."""
from __future__ import annotations

from typing import Any

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from .live_control import live_control_status


def live_execution_readiness(db: Any) -> dict[str, Any]:
    """Fail-closed report; a passing account check never enables trading."""
    checks: list[dict[str, Any]] = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    client = None
    try:
        client = create_live_readonly_client()
        account = client.get_account()
        account_id = str(getattr(account, "id", "") or "").strip()
        add("live_account", bool(account_id), "Dedicated Live credentials and account identity verified." if account_id else "Live account ID missing.")
        trading_blocked = bool(getattr(account, "trading_blocked", True))
        account_blocked = bool(getattr(account, "account_blocked", True))
        add("broker_restrictions", not trading_blocked and not account_blocked, "Account reports restrictions." if trading_blocked or account_blocked else "No account-level trading block reported.")
    except Exception as exc:
        add("live_account", False, f"Read-only Live account check failed: {type(exc).__name__}.")
        add("broker_restrictions", False, "Cannot verify account trading restrictions.")

    status = live_control_status(db)
    add("separate_live_control", isinstance(status.get("revision"), int), "Dedicated Live control state found.")
    try:
        if client is None:
            raise RuntimeError("Live account unavailable.")
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        add("broker_open_orders", len(open_orders) == 0,
            "No open broker orders returned in queried window." if not open_orders else
            f"{len(open_orders)} open broker orders require reconciliation.")
    except Exception:
        add("broker_open_orders", False, "Cannot establish broker open-order status.")
    # Deliberately hard-block until a reviewed execution engine and order safeguards exist.
    add("reviewed_live_executor", False, "Real-order executor is not implemented or authorized.")
    add("order_idempotency", False, "Persistent per-session order idempotency and recovery not implemented.")
    add("risk_limits", False, "Position, buying-power, daily-loss and emergency-stop checks not implemented.")
    add("plan_validation", False, "Winner plan parity and current-session signal validation not implemented.")
    add("operator_activation", False, "Separate real-order activation and second confirmation not implemented.")
    return {
        "market": "live",
        "ready_for_real_orders": False,
        "order_submission_enabled": False,
        "control_mode": status["control_mode"],
        "checks": checks,
        "message": "Live real-order execution is disabled. Monitoring controls do not activate trading.",
    }
