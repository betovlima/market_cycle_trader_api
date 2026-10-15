"""Preflight for future Live execution. Never sends broker orders."""
from __future__ import annotations

from typing import Any

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from .live_control import live_control_status
from .live_order_execution import order_submission_opted_in


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
    # The separate operator opt-in must be reported truthfully, even before
    # the production executor has been certified by an end-to-end dry-run.
    authorized = order_submission_opted_in()
    # Deliberately fail closed until end-to-end validation and reconciliation are reviewed.
    add("reviewed_live_executor", False, "Execution code exists but has not completed a production-like end-to-end review.")
    add("order_idempotency", False, "Unique order ledger exists; complete broker recovery verification is still pending.")
    add("risk_limits", False, "Risk checks exist but still require broker-level scenario validation.")
    add("plan_validation", False, "Winner signals are generated; end-to-end session parity is not yet certified.")
    add("operator_activation", authorized, "Explicit server opt-in configured." if authorized else "Real-order server opt-in is disabled.")
    return {
        "market": "live",
        "ready_for_real_orders": False,
        "order_submission_enabled": authorized and status["control_mode"] in {"active", "exit_only"},
        "control_mode": status["control_mode"],
        "checks": checks,
        "message": ("WARNING: Live real-money order authorization is set before full certification." if authorized else "Live real-order submission is disabled by default. Winner monitoring and signals are read-only."),
    }
