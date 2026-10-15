"""Live Winner readiness snapshots, independent of Paper plans and Paper broker.

This is a non-executing prerequisite for a future Live Winner scheduler.
The Winner metadata is not a decision signal. No plan can execute from here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .strategy_lab import get_trader_winner_summary

COLLECTION = "live_winner_preflight_snapshots"


def capture_live_winner_preflight(db: Any) -> dict[str, Any]:
    control = live_control_status(db)
    if control["control_mode"] not in {"active", "exit_only"}:
        return {"status": "skipped", "reason": "live_not_active"}
    client = create_live_readonly_client()
    account = client.get_account()
    clock = client.get_clock()
    winner = get_trader_winner_summary(db)
    if not isinstance(winner, dict) or not winner.get("name"):
        raise RuntimeError("No active Winner metadata available for Live inspection.")
    now = datetime.now(timezone.utc)
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("Live account identity is missing.")
    document = {
        "market": "live",
        "captured_at": now,
        "live_account_id": account_id,
        "winner_name": winner["name"],
        "control_mode": control["control_mode"],
        "broker_timestamp": getattr(clock, "timestamp", None),
        "market_open": bool(getattr(clock, "is_open", False)),
        "next_market_open": getattr(clock, "next_open", None),
        "next_market_close": getattr(clock, "next_close", None),
        "phase": "winner_preflight_only",
        "winner_decision_generated": False,
        "order_submission_enabled": False,
    }
    db[COLLECTION].insert_one(document.copy())
    return {
        **document,
        "captured_at": now.isoformat(),
        "broker_timestamp": str(document["broker_timestamp"]) if document["broker_timestamp"] else None,
        "next_market_open": str(document["next_market_open"]) if document["next_market_open"] else None,
        "next_market_close": str(document["next_market_close"]) if document["next_market_close"] else None,
        "status": "captured",
    }
