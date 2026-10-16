"""Explicit read-only recovery of a Live session that missed its order window.

Never re-arms a stale session for trading. Requires a fresh intraday Winner
calculation, strictly isolated from the Live broker order submission path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .live_order_execution import ORDER_LEDGER
from .live_winner_session import LIVE_RUNS
from .live_winner_events import append_live_event

EASTERN = ZoneInfo("America/New_York")


def queue_expired_live_analysis(
    db: Any, *, execution_session: str, actor_email: str,
) -> dict[str, Any]:
    """Queue one *non-executable* recalibration of today's expired Live session."""
    if not actor_email:
        raise RuntimeError("An authenticated operator is required.")
    if live_control_status(db)["control_mode"] != "active":
        raise RuntimeError("Live Trader must be active for an analysis-only recovery.")

    broker = create_live_readonly_client()
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("Live Alpaca account identity missing.")
    clock = broker.get_clock()
    timestamp = getattr(clock, "timestamp", None)
    if not bool(getattr(clock, "is_open", False)) or timestamp is None or timestamp.tzinfo is None:
        raise RuntimeError("Analysis-only intraday recovery requires an open regular session.")
    session = timestamp.astimezone(EASTERN).date().isoformat()
    if execution_session != session:
        raise RuntimeError("Only the current regular trading session can be reanalyzed.")
    run_id = f"{account_id}:{session}"
    run = db[LIVE_RUNS].find_one({"_id": run_id})
    if not run or run.get("status") != "review_required" or run.get("phase") != "execution_window_expired":
        raise RuntimeError("No eligible expired Live Winner session; nothing was reset.")
    if (not bool(run.get("winner_signal_prepared"))
        or run.get("live_account_id") != account_id):
        raise RuntimeError("Expired Live Winner signal/account identity cannot be verified.")
    if db[ORDER_LEDGER].find_one({"broker_account_id": account_id, "execution_session": session}):
        raise RuntimeError("Existing Live order intent found; broker reconciliation is required.")
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
        raise RuntimeError("Outstanding Alpaca Live orders require reconciliation.")

    now = datetime.now(timezone.utc)
    result = db[LIVE_RUNS].update_one(
        {"_id": run_id, "status": "review_required", "phase": "execution_window_expired"},
        {"$set": {
            "status": "analysis_only_requested",
            "phase": "waiting_for_fresh_intraday_analysis",
            "recovery_requested_at": now,
            "recovery_requested_by": actor_email,
            "order_submission_enabled": False,
        }},
    )
    if result.matched_count != 1:
        raise RuntimeError("Live recovery request changed concurrently.")
    append_live_event(db, account_id=account_id, session=session,
                      stage="late_analysis_requested", message="Recuperacao somente para analise solicitada.")
    return {
        "status": "analysis_only_requested", "execution_session": session,
        "previous_status": "review_required",
        "message": "The Live worker will perform a fresh intraday Winner analysis. This cannot submit brokerage orders.",
        "order_submission_enabled": False,
    }
