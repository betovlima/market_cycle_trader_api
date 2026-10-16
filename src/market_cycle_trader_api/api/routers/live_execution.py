"""Dedicated Alpaca LIVE execution API. Uses only the Winner Live session.

This router never invokes the Paper trading endpoint or accepts arbitrary
tickers, quantities or order sizes from clients.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...infrastructure.trading.alpaca_live import create_live_readonly_client
from ...services.live_control import live_control_status
from ...services.live_winner_session import LIVE_RUNS
from ...services.live_order_execution import execute_recalibrated_live_winner_session

router = APIRouter(prefix="/api/live-execution", tags=["alpaca-live-execution"])
EASTERN = ZoneInfo("America/New_York")


@router.post("/winner/execute")
def execute_current_winner_live(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Execute exactly the verified current session's Winner plan on Alpaca Live.

    The broker and current clock establish session identity; the model and the
    persisted plan determine the action and target. The executor independently
    verifies broker account, chosen Winner, current holdings, exchange window,
    open orders, order ledger, and buying power before submitting.
    """
    db = database()
    if live_control_status(db)["control_mode"] not in {"active", "exit_only"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Live Trader is not active.")

    try:
        broker = create_live_readonly_client()
        account_id = str(getattr(broker.get_account(), "id", "") or "").strip()
        clock = broker.get_clock()
        timestamp: datetime | None = getattr(clock, "timestamp", None)
        if not account_id or timestamp is None or timestamp.tzinfo is None:
            raise RuntimeError("Alpaca Live account or clock is incomplete.")
        if not bool(getattr(clock, "is_open", False)):
            raise HTTPException(status_code=409, detail="Alpaca Live market is closed.")
        session = timestamp.astimezone(EASTERN).date().isoformat()
        run = db[LIVE_RUNS].find_one({"_id": f"{account_id}:{session}"})
        if not run or run.get("status") != "recalibrated_no_orders":
            raise HTTPException(status_code=409, detail="No executable current-session Live Winner plan.")
        result = execute_recalibrated_live_winner_session(db, session)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if result.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Live Winner did not execute: {result.get('status')}")
    return result
