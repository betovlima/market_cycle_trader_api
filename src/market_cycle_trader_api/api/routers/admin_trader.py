from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...services.paper_market_scheduler import (
    list_admin_operation_logs,
    paper_market_robot_status,
    set_trader_control_mode,
)

router = APIRouter(prefix="/api/admin/trader-control", tags=["admin-trader-control"])


class TraderControlRequest(BaseModel):
    mode: Literal["active", "paused", "exit_only", "stopped"]
    cancel_pending_run: bool = False
    reason: str | None = Field(default=None, max_length=500)



@router.get("/status")
def read_trader_control_status() -> dict[str, Any]:
    return paper_market_robot_status(database())


@router.post("/mode")
def update_trader_control_mode(
    payload: TraderControlRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return set_trader_control_mode(
            database(),
            mode=payload.mode,
            reason=payload.reason,
            actor_email=identity.email,
            cancel_pending_run=payload.cancel_pending_run,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/history")
def read_trader_control_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    items = list_admin_operation_logs(database(), limit=limit)
    return {"count": len(items), "items": items}
