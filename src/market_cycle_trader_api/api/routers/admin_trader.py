from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...services.paper_plan_reconciliation import (
    preview_paper_plan_reconciliation,
    close_obsolete_paper_plans,
)
from ...services.paper_market_scheduler import (
    execute_manual_current_session_plan,
    list_admin_operation_logs,
    paper_market_manual_recovery_status,
    paper_market_robot_status,
    prepare_manual_current_session_plan,
    set_trader_control_mode,
)

router = APIRouter(prefix="/api/admin/trader-control", tags=["admin-trader-control"])


class TraderControlRequest(BaseModel):
    mode: Literal["active", "paused", "exit_only", "stopped"]
    cancel_pending_run: bool = False
    reason: str | None = Field(default=None, max_length=500)


class ManualRecoveryExecuteRequest(BaseModel):
    confirm: Literal["EXECUTE_TODAY"]
    plan_id: str | None = Field(default=None, max_length=160)


class HistoricalPlanReconciliationRequest(BaseModel):
    confirm: Literal["RECONCILE_EXPIRED_PREPARED_PLANS"]
    plan_ids: list[str] = Field(min_length=1, max_length=100)


@router.get("/plan-reconciliation/preview")
def preview_historical_paper_plans(
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Read-only, with explicit eligibility and blockers for each prepared plan."""
    return preview_paper_plan_reconciliation(database())


@router.post("/plan-reconciliation/close")
def close_historical_paper_plans(
    payload: HistoricalPlanReconciliationRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """An explicitly confirmed, audited close of eligible historical HOLD plans."""
    try:
        return close_obsolete_paper_plans(
            database(), plan_ids=payload.plan_ids, actor_email=identity.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/status")
def read_trader_control_status(
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    db = database()
    return {
        **paper_market_robot_status(db),
        "manual_recovery": paper_market_manual_recovery_status(db),
    }


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


@router.post("/manual-recovery/prepare")
def prepare_current_session_recovery(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return prepare_manual_current_session_plan(
            database(),
            actor_email=identity.email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/manual-recovery/execute")
def execute_current_session_recovery(
    payload: ManualRecoveryExecuteRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return execute_manual_current_session_plan(
            database(),
            plan_id=payload.plan_id,
            actor_email=identity.email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/history")
def read_trader_control_history(
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    items = list_admin_operation_logs(database(), limit=limit)
    return {"count": len(items), "items": items}
