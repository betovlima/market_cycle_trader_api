"""Live visibility API. Never submit, modify or cancel brokerage orders."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...services.live_control import LiveControlConflict, set_live_control_mode
from ...services.live_execution_readiness import live_execution_readiness
from ...services.live_execution_dry_run import preview_live_intent
from ...services.live_winner_preflight import capture_live_winner_preflight
from ...services.live_expired_recovery import queue_expired_live_analysis
from ...services.live_winner_events import list_live_events, export_live_events
from ...services.live_workflow_monitor import live_workflow_monitor
from ...services.live_manual_reconciliation import (
    ManualReconciliationBlocked, manual_reconciliation_preview,
    reconcile_manual_live_rotation,
)
from ...services.live_portfolio import (
    build_live_portfolio_export,
    live_portfolio_snapshot,
    live_robot_status,
)

router = APIRouter(prefix="/api/live-market", tags=["live-market-readonly"])


@router.get("/portfolio")
def portfolio() -> dict[str, Any]:
    try:
        return live_portfolio_snapshot(database())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/robot/status")
def robot_status() -> dict[str, Any]:
    return live_robot_status(database())


class LiveControlModeRequest(BaseModel):
    mode: Literal["active", "paused", "exit_only", "stopped"]
    expected_revision: int = Field(ge=0)
    reason: str = Field(min_length=3, max_length=500)


@router.post("/robot/mode")
def update_robot_mode(
    payload: LiveControlModeRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Set the operational Alpaca Live Trader mode, independently of Paper."""
    try:
        return {
            **set_live_control_mode(
                database(),
                mode=payload.mode,
                expected_revision=payload.expected_revision,
                actor_email=(identity.email or "").strip() or identity.subject,
                reason=payload.reason,
            ),
            "message": "Live Trader operating mode updated.",
        }
    except (LiveControlConflict, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


class LiveExpiredAnalysisRequest(BaseModel):
    execution_session: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confirm: Literal["LIVE_ANALYSIS_ONLY_NO_ORDERS"]


@router.post("/winner/recover-expired-analysis")
def recover_expired_live_analysis(
    payload: LiveExpiredAnalysisRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Schedule a new intraday audit decision, NEVER a late broker order."""
    try:
        return queue_expired_live_analysis(
            database(),
            execution_session=payload.execution_session,
            # Password-based admin sessions have no email. Preserve the
            # authenticated actor's stable subject for the Live audit.
            actor_email=(identity.email or "").strip() or identity.subject,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/winner/monitor")
def winner_workflow_monitor(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
    execution_session: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Read-only Live workflow and recovery status for the current Alpaca account."""
    try:
        return live_workflow_monitor(
            database(), session=execution_session, limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc),
        ) from exc


@router.get("/winner/events")
def winner_live_events(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
    execution_session: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    level: Literal["all", "info", "warning", "error"] = "all",
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    """Sanitized, account-scoped lifecycle events; never brokers orders."""
    try:
        return list_live_events(database(), session=execution_session, level=level, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/winner/events/export.zip")
def export_winner_live_events(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
    execution_session: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> Response:
    """Download sanitized Winner Live diagnostics for the selected session."""
    try:
        payload = export_live_events(database(), session=execution_session)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="mct_live_winner_logs_{timestamp}.zip"'},
    )



@router.get("/winner/manual-reconciliation/preview")
def preview_manual_live_rotation(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
    execution_session: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, Any]:
    """Read-only validation against account, broker fills and internal state."""
    return manual_reconciliation_preview(database(), execution_session=execution_session)


class ManualLiveReconciliationRequest(BaseModel):
    execution_session: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confirmation_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    confirm: Literal["RECONCILE_MANUAL_FILLS_NO_ORDERS"]


@router.post("/winner/manual-reconciliation/apply")
def apply_manual_live_rotation(
    payload: ManualLiveReconciliationRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Adopt only fully proved manual Alpaca fills while paused; no orders."""
    try:
        return reconcile_manual_live_rotation(
            database(), execution_session=payload.execution_session,
            confirmation_token=payload.confirmation_token,
            actor=(identity.email or "").strip() or identity.subject,
        )
    except ManualReconciliationBlocked as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/winner/preflight")
def winner_preflight() -> dict[str, Any]:
    """Read-only Live Winner metadata preflight, never an order plan."""
    try:
        return capture_live_winner_preflight(database())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

@router.get("/execution/readiness")
def execution_readiness() -> dict[str, Any]:
    """Operator preflight only. Never activates order execution."""
    return live_execution_readiness(database())


class LiveDryRunRequest(BaseModel):
    target: str = Field(min_length=1, max_length=8, description="Hypothetical CASH or ticker target, NOT a Winner signal.")
    max_allocation_fraction: float = Field(default=0.1, gt=0, le=1)


@router.post("/execution/dry-run")
def execution_dry_run(payload: LiveDryRunRequest) -> dict[str, Any]:
    """Read-only hypothetical scenario, always blocks real order execution."""
    try:
        return preview_live_intent(
            database(),
            target=payload.target,
            max_allocation_fraction=payload.max_allocation_fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/portfolio/export.zip")
def export_portfolio() -> Response:
    try:
        payload = build_live_portfolio_export(database())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": (
            f'attachment; filename="mct_live_transaction_audit_{timestamp}.zip"'
        )},
    )
