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
from ...services.live_order_execution import execute_recalibrated_live_winner_session
from ...services.live_winner_session import LIVE_RUNS
from ...services.live_execution_dry_run import preview_live_intent
from ...services.live_winner_preflight import capture_live_winner_preflight
from ...services.live_expired_recovery import queue_expired_live_analysis
from ...services.live_winner_events import list_live_events, export_live_events
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
    confirm: Literal["LIVE_MONITORING_ONLY"]


@router.post("/robot/mode")
def update_robot_mode(
    payload: LiveControlModeRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Control Live monitoring policy. This endpoint NEVER submits real orders."""
    try:
        return {
            **set_live_control_mode(
                database(),
                mode=payload.mode,
                expected_revision=payload.expected_revision,
                actor_email=identity.email,
                reason=payload.reason,
            ),
            "message": "Live monitoring policy changed; real-order execution remains disabled.",
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


@router.post("/winner/preflight")
def winner_preflight() -> dict[str, Any]:
    """Read-only Live Winner metadata preflight, never an order plan."""
    try:
        return capture_live_winner_preflight(database())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

class LiveExecuteWinnerRequest(BaseModel):
    execution_session: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    expected_target_asset: str = Field(min_length=1, max_length=8)
    expected_winner_configuration_hash: str = Field(min_length=64, max_length=64)
    confirm: Literal["EXECUTE_WINNER_LIVE_REAL_ORDER"]


@router.post("/winner/execute-current-plan")
def execute_live_current_plan(
    payload: LiveExecuteWinnerRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    """Operator-triggered execution of an eligible *current-session* Live Winner plan.

    Reuses the same session, identity, position, risk, order idempotency,
    Live authorization and exchange-window validations as the scheduler.
    Never converts a completed analysis-only session into an executable one.
    """
    db = database()
    run = db[LIVE_RUNS].find_one({"execution_session": payload.execution_session})
    if not run or run.get("status") != "recalibrated_no_orders":
        raise HTTPException(status_code=409, detail="No executable Winner plan for this session.")
    if (str(run.get("target_asset") or "").upper() != payload.expected_target_asset.upper()
        or str(run.get("winner_configuration_hash") or "") != payload.expected_winner_configuration_hash):
        raise HTTPException(status_code=409, detail="Winner plan changed. Refresh and confirm again.")
    try:
        result = execute_recalibrated_live_winner_session(db, payload.execution_session)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Live execution not performed: {result.get('status')}")
    return result


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
