"""Live visibility API. Never submit, modify or cancel brokerage orders."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...services.live_control import LiveControlConflict, set_live_control_mode
from ...services.live_execution_readiness import live_execution_readiness
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


@router.get("/execution/readiness")
def execution_readiness() -> dict[str, Any]:
    """Operator preflight only. Never activates order execution."""
    return live_execution_readiness(database())


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
