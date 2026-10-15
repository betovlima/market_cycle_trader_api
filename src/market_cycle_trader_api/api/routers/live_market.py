"""Live visibility API. Never submit, modify or cancel brokerage orders."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from ...core.runtime import database
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
