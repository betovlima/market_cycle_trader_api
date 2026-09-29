"""Swagger /docs endpoints for refreshed Alpaca Control shadow jobs.

All routes are admin-only via main.py's router dependency. This preview runs
using newly downloaded RAW/SIP files in MCT dados and cannot submit orders.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ...core.runtime import database
from ...services.control_shadow_jobs import (
    ControlShadowConflict,
    ControlShadowNotFound,
    ControlShadowUnavailable,
    get_control_shadow_job,
    get_control_shadow_logs,
    start_control_shadow_job,
)


router = APIRouter(
    prefix="/api/admin/control-shadow",
    tags=["Control Shadow — no orders"],
)


class StartControlShadowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["REFRESH_ALPACA_CONTROL_SHADOW_NO_ORDERS"] = Field(
        description="Explicitly confirm fresh Alpaca data download for a no-order preview."
    )
    current_asset: str = Field(
        default="CASH",
        min_length=1,
        max_length=10,
        description="Hypothetical starting state, not read from the real Trader portfolio.",
    )
    holding_sessions: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="Completed sessions held in the hypothetical position.",
    )


@router.post(
    "/jobs",
    status_code=202,
    summary="Download fresh Alpaca data and run Control shadow (no orders)",
    description=(
        "Download NEW Alpaca RAW/SIP daily bars and corporate actions for the "
        "Control v1.0.6 universe; create MCT API dados/control_shadow/snapshots "
        "automatically and calculate one hypothetical decision. Progress is "
        "printed in the PyCharm/Uvicorn console and saved for polling. HTTP "
        "input cannot select a filesystem path, account or production Strategy. "
        "This endpoint never submits trading orders."
    ),
)
def start_shadow_job(payload: StartControlShadowRequest) -> dict[str, Any]:
    try:
        return start_control_shadow_job(
            database(),
            current_asset=payload.current_asset,
            holding_sessions=payload.holding_sessions,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ControlShadowConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}",
    summary="Read Control shadow status and final decision",
)
def read_shadow_job(job_id: str) -> dict[str, Any]:
    try:
        return get_control_shadow_job(database(), job_id)
    except ControlShadowNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}/logs",
    summary="Read Control shadow progress and console log messages",
)
def read_shadow_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_control_shadow_logs(database(), job_id)
    except ControlShadowNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
