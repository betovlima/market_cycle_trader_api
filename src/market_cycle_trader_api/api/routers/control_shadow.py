"""Swagger /docs endpoints for frozen TCC Control shadow jobs.

All routes are admin-only via main.py's router dependency. This preview runs
with verified frozen CSV input on the API server, and cannot submit orders.
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

    confirm: Literal["RUN_FROZEN_SHADOW_NO_ORDERS"] = Field(
        description="Explicitly confirm frozen-input preview without Alpaca orders."
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
    summary="Start frozen Control shadow job (no orders)",
    description=(
        "Run the TCC v1.0.6 Control training/calibration and calculate one "
        "hypothetical decision. The API reads only a server-configured, "
        "SHA-verified frozen snapshot. Its progress is printed in the Uvicorn "
        "console and saved for polling. No HTTP input can select a filesystem "
        "path, an Alpaca account or a production Strategy."
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
