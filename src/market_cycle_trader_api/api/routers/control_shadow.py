"""Swagger /docs endpoints for refreshed Alpaca Control shadow jobs.

All routes are admin-only via main.py's router dependency. This preview runs
using newly downloaded RAW/SIP files in MCT dados and cannot submit orders.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ...core.runtime import database
from ...services.control_shadow_liquidity_jobs import (
    LiquidityConflict,
    LiquidityInvalid,
    LiquidityNotFound,
    start_liquidity_research,
    get_liquidity_research,
)
from ...services.control_shadow_sensitivity_jobs import (
    SensitivityConflict,
    SensitivityInvalid,
    SensitivityNotFound,
    start_control_sensitivity,
    get_control_sensitivity,
)
from ...services.control_shadow_execution_jobs import (
    ExecutionConflict,
    ExecutionInvalid,
    ExecutionNotFound,
    start_execution_feasibility,
    get_execution_feasibility,
)
from ...services.control_shadow_validation_jobs import (
    SnapshotValidationConflict,
    SnapshotValidationInvalid,
    SnapshotValidationNotFound,
    get_snapshot_validation,
    start_snapshot_validation,
)
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


class StartControlValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["VALIDATE_EXISTING_CONTROL_SNAPSHOT_NO_ORDERS"] = Field(
        description="Audit a previously completed MCT snapshot and replay Control; no new download and no orders."
    )
    source_job_id: str = Field(
        min_length=31, max_length=31,
        pattern=r"^control-shadow-[a-f0-9]{16}$",
        description="The existing Control Shadow job ID, never a file path."
    )
    expected_snapshot_sha256: str | None = Field(
        default=None,
        min_length=64, max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
        description=(
            "Original snapshot_sha256 from the completed Shadow result. "
            "Required when that source job is absent from the current MongoDB; "
            "also cross-checked when MongoDB contains the original. "
            "No disk path is accepted."
        ),
    )


@router.post(
    "/validation/jobs",
    status_code=202,
    summary="Validate an existing Control snapshot (no download or orders)",
    description=(
        "Verify every source SHA-256; reproduce all four calibration margins "
        "and execute the official expanding walk-forward Control OOS replay. "
        "If the original MongoDB job is missing, supply the independent "
        "snapshot SHA-256 recorded in its completed result: the API can verify "
        "the existing MCT local files directly, without redownloading. "
        "The API writes CSV/JSON reports beside the original MCT snapshot. "
        "Progress appears in PyCharm/Uvicorn and in the validation job logs. "
        "This is research-only and cannot create Alpaca orders or change Winner."
    ),
)
def start_validation_job(payload: StartControlValidationRequest) -> dict[str, Any]:
    try:
        return start_snapshot_validation(
            database(),
            source_job_id=payload.source_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SnapshotValidationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SnapshotValidationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SnapshotValidationInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/validation/jobs/{job_id}",
    summary="Read validation status, margin diagnostics and OOS metrics",
)
def read_validation_job(job_id: str) -> dict[str, Any]:
    try:
        return get_snapshot_validation(database(), job_id)
    except SnapshotValidationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/validation/jobs/{job_id}/logs",
    summary="Read validation progress and console logs",
)
def read_validation_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_snapshot_validation(database(), job_id, logs_only=True)
    except SnapshotValidationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class StartControlExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["SIMULATE_CONTROL_EXECUTION_FEASIBILITY_NO_ORDERS"] = Field(
        description="Research-only execution model; does not submit orders or redownload Alpaca."
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
        description="Completed v10.8.41 numerically verified Control validation job ID.",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        description="Original immutable snapshot SHA-256 as independent confirmation.",
    )


@router.post(
    "/execution/jobs",
    status_code=202,
    summary="OOS Control with volume-constrained modeled partial fills — NO orders",
    description=(
        "Research-only rerun of the same chronological LightGBM Control with a "
        "state-aware, single-position cash portfolio and a fixed conservative "
        "execution-cost/liquidity scenario. Historical next-session volume caps "
        "fills ex post but never selects a signal. Requires a completed and "
        "reproduced v10.8.41 source validation. No Alpaca refresh or orders."
    ),
)
def start_execution_job(payload: StartControlExecutionRequest) -> dict[str, Any]:
    try:
        return start_execution_feasibility(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExecutionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExecutionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExecutionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/execution/jobs/{job_id}",
    summary="Control feasibility research report and status (no orders)",
)
def read_execution_job(job_id: str) -> dict[str, Any]:
    try:
        return get_execution_feasibility(database(), job_id)
    except ExecutionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/execution/jobs/{job_id}/logs",
    summary="Control feasibility research progress and console logs",
)
def read_execution_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_execution_feasibility(database(), job_id, logs_only=True)
    except ExecutionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class StartControlSensitivityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["COMPARE_FIXED_CONTROL_EXECUTION_SCENARIOS_NO_ORDERS"] = Field(
        description="Run five predeclared research scenarios; no tuning, download or trading."
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
        description="Numerically reproduced Control v10.8.41 reference.",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
        description="Matching completed 10% Control v10.8.42 execution reference.",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        description="Immutable source snapshot SHA-256 confirmation.",
    )


@router.post(
    "/sensitivity/jobs",
    status_code=202,
    summary="Fixed Control execution sensitivity (1%, 5%, 10%, no-cap, costs) — NO orders",
    description=(
        "Reuses the same SHA-verified RAW/SIP Control snapshot and single "
        "chronological LightGBM fit for five PREDECLARED state-aware research "
        "scenarios. Requires completed matching v10.8.41 and v10.8.42 jobs. "
        "No download, auto tuning, orders or Winner promotion."
    ),
)
def start_sensitivity_job(payload: StartControlSensitivityRequest) -> dict[str, Any]:
    try:
        return start_control_sensitivity(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SensitivityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SensitivityConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SensitivityInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/sensitivity/jobs/{job_id}",
    summary="Fixed Control sensitivity scenario results (no orders)",
)
def read_sensitivity_job(job_id: str) -> dict[str, Any]:
    try:
        return get_control_sensitivity(database(), job_id)
    except SensitivityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/sensitivity/jobs/{job_id}/logs",
    summary="Fixed Control sensitivity progress and diagnostic logs",
)
def read_sensitivity_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_control_sensitivity(database(), job_id, logs_only=True)
    except SensitivityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class StartControlLiquidityResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_PRIOR_CLOSE_LIQUIDITY_NO_ORDERS"] = Field(
        description="Compare two predeclared Control policies with identical capacity-constrained research accounting."
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
    )
    source_sensitivity_job_id: str = Field(
        min_length=36, max_length=36,
        pattern=r"^control-sensitivity-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/liquidity/jobs",
    status_code=202,
    summary="Paired original Control vs prior-close liquidity-aware Control — research only",
    description=(
        "One unchanged LightGBM Control training/calibration; two independent "
        "10% capacity-constrained OOS accountings. Uses only historical "
        "volume and completed close for candidate utility adjustment. "
        "No Alpaca calls, order placement, or Winner promotion."
    ),
)
def start_liquidity_job(payload: StartControlLiquidityResearchRequest) -> dict[str, Any]:
    try:
        return start_liquidity_research(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            source_sensitivity_job_id=payload.source_sensitivity_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LiquidityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LiquidityConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LiquidityInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/liquidity/jobs/{job_id}",
    summary="Read paired Control liquidity-policy research results (no orders)",
)
def read_liquidity_job(job_id: str) -> dict[str, Any]:
    try:
        return get_liquidity_research(database(), job_id)
    except LiquidityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/liquidity/jobs/{job_id}/logs",
    summary="Read prior-close liquidity-policy research progress",
)
def read_liquidity_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_liquidity_research(database(), job_id, logs_only=True)
    except LiquidityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
