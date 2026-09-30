"""Swagger /docs endpoints for refreshed Alpaca Control shadow jobs.

All routes are admin-only via main.py's router dependency. This preview runs
using newly downloaded RAW/SIP files in MCT dados and cannot submit orders.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ...core.runtime import database
from ...services.control_shadow_deep_rank_jobs import (
    DeepRankConflict, DeepRankInvalid, DeepRankNotFound,
    start_deep_ranking_research, get_deep_ranking_research,
)
from ...services.control_shadow_counterfactual_advantage_jobs import (
    AdvantageConflict, AdvantageInvalid, AdvantageNotFound,
    start_counterfactual_advantage_research,
    get_counterfactual_advantage_research,
)
from ...services.control_shadow_policy_rollout_jobs import (
    RolloutConflict, RolloutInvalid, RolloutNotFound,
    start_policy_rollout_advantage_research,
    get_policy_rollout_advantage_research,
)
from ...services.control_shadow_rollout_signature_jobs import (
    SignatureConflict, SignatureInvalid, SignatureNotFound,
    start_rollout_signature_research,
    get_rollout_signature_research,
)
from ...services.control_shadow_reduced_signature_jobs import (
    ReducedSignatureConflict, ReducedSignatureInvalid, ReducedSignatureNotFound,
    start_reduced_rollout_signature_research,
    get_reduced_rollout_signature_research,
)
from ...services.control_shadow_reduced_meta_veto_jobs import (
    MetaVetoConflict, MetaVetoInvalid, MetaVetoNotFound,
    start_reduced_signature_meta_veto_research,
    get_reduced_signature_meta_veto_research,
)
from ...services.control_shadow_tcn_jobs import (
    TCNConflict, TCNInvalid, TCNNotFound,
    start_tcn_research, get_tcn_research,
)
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


class StartControlTCNResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_FIXED_CAUSAL_TCN_NO_ORDERS"] = Field(
        description="Fixed TinyTCN with mature labels and identical liquidity-cap accounting. No tuning or real orders."
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
    )
    source_liquidity_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-liquidity-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/deep-learning/jobs", status_code=202,
    summary="Exploratory causal TCN, three purged OOS folds, liquidity execution — NO orders",
    description=(
        "Trains a fixed pooled CPU TinyTCN on the verified snapshot. "
        "Selects epochs using pre-test calibration only, keeps 60-session "
        "label maturity and identical 10% execution constraints. Historical "
        "Control/Liquidity baselines are read-only references; no Alpaca or orders."
    ),
)
def start_deep_learning_job(payload: StartControlTCNResearchRequest) -> dict[str, Any]:
    try:
        return start_tcn_research(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            source_liquidity_job_id=payload.source_liquidity_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TCNNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TCNConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TCNInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/deep-learning/jobs/{job_id}",
    summary="Read fixed TinyTCN historical model and portfolio diagnostics",
)
def read_deep_learning_job(job_id: str) -> dict[str, Any]:
    try:
        return get_tcn_research(database(), job_id)
    except TCNNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/deep-learning/jobs/{job_id}/logs",
    summary="Read fixed TinyTCN progress and logs",
)
def read_deep_learning_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_tcn_research(database(), job_id, logs_only=True)
    except TCNNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class StartControlDeepRankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_PAIRWISE_DEEP_RANK_NO_ORDERS"] = Field(
        description=(
            "Train a fixed temporal pairwise ranker. Original LightGBM owns "
            "absolute utility thresholds and the existing liquidity overlay "
            "and execution accounting remain unchanged."
        )
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
    )
    source_liquidity_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-liquidity-[a-f0-9]{16}$",
    )
    source_tcn_job_id: str = Field(
        min_length=28, max_length=28,
        pattern=r"^control-tcn-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/deep-ranking/jobs",
    status_code=202,
    summary="Pairwise Deep Ranking + original Control utility — research only",
    description=(
        "Learns within-date candidate ordering on fully matured past labels. "
        "Deep Learning controls ranking only; original fold-specific LightGBM "
        "controls absolute utility/CASH/switch thresholds and the v10.8.44 "
        "liquidity overlay plus v10.8.42 execution model stay fixed. "
        "No Alpaca refresh, production strategy update or orders."
    ),
)
def start_deep_ranking_job(
    payload: StartControlDeepRankingRequest,
) -> dict[str, Any]:
    try:
        return start_deep_ranking_research(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            source_liquidity_job_id=payload.source_liquidity_job_id,
            source_tcn_job_id=payload.source_tcn_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DeepRankNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DeepRankConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeepRankInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/deep-ranking/jobs/{job_id}",
    summary="Read pairwise Deep Ranking historical diagnostics and portfolio",
)
def read_deep_ranking_job(job_id: str) -> dict[str, Any]:
    try:
        return get_deep_ranking_research(database(), job_id)
    except DeepRankNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/deep-ranking/jobs/{job_id}/logs",
    summary="Read pairwise Deep Ranking training and replay progress",
)
def read_deep_ranking_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_deep_ranking_research(
            database(), job_id, logs_only=True,
        )
    except DeepRankNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class StartControlCounterfactualAdvantageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_COUNTERFACTUAL_ADVANTAGE_NO_ORDERS"] = Field(
        description=(
            "Run the fixed v10.8.47 Control meta-veto research. Control remains "
            "the default action; the model can only veto asset-to-asset rotations."
        )
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
    )
    source_liquidity_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-liquidity-[a-f0-9]{16}$",
    )
    source_tcn_job_id: str = Field(
        min_length=28, max_length=28,
        pattern=r"^control-tcn-[a-f0-9]{16}$",
    )
    source_ranking_job_id: str = Field(
        min_length=29, max_length=29,
        pattern=r"^control-rank-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/counterfactual-advantage/jobs",
    status_code=202,
    summary="Control counterfactual advantage meta-veto — research only",
    description=(
        "Reuses the exact SHA-verified v10.8.41/42/44/45/46 chain. The same "
        "LightGBM/Liquidity-Aware policies are replayed first as the v10.8.44 "
        "baseline and then with a fixed candidate-vs-incumbent meta-veto. "
        "The job aborts if v10.8.44 ending capital is not reproduced exactly. "
        "No Alpaca refresh, strategy registration, Winner promotion or orders."
    ),
)
def start_counterfactual_advantage_job(
    payload: StartControlCounterfactualAdvantageRequest,
) -> dict[str, Any]:
    try:
        return start_counterfactual_advantage_research(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            source_liquidity_job_id=payload.source_liquidity_job_id,
            source_tcn_job_id=payload.source_tcn_job_id,
            source_ranking_job_id=payload.source_ranking_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AdvantageNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AdvantageConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AdvantageInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/counterfactual-advantage/jobs/{job_id}",
    summary="Read v10.8.47 meta-veto diagnostics and paired portfolio results",
)
def read_counterfactual_advantage_job(job_id: str) -> dict[str, Any]:
    try:
        return get_counterfactual_advantage_research(database(), job_id)
    except AdvantageNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/counterfactual-advantage/jobs/{job_id}/logs",
    summary="Read v10.8.47 counterfactual advantage progress and logs",
)
def read_counterfactual_advantage_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_counterfactual_advantage_research(
            database(), job_id, logs_only=True,
        )
    except AdvantageNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class StartControlPolicyRolloutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_POLICY_ROLLOUT_ADVANTAGE_NO_ORDERS"] = Field(
        description=(
            "Run fixed v10.8.48 paired execution-aware rollouts. The Control "
            "remains the default action and the model may only veto an "
            "asset-to-asset rotation after pre-OOS calibration skill."
        )
    )
    source_validation_job_id: str = Field(
        min_length=35, max_length=35,
        pattern=r"^control-validation-[a-f0-9]{16}$",
    )
    source_execution_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-execution-[a-f0-9]{16}$",
    )
    source_liquidity_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-liquidity-[a-f0-9]{16}$",
    )
    source_tcn_job_id: str = Field(
        min_length=28, max_length=28,
        pattern=r"^control-tcn-[a-f0-9]{16}$",
    )
    source_ranking_job_id: str = Field(
        min_length=29, max_length=29,
        pattern=r"^control-rank-[a-f0-9]{16}$",
    )
    source_advantage_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-advantage-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/policy-rollout/jobs",
    status_code=202,
    summary="Execution-aware Control policy-rollout advantage — research only",
    description=(
        "Builds exact 20-session paired rollouts from the same audited "
        "historical portfolio state: execute the Control rotation once versus "
        "HOLD once, then return both branches to the same Liquidity-Aware "
        "Control policy. Fold 1 cannot use a meta-model; folds 2/3 may use only "
        "fully matured prior-fold OOS labels. Aborts unless v10.8.44 is "
        "reproduced exactly. No Alpaca refresh, tuning, Winner promotion or orders."
    ),
)
def start_policy_rollout_job(
    payload: StartControlPolicyRolloutRequest,
) -> dict[str, Any]:
    try:
        return start_policy_rollout_advantage_research(
            database(),
            source_validation_job_id=payload.source_validation_job_id,
            source_execution_job_id=payload.source_execution_job_id,
            source_liquidity_job_id=payload.source_liquidity_job_id,
            source_tcn_job_id=payload.source_tcn_job_id,
            source_ranking_job_id=payload.source_ranking_job_id,
            source_advantage_job_id=payload.source_advantage_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RolloutNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RolloutConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RolloutInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/policy-rollout/jobs/{job_id}",
    summary="Read v10.8.48 rollout labels, calibration and portfolio result",
)
def read_policy_rollout_job(job_id: str) -> dict[str, Any]:
    try:
        return get_policy_rollout_advantage_research(database(), job_id)
    except RolloutNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/policy-rollout/jobs/{job_id}/logs",
    summary="Read v10.8.48 policy-rollout research progress and logs",
)
def read_policy_rollout_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_policy_rollout_advantage_research(
            database(), job_id, logs_only=True,
        )
    except RolloutNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class StartControlRolloutSignatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_ROLLOUT_DECISION_SIGNATURE_NO_ORDERS"] = Field(
        description=(
            "Analyze the fixed v10.8.48 paired rollout events using causal "
            "decision-time features and small fixed models. This endpoint "
            "creates no trading policy."
        )
    )
    source_rollout_job_id: str = Field(
        min_length=32, max_length=32,
        pattern=r"^control-rollout-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/rollout-signature/jobs",
    status_code=202,
    summary="Diagnose predictive signature of v10.8.48 rollout decisions",
    description=(
        "Builds a 321-event causal decision dataset from the verified v10.8.48 "
        "paired rollouts, computes feature diagnostics, and evaluates fixed "
        "Logistic Regression, shallow Decision Tree, small Random Forest and "
        "small LightGBM with chronological fold transfer. No policy creation, "
        "hyperparameter search, Alpaca refresh, Winner change or orders."
    ),
)
def start_rollout_signature_job(
    payload: StartControlRolloutSignatureRequest,
) -> dict[str, Any]:
    try:
        return start_rollout_signature_research(
            database(),
            source_rollout_job_id=payload.source_rollout_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SignatureConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SignatureInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/rollout-signature/jobs/{job_id}",
    summary="Read v10.8.49 diagnostic results",
)
def read_rollout_signature_job(job_id: str) -> dict[str, Any]:
    try:
        return get_rollout_signature_research(database(), job_id)
    except SignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/rollout-signature/jobs/{job_id}/logs",
    summary="Read v10.8.49 diagnostic progress and logs",
)
def read_rollout_signature_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_rollout_signature_research(
            database(), job_id, logs_only=True,
        )
    except SignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class StartControlReducedSignatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_REDUCED_ROLLOUT_SIGNATURE_NO_ORDERS"] = Field(
        description=(
            "Confirm the frozen 9-feature v10.8.50 diagnostic. This endpoint "
            "does not create or promote a trading policy."
        )
    )
    source_signature_job_id: str = Field(
        min_length=34, max_length=34,
        pattern=r"^control-signature-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/reduced-signature/jobs",
    status_code=202,
    summary="Confirm the reduced 9-feature rollout signature — diagnostic only",
    description=(
        "Consumes the verified signal-positive v10.8.49 dataset and evaluates "
        "two fixed reduced Logistic Regression models plus nine explanatory "
        "univariate models. Confirmation requires balanced accuracy > 0.50 "
        "and ROC AUC > 0.50 on both chronological transfer tests. No tuning, "
        "policy creation, Alpaca refresh, Winner change or orders."
    ),
)
def start_reduced_signature_job(
    payload: StartControlReducedSignatureRequest,
) -> dict[str, Any]:
    try:
        return start_reduced_rollout_signature_research(
            database(),
            source_signature_job_id=payload.source_signature_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ReducedSignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReducedSignatureConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ReducedSignatureInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/reduced-signature/jobs/{job_id}",
    summary="Read v10.8.50 reduced signature confirmation results",
)
def read_reduced_signature_job(job_id: str) -> dict[str, Any]:
    try:
        return get_reduced_rollout_signature_research(database(), job_id)
    except ReducedSignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/reduced-signature/jobs/{job_id}/logs",
    summary="Read v10.8.50 reduced signature progress and logs",
)
def read_reduced_signature_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_reduced_rollout_signature_research(
            database(), job_id, logs_only=True,
        )
    except ReducedSignatureNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class StartControlReducedMetaVetoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: Literal["RESEARCH_CONTROL_REDUCED_META_VETO_NO_ORDERS"] = Field(
        description=(
            "Run the frozen v10.8.51 reduced-signature fail-safe meta-veto. "
            "Research only; never creates or submits orders."
        )
    )
    source_reduced_job_id: str = Field(
        min_length=32, max_length=32,
        pattern=r"^control-reduced-[a-f0-9]{16}$",
    )
    expected_snapshot_sha256: str = Field(
        min_length=64, max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


@router.post(
    "/reduced-meta-veto/jobs",
    status_code=202,
    summary="Run v10.8.51 reduced-signature Meta-Veto — research only",
    description=(
        "Uses the confirmed v10.8.50 9-feature signature with fixed balanced "
        "Logistic Regression C=0.25. Fold 1 is Control-only; Fold 2 learns "
        "only Fold 1; Fold 3 learns only Folds 1+2. A fold is enabled only "
        "after chronological calibration BA>=0.52 and AUC>=0.52, and a "
        "rotation is vetoed only when P(ROTATE better)<=0.35. CASH transitions "
        "remain unchanged. No tuning, Alpaca refresh, Winner change or orders."
    ),
)
def start_reduced_meta_veto_job(
    payload: StartControlReducedMetaVetoRequest,
) -> dict[str, Any]:
    try:
        return start_reduced_signature_meta_veto_research(
            database(),
            source_reduced_job_id=payload.source_reduced_job_id,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except ControlShadowUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MetaVetoNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MetaVetoConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MetaVetoInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/reduced-meta-veto/jobs/{job_id}",
    summary="Read v10.8.51 reduced-signature Meta-Veto result",
)
def read_reduced_meta_veto_job(job_id: str) -> dict[str, Any]:
    try:
        return get_reduced_signature_meta_veto_research(database(), job_id)
    except MetaVetoNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/reduced-meta-veto/jobs/{job_id}/logs",
    summary="Read v10.8.51 reduced-signature Meta-Veto progress and logs",
)
def read_reduced_meta_veto_logs(job_id: str) -> dict[str, Any]:
    try:
        return get_reduced_signature_meta_veto_research(
            database(), job_id, logs_only=True,
        )
    except MetaVetoNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

