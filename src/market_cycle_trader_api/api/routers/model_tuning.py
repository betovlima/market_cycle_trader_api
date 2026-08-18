from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.model_tuning import ModelTuningAdoptRequest, ModelTuningStartRequest
from ...services.model_tuning_validation import (
    ModelTuningValidationConflict,
    ModelTuningValidationNotFound,
    certify_temporal_policy_candidate,
    get_tuning_validation,
    validate_temporal_policy_champion,
)
from ...services.model_tuning import (
    ModelTuningConflict,
    ModelTuningNotFound,
    adopt_model_tuning_candidate,
    build_model_tuning_export,
    get_latest_model_tuning_run,
    get_model_tuning_candidate_log,
    get_model_tuning_campaign_log,
    get_model_tuning_run,
    list_model_tuning_baselines,
    list_model_tuning_history,
    list_model_tuning_sources,
    request_model_tuning_stop,
    start_model_tuning,
    tuning_catalog,
)

router = APIRouter(prefix="/api/admin/model-tuning", tags=["model-tuning"])


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelTuningNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, (ModelTuningConflict, ModelTuningValidationConflict)):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ModelTuningValidationNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.get("/catalog")
def get_tuning_catalog() -> dict[str, Any]:
    return tuning_catalog(database())



@router.get("/latest")
def get_latest_tuning() -> dict[str, Any] | None:
    return get_latest_model_tuning_run(database())


@router.get("/baselines")
def get_tuning_baselines(limit: int = 20) -> dict[str, Any]:
    try:
        items = list_model_tuning_baselines(database(), limit=limit)
        return {"items": items, "count": len(items)}
    except (ModelTuningConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc




@router.get("/history")
def get_tuning_history(limit: int = 100) -> dict[str, Any]:
    try:
        items = list_model_tuning_history(database(), limit=limit)
        return {"items": items, "count": len(items)}
    except (ModelTuningConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/sources")
def get_tuning_sources(limit: int = 20) -> dict[str, Any]:
    try:
        items = list_model_tuning_sources(database(), limit=limit)
        return {"items": items, "count": len(items)}
    except (ModelTuningConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("", status_code=202)
def create_tuning(
    payload: ModelTuningStartRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return start_model_tuning(
            database(),
            method=payload.method,
            candidate_count=payload.candidate_count,
            caro_candidate_count=payload.caro_candidate_count,
            seed=payload.seed,
            baseline_job_id=payload.baseline_job_id,
            source_tuning_run_id=payload.source_tuning_run_id,
            anchor_candidate_id=payload.anchor_candidate_id,
            tuning_target=payload.tuning_target,
            probability_config=payload.probability.model_dump(mode="python") if payload.probability is not None else None,
            fold_protocol=payload.fold_protocol.model_dump(mode="python") if payload.fold_protocol is not None else None,
            explicit_start_confirmation=payload.explicit_start_confirmation,
            actor_email=identity.email,
        )
    except (ModelTuningConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}")
def get_tuning(run_id: str) -> dict[str, Any]:
    try:
        return get_model_tuning_run(database(), run_id)
    except ModelTuningNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/log")
def get_tuning_log(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return get_model_tuning_campaign_log(database(), run_id)
    except ModelTuningNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/candidates/{candidate_id}/log")
def get_tuning_candidate_log(
    run_id: str,
    candidate_id: int,
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return get_model_tuning_candidate_log(database(), run_id, candidate_id)
    except ModelTuningNotFound as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/stop")
def stop_tuning(run_id: str) -> dict[str, Any]:
    try:
        return request_model_tuning_stop(database(), run_id)
    except (ModelTuningNotFound, ModelTuningConflict) as exc:
        raise _translate_error(exc) from exc




@router.get("/{run_id}/export.zip")
def export_tuning(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> Response:
    try:
        content = build_model_tuning_export(database(), run_id)
    except ModelTuningNotFound as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="model_tuning_{run_id}.zip"'},
    )


@router.post("/{run_id}/candidates/{candidate_id}/adopt")
def adopt_tuning_candidate(
    run_id: str,
    candidate_id: int,
    payload: ModelTuningAdoptRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return adopt_model_tuning_candidate(
            database(),
            run_id,
            candidate_id,
            reason=payload.reason,
            actor_email=identity.email,
        )
    except (ModelTuningNotFound, ModelTuningConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/candidates/{candidate_id}/validate-champion")
def validate_tuning_champion(
    run_id: str,
    candidate_id: int,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return validate_temporal_policy_champion(
            database(),
            run_id,
            candidate_id,
            actor_email=identity.email,
        )
    except (ModelTuningValidationNotFound, ModelTuningValidationConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/candidates/{candidate_id}/certify")
def certify_tuning_candidate(
    run_id: str,
    candidate_id: int,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return certify_temporal_policy_candidate(
            database(),
            run_id,
            candidate_id,
            actor_email=identity.email,
        )
    except (ModelTuningValidationNotFound, ModelTuningValidationConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/candidates/{candidate_id}/validation")
def get_candidate_validation(
    run_id: str,
    candidate_id: int,
    _identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any] | None:
    try:
        return get_tuning_validation(database(), run_id, candidate_id)
    except (ModelTuningValidationNotFound, ModelTuningValidationConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc
