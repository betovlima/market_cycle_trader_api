from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ...auth.security import SessionIdentity, require_capability
from ...core.runtime import database
from ...schemas.temporal_rotation_quality import (
    DEFAULT_CHALLENGER_QUALITY_FLOORS,
    DEFAULT_DRAWDOWN_TRIGGERS,
    DEFAULT_ROTATION_SCORE_TOLERANCES,
    TemporalRotationQualityCaroConfig,
    TemporalRotationQualityResearchGate,
    TemporalRotationQualityResearchRequest,
    TemporalRotationQualityResearchStartRequest,
    TemporalRotationQualityValidationRequest,
    TemporalRotationQualityDiagnosticRequest,
)
from ...services.temporal_rotation_quality_analytics import (
    get_rotation_quality_analytics,
    get_rotation_quality_rotation_period,
    list_rotation_quality_analytics_processings,
)
from ...services.temporal_rotation_quality_diagnostics import (
    DEFAULT_DIAGNOSTIC_FEATURES,
    DIAGNOSTIC_FEATURES,
    build_temporal_rotation_quality_diagnostic_export,
    get_temporal_rotation_quality_diagnostic,
    list_temporal_rotation_quality_diagnostics,
    request_temporal_rotation_quality_diagnostic_stop,
    start_temporal_rotation_quality_diagnostic,
)
from ...services.temporal_rotation_quality import (
    TemporalRotationQualityConflict,
    TemporalRotationQualityNotFound,
    build_temporal_rotation_quality_export,
    build_temporal_rotation_quality_validation_export,
    get_temporal_rotation_quality_candidates,
    get_temporal_rotation_quality_research,
    get_temporal_rotation_quality_validation,
    list_temporal_rotation_quality_research,
    list_temporal_rotation_quality_validations,
    run_temporal_rotation_quality_research,
    start_temporal_rotation_quality_research,
    start_temporal_rotation_quality_validation,
)

router = APIRouter(
    prefix="/api/temporal-rotation-quality-research",
    tags=["temporal-rotation-quality-research"],
)
require_temporal_view = require_capability("temporal_intelligence.view")
require_temporal_export = require_capability("temporal_intelligence.export")
require_research_manage = require_capability("research.manage")


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TemporalRotationQualityNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, TemporalRotationQualityConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post("", status_code=status.HTTP_201_CREATED)
def create_temporal_rotation_quality_research(
    payload: TemporalRotationQualityResearchStartRequest,
    identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    """Backward-compatible default grid research."""
    try:
        return run_temporal_rotation_quality_research(
            database(),
            payload.to_research_request(),
            actor_email=identity.email,
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("/advanced", status_code=status.HTTP_201_CREATED)
def create_temporal_rotation_quality_research_advanced(
    payload: TemporalRotationQualityResearchRequest,
    identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    """Synchronous advanced research retained for API compatibility and small experiments."""
    try:
        return run_temporal_rotation_quality_research(
            database(),
            payload,
            actor_email=identity.email,
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def start_temporal_rotation_quality_research_run(
    payload: TemporalRotationQualityResearchRequest,
    identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    """Start configurable Grid, Manual or Unified Adaptive CARO research for the frontend console."""
    try:
        return start_temporal_rotation_quality_research(
            database(),
            payload,
            actor_email=identity.email,
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/config")
def temporal_rotation_quality_configuration(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    caro = TemporalRotationQualityCaroConfig().model_dump()
    research_gate = TemporalRotationQualityResearchGate().model_dump()
    return {
        "experiment": "drawdown_adaptive_rotation_quality_gate",
        "search_methods": [
            {"id": "caro", "label": "Unified Adaptive CARO"},
            {"id": "grid", "label": "Grid Search"},
            {"id": "manual", "label": "Manual"},
        ],
        "defaults": {
            "search_method": "caro",
            "focus_month": None,
            "control_tolerance_usd": 1.0,
            "strong_challenger_override": {
                "enabled": True,
                "baseline_drawdown_trigger": -0.05,
                "baseline_rotation_score_tolerance": -0.10,
                "challenger_quality_floors": list(DEFAULT_CHALLENGER_QUALITY_FLOORS),
            },
            "grid": {
                "drawdown_triggers": list(DEFAULT_DRAWDOWN_TRIGGERS),
                "rotation_score_tolerances": list(DEFAULT_ROTATION_SCORE_TOLERANCES),
            },
            "caro": caro,
            "research_gate": research_gate,
            "validation": {
                "kind": "validation",
                "fold_count": 5,
                "required_fold_wins": 4,
                "minimum_capital_lift": 0.0,
                "minimum_sharpe_delta": 0.0,
                "minimum_max_drawdown_delta": 0.0,
            },
            "certification": {
                "kind": "certification",
                "fold_count": 7,
                "required_fold_wins": 6,
                "minimum_capital_lift": 0.0,
                "minimum_sharpe_delta": 0.0,
                "minimum_max_drawdown_delta": 0.0,
            },
        },
        "limits": {
            "fold_count_min": 2,
            "fold_count_max": 20,
            "candidate_ids_max": 20,
            "grid_values_max": 64,
            "manual_candidates_max": 2000,
            "caro_trials_min": 4,
            "caro_trials_max": 2000,
        },
        "diagnostics": {
            "defaults": {
                "lookback_sessions": 5,
                "feature_names": list(DEFAULT_DIAGNOSTIC_FEATURES),
                "minimum_group_samples": 3,
                "outcome_neutral_band": 0.0,
                "top_feature_count": 20,
            },
            "features": [{"id": feature, "label": feature.replace("_", " ").title()} for feature in DIAGNOSTIC_FEATURES],
            "limits": {
                "lookback_sessions_min": 1,
                "lookback_sessions_max": 60,
                "minimum_group_samples_min": 2,
                "minimum_group_samples_max": 100,
                "top_feature_count_min": 1,
                "top_feature_count_max": 100,
                "outcome_neutral_band_min": 0.0,
                "outcome_neutral_band_max": 0.20,
            },
        },
        "decision_features": [
            "simulated strategy drawdown before decision",
            "entry_rank_score of simulated incumbent",
            "entry_rank_score of original Temporal target",
            "absolute entry_rank_score of challenger when Strong Challenger Override is enabled",
        ],
        "future_information_used_for_decision": False,
        "control_included": True,
    }


@router.get("/analytics/processings")
def rotation_quality_dashboard_processings(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return list_rotation_quality_analytics_processings(database(), limit=limit)


@router.get("/analytics/processings/{processing_id}")
def rotation_quality_dashboard_analytics(
    processing_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    candidate_id: str = Query(min_length=1, max_length=80),
) -> dict[str, Any]:
    return get_rotation_quality_analytics(database(), processing_id, candidate_id)


@router.get("/analytics/processings/{processing_id}/rotation-period")
def rotation_quality_dashboard_rotation_period(
    processing_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    candidate_id: str = Query(min_length=1, max_length=80),
    year: int = Query(ge=2000, le=2200),
    month: int = Query(ge=1, le=12),
) -> dict[str, Any]:
    return get_rotation_quality_rotation_period(
        database(), processing_id, candidate_id, year=year, month=month
    )


@router.get("")
def temporal_rotation_quality_history(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    source_run_id: str | None = Query(default=None, min_length=1, max_length=160),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    items = list_temporal_rotation_quality_research(
        database(),
        source_run_id=source_run_id,
        limit=limit,
    )
    return {"items": items, "count": len(items)}


@router.post("/{research_id}/validate", status_code=status.HTTP_202_ACCEPTED)
def validate_temporal_rotation_quality_research(
    research_id: str,
    payload: TemporalRotationQualityValidationRequest,
    identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    """Start a validation or certification with frozen Rotation Quality parameters."""
    try:
        return start_temporal_rotation_quality_validation(
            database(),
            research_id,
            payload,
            actor_email=identity.email,
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations")
def temporal_rotation_quality_validations(
    research_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        items = list_temporal_rotation_quality_validations(database(), research_id, limit=limit)
        return {"research_id": research_id, "items": items, "count": len(items)}
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations/{validation_id}")
def temporal_rotation_quality_validation(
    research_id: str,
    validation_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_rotation_quality_validation(database(), research_id, validation_id)
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations/{validation_id}/export.zip")
def export_temporal_rotation_quality_validation(
    research_id: str,
    validation_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_export)],
) -> Response:
    try:
        content = build_temporal_rotation_quality_validation_export(database(), research_id, validation_id)
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict) as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="temporal_rotation_quality_{validation_id}.zip"'
        },
    )


@router.post("/{research_id}/validations/{validation_id}/diagnostics", status_code=status.HTTP_202_ACCEPTED)
def start_rotation_quality_diagnostic(
    research_id: str,
    validation_id: str,
    payload: TemporalRotationQualityDiagnosticRequest,
    identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    try:
        return start_temporal_rotation_quality_diagnostic(
            database(),
            research_id,
            validation_id,
            payload,
            actor_email=identity.email,
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations/{validation_id}/diagnostics")
def rotation_quality_diagnostic_history(
    research_id: str,
    validation_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        items = list_temporal_rotation_quality_diagnostics(
            database(), research_id, validation_id, limit=limit
        )
        return {"research_id": research_id, "validation_id": validation_id, "items": items, "count": len(items)}
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}")
def rotation_quality_diagnostic_detail(
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_rotation_quality_diagnostic(database(), research_id, validation_id, diagnostic_id)
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.post("/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}/stop")
def stop_rotation_quality_diagnostic(
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_research_manage)],
) -> dict[str, Any]:
    try:
        return request_temporal_rotation_quality_diagnostic_stop(
            database(), research_id, validation_id, diagnostic_id
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict) as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}/export.zip")
def export_rotation_quality_diagnostic(
    research_id: str,
    validation_id: str,
    diagnostic_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_export)],
) -> Response:
    try:
        content = build_temporal_rotation_quality_diagnostic_export(
            database(), research_id, validation_id, diagnostic_id
        )
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict) as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="temporal_rotation_quality_diagnostic_{diagnostic_id}.zip"'
        },
    )


@router.get("/{research_id}")
def temporal_rotation_quality_research(
    research_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_rotation_quality_research(database(), research_id)
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/candidates")
def temporal_rotation_quality_candidates(
    research_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=100, ge=1, le=2000),
) -> dict[str, Any]:
    try:
        return get_temporal_rotation_quality_candidates(database(), research_id, limit=limit)
    except TemporalRotationQualityNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{research_id}/export.zip")
def export_temporal_rotation_quality_research(
    research_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_export)],
) -> Response:
    try:
        content = build_temporal_rotation_quality_export(database(), research_id)
    except (TemporalRotationQualityNotFound, TemporalRotationQualityConflict) as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="temporal_rotation_quality_{research_id}.zip"'
        },
    )
