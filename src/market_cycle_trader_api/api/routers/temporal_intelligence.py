from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ...auth.security import SessionIdentity, require_capability
from ...core.runtime import database
from ...services.temporal_intelligence import (
    TemporalIntelligenceConflict,
    TemporalIntelligenceNotFound,
    build_temporal_intelligence_export,
    get_latest_temporal_intelligence_run,
    get_temporal_intelligence_run,
    list_temporal_intelligence_history,
    materialize_temporal_intelligence_strategy,
    start_temporal_intelligence,
    stop_temporal_intelligence,
)

router = APIRouter(prefix="/api/temporal-intelligence", tags=["temporal-intelligence"])
require_temporal_view = require_capability("temporal_intelligence.view")
require_temporal_start = require_capability("temporal_intelligence.start")
require_temporal_stop = require_capability("temporal_intelligence.stop")
require_temporal_export = require_capability("temporal_intelligence.export")
require_temporal_materialize_strategy = require_capability("temporal_intelligence.materialize_strategy")


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TemporalIntelligenceNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, TemporalIntelligenceConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.get("/latest")
def latest_temporal_intelligence(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any] | None:
    return get_latest_temporal_intelligence_run(database())


@router.get("/history")
def temporal_intelligence_history(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    items = list_temporal_intelligence_history(database(), limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/{run_id}")
def temporal_intelligence_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_intelligence_run(database(), run_id)
    except TemporalIntelligenceNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/export.zip")
def export_temporal_intelligence(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_export)],
) -> Response:
    try:
        content = build_temporal_intelligence_export(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="temporal_intelligence_{run_id}.zip"'},
    )


@router.post("/{run_id}/strategy", status_code=201)
def create_strategy_from_temporal_intelligence(
    run_id: str,
    identity: Annotated[SessionIdentity, Depends(require_temporal_materialize_strategy)],
) -> dict[str, Any]:
    try:
        return materialize_temporal_intelligence_strategy(database(), run_id, actor_email=identity.email)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("", status_code=202)
def create_temporal_intelligence(
    identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return start_temporal_intelligence(database(), actor_email=identity.email)
    except (TemporalIntelligenceConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/stop")
def stop_temporal_intelligence_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_stop)],
) -> dict[str, Any]:
    try:
        return stop_temporal_intelligence(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc
