from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth.security import SessionIdentity, require_capability
from ..core.runtime import database
from ..services.temporal_intelligence import (
    TemporalIntelligenceConflict,
    TemporalIntelligenceNotFound,
    validate_temporal_research_processing,
)
from .errors import MilpDecisionError
from .materialization import materialize
from .persistence import public_document
from .schemas import MilpDecisionRequest
from .service import latest, run

router = APIRouter(prefix="/api/temporal-intelligence", tags=["milp-decision-optimization"])
require_view = require_capability("temporal_intelligence.view")
require_start = require_capability("temporal_intelligence.start")
require_materialize = require_capability("temporal_intelligence.materialize_strategy")


def _unprocessable(exc: Exception) -> HTTPException:
    if isinstance(exc, TemporalIntelligenceNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, TemporalIntelligenceConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


@router.get("/{run_id}/decision-optimization/latest")
def latest_decision_optimization(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_view)],
    processing_id: str = Query(..., min_length=1),
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, processing_id)
        return latest(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound, MilpDecisionError, ValueError) as exc:
        raise _unprocessable(exc) from exc


@router.post("/{run_id}/decision-optimization")
def decision_optimization(
    run_id: str,
    payload: MilpDecisionRequest,
    _identity: Annotated[SessionIdentity, Depends(require_start)],
) -> dict[str, Any]:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, payload.processing_id)
        return public_document(run(
            db,
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
        )) or {}
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound, MilpDecisionError, ValueError, RuntimeError) as exc:
        raise _unprocessable(exc) from exc


@router.post("/{run_id}/decision-optimization/{optimization_id}/strategy", status_code=201)
def create_strategy_from_decision_optimization(
    run_id: str,
    optimization_id: str,
    identity: Annotated[SessionIdentity, Depends(require_materialize)],
) -> dict[str, Any]:
    try:
        return materialize(database(), run_id, optimization_id, actor_email=identity.email)
    except (MilpDecisionError, ValueError, RuntimeError) as exc:
        raise _unprocessable(exc) from exc
