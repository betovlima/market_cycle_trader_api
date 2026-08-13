from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.model_research import (
    ResearchModelFamily,
    ModelResearchJobRequest,
    ModelResearchSettingsUpdateRequest,
)
from ...services.model_research import (
    ModelResearchSettingsConflict,
    list_model_research_executions,
    list_model_research_settings_history,
    public_model_research_catalog,
    public_model_research_settings,
    update_model_research_settings,
)
from ...services.strategy_lab import get_research_strategy_model_snapshot
from .jobs import queue_backtest_job

router = APIRouter(prefix="/api/admin/model-research", tags=["model-research"])


@router.get("")
def get_model_research_catalog() -> dict[str, Any]:
    return public_model_research_catalog(database())


@router.get("/settings")
def get_model_research_configuration() -> dict[str, Any]:
    return public_model_research_settings(database())


@router.patch("/settings/{model_family}")
def patch_model_research_configuration(
    model_family: ResearchModelFamily,
    payload: ModelResearchSettingsUpdateRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return update_model_research_settings(
            database(),
            model_family,
            payload,
            actor_email=identity.email,
        )
    except ModelResearchSettingsConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/settings/history")
def get_model_research_configuration_history(
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    return list_model_research_settings_history(database(), limit=limit)


@router.get("/executions")
def get_model_research_executions(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return list_model_research_executions(database(), limit=limit)


@router.post("/jobs", status_code=202)
def create_model_research_job(request: ModelResearchJobRequest) -> dict[str, Any]:
    
    
    db = database()
    bound = get_research_strategy_model_snapshot(db)
    if request.model_family != bound["family"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Backtest model is owned by the selected Strategy. Save the model on that Strategy first.",
        )
    return queue_backtest_job()
