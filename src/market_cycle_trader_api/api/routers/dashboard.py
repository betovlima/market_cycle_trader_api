from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from ...auth.security import SessionIdentity, require_portfolio_session
from ...core.runtime import database
from ...services.dashboard import (
    dashboard_job_detail,
    dashboard_strategy_intelligence,
    dashboard_summary,
    dashboard_tuning_candidate_detail,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_dashboard_summary(
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, Any]:
    

    return dashboard_summary(database(), limit=limit)


@router.get("/jobs/{job_id}")
def get_dashboard_job(job_id: str) -> dict[str, Any]:
    

    return dashboard_job_detail(database(), job_id)


@router.get("/strategy-intelligence")
def get_dashboard_strategy_intelligence(
    _identity: Annotated[SessionIdentity, Depends(require_portfolio_session)],
    job_id: str | None = None,
) -> dict[str, Any]:
    

    return dashboard_strategy_intelligence(database(), job_id=job_id)


@router.get("/strategy-intelligence/tuning/{run_id}/candidates/{candidate_id}")
def get_dashboard_tuning_candidate_detail(
    run_id: str,
    candidate_id: int,
    _identity: Annotated[SessionIdentity, Depends(require_portfolio_session)],
) -> dict[str, Any]:
    return dashboard_tuning_candidate_detail(database(), run_id, candidate_id)
