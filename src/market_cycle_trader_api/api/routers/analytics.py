from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from ...auth.security import SessionIdentity, require_portfolio_session, require_trader_session
from ...core.runtime import database
from ...services.analytics import (
    backtest_analytics,
    completed_backtests,
    completed_processings,
    portfolio_analytics,
    processing_analytics,
    processing_rotation_period_analysis,
    rotation_period_analysis,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
AuthenticatedSession = Annotated[SessionIdentity, Depends(require_trader_session)]
PortfolioSession = Annotated[SessionIdentity, Depends(require_portfolio_session)]




@router.get("/processings")
def list_completed_processings(
    _: AuthenticatedSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    return completed_processings(database(), limit=limit)


@router.get("/processings/{processing_id}")
def get_processing_analytics(processing_id: str, _: AuthenticatedSession) -> dict[str, Any]:
    return processing_analytics(database(), processing_id)


@router.get("/processings/{processing_id}/rotation-period")
def get_processing_rotation_period(
    processing_id: str,
    _: AuthenticatedSession,
    year: Annotated[int, Query(ge=2000, le=2200)],
    month: Annotated[int, Query(ge=1, le=12)],
) -> dict[str, Any]:
    return processing_rotation_period_analysis(database(), processing_id, year=year, month=month)


@router.get("/backtests")
def list_completed_backtests(
    _: AuthenticatedSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    return completed_backtests(database(), limit=limit)


@router.get("/backtests/{job_id}")
def get_backtest_analytics(job_id: str, _: AuthenticatedSession) -> dict[str, Any]:
    return backtest_analytics(database(), job_id)


@router.get("/backtests/{job_id}/rotation-period")
def get_backtest_rotation_period(
    job_id: str,
    _: AuthenticatedSession,
    year: Annotated[int, Query(ge=2000, le=2200)],
    month: Annotated[int, Query(ge=1, le=12)],
) -> dict[str, Any]:
    return rotation_period_analysis(database(), job_id, year=year, month=month)


@router.get("/portfolio")
def get_portfolio_analytics(_: PortfolioSession) -> dict[str, Any]:
    return portfolio_analytics(database())
