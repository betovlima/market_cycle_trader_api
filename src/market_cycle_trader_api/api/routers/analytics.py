from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from ...auth.security import SessionIdentity, require_portfolio_session, require_trader_session
from ...core.runtime import database
from ...services.analytics import asset_strategy_comparison, backtest_analytics, completed_backtests, portfolio_analytics

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
AuthenticatedSession = Annotated[SessionIdentity, Depends(require_trader_session)]
PortfolioSession = Annotated[SessionIdentity, Depends(require_portfolio_session)]


@router.get("/backtests")
def list_completed_backtests(
    _: AuthenticatedSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    return completed_backtests(database(), limit=limit)


@router.get("/backtests/{job_id}")
def get_backtest_analytics(job_id: str, _: AuthenticatedSession) -> dict[str, Any]:
    return backtest_analytics(database(), job_id)


@router.get("/backtests/{job_id}/assets/{asset}")
def get_backtest_asset_comparison(
    job_id: str,
    asset: str,
    _: AuthenticatedSession,
) -> dict[str, Any]:
    return asset_strategy_comparison(database(), job_id, asset)


@router.get("/portfolio")
def get_portfolio_analytics(_: PortfolioSession) -> dict[str, Any]:
    return portfolio_analytics(database())
