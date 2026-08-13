from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from ...core.runtime import database
from ...services.public_paper_portfolio import public_paper_portfolio_snapshot
from ...services.paper_market_scheduler import paper_market_robot_status

router = APIRouter(prefix="/api/paper-market", tags=["paper-market-public"])


@router.get("/public-portfolio")
def public_paper_market_portfolio() -> dict[str, Any]:
    

    try:
        return public_paper_portfolio_snapshot(database())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/public-robot-status")
def public_robot_status() -> dict[str, Any]:
    

    return paper_market_robot_status(database(), public=True)
