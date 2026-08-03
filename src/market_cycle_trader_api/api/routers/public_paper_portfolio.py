from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from ...core.runtime import database
from ...services.public_paper_portfolio import public_paper_portfolio_snapshot

router = APIRouter(prefix="/api/paper-market", tags=["paper-market-public"])


@router.get("/public-portfolio")
def public_paper_market_portfolio() -> dict[str, Any]:
    """Read-only sanitized portfolio data for the public frontend."""

    try:
        return public_paper_portfolio_snapshot(database())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
