from __future__ import annotations

import os
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ...core.runtime import database
from ...schemas.paper_market import CancelPaperMarketRequest, StartNextSessionRequest
from ...services.paper_portfolio import paper_portfolio_snapshot
from ...services.paper_market_scheduler import (
    arm_next_session,
    cancel_paper_market_run,
    latest_paper_market_run,
)

router = APIRouter(prefix="/api/paper-market", tags=["paper-market"])


def _configured_token() -> str:
    value = str(os.getenv("PAPER_MARKET_API_TOKEN") or "").strip()
    if len(value) < 16:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "PAPER_MARKET_API_TOKEN must be configured with at least 16 characters "
                "before the paper-market API can be used."
            ),
        )
    return value


def require_paper_market_token(
    supplied: Annotated[
        str,
        Header(alias="X-Paper-Market-Token"),
    ],
) -> None:
    expected = _configured_token()
    actual = str(supplied)
    if not secrets.compare_digest(actual, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid paper-market API token.",
        )


@router.post("/start-next-session", status_code=status.HTTP_202_ACCEPTED)
def start_next_session(
    _: StartNextSessionRequest,
    __: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    try:
        return arm_next_session(database())
    except RuntimeError as exc:
        code = status.HTTP_409_CONFLICT if "already active" in str(exc) else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=code, detail="Unable to start paper execution.") from exc


@router.get("/status")
def paper_market_status(
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any] | None:
    return latest_paper_market_run(database())




@router.get("/portfolio")
def paper_market_portfolio(
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    try:
        snapshot = paper_portfolio_snapshot(database())
        clock = snapshot.get("market_clock") or {}
        run = snapshot.get("next_session_run") or {}
        return {
            "status": str(snapshot.get("status") or "ready"),
            "portfolio_value": snapshot.get("portfolio_value"),
            "available_cash": snapshot.get("available_cash"),
            "market_value": snapshot.get("market_value"),
            "realized_pnl": snapshot.get("realized_pnl"),
            "unrealized_pnl": snapshot.get("unrealized_pnl"),
            "total_pnl": snapshot.get("total_pnl"),
            "total_return": snapshot.get("total_return"),
            "market_open": bool(clock.get("is_open")),
            "execution_status": str(run.get("status") or "idle"),
        }
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Portfolio data is temporarily unavailable.",
        ) from exc


@router.post("/{run_id}/cancel")
def cancel_next_session(
    run_id: str,
    _: CancelPaperMarketRequest,
    __: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    try:
        return cancel_paper_market_run(database(), run_id)
    except RuntimeError as exc:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(exc) else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=code, detail="Unable to cancel paper execution.") from exc
