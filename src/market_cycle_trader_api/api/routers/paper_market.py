from __future__ import annotations

import os
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ...core.runtime import database
from ...schemas.paper_market import (
    CancelPaperMarketRequest,
    StartNextSessionRequest,
    StopPaperRobotRequest,
)
from ...services.paper_portfolio import paper_portfolio_snapshot
from ...services.paper_market_scheduler import (
    arm_next_session,
    cancel_paper_market_run,
    latest_paper_market_run,
    paper_market_robot_status,
    stop_continuous_robot,
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
        detail = str(exc)
        code = status.HTTP_409_CONFLICT if "already active" in detail else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=code, detail=detail) from exc


@router.get("/status")
def paper_market_status(
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any] | None:
    return latest_paper_market_run(database())


@router.get("/robot/status")
def robot_status(
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    return paper_market_robot_status(database())


@router.post("/robot/stop")
def stop_robot(
    request: StopPaperRobotRequest,
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    return stop_continuous_robot(
        database(),
        cancel_pending_run=request.cancel_pending_run,
    )




@router.get("/portfolio")
def paper_market_portfolio(
    _: Annotated[None, Depends(require_paper_market_token)],
) -> dict[str, Any]:
    try:
        return paper_portfolio_snapshot(database())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
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
        detail = str(exc)
        code = status.HTTP_404_NOT_FOUND if "not found" in detail else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=code, detail=detail) from exc
