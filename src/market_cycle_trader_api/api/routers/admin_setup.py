from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.runtime import database, refresh_locked_configuration_status
from ...schemas.admin_setup import InitializeApplicationRequest
from ...services.admin_setup import initialize_application, setup_status
from .parameter_bootstrap import require_parameter_bootstrap_token

router = APIRouter(prefix="/api/admin/setup", tags=["administration"])


@router.get("/status")
def get_setup_status(
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    return setup_status(database())


@router.post("/initialize")
def initialize_everything(
    request: InitializeApplicationRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    if not request.confirm_paper:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="confirm_paper must be true.",
        )
    try:
        result = initialize_application(
            database(),
            arm_market=request.arm_next_session,
        )
        refresh_locked_configuration_status()
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
