from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ...core.runtime import database, refresh_locked_configuration_status
from ...schemas.parameter_bootstrap import ParameterBootstrapRequest
from ...services.parameter_bootstrap import (
    apply_parameter_documents,
    parameter_status,
)
from .admin_auth import require_admin_token

router = APIRouter(prefix="/api/admin/parameters", tags=["administration"])


@router.get("/status")
def read_parameter_status(
    _: Annotated[None, Depends(require_admin_token)],
) -> dict[str, Any]:
    return parameter_status(database())


@router.post("/bootstrap")
def bootstrap_parameters(
    request: ParameterBootstrapRequest,
    _: Annotated[None, Depends(require_admin_token)],
) -> dict[str, Any]:
    result = apply_parameter_documents(
        database(),
        strategy_configuration=request.strategy_configuration,
        paper_trading_configuration=request.paper_trading_configuration,
        replace_existing=request.replace_existing,
        note=request.note,
        source="parameter-bootstrap-api",
    )
    refresh_locked_configuration_status()
    return result
