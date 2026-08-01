from __future__ import annotations

import os
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ...core.runtime import database, refresh_locked_configuration_status
from ...schemas.parameter_bootstrap import BootstrapParametersRequest
from ...services.parameter_bootstrap import (
    bootstrap_missing_parameterizations,
    parameterization_status,
)

router = APIRouter(prefix="/api/admin/parameters", tags=["administration"])


def _configured_token() -> str:
    value = str(os.getenv("PARAMETER_BOOTSTRAP_API_TOKEN") or "").strip()
    if len(value) < 24:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "PARAMETER_BOOTSTRAP_API_TOKEN must be configured with at least "
                "24 characters before the parameter bootstrap API can be used."
            ),
        )
    return value


def require_parameter_bootstrap_token(
    supplied: Annotated[
        str,
        Header(alias="X-Parameter-Bootstrap-Token"),
    ],
) -> None:
    expected = _configured_token()
    actual = str(supplied)
    if not secrets.compare_digest(actual, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid parameter bootstrap API token.",
        )


@router.get("/status")
def get_parameterization_status(
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    items = parameterization_status(database())
    return {
        "mode": "insert_missing_repair_invalid_preserve_valid_api_configuration",
        "all_present": all(item["status"] != "missing" for item in items),
        "all_valid": all(item["valid"] for item in items),
        "items": items,
    }


@router.post("/bootstrap")
def bootstrap_parameters(
    _: BootstrapParametersRequest,
    __: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    result = bootstrap_missing_parameterizations(
        database(),
        source="parameter-bootstrap-api",
    )
    refresh_locked_configuration_status()
    return result
