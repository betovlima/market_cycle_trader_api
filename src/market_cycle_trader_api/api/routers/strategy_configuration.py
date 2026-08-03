from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from ...core.runtime import database, refresh_locked_configuration_status
from ...schemas.strategy_configuration import (
    StrategyConfigurationPatchRequest,
    StrategyConfigurationReplaceRequest,
    StrategyConfigurationResetRequest,
    StrategyConfigurationRestoreRequest,
    StrategyWinnerInstallRequest,
)
from ...services.strategy_configuration import (
    StrategyConfigurationConflict,
    StrategyConfigurationError,
    StrategyConfigurationNotFound,
    get_strategy_configuration,
    install_winner_strategy_configuration,
    list_strategy_configuration_history,
    patch_strategy_configuration,
    replace_strategy_configuration,
    reset_strategy_configuration,
    restore_strategy_configuration,
)
from .parameter_bootstrap import require_parameter_bootstrap_token

router = APIRouter(
    prefix="/api/admin/strategy-configuration",
    tags=["strategy configuration"],
)


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, StrategyConfigurationConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, StrategyConfigurationNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(include_url=False),
        )
    if isinstance(exc, StrategyConfigurationError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected strategy configuration error.",
    )


@router.get("")
def read_strategy_configuration(
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        return get_strategy_configuration(database())
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.patch("")
def update_strategy_configuration(
    request: StrategyConfigurationPatchRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        result = patch_strategy_configuration(
            database(),
            request.changes,
            note=request.note,
            source="strategy-configuration-api",
            expected_revision=request.expected_revision,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.put("")
def replace_active_strategy_configuration(
    request: StrategyConfigurationReplaceRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        result = replace_strategy_configuration(
            database(),
            request.configuration,
            note=request.note,
            source="strategy-configuration-api",
            expected_revision=request.expected_revision,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/winner/install")
def install_winner_strategy(
    request: StrategyWinnerInstallRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        result = install_winner_strategy_configuration(
            database(),
            note=request.note,
            source="winner-v1.13.2-install-api",
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/reset")
def reset_active_strategy_configuration(
    request: StrategyConfigurationResetRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        result = reset_strategy_configuration(
            database(),
            note=request.note,
            source="strategy-configuration-api",
            expected_revision=request.expected_revision,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.get("/history")
def read_strategy_configuration_history(
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    items = list_strategy_configuration_history(database(), limit=limit)
    return {"count": len(items), "items": items}


@router.post("/history/{history_id}/restore")
def restore_archived_strategy_configuration(
    history_id: str,
    request: StrategyConfigurationRestoreRequest,
    _: Annotated[None, Depends(require_parameter_bootstrap_token)],
) -> dict[str, Any]:
    try:
        result = restore_strategy_configuration(
            database(),
            history_id,
            note=request.note,
            source="strategy-configuration-api",
            expected_revision=request.expected_revision,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyConfigurationError, ValidationError) as exc:
        raise _translate_error(exc) from exc
