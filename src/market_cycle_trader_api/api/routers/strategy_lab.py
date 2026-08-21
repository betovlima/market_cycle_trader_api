from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from ...auth.security import SessionIdentity, require_admin_session, require_trader_session
from ...core.runtime import database, refresh_locked_configuration_status
from ...schemas.strategy_lab import (
    StrategyCandidateRequest,
    StrategyCreateRequest,
    StrategyDeleteRequest,
    StrategyModelUpdateRequest,
    StrategyPromoteRequest,
    StrategySelectRequest,
    StrategyUpdateRequest,
)
from ...services.strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    create_strategy,
    delete_strategy,
    get_strategy,
    get_strategy_control,
    list_strategies,
    mark_strategy_as_candidate,
    promote_strategy_to_trader,
    select_model_tuning_strategy,
    select_research_strategy,
    update_strategy,
    update_strategy_model,
)

router = APIRouter(prefix="/api/admin/strategies", tags=["admin-strategies"])
AdminIdentity = Annotated[SessionIdentity, Depends(require_admin_session)]
ViewerIdentity = Annotated[SessionIdentity, Depends(require_trader_session)]


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, StrategyLabConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, StrategyLabNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(include_url=False),
        )
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, StrategyLabError):
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected strategy management error.")


@router.get("")
def read_strategies(_: ViewerIdentity) -> dict[str, Any]:
    try:
        return list_strategies(database())
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.get("/control")
def read_strategy_control(_: ViewerIdentity) -> dict[str, Any]:
    try:
        return get_strategy_control(database())
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("", status_code=201)
def create_strategy_profile(
    payload: StrategyCreateRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return create_strategy(
            database(),
            name=payload.name,
            description=payload.description,
            clone_from_strategy_id=payload.clone_from_strategy_id,
            actor_email=identity.email,
        )
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{strategy_id}")
def read_strategy(strategy_id: str, _: ViewerIdentity) -> dict[str, Any]:
    try:
        return get_strategy(database(), strategy_id)
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.put("/{strategy_id}")
def replace_strategy(
    strategy_id: str,
    payload: StrategyUpdateRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        result = update_strategy(
            database(),
            strategy_id,
            configuration=payload.build_configuration(),
            name=payload.name,
            description=payload.description,
            note=payload.note,
            expected_revision=payload.expected_revision,
            actor_email=identity.email,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.put("/{strategy_id}/model")
def replace_strategy_model(
    strategy_id: str,
    payload: StrategyModelUpdateRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        result = update_strategy_model(
            database(),
            strategy_id,
            model_family=payload.model_family,
            values=payload.values,
            note=payload.note,
            expected_strategy_revision=payload.expected_strategy_revision,
            actor_email=identity.email,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyLabError, ValidationError, ValueError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{strategy_id}/select-for-strategy-research")
def select_strategy_for_strategy_research(
    strategy_id: str,
    payload: StrategySelectRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        result = select_research_strategy(
            database(),
            strategy_id,
            expected_control_revision=payload.expected_control_revision,
            note=payload.note,
            actor_email=identity.email,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{strategy_id}/select-for-model-tuning")
def select_strategy_for_model_tuning(
    strategy_id: str,
    payload: StrategySelectRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return select_model_tuning_strategy(
            database(),
            strategy_id,
            expected_control_revision=payload.expected_control_revision,
            note=payload.note,
            actor_email=identity.email,
        )
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{strategy_id}/select-for-backtest")
def select_strategy_for_backtest(
    strategy_id: str,
    payload: StrategySelectRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        result = select_research_strategy(
            database(),
            strategy_id,
            expected_control_revision=payload.expected_control_revision,
            note=payload.note,
            actor_email=identity.email,
        )
        refresh_locked_configuration_status()
        return result
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{strategy_id}/mark-as-candidate")
def mark_candidate(
    strategy_id: str,
    payload: StrategyCandidateRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return mark_strategy_as_candidate(
            database(),
            strategy_id,
            expected_strategy_revision=payload.expected_strategy_revision,
            model_family=payload.model_family,
            note=payload.note,
            actor_email=identity.email,
        )
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.post("/{strategy_id}/promote-to-trader")
def promote_strategy(
    strategy_id: str,
    payload: StrategyPromoteRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return promote_strategy_to_trader(
            database(),
            strategy_id,
            expected_control_revision=payload.expected_control_revision,
            expected_strategy_revision=payload.expected_strategy_revision,
            note=payload.note,
            actor_email=identity.email,
        )
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc


@router.delete("/{strategy_id}")
def remove_strategy(
    strategy_id: str,
    payload: StrategyDeleteRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return delete_strategy(
            database(),
            strategy_id,
            note=payload.note,
            actor_email=identity.email,
        )
    except (StrategyLabError, ValidationError) as exc:
        raise _translate_error(exc) from exc
