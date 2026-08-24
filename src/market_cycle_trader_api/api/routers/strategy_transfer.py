from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.strategy_transfer import StrategyTransferExportRequest, StrategyTransferImportRequest
from ...services.strategy_transfer import (
    StrategyTransferConflict,
    StrategyTransferError,
    StrategyTransferNotFound,
    export_strategy_transfer_package,
    import_strategy_transfer_package,
)

router = APIRouter(prefix="/api/admin/strategy-transfer", tags=["administration"])
AdminIdentity = Annotated[SessionIdentity, Depends(require_admin_session)]


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, StrategyTransferNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, StrategyTransferConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, StrategyTransferError):
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected Strategy transfer error.")


@router.post("/export")
def export_strategy_transfer(
    payload: StrategyTransferExportRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return export_strategy_transfer_package(
            database(),
            strategy_id=payload.strategy_id,
            strategy_sequence=payload.strategy_sequence,
            include_market_snapshot=payload.include_market_snapshot,
            actor_email=identity.email,
        )
    except StrategyTransferError as exc:
        raise _translate_error(exc) from exc


@router.post("/import", status_code=status.HTTP_201_CREATED)
def import_strategy_transfer(
    _: StrategyTransferImportRequest,
    identity: AdminIdentity,
) -> dict[str, Any]:
    try:
        return import_strategy_transfer_package(database(), actor_email=identity.email)
    except StrategyTransferError as exc:
        raise _translate_error(exc) from exc
