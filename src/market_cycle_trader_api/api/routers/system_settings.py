from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.system_settings import SystemSettingsUpdateRequest
from ...services.system_settings import (
    SystemSettingsConflict,
    get_system_settings,
    list_system_settings_history,
    update_system_settings,
)

router = APIRouter(prefix="/api/admin/system-settings", tags=["admin-system-settings"])


@router.get("")
def read_system_settings() -> dict[str, Any]:
    return get_system_settings(database())


@router.patch("")
def patch_system_settings(
    payload: SystemSettingsUpdateRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return update_system_settings(
            database(),
            payload,
            actor_email=identity.email,
        )
    except SystemSettingsConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/history")
def read_system_settings_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    items = list_system_settings_history(database(), limit=limit)
    return {"count": len(items), "items": items}
