from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.temporal_research_settings import TemporalResearchSettingsUpdateRequest
from ...services.temporal_research_settings import (
    TemporalResearchSettingsConflict,
    get_temporal_research_settings,
    list_temporal_research_settings_history,
    update_temporal_research_settings,
)

router = APIRouter(prefix="/api/admin/temporal-research-settings", tags=["admin-temporal-research-settings"])


@router.get("")
def read_temporal_research_settings() -> dict[str, Any]:
    return get_temporal_research_settings(database())


@router.patch("")
def patch_temporal_research_settings(
    payload: TemporalResearchSettingsUpdateRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return update_temporal_research_settings(database(), payload, actor_email=identity.email)
    except TemporalResearchSettingsConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/history")
def read_temporal_research_settings_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    items = list_temporal_research_settings_history(database(), limit=limit)
    return {"count": len(items), "items": items}
