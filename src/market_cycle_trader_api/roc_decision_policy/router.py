from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth.security import SessionIdentity, require_capability
from ..core.runtime import database
from .config import RocDecisionPolicySettingsConflict, get_settings, update_settings
from .errors import RocDecisionPolicyError
from .persistence import latest_raw, public_summary
from .schemas import RocDecisionPolicyRunRequest, RocDecisionPolicySettingsUpdateRequest
from .service import run

router = APIRouter(prefix="/api/roc-decision-policy", tags=["roc-decision-policy"])
require_view = require_capability("temporal_intelligence.view")
require_start = require_capability("temporal_intelligence.start")
require_manage = require_capability("admin.manage")


@router.get("/settings")
def roc_policy_settings(_identity: Annotated[SessionIdentity, Depends(require_view)]) -> dict[str, Any]:
    return get_settings(database())


@router.patch("/settings")
def update_roc_policy_settings(
    payload: RocDecisionPolicySettingsUpdateRequest,
    identity: Annotated[SessionIdentity, Depends(require_manage)],
) -> dict[str, Any]:
    try:
        return update_settings(database(), payload, actor_email=identity.email)
    except RocDecisionPolicySettingsConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{run_id}/latest")
def latest_roc_policy(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_view)],
) -> dict[str, Any] | None:
    return public_summary(latest_raw(database(), run_id))


@router.post("/{run_id}/run")
def run_roc_policy(
    run_id: str,
    payload: RocDecisionPolicyRunRequest,
    _identity: Annotated[SessionIdentity, Depends(require_start)],
) -> dict[str, Any]:
    try:
        return run(
            database(),
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
        )
    except (RocDecisionPolicyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
