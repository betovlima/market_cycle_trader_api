from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.encoders import jsonable_encoder

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...schemas.asset_discovery import AssetDiscoverySettingsUpdateRequest
from ...services.asset_discovery import (
    asset_discovery_status,
    start_asset_discovery,
    stop_asset_discovery,
)
from ...services.asset_discovery_export import build_asset_discovery_export
from ...services.asset_discovery_settings import (
    AssetDiscoveryConflict,
    get_asset_discovery_settings,
    update_asset_discovery_settings,
)
from ...services.asset_discovery_store import list_candidates, list_runs

router = APIRouter(prefix="/api/admin/asset-discovery", tags=["admin-asset-discovery"])


@router.get("/status")
def read_status() -> dict[str, Any]:
    return asset_discovery_status(database())


@router.get("/candidates")
def read_candidates(
    status_filter: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None, max_length=20),
    limit: int = Query(default=250, ge=1, le=1000),
) -> dict[str, Any]:
    items = list_candidates(database(), status=status_filter, query=q, limit=limit)
    return {"count": len(items), "items": items}


@router.get("/runs")
def read_runs(limit: int = Query(default=30, ge=1, le=100)) -> dict[str, Any]:
    items = list_runs(database(), limit=limit)
    return {"count": len(items), "items": items}




@router.get("/export")
def export_analysis(
    front_version: str | None = Query(default=None, max_length=32, pattern=r"^[A-Za-z0-9._-]+$"),
) -> Response:
    payload = build_asset_discovery_export(database(), front_version=front_version)
    generated_at = str(jsonable_encoder(payload["generated_at"]))
    timestamp = generated_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    timestamp = timestamp.replace(".", "").replace(" ", "T")[:15] + "Z"
    filename = f"asset_discovery_analysis_{timestamp}.json"
    body = json.dumps(jsonable_encoder(payload), ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/settings")
def read_settings() -> dict[str, Any]:
    return get_asset_discovery_settings(database())


@router.patch("/settings")
def patch_settings(
    payload: AssetDiscoverySettingsUpdateRequest,
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return update_asset_discovery_settings(database(), payload, actor_email=identity.email)
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
def start_manual_analysis(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
) -> dict[str, Any]:
    try:
        return start_asset_discovery(database(), source="manual", actor_email=identity.email)
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/stop", status_code=status.HTTP_202_ACCEPTED)
def stop_analysis() -> dict[str, Any]:
    return stop_asset_discovery(database())
