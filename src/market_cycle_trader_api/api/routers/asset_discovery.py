from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.encoders import jsonable_encoder

from ...auth.security import SessionIdentity, require_capability
from ...core.runtime import database
from ...schemas.asset_discovery import AssetDiscoveryCreateStrategyRequest, AssetDiscoveryStartRequest
from ...services.asset_discovery import (
    AssetDiscoveryConflict,
    create_research_strategy_from_discovery,
    export_asset_discovery,
    get_asset_discovery_status,
    get_discovery_catalog,
    start_asset_discovery,
    start_marginal_capital_replay,
    stop_asset_discovery,
)

router = APIRouter(prefix="/api/asset-discovery", tags=["asset-discovery"])


@router.get("/status", dependencies=[Depends(require_capability("asset_discovery.view"))])
def read_status() -> dict[str, Any]:
    return get_asset_discovery_status(database())


@router.get("/catalog", dependencies=[Depends(require_capability("asset_discovery.view"))])
def read_catalog() -> dict[str, Any]:
    return get_discovery_catalog(database())


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
def start_campaign(
    payload: AssetDiscoveryStartRequest,
    identity: Annotated[SessionIdentity, Depends(require_capability("asset_discovery.start"))],
) -> dict[str, Any]:
    try:
        return start_asset_discovery(database(), research_size=payload.research_size)
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/marginal-replay", status_code=status.HTTP_202_ACCEPTED)
def start_marginal_replay(
    _: Annotated[SessionIdentity, Depends(require_capability("asset_discovery.start"))],
) -> dict[str, Any]:
    try:
        return start_marginal_capital_replay(database())
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/stop", status_code=status.HTTP_202_ACCEPTED)
def stop_campaign(
    _: Annotated[SessionIdentity, Depends(require_capability("asset_discovery.stop"))],
) -> dict[str, Any]:
    return stop_asset_discovery(database())


@router.post("/create-strategy", status_code=status.HTTP_201_CREATED)
def create_research_strategy(
    payload: AssetDiscoveryCreateStrategyRequest,
    identity: Annotated[SessionIdentity, Depends(require_capability("asset_discovery.create_strategy"))],
) -> dict[str, Any]:
    try:
        return create_research_strategy_from_discovery(
            database(),
            run_id=payload.run_id,
            symbols=payload.symbols,
            actor_email=identity.email,
        )
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/export")
def export_campaign(
    _: Annotated[SessionIdentity, Depends(require_capability("asset_discovery.export"))],
    front_version: str | None = Query(default=None, max_length=32, pattern=r"^[A-Za-z0-9._-]+$"),
) -> Response:
    try:
        payload = export_asset_discovery(database(), front_version=front_version)
    except AssetDiscoveryConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    generated = jsonable_encoder(payload)
    timestamp = str(generated.get("generated_at") or "").replace("-", "").replace(":", "").replace("+00:00", "Z")
    timestamp = timestamp.replace(".", "").replace(" ", "T")[:15] + "Z"
    body = json.dumps(generated, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="asset_discovery_ranker_{timestamp}.json"',
            "Cache-Control": "no-store",
        },
    )
