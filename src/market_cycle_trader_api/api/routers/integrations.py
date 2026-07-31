from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...infrastructure.market_data.alpaca import test_connection as test_alpaca_market_data_connection
from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import (
    delete_alpaca_credentials,
    get_alpaca_credentials,
    get_alpaca_integration_status,
    save_alpaca_credentials,
)
from ...schemas.requests import AlpacaConnectionTestRequest, AlpacaCredentialsRequest
from ...services.serialization import iso_value

router = APIRouter(tags=["integrations"])

@router.get("/api/integrations/alpaca")
def get_alpaca_integration() -> dict[str, Any]:
    return iso_value(get_alpaca_integration_status(database()))


@router.put("/api/integrations/alpaca")
def put_alpaca_integration(
    payload: AlpacaCredentialsRequest,
) -> dict[str, Any]:
    try:
        return iso_value(
            save_alpaca_credentials(
                database(),
                api_key_id=payload.api_key_id,
                secret_key=payload.secret_key,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/integrations/alpaca/test")
def test_alpaca_integration(
    payload: AlpacaConnectionTestRequest,
) -> dict[str, Any]:
    try:
        credentials = get_alpaca_credentials(database())
        result = test_alpaca_market_data_connection(
            api_key_id=credentials["api_key_id"],
            secret_key=credentials["secret_key"],
            feed=payload.feed,
        )
        return iso_value(result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Alpaca connection failed: {exc}") from exc


@router.delete("/api/integrations/alpaca", status_code=204)
def remove_alpaca_integration() -> Response:
    delete_alpaca_credentials(database())
    return Response(status_code=204)
