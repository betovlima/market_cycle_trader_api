from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...core.config import API_VERSION, ENGINE_PATH
from ...core.runtime import MONGO_STATUS

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> JSONResponse:
    mongo_available = bool(MONGO_STATUS.get("available"))
    configuration_available = bool(MONGO_STATUS.get("configuration_available"))
    engine_available = ENGINE_PATH.is_file()
    ready = mongo_available and configuration_available and engine_available
    payload: dict[str, Any] = {
        "status": "ok" if ready else "degraded",
        "api_version": API_VERSION,
        "services": {
            "mongodb": mongo_available,
            "locked_configuration": configuration_available,
            "engine": engine_available,
        },
        "storage": "mongodb-only",
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
