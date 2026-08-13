from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...core.config import API_VERSION, ENGINE_PATH
from ...core.runtime import MONGO_STATUS

router = APIRouter(tags=["health"])


def _readiness_payload() -> tuple[bool, dict[str, Any]]:
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
    if not ready:
        payload["message"] = (
            "The process is alive, but the API is not operationally ready yet. "
            "Run the parameter bootstrap when the locked MongoDB configuration is missing."
        )
    return ready, payload


@router.get("/api/health/live")
def liveness() -> dict[str, str]:
    





    return {
        "status": "ok",
        "api_version": API_VERSION,
    }


@router.get("/api/health/ready")
def readiness() -> JSONResponse:
    

    ready, payload = _readiness_payload()
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@router.get("/api/health")
def health() -> JSONResponse:
    

    return readiness()
