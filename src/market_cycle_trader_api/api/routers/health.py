from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...core.config import API_VERSION, ENGINE_PATH
from ...core.runtime import MONGO_STATUS, close_mongo, compute_runtime_status, initialize_mongo

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> JSONResponse:
    ready = bool(MONGO_STATUS.get("available")) and ENGINE_PATH.is_file()
    payload: dict[str, Any] = {
        "status": "ok" if ready else "degraded",
        "api_version": API_VERSION,
        "mongo": dict(MONGO_STATUS),
        "engine": {
            "available": ENGINE_PATH.is_file(),
            "entrypoint": ENGINE_PATH.name,
        },
        "storage": "mongodb-only",
        "compute": compute_runtime_status(),
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@router.post("/api/cache/reconnect")
def reconnect_cache() -> dict[str, Any]:
    close_mongo()
    initialize_mongo()
    return dict(MONGO_STATUS)
