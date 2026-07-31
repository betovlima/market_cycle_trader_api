from __future__ import annotations

import os
import subprocess
from typing import Any

from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    MONGO_DATABASE,
    MONGO_URI,
    create_client,
    ensure_database,
    get_database,
    utc_now,
)

_MONGO_CLIENT: MongoClient | None = None
_MONGO_DB: Database | None = None
MONGO_STATUS: dict[str, Any] = {
    "available": False,
    "database": MONGO_DATABASE,
    "message": "MongoDB has not been initialized.",
}


def database() -> Database:
    if _MONGO_DB is None:
        raise HTTPException(status_code=503, detail="MongoDB is unavailable.")
    return _MONGO_DB


def initialize_mongo() -> None:
    global _MONGO_CLIENT, _MONGO_DB, MONGO_STATUS
    try:
        client = create_client()
        db = get_database(client)
        ensure_database(db)
        db[JOBS_COLLECTION].update_many(
            {"status": {"$in": ["queued", "running"]}},
            {
                "$set": {
                    "status": "interrupted",
                    "stage": "The API was restarted before the run finished.",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                },
                "$unset": {"process_id": ""},
            },
        )
        _MONGO_CLIENT = client
        _MONGO_DB = db
        MONGO_STATUS.clear()
        MONGO_STATUS.update({
            "available": True,
            "configured": bool(MONGO_URI),
            "database": MONGO_DATABASE,
            "message": "MongoDB is available.",
        })
    except PyMongoError as exc:
        _MONGO_CLIENT = None
        _MONGO_DB = None
        MONGO_STATUS.clear()
        MONGO_STATUS.update({
            "available": False,
            "configured": bool(MONGO_URI),
            "database": MONGO_DATABASE,
            "message": f"MongoDB is unavailable: {exc}",
        })


def close_mongo() -> None:
    global _MONGO_CLIENT, _MONGO_DB
    if _MONGO_CLIENT is not None:
        _MONGO_CLIENT.close()
    _MONGO_CLIENT = None
    _MONGO_DB = None


def compute_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "platform": "railway"
        if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID")
        else "local",
        "gpu_name": None,
        "nvidia_driver_visible": False,
        "qrdqn": {
            "device_available": "cpu",
            "cuda_available": False,
            "torch_version": None,
            "torch_cuda_version": None,
        },
        "xgboost": {
            "device_available": "cpu",
            "cuda_available": False,
            "version": None,
            "cuda_build": None,
        },
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode == 0 and names:
            status["nvidia_driver_visible"] = True
            status["gpu_name"] = names[0]
    except Exception:
        pass
    try:
        import torch
        status["qrdqn"]["torch_version"] = str(torch.__version__)
        status["qrdqn"]["torch_cuda_version"] = str(torch.version.cuda) if torch.version.cuda is not None else None
        status["qrdqn"]["cuda_available"] = bool(torch.cuda.is_available())
        if status["qrdqn"]["cuda_available"]:
            status["qrdqn"]["device_available"] = "cuda"
            if not status["gpu_name"]:
                status["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        status["qrdqn"]["error"] = str(exc)
    try:
        import xgboost as xgb
        status["xgboost"]["version"] = str(xgb.__version__)
        build = xgb.build_info()
        cuda_value = build.get("USE_CUDA") if isinstance(build, dict) else None
        if cuda_value is None:
            cuda_build = None
        elif isinstance(cuda_value, bool):
            cuda_build = cuda_value
        else:
            cuda_build = str(cuda_value).strip().lower() in {"1", "true", "yes", "on"}
        status["xgboost"]["cuda_build"] = cuda_build
        status["xgboost"]["cuda_available"] = bool(cuda_build and status["nvidia_driver_visible"])
        if status["xgboost"]["cuda_available"]:
            status["xgboost"]["device_available"] = "cuda"
    except Exception as exc:
        status["xgboost"]["error"] = str(exc)
    return status
