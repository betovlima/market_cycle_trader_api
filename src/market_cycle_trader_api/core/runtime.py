from __future__ import annotations

import logging
from threading import RLock
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    create_client,
    ensure_database,
    get_database,
    get_settings,
    mongo_connection_status,
    mongo_database_name,
    utc_now,
)
from ..schemas.requests import BacktestRequest

logger = logging.getLogger(__name__)

_MONGO_CLIENT: MongoClient | None = None
_MONGO_DB: Database | None = None
_MONGO_LOCK = RLock()

_initial_connection = mongo_connection_status()
MONGO_STATUS: dict[str, Any] = {
    "available": False,
    "configured": bool(_initial_connection["configured"]),
    "database": _initial_connection["database"],
    "configuration_available": False,
    "message": "MongoDB has not been initialized.",
}


def database() -> Database:
    """Return the active database and retry one failed startup connection lazily."""

    if _MONGO_DB is None:
        initialize_mongo()
    if _MONGO_DB is None:
        raise HTTPException(status_code=503, detail="MongoDB is unavailable.")
    return _MONGO_DB


def _validate_configuration(db: Database) -> None:
    BacktestRequest.model_validate(get_settings(db))


def refresh_locked_configuration_status() -> bool:
    if _MONGO_DB is None:
        MONGO_STATUS["configuration_available"] = False
        MONGO_STATUS["configuration_message"] = "MongoDB is unavailable."
        return False
    try:
        _validate_configuration(_MONGO_DB)
    except (RuntimeError, ValidationError) as exc:
        MONGO_STATUS["configuration_available"] = False
        MONGO_STATUS["configuration_message"] = (
            f"Configuration is unavailable or invalid: {exc}"
        )
        return False
    MONGO_STATUS["configuration_available"] = True
    MONGO_STATUS["configuration_message"] = "Configuration is valid."
    return True


def initialize_mongo() -> None:
    global _MONGO_CLIENT, _MONGO_DB

    with _MONGO_LOCK:
        if _MONGO_DB is not None:
            return

        connection = mongo_connection_status()
        client: MongoClient | None = None
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

            configuration_available = True
            configuration_message = "Configuration is valid."
            try:
                _validate_configuration(db)
            except (RuntimeError, ValidationError) as exc:
                configuration_available = False
                configuration_message = (
                    f"Configuration is unavailable or invalid: {exc}"
                )

            _MONGO_CLIENT = client
            _MONGO_DB = db
            MONGO_STATUS.clear()
            MONGO_STATUS.update(
                {
                    "available": True,
                    "configured": True,
                    "database": mongo_database_name(),
                    "configuration_available": configuration_available,
                    "configuration_message": configuration_message,
                    "message": "MongoDB is available.",
                }
            )
        except (PyMongoError, RuntimeError, ValueError) as exc:
            if client is not None:
                client.close()
            _MONGO_CLIENT = None
            _MONGO_DB = None
            MONGO_STATUS.clear()
            MONGO_STATUS.update(
                {
                    "available": False,
                    "configured": bool(connection["configured"]),
                    "database": connection["database"],
                    "configuration_available": False,
                    "message": "MongoDB is unavailable.",
                    "error_type": type(exc).__name__,
                }
            )
            logger.error("MongoDB initialization failed: %s: %s", type(exc).__name__, exc)


def reconnect_mongo() -> bool:
    """Close the current client and resolve MONGO_URL from the environment again."""

    close_mongo()
    initialize_mongo()
    return _MONGO_DB is not None


def close_mongo() -> None:
    global _MONGO_CLIENT, _MONGO_DB

    with _MONGO_LOCK:
        if _MONGO_CLIENT is not None:
            _MONGO_CLIENT.close()
        _MONGO_CLIENT = None
        _MONGO_DB = None
