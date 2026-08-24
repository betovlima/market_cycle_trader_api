from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
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
    "configured": bool(MONGO_URI and MONGO_DATABASE),
    "database": MONGO_DATABASE or None,
    "configuration_available": False,
    "message": "MongoDB has not been initialized.",
}


def database() -> Database:
    if _MONGO_DB is None:
        raise HTTPException(status_code=503, detail="MongoDB is unavailable.")
    return _MONGO_DB


def refresh_locked_configuration_status() -> bool:
    

    if _MONGO_DB is None:
        MONGO_STATUS["configuration_available"] = False
        MONGO_STATUS["configuration_message"] = "MongoDB is unavailable."
        return False

    try:
        from ..services.strategy_lab import get_research_strategy_context, get_trader_winner_context
        get_research_strategy_context(_MONGO_DB)
        get_trader_winner_context(_MONGO_DB)
    except (RuntimeError, ValidationError) as exc:
        MONGO_STATUS["configuration_available"] = False
        MONGO_STATUS["configuration_message"] = (
            f"Strategy catalog is unavailable or invalid: {exc}"
        )
        return False

    MONGO_STATUS["configuration_available"] = True
    MONGO_STATUS["configuration_message"] = "Research strategy and Trader winner are valid."
    return True


def initialize_mongo(*, role: str = "api") -> None:
    global _MONGO_CLIENT, _MONGO_DB, MONGO_STATUS
    try:
        client = create_client()
        db = get_database(client)
        ensure_database(db)
        normalized_role = str(role or "api").strip().lower()
        if normalized_role == "api":
            
            
            
            
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
        configuration_message = "Research strategy and Trader winner are valid."
        try:
            from ..services.strategy_lab import get_research_strategy_context, get_trader_winner_context
            get_research_strategy_context(db)
            get_trader_winner_context(db)
        except (RuntimeError, ValidationError) as exc:
            configuration_available = False
            configuration_message = f"Strategy catalog is unavailable or invalid: {exc}"

        _MONGO_CLIENT = client
        _MONGO_DB = db
        MONGO_STATUS.clear()
        MONGO_STATUS.update(
            {
                "available": True,
                "configured": True,
                "database": MONGO_DATABASE,
                "configuration_available": configuration_available,
                "configuration_message": configuration_message,
                "message": "MongoDB is available.",
            }
        )
    except (PyMongoError, RuntimeError) as exc:
        _MONGO_CLIENT = None
        _MONGO_DB = None
        MONGO_STATUS.clear()
        MONGO_STATUS.update(
            {
                "available": False,
                "configured": bool(MONGO_URI and MONGO_DATABASE),
                "database": MONGO_DATABASE or None,
                "configuration_available": False,
                "message": f"MongoDB is unavailable: {exc}",
            }
        )


def close_mongo() -> None:
    global _MONGO_CLIENT, _MONGO_DB
    if _MONGO_CLIENT is not None:
        _MONGO_CLIENT.close()
    _MONGO_CLIENT = None
    _MONGO_DB = None
