from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from ...core.config import ACTIVE_STRATEGY_MODE, STRATEGY_CATALOG
from ...core.runtime import MONGO_STATUS, database
from ...infrastructure.persistence.mongo_repository import DEFAULT_SETTINGS, SETTINGS_COLLECTION, SETTINGS_SCHEMA_VERSION, get_settings, update_settings, utc_now
from ...schemas.requests import normalize_assets, validate_json_configuration
from ...services.serialization import iso_value

router = APIRouter(tags=["configuration"])


@router.get("/api/strategies")
def get_strategy_catalog() -> dict[str, Any]:
    return {"active_strategy_mode": ACTIVE_STRATEGY_MODE, "strategies": [iso_value(metadata) for metadata in STRATEGY_CATALOG.values()]}


@router.get("/api/config")
def get_config() -> dict[str, Any]:
    return {**iso_value(get_settings(database())), "mongo_status": dict(MONGO_STATUS)}


@router.put("/api/config")
def save_config(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if "assets" in changes:
        changes["assets"] = normalize_assets(changes["assets"])
    normalized, _ = validate_json_configuration(database(), changes)
    return iso_value(update_settings(database(), normalized))


@router.post("/api/config/json/validate")
def validate_config_json(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
    normalized_changes, effective = validate_json_configuration(database(), changes)
    return {"valid": True, "changes": iso_value(normalized_changes), "effective_config": iso_value(effective)}


@router.put("/api/config/json/apply")
def apply_config_json(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
    db = database()
    normalized_changes, _ = validate_json_configuration(db, changes)
    saved = update_settings(db, normalized_changes)
    return {"applied": True, "changes": iso_value(normalized_changes), "config": iso_value(saved)}


@router.post("/api/config/reset")
def reset_config() -> dict[str, Any]:
    db = database()
    now = utc_now()
    db[SETTINGS_COLLECTION].replace_one({"_id": "default"}, {"_id": "default", **DEFAULT_SETTINGS, "created_at": now, "updated_at": now, "schema_version": SETTINGS_SCHEMA_VERSION}, upsert=True)
    return iso_value(get_settings(db))
