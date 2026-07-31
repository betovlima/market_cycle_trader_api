from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from ...core.config import ACTIVE_STRATEGY_MODE, STRATEGY_CATALOG
from ...core.runtime import MONGO_STATUS, database
from ...infrastructure.persistence.mongo_repository import (
    DEFAULT_SETTINGS,
    SETTINGS_COLLECTION,
    SETTINGS_SCHEMA_VERSION,
    delete_parameter_profile,
    get_parameter_profile,
    get_parameter_profiles,
    get_settings,
    save_parameter_profile,
    update_settings,
    utc_now,
)
from ...schemas.requests import (
    ParameterProfileRequest,
    normalize_assets,
    normalize_backends,
    normalize_exit_risk_backends,
    validate_json_configuration,
)
from ...services.serialization import iso_value

router = APIRouter(tags=["configuration"])

@router.get("/api/strategies")
def get_strategy_catalog() -> dict[str, Any]:
    return {
        "active_strategy_mode": ACTIVE_STRATEGY_MODE,
        "strategies": [
            iso_value(metadata)
            for metadata in STRATEGY_CATALOG.values()
        ],
    }


@router.get("/api/config")
def get_config() -> dict[str, Any]:
    settings = get_settings(database())
    return {
        **iso_value(settings),
        "mongo_status": dict(MONGO_STATUS),
    }


@router.put("/api/config")
def save_config(
    changes: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    if "assets" in changes:
        changes["assets"] = normalize_assets(changes["assets"])
    if "model_backends" in changes:
        changes["model_backends"] = normalize_backends(
            changes["model_backends"]
        )
    if "exit_risk_model_backends" in changes:
        changes["exit_risk_model_backends"] = normalize_exit_risk_backends(
            changes["exit_risk_model_backends"]
        )
    return iso_value(update_settings(database(), changes))


@router.post("/api/config/json/validate")
def validate_config_json(
    changes: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    normalized_changes, effective = validate_json_configuration(
        database(), changes
    )
    return {
        "valid": True,
        "changes": iso_value(normalized_changes),
        "effective_config": iso_value(effective),
    }


@router.put("/api/config/json/apply")
def apply_config_json(
    changes: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    db = database()
    normalized_changes, _ = validate_json_configuration(db, changes)
    saved = update_settings(db, normalized_changes)
    return {
        "applied": True,
        "changes": iso_value(normalized_changes),
        "config": iso_value(saved),
    }


@router.post("/api/config/reset")
def reset_config() -> dict[str, Any]:
    db = database()
    db[SETTINGS_COLLECTION].replace_one(
        {"_id": "default"},
        {
            "_id": "default",
            **DEFAULT_SETTINGS,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "schema_version": SETTINGS_SCHEMA_VERSION,
        },
        upsert=True,
    )
    return iso_value(get_settings(db))


@router.get("/api/config/profiles")
def list_profiles(
    timeframe: str | None = None,
) -> dict[str, Any]:
    db = database()
    settings = get_settings(db)
    symbols = settings.get("assets", [])
    profiles = get_parameter_profiles(
        db,
        symbols=symbols,
        timeframe=timeframe,
    )
    return {
        "general": iso_value(settings),
        "profiles": iso_value(profiles),
    }


@router.get("/api/config/profiles/{symbol}/{timeframe}")
def read_profile(
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    profile = get_parameter_profile(
        database(),
        symbol=symbol,
        timeframe=timeframe,
    )
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Parameter profile not found.",
        )
    return iso_value(profile)


@router.put("/api/config/profiles/{symbol}/{timeframe}")
def write_profile(
    symbol: str,
    timeframe: str,
    payload: ParameterProfileRequest,
) -> dict[str, Any]:
    if payload.symbol != symbol.upper():
        raise HTTPException(
            status_code=422,
            detail="Path symbol and payload symbol must match.",
        )
    if payload.timeframe != timeframe:
        raise HTTPException(
            status_code=422,
            detail="Path timeframe and payload timeframe must match.",
        )
    profile = save_parameter_profile(
        database(),
        symbol=symbol,
        timeframe=timeframe,
        parameters=payload.parameters,
        profile_name=payload.profile_name,
        source_job_id=payload.source_job_id,
        validation_status=payload.validation_status,
    )
    return iso_value(profile)


@router.delete("/api/config/profiles/{symbol}/{timeframe}")
def remove_profile(
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    removed = delete_parameter_profile(
        database(),
        symbol=symbol,
        timeframe=timeframe,
    )
    return {"removed": removed}
