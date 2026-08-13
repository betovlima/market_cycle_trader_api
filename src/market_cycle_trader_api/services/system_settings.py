from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import (
    SYSTEM_SETTINGS_COLLECTION,
    SYSTEM_SETTINGS_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.system_settings import SystemSettingsUpdateRequest

SYSTEM_SETTINGS_ID = "market-cycle-runtime"
DEFAULT_TRAINING_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "automatic_training_enabled": True,
    "model_threads": 8,
    "numeric_threads": 4,
    "max_concurrent_jobs": 1,
    "timeout_seconds": 21_600,
}

ConfigurationModel = TypeVar("ConfigurationModel", bound=BaseModel)


class SystemSettingsConflict(RuntimeError):
    pass


def _detected_cpu_count() -> int:
    detected = max(1, int(os.cpu_count() or 1))
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if not cpu_max.exists():
        return detected
    try:
        quota_text, period_text = cpu_max.read_text(encoding="utf-8").strip().split()[:2]
        if quota_text == "max":
            return detected
        quota = int(quota_text)
        period = int(period_text)
        if quota <= 0 or period <= 0:
            return detected
        constrained = max(1, int((quota + period - 1) // period))
        return min(detected, constrained)
    except (OSError, ValueError, IndexError):
        return detected


def _normalized_training(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    merged = {**DEFAULT_TRAINING_SETTINGS, **source}
    return {
        "enabled": bool(merged["enabled"]),
        "automatic_training_enabled": bool(merged["automatic_training_enabled"]),
        "model_threads": max(1, min(64, int(merged["model_threads"]))),
        "numeric_threads": max(1, min(64, int(merged["numeric_threads"]))),
        
        
        "max_concurrent_jobs": 1,
        "timeout_seconds": max(300, min(86_400, int(merged["timeout_seconds"]))),
    }


def ensure_system_settings(db: Any) -> dict[str, Any]:
    now = utc_now()
    db[SYSTEM_SETTINGS_COLLECTION].update_one(
        {"_id": SYSTEM_SETTINGS_ID},
        {
            "$setOnInsert": {
                "revision": 1,
                "training": deepcopy(DEFAULT_TRAINING_SETTINGS),
                "created_at": now,
                "updated_at": now,
                "updated_by": None,
            }
        },
        upsert=True,
    )
    document = db[SYSTEM_SETTINGS_COLLECTION].find_one({"_id": SYSTEM_SETTINGS_ID})
    if document is None:
        raise RuntimeError("System settings could not be initialized.")

    normalized = _normalized_training(document.get("training"))
    if document.get("training") != normalized:
        db[SYSTEM_SETTINGS_COLLECTION].update_one(
            {"_id": SYSTEM_SETTINGS_ID},
            {"$set": {"training": normalized, "updated_at": now}},
        )
        document = {**document, "training": normalized, "updated_at": now}
    return document


def public_system_settings(document: dict[str, Any]) -> dict[str, Any]:
    training = _normalized_training(document.get("training"))
    return {
        "revision": int(document.get("revision") or 1),
        "training": training,
        "runtime": {
            "detected_cpu_count": _detected_cpu_count(),
            "configured_model_threads": int(training["model_threads"]),
            "configured_numeric_threads": int(training["numeric_threads"]),
            "winner_compute_locked": True,
        },
        "updated_at": bson_value(document.get("updated_at")),
        "updated_by": document.get("updated_by"),
    }


def get_system_settings(db: Any) -> dict[str, Any]:
    return public_system_settings(ensure_system_settings(db))


def update_system_settings(
    db: Any,
    payload: SystemSettingsUpdateRequest,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    previous = ensure_system_settings(db)
    current_revision = int(previous.get("revision") or 1)
    if payload.expected_revision != current_revision:
        raise SystemSettingsConflict(
            f"Expected revision {payload.expected_revision}, current revision {current_revision}."
        )

    training = _normalized_training(previous.get("training"))
    training.update(payload.training.model_dump(exclude_none=True))
    training = _normalized_training(training)
    now = utc_now()
    actor = str(actor_email or "").strip().lower() or None
    updated = db[SYSTEM_SETTINGS_COLLECTION].find_one_and_update(
        {"_id": SYSTEM_SETTINGS_ID, "revision": current_revision},
        {
            "$set": {
                "training": training,
                "updated_at": now,
                "updated_by": actor,
            },
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise SystemSettingsConflict("System settings changed before this update was applied.")

    db[SYSTEM_SETTINGS_HISTORY_COLLECTION].insert_one(
        bson_value(
            {
                "settings_id": SYSTEM_SETTINGS_ID,
                "previous_revision": current_revision,
                "revision": current_revision + 1,
                "reason": payload.reason.strip(),
                "updated_at": now,
                "updated_by": actor,
                "training": training,
            }
        )
    )
    return public_system_settings(updated)


def list_system_settings_history(db: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    cursor = (
        db[SYSTEM_SETTINGS_HISTORY_COLLECTION]
        .find({"settings_id": SYSTEM_SETTINGS_ID})
        .sort("updated_at", -1)
        .limit(max(1, min(int(limit), 200)))
    )
    return [
        {
            "previous_revision": int(item.get("previous_revision") or 1),
            "revision": int(item.get("revision") or 1),
            "reason": str(item.get("reason") or "Settings updated"),
            "updated_at": bson_value(item.get("updated_at")),
            "updated_by": item.get("updated_by"),
            "training": _normalized_training(item.get("training")),
        }
        for item in cursor
    ]


def apply_training_runtime_settings(
    db: Any,
    configuration: ConfigurationModel,
) -> ConfigurationModel:
    







    del db
    return type(configuration).model_validate(
        configuration.model_dump(mode="python")
    )
