from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib import resources
from typing import Any

from pymongo import ReturnDocument

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    ROC_DECISION_POLICY_SETTINGS_COLLECTION as SETTINGS_COLLECTION,
    ROC_DECISION_POLICY_SETTINGS_HISTORY_COLLECTION as SETTINGS_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from .schemas import RocDecisionPolicySettings, RocDecisionPolicySettingsUpdateRequest

SETTINGS_ID = "roc-decision-policy"
SETTINGS_SCHEMA_VERSION = 1
PARAMETERIZATION_FILE = "004_roc_decision_policy.json"


class RocDecisionPolicySettingsConflict(RuntimeError):
    pass


def _seed_settings() -> dict[str, Any]:
    package = resources.files("market_cycle_trader_api.parameterizations")
    raw = json.loads(package.joinpath(PARAMETERIZATION_FILE).read_text(encoding="utf-8"))
    return RocDecisionPolicySettings.model_validate(raw).model_dump(mode="python")


def _validated_settings(raw: Any) -> dict[str, Any]:
    return RocDecisionPolicySettings.model_validate(raw).model_dump(mode="python")


def _settings_hash(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_settings(db: Any) -> dict[str, Any]:
    now = utc_now()
    seed = _seed_settings()
    db[SETTINGS_COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {"$setOnInsert": {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "revision": 1,
            **deepcopy(seed),
            "settings_hash": _settings_hash(seed),
            "created_at": now,
            "updated_at": now,
            "updated_by": None,
            "seeded_api_version": API_VERSION,
            "bootstrap_source": PARAMETERIZATION_FILE,
        }},
        upsert=True,
    )
    document = db[SETTINGS_COLLECTION].find_one({"_id": SETTINGS_ID})
    if document is None:
        raise RuntimeError("ROC Decision Policy settings could not be initialized.")
    raw = {key: document.get(key) for key in seed}
    settings = _validated_settings(raw)
    expected_hash = _settings_hash(settings)
    if raw != settings or document.get("settings_hash") != expected_hash:
        db[SETTINGS_COLLECTION].update_one(
            {"_id": SETTINGS_ID},
            {"$set": {**settings, "settings_hash": expected_hash, "updated_at": now}},
        )
        document = {**document, **settings, "settings_hash": expected_hash, "updated_at": now}
    return document


def settings_snapshot(db: Any) -> dict[str, Any]:
    document = ensure_settings(db)
    seed = _seed_settings()
    settings = _validated_settings({key: document.get(key) for key in seed})
    return {
        "settings_id": SETTINGS_ID,
        "schema_version": int(document.get("schema_version") or SETTINGS_SCHEMA_VERSION),
        "revision": int(document.get("revision") or 1),
        "settings_hash": str(document.get("settings_hash") or _settings_hash(settings)),
        "settings": settings,
    }


def get_settings(db: Any) -> dict[str, Any]:
    document = ensure_settings(db)
    return {
        **settings_snapshot(db),
        "updated_at": bson_value(document.get("updated_at")),
        "updated_by": document.get("updated_by"),
    }


def update_settings(db: Any, payload: RocDecisionPolicySettingsUpdateRequest, *, actor_email: str | None) -> dict[str, Any]:
    previous = ensure_settings(db)
    revision = int(previous.get("revision") or 1)
    if payload.expected_revision != revision:
        raise RocDecisionPolicySettingsConflict(f"Expected revision {payload.expected_revision}, current revision {revision}.")
    settings = _validated_settings(payload.settings.model_dump(mode="python"))
    now = utc_now()
    actor = str(actor_email or "").strip().lower() or None
    updated = db[SETTINGS_COLLECTION].find_one_and_update(
        {"_id": SETTINGS_ID, "revision": revision},
        {"$set": {
            **settings,
            "settings_hash": _settings_hash(settings),
            "updated_at": now,
            "updated_by": actor,
        }, "$inc": {"revision": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise RocDecisionPolicySettingsConflict("ROC Decision Policy settings changed before this update was applied.")
    db[SETTINGS_HISTORY_COLLECTION].insert_one(bson_value({
        "settings_id": SETTINGS_ID,
        "previous_revision": revision,
        "revision": revision + 1,
        "reason": payload.reason,
        "settings": settings,
        "settings_hash": _settings_hash(settings),
        "updated_at": now,
        "updated_by": actor,
    }))
    return get_settings(db)
