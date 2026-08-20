from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib import resources
from typing import Any

from pymongo import ReturnDocument

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_RESEARCH_SETTINGS_COLLECTION,
    TEMPORAL_RESEARCH_SETTINGS_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.temporal_research_settings import (
    TemporalResearchSettingsUpdateRequest,
    TemporalWinnerTransitionResearchSettings,
)

SETTINGS_ID = "winner-transition"
SETTINGS_SCHEMA_VERSION = 1
PARAMETERIZATION_FILE = "003_temporal_winner_transition_research.json"


class TemporalResearchSettingsConflict(RuntimeError):
    pass


def _seed_settings() -> dict[str, Any]:
    package = resources.files("market_cycle_trader_api.parameterizations")
    raw = json.loads(package.joinpath(PARAMETERIZATION_FILE).read_text(encoding="utf-8"))
    return TemporalWinnerTransitionResearchSettings.model_validate(raw).model_dump(mode="python")


def _validated_settings(raw: Any) -> dict[str, Any]:
    return TemporalWinnerTransitionResearchSettings.model_validate(raw).model_dump(mode="python")


def _settings_hash(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_temporal_research_settings(db: Any) -> dict[str, Any]:
    now = utc_now()
    seed = _seed_settings()
    db[TEMPORAL_RESEARCH_SETTINGS_COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {
            "$setOnInsert": {
                "schema_version": SETTINGS_SCHEMA_VERSION,
                "revision": 1,
                **deepcopy(seed),
                "settings_hash": _settings_hash(seed),
                "created_at": now,
                "updated_at": now,
                "updated_by": None,
                "seeded_api_version": API_VERSION,
                "bootstrap_source": PARAMETERIZATION_FILE,
            }
        },
        upsert=True,
    )
    document = db[TEMPORAL_RESEARCH_SETTINGS_COLLECTION].find_one({"_id": SETTINGS_ID})
    if document is None:
        raise RuntimeError("Temporal research settings could not be initialized.")
    raw_settings = {"risk": document.get("risk"), "confidence": document.get("confidence")}
    settings = _validated_settings(raw_settings)
    expected_hash = _settings_hash(settings)
    if raw_settings != settings or document.get("settings_hash") != expected_hash:
        db[TEMPORAL_RESEARCH_SETTINGS_COLLECTION].update_one(
            {"_id": SETTINGS_ID},
            {"$set": {**settings, "settings_hash": expected_hash, "updated_at": now}},
        )
        document = {**document, **settings, "settings_hash": expected_hash, "updated_at": now}
    return document


def temporal_research_settings_snapshot(db: Any) -> dict[str, Any]:
    document = ensure_temporal_research_settings(db)
    settings = _validated_settings({"risk": document.get("risk"), "confidence": document.get("confidence")})
    return {
        "settings_id": SETTINGS_ID,
        "schema_version": int(document.get("schema_version") or SETTINGS_SCHEMA_VERSION),
        "revision": int(document.get("revision") or 1),
        "settings_hash": str(document.get("settings_hash") or _settings_hash(settings)),
        "settings": settings,
    }


def get_temporal_research_settings(db: Any) -> dict[str, Any]:
    document = ensure_temporal_research_settings(db)
    snapshot = temporal_research_settings_snapshot(db)
    return {
        **snapshot,
        "updated_at": bson_value(document.get("updated_at")),
        "updated_by": document.get("updated_by"),
    }


def update_temporal_research_settings(
    db: Any,
    payload: TemporalResearchSettingsUpdateRequest,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    previous = ensure_temporal_research_settings(db)
    revision = int(previous.get("revision") or 1)
    if payload.expected_revision != revision:
        raise TemporalResearchSettingsConflict(
            f"Expected revision {payload.expected_revision}, current revision {revision}."
        )
    settings = _validated_settings({"risk": previous.get("risk"), "confidence": previous.get("confidence")})
    patch = payload.settings.model_dump(exclude_none=True, mode="python")
    for group, values in patch.items():
        settings[group] = deepcopy(values)
    settings = _validated_settings(settings)
    now = utc_now()
    actor = str(actor_email or "").strip().lower() or None
    updated = db[TEMPORAL_RESEARCH_SETTINGS_COLLECTION].find_one_and_update(
        {"_id": SETTINGS_ID, "revision": revision},
        {
            "$set": {
                **settings,
                "settings_hash": _settings_hash(settings),
                "updated_at": now,
                "updated_by": actor,
            },
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise TemporalResearchSettingsConflict(
            "Temporal research settings changed before this update was applied."
        )
    db[TEMPORAL_RESEARCH_SETTINGS_HISTORY_COLLECTION].insert_one(
        bson_value({
            "settings_id": SETTINGS_ID,
            "previous_revision": revision,
            "revision": revision + 1,
            "reason": payload.reason,
            "settings": settings,
            "settings_hash": _settings_hash(settings),
            "updated_at": now,
            "updated_by": actor,
        })
    )
    return get_temporal_research_settings(db)


def list_temporal_research_settings_history(db: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    cursor = (
        db[TEMPORAL_RESEARCH_SETTINGS_HISTORY_COLLECTION]
        .find({"settings_id": SETTINGS_ID}, {"_id": 0})
        .sort("updated_at", -1)
        .limit(max(1, min(int(limit), 200)))
    )
    return [bson_value(item) for item in cursor]
