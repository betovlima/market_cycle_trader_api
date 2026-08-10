from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import (
    ASSET_DISCOVERY_SETTINGS_COLLECTION,
    ASSET_DISCOVERY_SETTINGS_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.asset_discovery import AssetDiscoverySettingsUpdateRequest

EASTERN = ZoneInfo("America/New_York")
SETTINGS_ID = "default"
DEFAULT_SETTINGS: dict[str, Any] = {
    "automatic_enabled": False,
    "batch_size": 8,
    "schedule_hours_et": [18, 20, 22],
    "recheck_days": 30,
    "max_scan_attempts": 80,
    "min_price": 5.0,
    "min_median_dollar_volume": 5_000_000.0,
    "min_nonzero_volume_ratio": 0.95,
    "behavior_lookback_days": 1125,
    "behavior_lookback_sessions": 756,
    "behavior_min_sessions": 63,
    "behavior_max_downside_tail_1pct": 0.12,
    "behavior_max_gap_downside_tail_1pct": 0.10,
    "behavior_max_annualized_volatility": 0.90,
    "behavior_max_drawdown": 0.75,
    "behavior_max_single_day_loss": 0.30,
    "behavior_max_single_gap_loss": 0.25,
    "behavior_max_10_session_loss": 0.35,
}


class AssetDiscoveryConflict(RuntimeError):
    pass


def normalized_asset_discovery_settings(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    merged = {**DEFAULT_SETTINGS, **source}
    hours = sorted(set(int(hour) for hour in merged["schedule_hours_et"]))
    return {
        "automatic_enabled": bool(merged["automatic_enabled"]),
        "batch_size": max(1, min(50, int(merged["batch_size"]))),
        "schedule_hours_et": [hour for hour in hours if 0 <= hour <= 23]
        or deepcopy(DEFAULT_SETTINGS["schedule_hours_et"]),
        "recheck_days": max(1, min(365, int(merged["recheck_days"]))),
        "max_scan_attempts": max(8, min(500, int(merged["max_scan_attempts"]))),
        "min_price": max(0.5, float(merged["min_price"])),
        "min_median_dollar_volume": max(0.0, float(merged["min_median_dollar_volume"])),
        "min_nonzero_volume_ratio": max(
            0.0, min(1.0, float(merged["min_nonzero_volume_ratio"]))
        ),
        "behavior_lookback_days": max(365, min(3650, int(merged["behavior_lookback_days"]))),
        "behavior_lookback_sessions": max(63, min(2520, int(merged["behavior_lookback_sessions"]))),
        "behavior_min_sessions": max(20, min(252, int(merged["behavior_min_sessions"]))),
        "behavior_max_downside_tail_1pct": max(0.01, min(1.0, float(merged["behavior_max_downside_tail_1pct"]))),
        "behavior_max_gap_downside_tail_1pct": max(0.01, min(1.0, float(merged["behavior_max_gap_downside_tail_1pct"]))),
        "behavior_max_annualized_volatility": max(0.10, min(5.0, float(merged["behavior_max_annualized_volatility"]))),
        "behavior_max_drawdown": max(0.10, min(0.99, float(merged["behavior_max_drawdown"]))),
        "behavior_max_single_day_loss": max(0.05, min(0.99, float(merged["behavior_max_single_day_loss"]))),
        "behavior_max_single_gap_loss": max(0.05, min(0.99, float(merged["behavior_max_single_gap_loss"]))),
        "behavior_max_10_session_loss": max(0.10, min(0.99, float(merged["behavior_max_10_session_loss"]))),
    }


def ensure_asset_discovery_settings(db: Any) -> dict[str, Any]:
    now = utc_now()
    db[ASSET_DISCOVERY_SETTINGS_COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {
            "$setOnInsert": {
                "revision": 1,
                "settings": deepcopy(DEFAULT_SETTINGS),
                "created_at": now,
                "updated_at": now,
                "updated_by": None,
            }
        },
        upsert=True,
    )
    document = db[ASSET_DISCOVERY_SETTINGS_COLLECTION].find_one({"_id": SETTINGS_ID})
    if document is None:
        raise RuntimeError("Asset Discovery settings could not be initialized.")
    normalized = normalized_asset_discovery_settings(document.get("settings"))
    if document.get("settings") != normalized:
        db[ASSET_DISCOVERY_SETTINGS_COLLECTION].update_one(
            {"_id": SETTINGS_ID}, {"$set": {"settings": normalized, "updated_at": now}}
        )
        document = {**document, "settings": normalized, "updated_at": now}
    return document


def _next_scheduled_at(settings: dict[str, Any]) -> Any:
    if not settings["automatic_enabled"]:
        return None
    now_et = utc_now().astimezone(EASTERN)
    for day_offset in range(0, 8):
        candidate_day = (now_et + timedelta(days=day_offset)).date()
        if candidate_day.weekday() >= 5:
            continue
        for hour in settings["schedule_hours_et"]:
            candidate = now_et.replace(
                year=candidate_day.year,
                month=candidate_day.month,
                day=candidate_day.day,
                hour=int(hour),
                minute=0,
                second=0,
                microsecond=0,
            )
            if candidate > now_et:
                return candidate.astimezone(ZoneInfo("UTC"))
    return None


def public_asset_discovery_settings(document: dict[str, Any]) -> dict[str, Any]:
    settings = normalized_asset_discovery_settings(document.get("settings"))
    return {
        "revision": int(document.get("revision") or 1),
        "automatic_enabled": settings["automatic_enabled"],
        "batch_size": settings["batch_size"],
        "schedule_hours_et": settings["schedule_hours_et"],
        "recheck_days": settings["recheck_days"],
        "next_scheduled_at": bson_value(_next_scheduled_at(settings)),
        "updated_at": bson_value(document.get("updated_at")),
        "updated_by": document.get("updated_by"),
    }


def get_asset_discovery_settings(db: Any) -> dict[str, Any]:
    return public_asset_discovery_settings(ensure_asset_discovery_settings(db))


def update_asset_discovery_settings(
    db: Any,
    payload: AssetDiscoverySettingsUpdateRequest,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    previous = ensure_asset_discovery_settings(db)
    revision = int(previous.get("revision") or 1)
    if payload.expected_revision != revision:
        raise AssetDiscoveryConflict(
            f"Expected revision {payload.expected_revision}, current revision {revision}."
        )

    settings = normalized_asset_discovery_settings(previous.get("settings"))
    settings.update(payload.settings.model_dump(exclude_none=True))
    settings = normalized_asset_discovery_settings(settings)
    now = utc_now()
    actor = str(actor_email or "").strip().lower() or None
    updated = db[ASSET_DISCOVERY_SETTINGS_COLLECTION].find_one_and_update(
        {"_id": SETTINGS_ID, "revision": revision},
        {
            "$set": {"settings": settings, "updated_at": now, "updated_by": actor},
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise AssetDiscoveryConflict(
            "Asset Discovery settings changed before this update was applied."
        )
    db[ASSET_DISCOVERY_SETTINGS_HISTORY_COLLECTION].insert_one(
        bson_value(
            {
                "settings_id": SETTINGS_ID,
                "previous_revision": revision,
                "revision": revision + 1,
                "reason": payload.reason.strip(),
                "settings": settings,
                "updated_at": now,
                "updated_by": actor,
            }
        )
    )
    return public_asset_discovery_settings(updated)
