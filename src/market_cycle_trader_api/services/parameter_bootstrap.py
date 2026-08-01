from __future__ import annotations

from copy import deepcopy
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    PARAMETER_BOOTSTRAP_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PAPER_TRADING_SETTINGS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    SETTINGS_SCHEMA_VERSION,
    STRATEGY_POLICY_COLLECTION,
    STRATEGY_POLICY_HISTORY_COLLECTION,
    bson_value,
    ensure_database,
)
from ..schemas.paper_trading import PaperTradingSettings
from ..schemas.requests import BacktestRequest
from ..schemas.strategy_policy import StrategyPolicy

REMOVED_CONFIGURATION_FIELDS = frozenset({
    "rotation_seed_ensemble_enabled",
    "rotation_seed_ensemble_method",
    "rotation_seed_ensemble_min_agreement",
})

POLICY_SOURCE_FIELDS = frozenset(
    {
        "start_date",
        "end_date",
        "training_start_date",
        "training_end_date",
        "training_history_start",
        "training_history_end",
        "market_data_provider",
        "alpaca_historical_feed",
        "alpaca_live_feed",
        "historical_feed",
        "live_feed",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _operational(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def _policy_payload(source: dict[str, Any]) -> dict[str, Any] | None:
    payload = {
        "training_start_date": source.get("training_start_date")
        or source.get("training_history_start")
        or source.get("start_date"),
        "training_end_date": source.get("training_end_date")
        or source.get("training_history_end")
        or source.get("end_date"),
        "market_data_provider": source.get("market_data_provider"),
        "historical_feed": source.get("historical_feed")
        or source.get("alpaca_historical_feed"),
        "live_feed": source.get("live_feed") or source.get("alpaca_live_feed"),
    }
    if any(payload[key] is None for key in ("training_start_date", "market_data_provider", "historical_feed", "live_feed")):
        return None
    try:
        validated = StrategyPolicy.model_validate(payload)
    except ValidationError:
        return None
    return bson_value(validated.model_dump(mode="python"))



def _active_deployment_sources() -> list[dict[str, Any]]:
    domain = str(os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    token = str(os.getenv("PARAMETER_BOOTSTRAP_API_TOKEN") or "").strip()
    if not domain or not token:
        return []
    request = Request(
        f"https://{domain}/api/admin/strategy-configuration",
        headers={"X-Parameter-Bootstrap-Token": token},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    sources: list[dict[str, Any]] = []
    for key in ("system_rules", "policy", "configuration"):
        item = payload.get(key)
        if isinstance(item, dict):
            sources.append(item)
    return sources

def _candidate_sources(db: Database) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = _active_deployment_sources()
    current = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if current:
        candidates.append(_operational(current))
    for record in db[SETTINGS_HISTORY_COLLECTION].find({}).sort("captured_at", -1).limit(100):
        document = record.get("document")
        if isinstance(document, dict):
            candidates.append(_operational(document))
    for record in db[COMPARISONS_COLLECTION].find({}).sort("updated_at", -1).limit(50):
        document = record.get("effective_config")
        if isinstance(document, dict):
            candidates.append(document)
    for record in db[JOBS_COLLECTION].find({}).sort("created_at", -1).limit(50):
        document = record.get("request")
        if isinstance(document, dict):
            candidates.append(document)
    for record in db[PAPER_TRADE_PLANS_COLLECTION].find({}).sort("created_at", -1).limit(50):
        document = record.get("system_rules")
        if isinstance(document, dict):
            candidates.append(document)
    return candidates


def _archive(db: Database, collection: str, document: dict[str, Any], source: str, change_type: str) -> None:
    copy = deepcopy(document)
    original_id = copy.pop("_id", None)
    target = SETTINGS_HISTORY_COLLECTION if collection == SETTINGS_COLLECTION else STRATEGY_POLICY_HISTORY_COLLECTION
    db[target].insert_one(
        {
            "captured_at": _utc_now(),
            "source": source,
            "change_type": change_type,
            "original_document_id": str(original_id),
            "original_revision": int(document.get("revision") or 1),
            "document": bson_value(copy),
        }
    )


def _ensure_policy(db: Database, source: str) -> dict[str, Any]:
    collection = db[STRATEGY_POLICY_COLLECTION]
    existing = collection.find_one({"_id": "active"})
    if existing is not None:
        StrategyPolicy.model_validate(_operational(existing))
        return {"status": "preserved", "collection": STRATEGY_POLICY_COLLECTION, "valid": True}

    payload = None
    for candidate in _candidate_sources(db):
        payload = _policy_payload(candidate)
        if payload is not None:
            break
    if payload is None:
        raise RuntimeError("A complete strategy runtime policy is required in MongoDB.")

    now = _utc_now()
    collection.insert_one(
        {
            "_id": "active",
            **payload,
            "created_at": now,
            "updated_at": now,
            "schema_version": 1,
            "revision": 1,
            "bootstrap_source": source,
        }
    )
    return {"status": "migrated", "collection": STRATEGY_POLICY_COLLECTION, "valid": True}


def _ensure_strategy(db: Database, source: str) -> dict[str, Any]:
    collection = db[SETTINGS_COLLECTION]
    documents = list(collection.find({}))
    current = next((item for item in documents if item.get("_id") == "default"), None)
    if current is None:
        raise RuntimeError("The active strategy configuration is missing from MongoDB.")

    payload = _operational(current)
    cleaned = {
        key: value
        for key, value in payload.items()
        if key not in POLICY_SOURCE_FIELDS and key not in REMOVED_CONFIGURATION_FIELDS
    }
    validated = BacktestRequest.model_validate(cleaned)
    next_payload = bson_value(validated.model_dump(mode="python"))
    extras = [item for item in documents if item.get("_id") != "default"]
    changed = payload != next_payload or int(current.get("schema_version") or 0) != SETTINGS_SCHEMA_VERSION

    if changed:
        _archive(db, SETTINGS_COLLECTION, current, source, "schema_migration")
        now = _utc_now()
        collection.replace_one(
            {"_id": "default"},
            {
                "_id": "default",
                **next_payload,
                "created_at": current.get("created_at") or now,
                "updated_at": now,
                "schema_version": SETTINGS_SCHEMA_VERSION,
                "revision": int(current.get("revision") or 1) + 1,
                "configuration_name": current.get("configuration_name") or "managed",
                "configuration_note": current.get("configuration_note") or "",
                "bootstrap_source": source,
            },
        )

    for extra in extras:
        _archive(db, SETTINGS_COLLECTION, extra, source, "duplicate_cleanup")
        collection.delete_one({"_id": extra.get("_id")})

    return {
        "status": "migrated" if changed or extras else "preserved",
        "collection": SETTINGS_COLLECTION,
        "valid": True,
    }


def _ensure_paper_settings(db: Database) -> dict[str, Any]:
    document = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise RuntimeError("Paper-trading settings are missing from MongoDB.")
    PaperTradingSettings.model_validate(_operational(document))
    return {"status": "preserved", "collection": PAPER_TRADING_SETTINGS_COLLECTION, "valid": True}


def parameterization_status(db: Database) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    checks = (
        (SETTINGS_COLLECTION, "default", BacktestRequest),
        (STRATEGY_POLICY_COLLECTION, "active", StrategyPolicy),
        (PAPER_TRADING_SETTINGS_COLLECTION, "default", PaperTradingSettings),
    )
    for collection, document_id, validator in checks:
        document = db[collection].find_one({"_id": document_id})
        if document is None:
            results.append({"collection": collection, "document_id": document_id, "status": "missing", "valid": False})
            continue
        try:
            validator.model_validate(_operational(document))
        except ValidationError as exc:
            results.append({"collection": collection, "document_id": document_id, "status": "invalid", "valid": False, "message": str(exc)})
        else:
            results.append({"collection": collection, "document_id": document_id, "status": "valid", "valid": True})
    return results


def bootstrap_missing_parameterizations(db: Database, *, source: str) -> dict[str, Any]:
    ensure_database(db)
    started_at = _utc_now()
    results = [
        _ensure_policy(db, source),
        _ensure_strategy(db, source),
        _ensure_paper_settings(db),
    ]
    finished_at = _utc_now()
    summary = {
        "preserved": sum(item["status"] == "preserved" for item in results),
        "migrated": sum(item["status"] == "migrated" for item in results),
        "invalid": sum(not item.get("valid", False) for item in results),
    }
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].insert_one(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "source": source,
            "mode": "validate_and_migrate_existing_database_configuration",
            "summary": summary,
            "results": results,
        }
    )
    return {
        "mode": "validate_and_migrate_existing_database_configuration",
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
        "results": results,
    }
