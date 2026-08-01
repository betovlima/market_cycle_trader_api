from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import Any

from pydantic import BaseModel, ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    PARAMETER_BOOTSTRAP_RUNS_COLLECTION,
    PAPER_TRADING_SETTINGS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    SETTINGS_SCHEMA_VERSION,
    bson_value,
    ensure_database,
)
from ..schemas.paper_trading import PaperTradingSettings
from ..core.system_rules import IMMUTABLE_STRATEGY_FIELDS
from ..schemas.requests import BacktestRequest

Validator = type[BaseModel]


@dataclass(frozen=True)
class ParameterizationDefinition:
    key: str
    filename: str
    collection: str
    document_id: str
    validator: Validator
    schema_version: int
    configuration_name: str
    configuration_note: str


DEFINITIONS: tuple[ParameterizationDefinition, ...] = (
    ParameterizationDefinition(
        key="xgboost_high_performance_seed_3042",
        filename="001_xgboost_high_performance_seed_3042.json",
        collection=SETTINGS_COLLECTION,
        document_id="default",
        validator=BacktestRequest,
        schema_version=SETTINGS_SCHEMA_VERSION,
        configuration_name="xgboost-api-managed-v1.12.16",
        configuration_note=(
            "Initial canonical XGBoost configuration. After installation, valid "
            "strategy parameters are managed through the protected administration API."
        ),
    ),
    ParameterizationDefinition(
        key="alpaca_paper_next_session",
        filename="002_alpaca_paper_next_session.json",
        collection=PAPER_TRADING_SETTINGS_COLLECTION,
        document_id="default",
        validator=PaperTradingSettings,
        schema_version=2,
        configuration_name="alpaca-paper-continuous-v1.12.16",
        configuration_note=(
            "Bundled paper-market execution settings for the next regular Alpaca session."
        ),
    ),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_raw(definition: ParameterizationDefinition) -> dict[str, Any]:
    package = resources.files("market_cycle_trader_api.parameterizations")
    text = package.joinpath(definition.filename).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Bundled parameterization {definition.filename} must contain one JSON object."
        )
    return raw


def _validated_payload(definition: ParameterizationDefinition) -> dict[str, Any]:
    validated = definition.validator.model_validate(_load_raw(definition))
    return bson_value(validated.model_dump(mode="python"))


def _operational_document(existing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in existing.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def _validate_existing(
    definition: ParameterizationDefinition,
    existing: dict[str, Any],
) -> tuple[bool, str]:
    try:
        definition.validator.model_validate(_operational_document(existing))
    except ValidationError as exc:
        return False, str(exc)
    return True, "Existing document is valid and was preserved unchanged."


def _new_document(
    definition: ParameterizationDefinition,
    *,
    source: str,
    created_at: datetime | None = None,
    revision: int = 1,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "_id": definition.document_id,
        **_validated_payload(definition),
        "created_at": created_at or now,
        "updated_at": now,
        "schema_version": definition.schema_version,
        "revision": max(1, int(revision)),
        "configuration_name": definition.configuration_name,
        "configuration_note": definition.configuration_note,
        "bootstrap_source": source,
    }


def _archive_documents(
    db: Database,
    documents: list[dict[str, Any]],
    *,
    source: str,
    note: str,
) -> int:
    if not documents:
        return 0
    captured_at = _utc_now()
    audit_documents: list[dict[str, Any]] = []
    for existing in documents:
        copy = deepcopy(existing)
        original_id = copy.pop("_id", None)
        audit_documents.append(
            {
                "captured_at": captured_at,
                "source": source,
                "change_type": "deployment_repair",
                "note": note,
                "original_document_id": str(original_id),
                "original_revision": int(existing.get("revision") or 1),
                "target_schema_version": SETTINGS_SCHEMA_VERSION,
                "changed_fields": [],
                "document": bson_value(copy),
            }
        )
    db[SETTINGS_HISTORY_COLLECTION].insert_many(audit_documents, ordered=True)
    return len(audit_documents)




def _migrated_strategy_payload(existing: dict[str, Any]) -> dict[str, Any]:
    payload = _operational_document(existing)
    for field in IMMUTABLE_STRATEGY_FIELDS:
        payload.pop(field, None)
    repetitions = int(payload.get("rotation_xgb_repetitions") or 1)
    payload.setdefault("rotation_seed_ensemble_enabled", repetitions >= 3)
    payload.setdefault("rotation_seed_ensemble_method", "majority_vote")
    payload.setdefault("rotation_seed_ensemble_min_agreement", 0.4)
    return payload


def _try_migrate_strategy_document(
    db: Database,
    definition: ParameterizationDefinition,
    existing: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    payload = _migrated_strategy_payload(existing)
    try:
        validated = BacktestRequest.model_validate(payload)
    except ValidationError:
        return None

    previous_payload = _operational_document(existing)
    next_payload = bson_value(validated.model_dump(mode="python"))
    if previous_payload == next_payload and int(existing.get("schema_version") or 0) == definition.schema_version:
        return None

    _archive_documents(
        db,
        [existing],
        source=source,
        note=(
            "Automatic schema v16 migration: fixed training/provider/feed rules were "
            "removed from editable MongoDB parameters and seed-ensemble fields were added."
        ),
    )
    now = _utc_now()
    document = {
        "_id": definition.document_id,
        **next_payload,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "schema_version": definition.schema_version,
        "revision": int(existing.get("revision") or 1) + 1,
        "configuration_name": "api-managed-xgboost-ensemble-v1.12.16",
        "configuration_note": (
            "Fixed 2016/SIP system rules with API-managed robust XGBoost ensemble parameters."
        ),
        "bootstrap_source": source,
    }
    db[definition.collection].replace_one({"_id": definition.document_id}, document)
    return {
        "key": definition.key,
        "collection": definition.collection,
        "document_id": definition.document_id,
        "status": "migrated_existing",
        "valid": True,
        "message": (
            "Existing strategy was migrated automatically to schema v16. Training starts "
            "at 2016-01-01 and Alpaca SIP/IEX rules are now immutable system rules."
        ),
    }

def _bootstrap_strategy(
    db: Database,
    definition: ParameterizationDefinition,
    *,
    source: str,
) -> dict[str, Any]:
    collection = db[definition.collection]
    existing_documents = list(collection.find({}))
    default = next(
        (document for document in existing_documents if document.get("_id") == definition.document_id),
        None,
    )
    extras = [
        document
        for document in existing_documents
        if document.get("_id") != definition.document_id
    ]

    if default is None:
        if existing_documents:
            _archive_documents(
                db,
                existing_documents,
                source=source,
                note=(
                    "No default strategy document existed. Old strategy documents were "
                    "archived and replaced by the validated initial configuration."
                ),
            )
            collection.delete_many({})
        collection.insert_one(_new_document(definition, source=source))
        return {
            "key": definition.key,
            "collection": definition.collection,
            "document_id": definition.document_id,
            "status": "inserted" if not existing_documents else "repaired_invalid",
            "valid": True,
            "message": (
                "One validated default strategy configuration was installed. Future "
                "parameter changes are preserved and managed through the administration API."
            ),
        }

    migration_result = _try_migrate_strategy_document(
        db, definition, default, source=source
    )
    if migration_result is not None:
        default = collection.find_one({"_id": definition.document_id}) or default
        if extras:
            _archive_documents(
                db,
                extras,
                source=source,
                note="Extra strategy documents were archived during the schema v16 migration.",
            )
            for document in extras:
                collection.delete_one({"_id": document.get("_id")})
        return migration_result

    valid, validation_message = _validate_existing(definition, default)
    if not valid:
        _archive_documents(
            db,
            existing_documents,
            source=source,
            note=(
                "The stored strategy did not match the current BacktestRequest schema. "
                "It was archived and replaced automatically during deployment."
            ),
        )
        created_at = default.get("created_at")
        revision = int(default.get("revision") or 1) + 1
        collection.delete_many({})
        collection.insert_one(
            _new_document(
                definition,
                source=source,
                created_at=created_at,
                revision=revision,
            )
        )
        return {
            "key": definition.key,
            "collection": definition.collection,
            "document_id": definition.document_id,
            "status": "repaired_invalid",
            "valid": True,
            "message": (
                "The invalid stored strategy was archived and replaced automatically "
                "by the validated initial configuration."
            ),
        }

    migrated = False
    messages: list[str] = []

    if extras:
        _archive_documents(
            db,
            extras,
            source=source,
            note=(
                "Extra strategy documents were archived and removed. The valid default "
                "API-managed strategy was preserved."
            ),
        )
        for document in extras:
            collection.delete_one({"_id": document.get("_id")})
        migrated = True
        messages.append(f"Archived and removed {len(extras)} extra strategy document(s).")

    metadata_updates: dict[str, Any] = {}
    if int(default.get("schema_version") or 0) != definition.schema_version:
        metadata_updates["schema_version"] = definition.schema_version
    if not default.get("revision"):
        metadata_updates["revision"] = 1
    if not default.get("configuration_name"):
        metadata_updates["configuration_name"] = "api-managed-xgboost-strategy"
    if not default.get("configuration_note"):
        metadata_updates["configuration_note"] = (
            "Valid strategy preserved during deployment; managed through the administration API."
        )
    if metadata_updates:
        metadata_updates["updated_at"] = _utc_now()
        metadata_updates["bootstrap_source"] = source
        collection.update_one({"_id": definition.document_id}, {"$set": metadata_updates})
        migrated = True
        messages.append("Updated strategy metadata without changing operational parameters.")

    return {
        "key": definition.key,
        "collection": definition.collection,
        "document_id": definition.document_id,
        "status": "migrated_existing" if migrated else "skipped_existing_valid",
        "valid": True,
        "message": " ".join(messages) if messages else validation_message,
    }


def _bootstrap_non_strategy(
    db: Database,
    definition: ParameterizationDefinition,
    *,
    source: str,
) -> dict[str, Any]:
    collection = db[definition.collection]
    existing = collection.find_one({"_id": definition.document_id})
    if existing is not None:
        valid, message = _validate_existing(definition, existing)
        if valid:
            return {
                "key": definition.key,
                "collection": definition.collection,
                "document_id": definition.document_id,
                "status": "skipped_existing_valid",
                "valid": True,
                "message": message,
            }
        if definition.collection == PAPER_TRADING_SETTINGS_COLLECTION:
            payload = _operational_document(existing)
            payload.setdefault("automatic_continuation_enabled", True)
            payload.setdefault("scheduler_poll_seconds", 10.0)
            payload.setdefault("preparation_retry_seconds", 60.0)
            try:
                validated = PaperTradingSettings.model_validate(payload)
            except ValidationError:
                return {
                    "key": definition.key,
                    "collection": definition.collection,
                    "document_id": definition.document_id,
                    "status": "skipped_existing_invalid",
                    "valid": False,
                    "message": message,
                }
            now = _utc_now()
            collection.replace_one(
                {"_id": definition.document_id},
                {
                    "_id": definition.document_id,
                    **bson_value(validated.model_dump(mode="python")),
                    "created_at": existing.get("created_at") or now,
                    "updated_at": now,
                    "schema_version": definition.schema_version,
                    "revision": int(existing.get("revision") or 1) + 1,
                    "configuration_name": definition.configuration_name,
                    "configuration_note": definition.configuration_note,
                    "bootstrap_source": source,
                },
            )
            return {
                "key": definition.key,
                "collection": definition.collection,
                "document_id": definition.document_id,
                "status": "migrated_existing",
                "valid": True,
                "message": (
                    "Paper settings were migrated automatically; completed runs will "
                    "continue by arming the next Alpaca regular session."
                ),
            }
        return {
            "key": definition.key,
            "collection": definition.collection,
            "document_id": definition.document_id,
            "status": "skipped_existing_invalid",
            "valid": False,
            "message": message,
        }

    collection.insert_one(_new_document(definition, source=source))
    return {
        "key": definition.key,
        "collection": definition.collection,
        "document_id": definition.document_id,
        "status": "inserted",
        "valid": True,
        "message": "Bundled validated document was inserted.",
    }


def parameterization_status(db: Database) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for definition in DEFINITIONS:
        existing = db[definition.collection].find_one({"_id": definition.document_id})
        if existing is None:
            results.append(
                {
                    "key": definition.key,
                    "collection": definition.collection,
                    "document_id": definition.document_id,
                    "status": "missing",
                    "valid": False,
                    "message": "Document does not exist and will be inserted during deployment.",
                }
            )
            continue
        valid, message = _validate_existing(definition, existing)
        results.append(
            {
                "key": definition.key,
                "collection": definition.collection,
                "document_id": definition.document_id,
                "status": "skipped_existing_valid" if valid else "skipped_existing_invalid",
                "valid": valid,
                "message": message,
            }
        )
    return results


def bootstrap_missing_parameterizations(
    db: Database,
    *,
    source: str,
) -> dict[str, Any]:
    """Create missing documents, repair invalid schemas, and preserve valid API changes."""

    ensure_database(db)
    started_at = _utc_now()
    results: list[dict[str, Any]] = []

    for definition in DEFINITIONS:
        if definition.collection == SETTINGS_COLLECTION:
            result = _bootstrap_strategy(db, definition, source=source)
        else:
            result = _bootstrap_non_strategy(db, definition, source=source)
        results.append(result)

    finished_at = _utc_now()
    summary = {
        "inserted": sum(item["status"] == "inserted" for item in results),
        "migrated_existing": sum(item["status"] == "migrated_existing" for item in results),
        "repaired_invalid": sum(item["status"] == "repaired_invalid" for item in results),
        "skipped_existing_valid": sum(
            item["status"] == "skipped_existing_valid" for item in results
        ),
        "skipped_existing_invalid": sum(
            item["status"] == "skipped_existing_invalid" for item in results
        ),
        "missing": sum(item["status"] == "missing" for item in results),
    }
    mode = "insert_missing_repair_invalid_preserve_valid_api_configuration"
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].insert_one(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "source": source,
            "mode": mode,
            "summary": summary,
            "results": results,
        }
    )
    return {
        "mode": mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
        "results": results,
    }
