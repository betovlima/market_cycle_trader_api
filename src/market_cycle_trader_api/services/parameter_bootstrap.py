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
    SETTINGS_SCHEMA_VERSION,
    bson_value,
    ensure_database,
)
from ..schemas.paper_trading import PaperTradingSettings
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
        configuration_name="xgboost-high-performance-seed-3042-alpaca-sip-v1.12.14",
        configuration_note=(
            "Canonical locked XGBoost strategy. Alpaca SIP is used for historical "
            "training/backtests and Alpaca IEX is reserved for recent/live market data."
        ),
    ),
    ParameterizationDefinition(
        key="alpaca_paper_next_session",
        filename="002_alpaca_paper_next_session.json",
        collection=PAPER_TRADING_SETTINGS_COLLECTION,
        document_id="default",
        validator=PaperTradingSettings,
        schema_version=1,
        configuration_name="alpaca-paper-next-session-v1.12.0",
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
    raw = _load_raw(definition)
    validated = definition.validator.model_validate(raw)
    return bson_value(validated.model_dump(mode="python"))


def _operational_document(existing: dict[str, Any]) -> dict[str, Any]:
    metadata_fields = {
        "_id",
        "created_at",
        "updated_at",
        "schema_version",
        "configuration_name",
        "configuration_note",
        "bootstrap_source",
    }
    return {
        key: value
        for key, value in existing.items()
        if key not in metadata_fields
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


def _canonical_document(
    definition: ParameterizationDefinition,
    *,
    source: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "_id": definition.document_id,
        **_validated_payload(definition),
        "created_at": created_at or now,
        "updated_at": now,
        "schema_version": definition.schema_version,
        "configuration_name": definition.configuration_name,
        "configuration_note": definition.configuration_note,
        "bootstrap_source": source,
    }


def _strategy_is_canonical(
    definition: ParameterizationDefinition,
    existing_documents: list[dict[str, Any]],
) -> bool:
    if len(existing_documents) != 1:
        return False
    existing = existing_documents[0]
    if existing.get("_id") != definition.document_id:
        return False
    if int(existing.get("schema_version") or 0) != definition.schema_version:
        return False
    if str(existing.get("configuration_name") or "") != definition.configuration_name:
        return False
    try:
        validated_existing = definition.validator.model_validate(
            _operational_document(existing)
        )
    except ValidationError:
        return False
    existing_payload = bson_value(validated_existing.model_dump(mode="python"))
    return existing_payload == _validated_payload(definition)


def _archive_strategy_documents(
    db: Database,
    documents: list[dict[str, Any]],
    *,
    source: str,
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
                "note": (
                    "Automatic v1.12.14 canonical reset before Railway deployment. "
                    "The previous strategy document was archived before replacement."
                ),
                "original_document_id": str(original_id),
                "target_schema_version": SETTINGS_SCHEMA_VERSION,
                "document": bson_value(copy),
            }
        )
    db[SETTINGS_HISTORY_COLLECTION].insert_many(audit_documents, ordered=True)
    return len(audit_documents)


def _reset_strategy_parameterization(
    db: Database,
    definition: ParameterizationDefinition,
    *,
    source: str,
) -> dict[str, Any]:
    collection = db[definition.collection]
    existing_documents = list(collection.find({}))

    if _strategy_is_canonical(definition, existing_documents):
        return {
            "key": definition.key,
            "collection": definition.collection,
            "document_id": definition.document_id,
            "status": "skipped_existing_valid",
            "valid": True,
            "message": (
                "The single stored strategy document already matches the canonical "
                "v1.12.14 Alpaca SIP/IEX configuration."
            ),
        }

    archived = _archive_strategy_documents(db, existing_documents, source=source)
    original_created_at = None
    for document in existing_documents:
        if document.get("_id") == definition.document_id:
            original_created_at = document.get("created_at")
            break

    canonical = _canonical_document(
        definition,
        source=source,
        created_at=original_created_at,
    )

    # This collection contains exactly one locked operational strategy. A full
    # reset prevents old schema fields and extra strategy documents from being
    # accumulated across releases.
    collection.delete_many({})
    collection.insert_one(canonical)

    stored = collection.find_one({"_id": definition.document_id})
    if stored is None:
        raise RuntimeError("Canonical strategy configuration was not inserted.")
    definition.validator.model_validate(_operational_document(stored))

    return {
        "key": definition.key,
        "collection": definition.collection,
        "document_id": definition.document_id,
        "status": "migrated_existing" if existing_documents else "inserted",
        "valid": True,
        "message": (
            "Canonical strategy reset completed: archived "
            f"{archived} previous document(s), removed all old strategy documents, "
            "and inserted one validated schema v14 configuration using Alpaca SIP "
            "for historical data and IEX for live/recent data."
        ),
    }


def _insert_or_preserve_non_strategy_parameterization(
    db: Database,
    definition: ParameterizationDefinition,
    *,
    source: str,
) -> dict[str, Any]:
    collection = db[definition.collection]
    existing = collection.find_one({"_id": definition.document_id})
    if existing is not None:
        valid, message = _validate_existing(definition, existing)
        return {
            "key": definition.key,
            "collection": definition.collection,
            "document_id": definition.document_id,
            "status": "skipped_existing_valid" if valid else "skipped_existing_invalid",
            "valid": valid,
            "message": message,
        }

    document = _canonical_document(definition, source=source)
    collection.insert_one(document)
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
        if definition.collection == SETTINGS_COLLECTION:
            existing_documents = list(db[definition.collection].find({}))
            canonical = _strategy_is_canonical(definition, existing_documents)
            results.append(
                {
                    "key": definition.key,
                    "collection": definition.collection,
                    "document_id": definition.document_id,
                    "status": (
                        "skipped_existing_valid"
                        if canonical
                        else "skipped_existing_invalid"
                        if existing_documents
                        else "missing"
                    ),
                    "valid": canonical,
                    "message": (
                        "The canonical locked strategy is installed."
                        if canonical
                        else "The strategy collection does not match the canonical release configuration."
                        if existing_documents
                        else "The strategy collection is empty."
                    ),
                }
            )
            continue

        existing = db[definition.collection].find_one({"_id": definition.document_id})
        if existing is None:
            results.append(
                {
                    "key": definition.key,
                    "collection": definition.collection,
                    "document_id": definition.document_id,
                    "status": "missing",
                    "valid": False,
                    "message": "Document does not exist and can be inserted by bootstrap.",
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
    """Install the canonical strategy and preserve valid paper settings.

    The strategy collection is intentionally reset to one validated document on
    every release that changes the canonical configuration. The operation is
    idempotent: later deploys skip the reset when the document already matches.
    """

    ensure_database(db)
    started_at = _utc_now()
    results: list[dict[str, Any]] = []

    for definition in DEFINITIONS:
        if definition.collection == SETTINGS_COLLECTION:
            result = _reset_strategy_parameterization(
                db,
                definition,
                source=source,
            )
        else:
            result = _insert_or_preserve_non_strategy_parameterization(
                db,
                definition,
                source=source,
            )
        results.append(result)

    finished_at = _utc_now()
    summary = {
        "inserted": sum(item["status"] == "inserted" for item in results),
        "migrated_existing": sum(
            item["status"] == "migrated_existing" for item in results
        ),
        "skipped_existing_valid": sum(
            item["status"] == "skipped_existing_valid" for item in results
        ),
        "skipped_existing_invalid": sum(
            item["status"] == "skipped_existing_invalid" for item in results
        ),
        "missing": sum(item["status"] == "missing" for item in results),
    }
    mode = "canonical_strategy_reset_and_insert_missing"
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
