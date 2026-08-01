from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Callable

from pydantic import BaseModel, ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    PARAMETER_BOOTSTRAP_RUNS_COLLECTION,
    PAPER_TRADING_SETTINGS_COLLECTION,
    SETTINGS_COLLECTION,
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
        configuration_name="xgboost-high-performance-cpu-seed-3042-v1.12.0",
        configuration_note=(
            "Bundled XGBoost-only CPU parameterization for the Alpaca paper-market runtime."
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
                    "message": "Document does not exist and can be inserted by the bootstrap API.",
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
    """Insert bundled documents only when their exact `_id` is absent.

    Existing documents are never replaced, merged, or repaired. This makes the
    operation safe to call repeatedly and prevents an HTTP request from
    silently changing a promoted strategy or live paper-market setting.
    """

    ensure_database(db)
    started_at = _utc_now()
    results: list[dict[str, Any]] = []

    for definition in DEFINITIONS:
        existing = db[definition.collection].find_one({"_id": definition.document_id})
        if existing is not None:
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
            continue

        payload = _validated_payload(definition)
        now = _utc_now()
        document = {
            "_id": definition.document_id,
            **payload,
            "created_at": now,
            "updated_at": now,
            "schema_version": definition.schema_version,
            "configuration_name": definition.configuration_name,
            "configuration_note": definition.configuration_note,
            "bootstrap_source": source,
        }

        # $setOnInsert keeps the operation atomic and idempotent if two callers
        # reach the endpoint at the same time.
        write = db[definition.collection].update_one(
            {"_id": definition.document_id},
            {"$setOnInsert": document},
            upsert=True,
        )
        inserted = write.upserted_id is not None
        if inserted:
            status = "inserted"
            valid = True
            message = "Bundled validated document was inserted."
        else:
            raced = db[definition.collection].find_one({"_id": definition.document_id})
            if raced is None:
                status = "missing"
                valid = False
                message = "Document was not inserted and could not be read after the write."
            else:
                valid, validation_message = _validate_existing(definition, raced)
                status = "skipped_existing_valid" if valid else "skipped_existing_invalid"
                message = (
                    "Another caller created the document first. " + validation_message
                )

        results.append(
            {
                "key": definition.key,
                "collection": definition.collection,
                "document_id": definition.document_id,
                "status": status,
                "valid": valid,
                "message": message,
            }
        )

    finished_at = _utc_now()
    summary = {
        "inserted": sum(item["status"] == "inserted" for item in results),
        "skipped_existing_valid": sum(
            item["status"] == "skipped_existing_valid" for item in results
        ),
        "skipped_existing_invalid": sum(
            item["status"] == "skipped_existing_invalid" for item in results
        ),
        "missing": sum(item["status"] == "missing" for item in results),
    }
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].insert_one(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "source": source,
            "mode": "insert_missing_documents_only",
            "summary": summary,
            "results": results,
        }
    )
    return {
        "mode": "insert_missing_documents_only",
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
        "results": results,
    }
