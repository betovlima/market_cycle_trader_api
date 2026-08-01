from __future__ import annotations

import json
from copy import deepcopy
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
        configuration_name="xgboost-high-performance-cpu-seed-3042-v1.12.13",
        configuration_note=(
            "Bundled high-performance XGBoost-only CPU parameterization using Alpaca for all market data."
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



def _strategy_schema_migration(
    db: Database,
    definition: ParameterizationDefinition,
    existing: dict[str, Any],
    *,
    source: str,
) -> tuple[dict[str, Any], bool]:
    """Apply idempotent strategy migrations required by the current release.

    Schema v13 removes the retired Yahoo Finance settings and guarantees that
    historical bootstrap, cache refresh, backtests, diagnostics, and paper
    planning use Alpaca market data only. Existing strategy hyperparameters,
    dates, costs, assets, and promoted seeds remain unchanged.
    """

    if definition.collection != SETTINGS_COLLECTION:
        return existing, False

    current_schema = int(existing.get("schema_version") or 0)
    operational = _operational_document(existing)
    updates: dict[str, Any] = {}
    unset_fields: dict[str, str] = {}
    migration_notes: list[str] = []

    v11_fields_missing = any(
        field not in operational
        for field in ("deterministic_execution", "numeric_thread_limit")
    )
    if current_schema < 11 or v11_fields_missing:
        updates.update(
            {
                "xgb_n_jobs": -1,
                "deterministic_execution": False,
                "numeric_thread_limit": 1,
            }
        )
        migration_notes.append(
            "restored the promoted high-performance XGBoost execution policy"
        )

    alpaca_only_defaults = {
        "market_data_provider": "alpaca",
        "market_data_history_backfill_enabled": True,
        "market_data_history_backfill_provider": "alpaca",
        "market_data_history_start_tolerance_days": 10,
        "market_data_require_complete_history": True,
    }
    for field, value in alpaca_only_defaults.items():
        if current_schema < 13 or operational.get(field) != value:
            updates[field] = value

    retired_yahoo_fields = (
        "yfinance_auto_adjust",
        "yfinance_repair",
        "yfinance_timeout",
        "yfinance_fallback_period",
    )
    for field in retired_yahoo_fields:
        if field in operational:
            unset_fields[field] = ""

    if current_schema < 13 or updates or unset_fields:
        migration_notes.append(
            "removed Yahoo Finance settings and enabled Alpaca-only historical backfill"
        )

    if not updates and not unset_fields and current_schema >= SETTINGS_SCHEMA_VERSION:
        return existing, False

    now = _utc_now()
    history = deepcopy(existing)
    history.pop("_id", None)
    history.update(
        {
            "captured_at": now,
            "note": "Automatic v1.12.13 Alpaca-only strategy migration before Railway deployment.",
            "source": source,
        }
    )
    db[SETTINGS_HISTORY_COLLECTION].insert_one(history)

    previous_note = str(existing.get("configuration_note") or "").strip()
    migration_note = (
        "Automatic v1.12.13 migration: " + "; ".join(migration_notes) + "."
    )
    updates.update(
        {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "updated_at": now,
            "configuration_note": (
                f"{previous_note} {migration_note}".strip()
                if previous_note
                else migration_note
            ),
            "bootstrap_source": source,
        }
    )
    mongo_update: dict[str, Any] = {"$set": updates}
    if unset_fields:
        mongo_update["$unset"] = unset_fields
    db[SETTINGS_COLLECTION].update_one(
        {"_id": definition.document_id},
        mongo_update,
        upsert=False,
    )
    migrated = db[SETTINGS_COLLECTION].find_one({"_id": definition.document_id})
    if migrated is None:
        raise RuntimeError("Strategy configuration disappeared during schema migration.")
    definition.validator.model_validate(_operational_document(migrated))
    return migrated, True


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
    """Insert missing documents and apply versioned, idempotent migrations.

    Existing promoted strategy values are preserved. Versioned migrations restore
    the promoted execution policy when needed, enforce Alpaca-only complete-history backfill,
    and snapshot the previous document before changing affected fields.
    """

    ensure_database(db)
    started_at = _utc_now()
    results: list[dict[str, Any]] = []

    for definition in DEFINITIONS:
        existing = db[definition.collection].find_one({"_id": definition.document_id})
        if existing is not None:
            migrated_existing, migrated = _strategy_schema_migration(
                db, definition, existing, source=source
            )
            valid, message = _validate_existing(definition, migrated_existing)
            results.append(
                {
                    "key": definition.key,
                    "collection": definition.collection,
                    "document_id": definition.document_id,
                    "status": (
                        "migrated_existing"
                        if migrated and valid
                        else "skipped_existing_valid"
                        if valid
                        else "skipped_existing_invalid"
                    ),
                    "valid": valid,
                    "message": (
                        "Existing strategy document was safely migrated to schema v13 with Alpaca-only complete-history backfill."
                        if migrated and valid
                        else message
                    ),
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
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].insert_one(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "source": source,
            "mode": "insert_missing_and_safe_schema_migrations",
            "summary": summary,
            "results": results,
        }
    )
    return {
        "mode": "insert_missing_and_safe_schema_migrations",
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
        "results": results,
    }
