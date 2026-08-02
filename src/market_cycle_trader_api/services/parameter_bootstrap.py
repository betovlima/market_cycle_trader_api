from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    PARAMETER_BOOTSTRAP_RUNS_COLLECTION,
    PAPER_SETTINGS_METADATA_FIELDS,
    PAPER_TRADING_SETTINGS_COLLECTION,
    PAPER_TRADING_SETTINGS_HISTORY_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    bson_value,
    ensure_database,
    utc_now,
)
from ..schemas.paper_trading import PaperTradingSettings
from ..schemas.requests import BacktestRequest
from .strategy_configuration import replace_strategy_configuration


def _operational(
    document: dict[str, Any],
    metadata_fields: frozenset[str],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in metadata_fields
    }


def parameter_status(db: Database) -> dict[str, Any]:
    strategy = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    paper = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})

    strategy_valid = False
    strategy_error = None
    if strategy is not None:
        try:
            BacktestRequest.model_validate(
                _operational(strategy, SETTINGS_METADATA_FIELDS)
            )
            strategy_valid = True
        except ValidationError as exc:
            strategy_error = str(exc)

    paper_valid = False
    paper_error = None
    if paper is not None:
        try:
            PaperTradingSettings.model_validate(
                _operational(paper, PAPER_SETTINGS_METADATA_FIELDS)
            )
            paper_valid = True
        except ValidationError as exc:
            paper_error = str(exc)

    return {
        "strategy": {
            "exists": strategy is not None,
            "valid": strategy_valid,
            "revision": int((strategy or {}).get("revision") or 0),
            "schema_version": int((strategy or {}).get("schema_version") or 0),
            "validation_error": strategy_error,
        },
        "paper_trading": {
            "exists": paper is not None,
            "valid": paper_valid,
            "revision": int((paper or {}).get("revision") or 0),
            "schema_version": int((paper or {}).get("schema_version") or 0),
            "validation_error": paper_error,
        },
    }


def _apply_paper_settings(
    db: Database,
    configuration: PaperTradingSettings,
    *,
    replace_existing: bool,
    note: str,
    source: str,
) -> dict[str, Any]:
    collection = db[PAPER_TRADING_SETTINGS_COLLECTION]
    previous = collection.find_one({"_id": "default"})
    if previous is not None and not replace_existing:
        return {
            "status": "skipped_existing",
            "revision": int(previous.get("revision") or 1),
        }

    now = utc_now()
    current_revision = int((previous or {}).get("revision") or 0)
    if previous is not None:
        archived = deepcopy(previous)
        archived.pop("_id", None)
        db[PAPER_TRADING_SETTINGS_HISTORY_COLLECTION].insert_one(
            {
                "captured_at": now,
                "source": source,
                "note": note,
                "original_revision": current_revision,
                "document": bson_value(archived),
            }
        )

    document = {
        "_id": "default",
        **bson_value(configuration.model_dump(mode="python")),
        "created_at": (previous or {}).get("created_at") or now,
        "updated_at": now,
        "schema_version": 1,
        "revision": current_revision + 1,
        "configuration_name": "api-managed-paper-trading",
        "configuration_note": note,
        "bootstrap_source": source,
    }
    collection.replace_one({"_id": "default"}, document, upsert=True)
    return {
        "status": "created" if previous is None else "updated",
        "revision": current_revision + 1,
    }


def apply_parameter_documents(
    db: Database,
    *,
    strategy_configuration: BacktestRequest | None,
    paper_trading_configuration: PaperTradingSettings | None,
    replace_existing: bool,
    note: str,
    source: str,
) -> dict[str, Any]:
    ensure_database(db)
    started_at = utc_now()
    results: dict[str, Any] = {}

    if strategy_configuration is not None:
        existing = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        if existing is not None and not replace_existing:
            results["strategy"] = {
                "status": "skipped_existing",
                "revision": int(existing.get("revision") or 1),
            }
        else:
            results["strategy"] = replace_strategy_configuration(
                db,
                strategy_configuration,
                note=note,
                source=source,
                expected_revision=(
                    int(existing.get("revision") or 1)
                    if existing is not None
                    else None
                ),
                allow_create=True,
            )

    if paper_trading_configuration is not None:
        results["paper_trading"] = _apply_paper_settings(
            db,
            paper_trading_configuration,
            replace_existing=replace_existing,
            note=note,
            source=source,
        )

    finished_at = utc_now()
    audit = {
        "started_at": started_at,
        "finished_at": finished_at,
        "source": source,
        "replace_existing": replace_existing,
        "note": note,
        "results": bson_value(results),
    }
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].insert_one(audit)
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "replace_existing": replace_existing,
        "results": results,
        "status": parameter_status(db),
    }
