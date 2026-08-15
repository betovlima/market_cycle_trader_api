from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from importlib import resources
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pydantic import ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    SETTINGS_SCHEMA_VERSION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestRequest

WINNER_PARAMETERIZATION = "winner-v1.13.2.json"
WINNER_CONFIGURATION_SHA256 = (
    "22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953"
)
WINNER_ACTIVE_PAPER_KEY = "alpaca-paper-next-session"


class StrategyConfigurationError(RuntimeError):
    pass


class StrategyConfigurationConflict(StrategyConfigurationError):
    pass


class StrategyConfigurationNotFound(StrategyConfigurationError):
    pass


def _operational_document(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def _configuration_hash(configuration: dict[str, Any]) -> str:
    canonical = dict(configuration)
    mode = str(canonical.get("strategy_mode") or "")
    if mode != "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE":
        canonical.pop("opportunity_utility_entry_threshold", None)
        canonical.pop("opportunity_utility_exit_threshold", None)
    if mode not in {"COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION", "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION", "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY"}:
        for field in (
            "allocation_lookback_days",
            "allocation_max_asset_weight",
            "allocation_cvar_confidence",
            "allocation_cvar_penalty",
            "allocation_turnover_penalty",
            "allocation_minimum_utility",
            "allocation_signal_scale",
        ):
            canonical.pop(field, None)
    encoded = json.dumps(
        bson_value(canonical),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _winner_configuration() -> tuple[BacktestRequest, str]:
    package = resources.files("market_cycle_trader_api.parameterizations")
    raw = json.loads(
        package.joinpath(WINNER_PARAMETERIZATION).read_text(encoding="utf-8")
    )
    if not isinstance(raw, dict):
        raise StrategyConfigurationError(
            f"Bundled winner file {WINNER_PARAMETERIZATION} must contain one JSON object."
        )

    validated = BacktestRequest.model_validate(raw)
    actual_hash = _configuration_hash(validated.model_dump(mode="json"))
    if actual_hash != WINNER_CONFIGURATION_SHA256:
        raise StrategyConfigurationError(
            "The bundled winner strategy does not match the validated winner-v1.13.2 "
            f"configuration hash. Expected {WINNER_CONFIGURATION_SHA256}, "
            f"received {actual_hash}."
        )
    return validated, actual_hash


def _metadata(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(document.get("schema_version") or 0),
        "revision": int(document.get("revision") or 1),
        "configuration_name": str(document.get("configuration_name") or ""),
        "configuration_note": str(document.get("configuration_note") or ""),
        "bootstrap_source": str(document.get("bootstrap_source") or ""),
        "winner_source_file": str(document.get("winner_source_file") or ""),
        "winner_configuration_hash": str(
            document.get("winner_configuration_hash") or ""
        ),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


def _public_configuration(document: dict[str, Any]) -> dict[str, Any]:
    validated = BacktestRequest.model_validate(_operational_document(document))
    configuration = validated.model_dump(mode="json")
    return {
        "configuration": configuration,
        "configuration_hash": _configuration_hash(configuration),
        "metadata": _metadata(document),
    }


def get_strategy_configuration(db: Database) -> dict[str, Any]:
    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise StrategyConfigurationNotFound(
            "The active strategy configuration does not exist."
        )
    return _public_configuration(document)


def _assert_no_active_backtest(db: Database) -> None:
    active = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1, "status": 1},
    )
    if active is not None:
        raise StrategyConfigurationConflict(
            "Strategy parameters cannot be changed while a backtest is queued or running. "
            f"Active job: {active.get('id', 'unknown')} ({active.get('status', 'unknown')})."
        )


def _assert_no_active_strategy_execution(db: Database) -> None:
    _assert_no_active_backtest(db)
    active_paper = db[PAPER_MARKET_RUNS_COLLECTION].find_one(
        {"active_key": WINNER_ACTIVE_PAPER_KEY},
        {"_id": 0, "run_id": 1, "status": 1},
    )
    if active_paper is not None:
        raise StrategyConfigurationConflict(
            "The winner strategy cannot be installed while a Paper run is active. "
            f"Active run: {active_paper.get('run_id', 'unknown')} "
            f"({active_paper.get('status', 'unknown')})."
        )


def _canonical_configuration() -> BacktestRequest:
    configuration, _ = _winner_configuration()
    return configuration


def _archive_previous(
    db: Database,
    document: dict[str, Any],
    *,
    source: str,
    note: str,
    change_type: str,
    changed_fields: list[str],
) -> str:
    archived = deepcopy(document)
    original_id = archived.pop("_id", None)
    result = db[SETTINGS_HISTORY_COLLECTION].insert_one(
        {
            "captured_at": utc_now(),
            "source": source,
            "change_type": change_type,
            "note": note,
            "original_document_id": str(original_id),
            "original_revision": int(document.get("revision") or 1),
            "target_schema_version": SETTINGS_SCHEMA_VERSION,
            "changed_fields": changed_fields,
            "document": bson_value(archived),
        }
    )
    return str(result.inserted_id)


def _changed_fields(
    previous: dict[str, Any],
    next_configuration: dict[str, Any],
) -> list[str]:
    previous_configuration = _operational_document(previous)
    return sorted(
        key
        for key in set(previous_configuration) | set(next_configuration)
        if bson_value(previous_configuration.get(key))
        != bson_value(next_configuration.get(key))
    )


def _replace_configuration(
    db: Database,
    configuration: BacktestRequest,
    *,
    note: str,
    source: str,
    change_type: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    _assert_no_active_backtest(db)
    collection = db[SETTINGS_COLLECTION]
    previous = collection.find_one({"_id": "default"})
    if previous is None:
        raise StrategyConfigurationNotFound(
            "The active strategy configuration does not exist."
        )

    current_revision = int(previous.get("revision") or 1)
    if expected_revision is not None and expected_revision != current_revision:
        raise StrategyConfigurationConflict(
            "The strategy configuration changed after it was read. "
            f"Expected revision {expected_revision}, current revision {current_revision}."
        )

    payload = bson_value(configuration.model_dump(mode="python"))
    changed_fields = _changed_fields(previous, payload)
    if not changed_fields:
        result = _public_configuration(previous)
        result.update(
            {
                "status": "unchanged",
                "changed_fields": [],
                "history_id": None,
                "message": "The supplied configuration matches the active configuration.",
            }
        )
        return result

    now = utc_now()
    next_revision = current_revision + 1
    document = {
        "_id": "default",
        **payload,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "revision": next_revision,
        "configuration_name": "api-managed-xgboost-strategy",
        "configuration_note": note,
        "bootstrap_source": source,
    }

    history_id = _archive_previous(
        db,
        previous,
        source=source,
        note=note,
        change_type=change_type,
        changed_fields=changed_fields,
    )

    query: dict[str, Any] = {"_id": "default"}
    if "revision" in previous:
        query["revision"] = current_revision
    else:
        query["revision"] = {"$exists": False}

    result = collection.replace_one(query, document, upsert=False)
    if result.matched_count != 1:
        raise StrategyConfigurationConflict(
            "The strategy configuration was updated concurrently. Read it again and retry."
        )

    stored = collection.find_one({"_id": "default"})
    if stored is None:
        raise StrategyConfigurationError(
            "The updated strategy configuration could not be read from MongoDB."
        )
    BacktestRequest.model_validate(_operational_document(stored))

    response = _public_configuration(stored)
    response.update(
        {
            "status": "updated",
            "changed_fields": changed_fields,
            "history_id": history_id,
            "message": (
                f"Strategy configuration updated to revision {next_revision}. "
                "New backtests and future paper plans will use the new parameters."
            ),
        }
    )
    return response


def patch_strategy_configuration(
    db: Database,
    changes: dict[str, Any],
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    current = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if current is None:
        raise StrategyConfigurationNotFound(
            "The active strategy configuration does not exist."
        )
    merged = {**_operational_document(current), **changes}
    validated = BacktestRequest.model_validate(merged)
    return _replace_configuration(
        db,
        validated,
        note=note,
        source=source,
        change_type="patch",
        expected_revision=expected_revision,
    )


def replace_strategy_configuration(
    db: Database,
    configuration: BacktestRequest,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _replace_configuration(
        db,
        configuration,
        note=note,
        source=source,
        change_type="replace",
        expected_revision=expected_revision,
    )


def reset_strategy_configuration(
    db: Database,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _replace_configuration(
        db,
        _canonical_configuration(),
        note=note,
        source=source,
        change_type="canonical_reset",
        expected_revision=expected_revision,
    )


def install_winner_strategy_configuration(
    db: Database,
    *,
    note: str,
    source: str,
) -> dict[str, Any]:
    

    _assert_no_active_strategy_execution(db)
    configuration, winner_hash = _winner_configuration()
    payload = bson_value(configuration.model_dump(mode="python"))
    collection = db[SETTINGS_COLLECTION]
    history_collection = db[SETTINGS_HISTORY_COLLECTION]
    previous_default = collection.find_one({"_id": "default"})
    now = utc_now()
    document = {
        "_id": "default",
        **payload,
        "created_at": now,
        "updated_at": now,
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "revision": 1,
        "configuration_name": "winner-v1.13.2",
        "configuration_note": note,
        "bootstrap_source": source,
        "winner_source_file": WINNER_PARAMETERIZATION,
        "winner_configuration_hash": winner_hash,
    }

    result = collection.replace_one({"_id": "default"}, document, upsert=True)
    if previous_default is not None and result.matched_count != 1:
        raise StrategyConfigurationConflict(
            "The active strategy configuration changed while winner-v1.13.2 was being installed."
        )

    extras = collection.delete_many({"_id": {"$ne": "default"}})
    history = history_collection.delete_many({})

    stored = collection.find_one({"_id": "default"})
    if stored is None:
        raise StrategyConfigurationError(
            "winner-v1.13.2 was written but could not be read back from MongoDB."
        )
    validated = BacktestRequest.model_validate(_operational_document(stored))
    stored_hash = _configuration_hash(validated.model_dump(mode="json"))
    if stored_hash != WINNER_CONFIGURATION_SHA256:
        raise StrategyConfigurationError(
            "The stored strategy does not match the validated winner-v1.13.2 hash."
        )

    from .strategy_lab import synchronize_bundled_winner_installation

    synchronize_bundled_winner_installation(
        db,
        validated,
        note=note,
        source=source,
    )

    response = _public_configuration(stored)
    response.update(
        {
            "status": "winner_installed",
            "source_file": WINNER_PARAMETERIZATION,
            "expected_configuration_hash": WINNER_CONFIGURATION_SHA256,
            "replaced_previous_default": previous_default is not None,
            "deleted_extra_strategy_documents": int(extras.deleted_count),
            "deleted_strategy_history_documents": int(history.deleted_count),
            "message": (
                "Old strategy configuration data was removed and winner-v1.13.2 "
                "was installed as revision 1. Backtest results, market bars, and Paper "
                "execution data were not deleted."
            ),
        }
    )
    return response


def list_strategy_configuration_history(
    db: Database,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    cursor = db[SETTINGS_HISTORY_COLLECTION].find({}).sort("captured_at", -1).limit(
        max(1, min(int(limit), 200))
    )
    items: list[dict[str, Any]] = []
    for record in cursor:
        stored_document = dict(record.get("document") or {})
        try:
            operational = BacktestRequest.model_validate(
                _operational_document(stored_document)
            ).model_dump(mode="json")
            valid = True
            validation_error = None
        except ValidationError as exc:
            operational = bson_value(_operational_document(stored_document))
            valid = False
            validation_error = str(exc)

        items.append(
            {
                "history_id": str(record.get("_id")),
                "captured_at": record.get("captured_at"),
                "source": record.get("source"),
                "change_type": record.get("change_type"),
                "note": record.get("note"),
                "original_document_id": record.get("original_document_id"),
                "original_revision": record.get("original_revision"),
                "changed_fields": list(record.get("changed_fields") or []),
                "valid_for_current_schema": valid,
                "validation_error": validation_error,
                "configuration_hash": _configuration_hash(operational),
                "configuration": operational,
            }
        )
    return items


def restore_strategy_configuration(
    db: Database,
    history_id: str,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    try:
        object_id = ObjectId(history_id)
    except (InvalidId, TypeError) as exc:
        raise StrategyConfigurationNotFound("Invalid strategy history id.") from exc

    record = db[SETTINGS_HISTORY_COLLECTION].find_one({"_id": object_id})
    if record is None:
        raise StrategyConfigurationNotFound(
            "The requested strategy history entry does not exist."
        )
    configuration = BacktestRequest.model_validate(
        _operational_document(dict(record.get("document") or {}))
    )
    return _replace_configuration(
        db,
        configuration,
        note=note,
        source=source,
        change_type="history_restore",
        expected_revision=expected_revision,
    )
