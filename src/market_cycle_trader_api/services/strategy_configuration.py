from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, TypeVar

from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel, ValidationError
from pymongo.database import Database

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    SETTINGS_SCHEMA_VERSION,
    STRATEGY_POLICY_COLLECTION,
    STRATEGY_POLICY_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestRequest
from ..schemas.strategy_policy import StrategyPolicy


class StrategyConfigurationError(RuntimeError):
    pass


class StrategyConfigurationConflict(StrategyConfigurationError):
    pass


class StrategyConfigurationNotFound(StrategyConfigurationError):
    pass


ModelT = TypeVar("ModelT", bound=BaseModel)


def _operational(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        bson_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(document.get("schema_version") or 0),
        "revision": int(document.get("revision") or 1),
        "configuration_name": str(document.get("configuration_name") or ""),
        "configuration_note": str(document.get("configuration_note") or ""),
        "bootstrap_source": str(document.get("bootstrap_source") or ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


def _public(document: dict[str, Any], validator: type[ModelT], field_name: str) -> dict[str, Any]:
    validated = validator.model_validate(_operational(document))
    payload = validated.model_dump(mode="json")
    return {
        field_name: payload,
        "configuration_hash": _hash(payload),
        "metadata": _metadata(document),
    }


def _assert_no_active_backtest(db: Database) -> None:
    active = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1, "status": 1},
    )
    if active is not None:
        raise StrategyConfigurationConflict(
            "Configuration cannot be changed while an analysis is active."
        )


def _archive(
    db: Database,
    collection: str,
    history_collection: str,
    document: dict[str, Any],
    *,
    source: str,
    note: str,
    change_type: str,
    changed_fields: list[str],
) -> str:
    copy = deepcopy(document)
    original_id = copy.pop("_id", None)
    result = db[history_collection].insert_one(
        {
            "captured_at": utc_now(),
            "source": source,
            "change_type": change_type,
            "note": note,
            "original_document_id": str(original_id),
            "original_revision": int(document.get("revision") or 1),
            "changed_fields": changed_fields,
            "document": bson_value(copy),
        }
    )
    return str(result.inserted_id)


def _changed_fields(previous: dict[str, Any], next_payload: dict[str, Any]) -> list[str]:
    before = _operational(previous)
    return sorted(
        key
        for key in set(before) | set(next_payload)
        if bson_value(before.get(key)) != bson_value(next_payload.get(key))
    )


def _replace(
    db: Database,
    *,
    collection: str,
    history_collection: str,
    document_id: str,
    validator: type[ModelT],
    payload: dict[str, Any],
    note: str,
    source: str,
    change_type: str,
    expected_revision: int | None,
    schema_version: int,
    response_field: str,
) -> dict[str, Any]:
    _assert_no_active_backtest(db)
    target = db[collection]
    previous = target.find_one({"_id": document_id})
    if previous is None:
        raise StrategyConfigurationNotFound("The active configuration does not exist.")
    current_revision = int(previous.get("revision") or 1)
    if expected_revision is not None and expected_revision != current_revision:
        raise StrategyConfigurationConflict(
            f"Expected revision {expected_revision}, current revision {current_revision}."
        )
    validated = validator.model_validate(payload)
    clean = bson_value(validated.model_dump(mode="python"))
    changed = _changed_fields(previous, clean)
    if not changed:
        response = _public(previous, validator, response_field)
        response.update({"status": "unchanged", "changed_fields": [], "history_id": None})
        return response
    history_id = _archive(
        db,
        collection,
        history_collection,
        previous,
        source=source,
        note=note,
        change_type=change_type,
        changed_fields=changed,
    )
    now = utc_now()
    next_revision = current_revision + 1
    stored = {
        "_id": document_id,
        **clean,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
        "schema_version": schema_version,
        "revision": next_revision,
        "configuration_name": previous.get("configuration_name") or "managed",
        "configuration_note": note,
        "bootstrap_source": source,
    }
    query: dict[str, Any] = {"_id": document_id}
    query["revision"] = current_revision if "revision" in previous else {"$exists": False}
    result = target.replace_one(query, stored, upsert=False)
    if result.matched_count != 1:
        raise StrategyConfigurationConflict("The configuration changed concurrently.")
    response = _public(stored, validator, response_field)
    response.update(
        {
            "status": "updated",
            "changed_fields": changed,
            "history_id": history_id,
        }
    )
    return response


def get_strategy_configuration(db: Database) -> dict[str, Any]:
    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise StrategyConfigurationNotFound("The active configuration does not exist.")
    return _public(document, BacktestRequest, "configuration")


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
        raise StrategyConfigurationNotFound("The active configuration does not exist.")
    merged = {**_operational(current), **changes}
    return _replace(
        db,
        collection=SETTINGS_COLLECTION,
        history_collection=SETTINGS_HISTORY_COLLECTION,
        document_id="default",
        validator=BacktestRequest,
        payload=merged,
        note=note,
        source=source,
        change_type="patch",
        expected_revision=expected_revision,
        schema_version=SETTINGS_SCHEMA_VERSION,
        response_field="configuration",
    )


def replace_strategy_configuration(
    db: Database,
    configuration: BacktestRequest,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _replace(
        db,
        collection=SETTINGS_COLLECTION,
        history_collection=SETTINGS_HISTORY_COLLECTION,
        document_id="default",
        validator=BacktestRequest,
        payload=configuration.model_dump(mode="python"),
        note=note,
        source=source,
        change_type="replace",
        expected_revision=expected_revision,
        schema_version=SETTINGS_SCHEMA_VERSION,
        response_field="configuration",
    )


def get_strategy_policy(db: Database) -> dict[str, Any]:
    document = db[STRATEGY_POLICY_COLLECTION].find_one({"_id": "active"})
    if document is None:
        raise StrategyConfigurationNotFound("The active runtime policy does not exist.")
    return _public(document, StrategyPolicy, "policy")


def replace_strategy_policy(
    db: Database,
    policy: StrategyPolicy,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _replace(
        db,
        collection=STRATEGY_POLICY_COLLECTION,
        history_collection=STRATEGY_POLICY_HISTORY_COLLECTION,
        document_id="active",
        validator=StrategyPolicy,
        payload=policy.model_dump(mode="python"),
        note=note,
        source=source,
        change_type="replace",
        expected_revision=expected_revision,
        schema_version=1,
        response_field="policy",
    )


def _history_items(
    db: Database,
    *,
    history_collection: str,
    validator: type[ModelT],
    response_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    cursor = db[history_collection].find({}).sort("captured_at", -1).limit(max(1, min(int(limit), 200)))
    items: list[dict[str, Any]] = []
    for record in cursor:
        stored = dict(record.get("document") or {})
        try:
            payload = validator.model_validate(_operational(stored)).model_dump(mode="json")
            valid = True
            error = None
        except ValidationError as exc:
            payload = bson_value(_operational(stored))
            valid = False
            error = str(exc)
        items.append(
            {
                "history_id": str(record.get("_id")),
                "captured_at": record.get("captured_at"),
                "source": record.get("source"),
                "change_type": record.get("change_type"),
                "note": record.get("note"),
                "original_revision": record.get("original_revision"),
                "changed_fields": list(record.get("changed_fields") or []),
                "valid_for_current_schema": valid,
                "validation_error": error,
                "configuration_hash": _hash(payload),
                response_field: payload,
            }
        )
    return items


def list_strategy_configuration_history(db: Database, *, limit: int) -> list[dict[str, Any]]:
    return _history_items(
        db,
        history_collection=SETTINGS_HISTORY_COLLECTION,
        validator=BacktestRequest,
        response_field="configuration",
        limit=limit,
    )


def list_strategy_policy_history(db: Database, *, limit: int) -> list[dict[str, Any]]:
    return _history_items(
        db,
        history_collection=STRATEGY_POLICY_HISTORY_COLLECTION,
        validator=StrategyPolicy,
        response_field="policy",
        limit=limit,
    )


def _restore(
    db: Database,
    *,
    history_collection: str,
    history_id: str,
    validator: type[ModelT],
    replace_func,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    try:
        object_id = ObjectId(history_id)
    except (InvalidId, TypeError) as exc:
        raise StrategyConfigurationNotFound("Invalid history id.") from exc
    record = db[history_collection].find_one({"_id": object_id})
    if record is None:
        raise StrategyConfigurationNotFound("The requested history entry does not exist.")
    validated = validator.model_validate(_operational(dict(record.get("document") or {})))
    return replace_func(
        db,
        validated,
        note=note,
        source=source,
        expected_revision=expected_revision,
    )


def restore_strategy_configuration(
    db: Database,
    history_id: str,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _restore(
        db,
        history_collection=SETTINGS_HISTORY_COLLECTION,
        history_id=history_id,
        validator=BacktestRequest,
        replace_func=replace_strategy_configuration,
        note=note,
        source=source,
        expected_revision=expected_revision,
    )


def restore_strategy_policy(
    db: Database,
    history_id: str,
    *,
    note: str,
    source: str,
    expected_revision: int | None,
) -> dict[str, Any]:
    return _restore(
        db,
        history_collection=STRATEGY_POLICY_HISTORY_COLLECTION,
        history_id=history_id,
        validator=StrategyPolicy,
        replace_func=replace_strategy_policy,
        note=note,
        source=source,
        expected_revision=expected_revision,
    )
