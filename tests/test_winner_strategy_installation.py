from __future__ import annotations

import copy
from typing import Any

import pytest

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
)
from market_cycle_trader_api.services.strategy_configuration import (
    WINNER_CONFIGURATION_SHA256,
    WINNER_PARAMETERIZATION,
    StrategyConfigurationConflict,
    _winner_configuration,
    install_winner_strategy_configuration,
)


class _ReplaceResult:
    def __init__(self, *, matched_count: int, upserted_id: str | None = None) -> None:
        self.matched_count = matched_count
        self.modified_count = matched_count
        self.upserted_id = upserted_id


class _DeleteResult:
    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


class _Collection:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}

    def find_one(self, query: dict[str, Any], *args, **kwargs):
        if "_id" in query:
            value = self.documents.get(str(query["_id"]))
            return copy.deepcopy(value) if value is not None else None

        if "status" in query:
            allowed = set(query["status"].get("$in", []))
            for document in self.documents.values():
                if document.get("status") in allowed:
                    return copy.deepcopy(document)
            return None

        if "active_key" in query:
            expected = query["active_key"]
            for document in self.documents.values():
                if document.get("active_key") == expected:
                    return copy.deepcopy(document)
            return None

        if not query:
            value = next(iter(self.documents.values()), None)
            return copy.deepcopy(value) if value is not None else None
        raise AssertionError(f"Unexpected query: {query}")

    def replace_one(
        self,
        query: dict[str, Any],
        document: dict[str, Any],
        *,
        upsert: bool,
    ) -> _ReplaceResult:
        document_id = str(query["_id"])
        existed = document_id in self.documents
        self.documents[document_id] = copy.deepcopy(document)
        return _ReplaceResult(
            matched_count=1 if existed else 0,
            upserted_id=None if existed else document_id,
        )

    def delete_many(self, query: dict[str, Any]) -> _DeleteResult:
        if not query:
            count = len(self.documents)
            self.documents.clear()
            return _DeleteResult(count)
        if query == {"_id": {"$ne": "default"}}:
            ids = [document_id for document_id in self.documents if document_id != "default"]
            for document_id in ids:
                self.documents.pop(document_id, None)
            return _DeleteResult(len(ids))
        raise AssertionError(f"Unexpected delete query: {query}")


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())


def test_bundled_winner_file_has_the_validated_configuration_hash() -> None:
    configuration, configuration_hash = _winner_configuration()

    assert WINNER_PARAMETERIZATION == "winner-v1.13.2.json"
    assert configuration_hash == WINNER_CONFIGURATION_SHA256
    assert configuration.strategy_mode == "COMPOUND_ROTATION_SWING_XGBOOST"
    assert configuration.rotation_accelerator == "cpu"
    assert configuration.random_state == 42
    assert configuration.rotation_target_horizons == [5, 10, 20, 40, 60]


def test_install_winner_removes_old_strategy_data_and_resets_revision() -> None:
    db = _Database()
    winner, _ = _winner_configuration()
    previous = winner.model_dump(mode="python")
    previous["rotation_xgb_n_estimators"] = 450
    previous.update(
        {
            "_id": "default",
            "revision": 8,
            "schema_version": 16,
            "configuration_name": "old-strategy",
        }
    )
    db[SETTINGS_COLLECTION].documents["default"] = previous
    db[SETTINGS_COLLECTION].documents["obsolete"] = {
        "_id": "obsolete",
        "strategy_mode": "old",
    }
    db[SETTINGS_HISTORY_COLLECTION].documents["history-1"] = {"_id": "history-1"}
    db[SETTINGS_HISTORY_COLLECTION].documents["history-2"] = {"_id": "history-2"}

    result = install_winner_strategy_configuration(
        db,
        note="Install the validated winner strategy.",
        source="test",
    )

    assert result["status"] == "winner_installed"
    assert result["source_file"] == "winner-v1.13.2.json"
    assert result["configuration_hash"] == WINNER_CONFIGURATION_SHA256
    assert result["replaced_previous_default"] is True
    assert result["deleted_extra_strategy_documents"] == 1
    assert result["deleted_strategy_history_documents"] == 2
    assert list(db[SETTINGS_COLLECTION].documents) == ["default"]
    assert db[SETTINGS_HISTORY_COLLECTION].documents == {}

    stored = db[SETTINGS_COLLECTION].documents["default"]
    assert stored["revision"] == 1
    assert stored["configuration_name"] == "winner-v1.13.2"
    assert stored["winner_source_file"] == "winner-v1.13.2.json"
    assert stored["winner_configuration_hash"] == WINNER_CONFIGURATION_SHA256
    assert stored["rotation_xgb_n_estimators"] == 300
    assert stored["rotation_target_horizons"] == [5, 10, 20, 40, 60]


def test_install_winner_is_blocked_by_active_backtest() -> None:
    db = _Database()
    db[JOBS_COLLECTION].documents["job"] = {
        "_id": "job",
        "id": "job-123",
        "status": "running",
    }

    with pytest.raises(StrategyConfigurationConflict, match="backtest"):
        install_winner_strategy_configuration(
            db,
            note="Install the validated winner strategy.",
            source="test",
        )

    assert db[SETTINGS_COLLECTION].documents == {}


def test_install_winner_is_blocked_by_active_paper_run() -> None:
    db = _Database()
    db[PAPER_MARKET_RUNS_COLLECTION].documents["paper"] = {
        "_id": "paper",
        "run_id": "paper-run-123",
        "active_key": "alpaca-paper-next-session",
        "status": "armed",
    }

    with pytest.raises(StrategyConfigurationConflict, match="Paper run"):
        install_winner_strategy_configuration(
            db,
            note="Install the validated winner strategy.",
            source="test",
        )

    assert db[SETTINGS_COLLECTION].documents == {}
