from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    PAPER_TRADING_SETTINGS_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
)
from market_cycle_trader_api.services.parameter_bootstrap import (
    bootstrap_missing_parameterizations,
    parameterization_status,
)


class _InsertResult:
    inserted_id = "inserted"


class _InsertManyResult:
    inserted_ids = ["audit"]


class _DeleteResult:
    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


class _Cursor(list):
    pass


class _Collection:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.audit: list[dict] = []

    def find(self, query: dict):
        if query:
            raise AssertionError(f"Unexpected query in fake collection: {query}")
        return _Cursor(copy.deepcopy(list(self.documents.values())))

    def find_one(self, query: dict, *args, **kwargs):
        document_id = query.get("_id")
        document = self.documents.get(document_id)
        return copy.deepcopy(document) if document is not None else None

    def insert_one(self, document: dict):
        item = copy.deepcopy(document)
        document_id = str(item.get("_id") or f"audit-{len(self.audit) + 1}")
        if "_id" in item:
            self.documents[document_id] = item
        else:
            self.audit.append(item)
        return _InsertResult()

    def insert_many(self, documents: list[dict], ordered: bool = True):
        self.audit.extend(copy.deepcopy(documents))
        return _InsertManyResult()

    def delete_many(self, query: dict):
        if query:
            raise AssertionError(f"Unexpected delete query in fake collection: {query}")
        count = len(self.documents)
        self.documents.clear()
        return _DeleteResult(count)


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())


class ParameterBootstrapTests(unittest.TestCase):
    def test_missing_status_reports_both_documents(self) -> None:
        db = _Database()
        items = parameterization_status(db)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item["status"] == "missing" for item in items))

    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_bootstrap_installs_one_canonical_strategy_and_is_idempotent(
        self, _ensure_database
    ) -> None:
        db = _Database()

        first = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(first["mode"], "canonical_strategy_reset_and_insert_missing")
        self.assertEqual(first["summary"]["inserted"], 2)

        strategy = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        paper = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertIsNotNone(strategy)
        self.assertIsNotNone(paper)
        self.assertEqual(strategy["random_state"], 3042)
        self.assertEqual(strategy["schema_version"], 14)
        self.assertEqual(strategy["alpaca_historical_feed"], "sip")
        self.assertEqual(strategy["alpaca_live_feed"], "iex")
        self.assertEqual(strategy["xgb_n_jobs"], -1)
        self.assertFalse(strategy["deterministic_execution"])
        self.assertTrue(paper["enabled"])

        second = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(second["summary"]["inserted"], 0)
        self.assertEqual(second["summary"]["migrated_existing"], 0)
        self.assertEqual(second["summary"]["skipped_existing_valid"], 2)
        self.assertEqual(len(db[SETTINGS_COLLECTION].documents), 1)

    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_strategy_drift_and_extra_documents_are_archived_and_replaced(
        self, _ensure_database
    ) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        drifted = db[SETTINGS_COLLECTION].documents["default"]
        drifted["random_state"] = 42
        drifted["alpaca_historical_feed"] = "sip"
        drifted["configuration_name"] = "manual-strategy"
        db[SETTINGS_COLLECTION].documents["old-strategy"] = {
            "_id": "old-strategy",
            "random_state": 7,
        }

        result = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(result["summary"]["migrated_existing"], 1)
        self.assertEqual(len(db[SETTINGS_COLLECTION].documents), 1)

        canonical = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(canonical["random_state"], 3042)
        self.assertEqual(canonical["alpaca_historical_feed"], "sip")
        self.assertEqual(canonical["alpaca_live_feed"], "iex")
        self.assertEqual(canonical["schema_version"], 14)

        history = db[SETTINGS_HISTORY_COLLECTION].audit
        self.assertEqual(len(history), 2)
        archived_ids = {item["original_document_id"] for item in history}
        self.assertEqual(archived_ids, {"default", "old-strategy"})

    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_valid_paper_settings_are_preserved(self, _ensure_database) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        paper = db[PAPER_TRADING_SETTINGS_COLLECTION].documents["default"]
        paper["paper_account_id"] = "account-123"

        result = bootstrap_missing_parameterizations(db, source="test")
        item = next(
            value
            for value in result["results"]
            if value["collection"] == PAPER_TRADING_SETTINGS_COLLECTION
        )
        self.assertTrue(item["valid"])
        self.assertEqual(item["status"], "skipped_existing_valid")
        stored = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(stored["paper_account_id"], "account-123")


if __name__ == "__main__":
    unittest.main()
