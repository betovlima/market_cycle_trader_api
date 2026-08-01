from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    PAPER_TRADING_SETTINGS_COLLECTION,
    SETTINGS_COLLECTION,
)
from market_cycle_trader_api.services.parameter_bootstrap import (
    bootstrap_missing_parameterizations,
    parameterization_status,
)


class _WriteResult:
    def __init__(self, upserted_id=None) -> None:
        self.upserted_id = upserted_id


class _InsertResult:
    inserted_id = "audit"


class _Collection:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.audit: list[dict] = []

    def find_one(self, query: dict, *args, **kwargs):
        document_id = query.get("_id")
        document = self.documents.get(document_id)
        return copy.deepcopy(document) if document is not None else None

    def update_one(self, query: dict, update: dict, *, upsert: bool = False):
        document_id = query.get("_id")
        if document_id in self.documents:
            if "$set" in update:
                self.documents[document_id].update(copy.deepcopy(update["$set"]))
            if "$unset" in update:
                for field in update["$unset"]:
                    self.documents[document_id].pop(field, None)
            return _WriteResult()
        if not upsert:
            return _WriteResult()
        document = copy.deepcopy(update.get("$setOnInsert", {}))
        if "$set" in update:
            document.update(copy.deepcopy(update["$set"]))
        self.documents[document_id] = document
        return _WriteResult(upserted_id=document_id)

    def insert_one(self, document: dict):
        self.audit.append(copy.deepcopy(document))
        return _InsertResult()


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
    def test_bootstrap_is_idempotent(self, _ensure_database) -> None:
        db = _Database()

        first = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(first["summary"]["inserted"], 2)
        self.assertEqual(first["summary"]["migrated_existing"], 0)
        self.assertEqual(first["summary"]["skipped_existing_valid"], 0)

        second = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(second["summary"]["inserted"], 0)
        self.assertEqual(second["summary"]["skipped_existing_valid"], 2)

        strategy = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        paper = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(strategy["random_state"], 3042)
        self.assertFalse(strategy["deterministic_execution"])
        self.assertEqual(strategy["numeric_thread_limit"], 1)
        self.assertEqual(strategy["xgb_n_jobs"], -1)
        self.assertTrue(strategy["market_data_history_backfill_enabled"])
        self.assertEqual(strategy["market_data_history_backfill_provider"], "alpaca")
        self.assertEqual(strategy["market_data_history_start_tolerance_days"], 10)
        self.assertTrue(strategy["market_data_require_complete_history"])
        self.assertEqual(strategy["schema_version"], 13)
        self.assertTrue(paper["enabled"])

    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_existing_valid_document_is_never_overwritten(self, _ensure_database) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        existing = db[SETTINGS_COLLECTION].documents["default"]
        existing["random_state"] = 42
        existing["configuration_name"] = "manually-promoted-seed-42"

        result = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(result["summary"]["inserted"], 0)
        self.assertEqual(result["summary"]["skipped_existing_valid"], 2)
        preserved = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(preserved["random_state"], 42)
        self.assertEqual(preserved["configuration_name"], "manually-promoted-seed-42")


    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_v10_strategy_document_restores_promoted_execution_policy(
        self, _ensure_database
    ) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        strategy = db[SETTINGS_COLLECTION].documents["default"]
        strategy["schema_version"] = 10
        strategy["random_state"] = 42
        strategy["configuration_name"] = "manually-promoted-seed-42"
        strategy["xgb_n_jobs"] = 1
        strategy["deterministic_execution"] = True
        strategy["numeric_thread_limit"] = 1

        result = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(result["summary"]["migrated_existing"], 1)

        migrated = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(migrated["random_state"], 42)
        self.assertEqual(migrated["configuration_name"], "manually-promoted-seed-42")
        self.assertEqual(migrated["xgb_n_jobs"], -1)
        self.assertFalse(migrated["deterministic_execution"])
        self.assertEqual(migrated["numeric_thread_limit"], 1)


    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_v12_strategy_document_migrates_to_alpaca_only_history_backfill(
        self, _ensure_database
    ) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        strategy = db[SETTINGS_COLLECTION].documents["default"]
        strategy["schema_version"] = 12
        strategy["market_data_provider"] = "alpaca"
        strategy["market_data_history_backfill_enabled"] = True
        strategy["market_data_history_backfill_provider"] = "yahoo"
        strategy["yfinance_auto_adjust"] = True
        strategy["yfinance_repair"] = False
        strategy["yfinance_timeout"] = 30
        strategy["yfinance_fallback_period"] = "max"
        strategy["random_state"] = 42

        result = bootstrap_missing_parameterizations(db, source="test")
        self.assertEqual(result["summary"]["migrated_existing"], 1)

        migrated = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        self.assertEqual(migrated["random_state"], 42)
        self.assertTrue(migrated["market_data_history_backfill_enabled"])
        self.assertEqual(migrated["market_data_history_backfill_provider"], "alpaca")
        self.assertEqual(migrated["market_data_history_start_tolerance_days"], 10)
        self.assertTrue(migrated["market_data_require_complete_history"])
        self.assertEqual(migrated["schema_version"], 13)
        self.assertEqual(migrated["market_data_provider"], "alpaca")
        self.assertNotIn("yfinance_auto_adjust", migrated)
        self.assertNotIn("yfinance_repair", migrated)
        self.assertNotIn("yfinance_timeout", migrated)
        self.assertNotIn("yfinance_fallback_period", migrated)

    @patch(
        "market_cycle_trader_api.services.parameter_bootstrap.ensure_database",
        return_value=None,
    )
    def test_legacy_paper_settings_without_account_id_are_valid_for_automatic_binding(
        self, _ensure_database
    ) -> None:
        db = _Database()
        bootstrap_missing_parameterizations(db, source="test")

        paper = db[PAPER_TRADING_SETTINGS_COLLECTION].documents["default"]
        paper.pop("paper_account_id", None)

        result = bootstrap_missing_parameterizations(db, source="test")
        item = next(
            value
            for value in result["results"]
            if value["collection"] == PAPER_TRADING_SETTINGS_COLLECTION
        )
        self.assertTrue(item["valid"])
        self.assertEqual(item["status"], "skipped_existing_valid")



if __name__ == "__main__":
    unittest.main()
