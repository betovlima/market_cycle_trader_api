from __future__ import annotations

import copy
import unittest

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    PAPER_TRADING_SETTINGS_COLLECTION,
    get_paper_trading_settings,
)
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings


class _Collection:
    def __init__(self, document: dict) -> None:
        self.document = copy.deepcopy(document)

    def find_one(self, query: dict):
        return copy.deepcopy(self.document) if query == {"_id": "default"} else None


class _Database:
    def __init__(self, document: dict) -> None:
        self.collection = _Collection(document)

    def __getitem__(self, name: str):
        if name != PAPER_TRADING_SETTINGS_COLLECTION:
            raise KeyError(name)
        return self.collection


class PaperSettingsMetadataTests(unittest.TestCase):
    def test_repository_removes_revision_and_other_metadata(self) -> None:
        db = _Database({
            "_id": "default",
            "enabled": True,
            "paper_account_id": None,
            "client_order_id_prefix": "mct-xgb-paper",
            "market_open_delay_seconds": 60,
            "market_execution_window_seconds": 900,
            "order_fill_timeout_seconds": 180,
            "order_poll_interval_seconds": 2.0,
            "cash_reserve_dollars": 0.0,
            "schema_version": 1,
            "revision": 1,
            "configuration_name": "paper",
            "configuration_note": "metadata",
            "bootstrap_source": "parameter-bootstrap-api",
            "created_at": "2026-08-02T00:00:00Z",
            "updated_at": "2026-08-02T00:00:00Z",
        })

        settings = get_paper_trading_settings(db)

        self.assertNotIn("revision", settings)
        self.assertNotIn("schema_version", settings)
        validated = PaperTradingSettings.model_validate(settings)
        self.assertTrue(validated.enabled)
        self.assertEqual(validated.premarket_analysis_minutes, 90)


if __name__ == "__main__":
    unittest.main()
