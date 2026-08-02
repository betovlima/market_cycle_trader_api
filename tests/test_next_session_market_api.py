from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from market_cycle_trader_api.schemas.paper_market import StartNextSessionRequest
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings
from market_cycle_trader_api.services.paper_market_scheduler import arm_next_session


class _InsertResult:
    inserted_id = "fake"


class _Collection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    def insert_one(self, document: dict) -> _InsertResult:
        self.documents.append(dict(document))
        return _InsertResult()

    def find_one(self, query: dict, *args, **kwargs):
        for document in reversed(self.documents):
            if all(document.get(key) == value for key, value in query.items()):
                return dict(document)
        return None


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())


class NextSessionMarketApiTests(unittest.TestCase):
    def test_start_request_requires_explicit_paper_confirmation(self) -> None:
        request = StartNextSessionRequest(confirm_paper=True)
        self.assertTrue(request.confirm_paper)
        with self.assertRaises(ValueError):
            StartNextSessionRequest(confirm_paper=False)

    def test_paper_settings_require_safe_execution_window(self) -> None:
        settings = PaperTradingSettings.model_validate(
            {
                "enabled": True,
                "client_order_id_prefix": "mct-xgb-paper",
                "market_open_delay_seconds": 60,
                "market_execution_window_seconds": 900,
                "order_fill_timeout_seconds": 180,
                "order_poll_interval_seconds": 2.0,
                "cash_reserve_dollars": 0.0,
                "automatic_continuation_enabled": True,
                "scheduler_poll_seconds": 10.0,
                "preparation_retry_seconds": 60.0,
            }
        )
        self.assertEqual(settings.market_execution_window_seconds, 900)

        invalid = settings.model_dump()
        invalid["market_execution_window_seconds"] = 60
        with self.assertRaises(ValueError):
            PaperTradingSettings.model_validate(invalid)

    def test_arming_uses_alpaca_next_open_and_persists_one_run(self) -> None:
        db = _Database()
        next_open = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
        readiness = {
            "clock": {
                "timestamp": datetime(2026, 8, 1, 2, 0, tzinfo=timezone.utc),
                "is_open": False,
                "next_open": next_open,
                "next_close": datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc),
            },
            "settings": {},
            "strategy_cash": 10_000.0,
            "managed_symbol": None,
        }
        with patch(
            "market_cycle_trader_api.services.paper_market_scheduler.paper_market_readiness",
            return_value=readiness,
        ):
            run = arm_next_session(db)

        self.assertEqual(run["status"], "armed")
        self.assertEqual(run["execution_session"], "2026-08-03")
        self.assertEqual(run["strategy_cash"], 10_000.0)
        self.assertEqual(len(db["paper_market_runs"].documents), 1)


if __name__ == "__main__":
    unittest.main()
