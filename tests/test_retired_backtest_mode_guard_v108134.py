"""Retired strategy modes remain readable but cannot create new jobs.

Run:
  python -m unittest tests.test_retired_backtest_mode_guard_v108134 -v
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from market_cycle_trader_api.api.routers.jobs import queue_backtest_job
from market_cycle_trader_api.api.routers.strategy_lab import _translate_error
from market_cycle_trader_api.services.strategy_lab import (
    UnsupportedBacktestStrategyMode,
    assert_supported_backtest_strategy,
    get_research_strategy_context,
    select_research_strategy,
    select_research_strategy_only,
)


OLD_MODE = "COMPOUND_ROTATION_SWING_TCC_U75_DAILY_CASH_V101"
SUPPORTED_MODE = "COMPOUND_ROTATION_SWING_TCC_U67_V1210"


class RetiredBacktestModeTests(unittest.TestCase):
    def test_retired_mode_rejected_without_auto_remapping(self):
        profile = {"_id": "historical", "name": "Historical daily cash",
                   "configuration": {"strategy_mode": OLD_MODE}}
        with self.assertRaises(UnsupportedBacktestStrategyMode) as raised:
            assert_supported_backtest_strategy(profile)
        detail = str(raised.exception)
        self.assertIn("RETIRED_BACKTEST_STRATEGY_MODE", detail)
        self.assertIn(OLD_MODE, detail)
        self.assertIn("Historical results remain accessible", detail)
        self.assertNotIn("will use", detail)
        self.assertEqual(_translate_error(raised.exception).status_code, 409)

    def test_supported_mode_is_not_rejected_by_retired_mode_guard(self):
        assert_supported_backtest_strategy({
            "configuration": {"strategy_mode": SUPPORTED_MODE}
        })

    @patch("market_cycle_trader_api.services.strategy_lab.ensure_strategy_catalog")
    def test_current_retired_research_profile_fails_before_pydantic(self, ensure_catalog):
        ensure_catalog.return_value = {"research_strategy_id": "historical"}
        db = MagicMock()
        db.__getitem__.return_value.find_one.return_value = {
            "_id": "historical", "configuration": {"strategy_mode": OLD_MODE}
        }
        with self.assertRaises(UnsupportedBacktestStrategyMode):
            get_research_strategy_context(db)

    @patch("market_cycle_trader_api.services.strategy_lab.ensure_strategy_catalog")
    @patch("market_cycle_trader_api.services.strategy_lab._assert_no_active_backtest")
    def test_research_selection_rejects_retired_mode_without_mutation(self, _no_active, catalog):
        catalog.return_value = {"revision": 4, "research_strategy_id": "valid"}
        db = MagicMock()
        db.__getitem__.return_value.find_one.return_value = {
            "_id": "historical", "configuration": {"strategy_mode": OLD_MODE}
        }
        for selector in (select_research_strategy, select_research_strategy_only):
            with self.assertRaises(UnsupportedBacktestStrategyMode):
                selector(db, "historical", expected_control_revision=4, note="test", actor_email=None)
        self.assertFalse(db.__getitem__.return_value.find_one_and_update.called)

    @patch("market_cycle_trader_api.api.routers.jobs.get_research_strategy_context")
    @patch("market_cycle_trader_api.api.routers.jobs.get_system_settings")
    @patch("market_cycle_trader_api.api.routers.jobs.database")
    def test_post_jobs_returns_409_without_inserting_any_job(self, database, settings, context):
        db = MagicMock()
        db.__getitem__.return_value.count_documents.return_value = 0
        db.__getitem__.return_value.find_one.return_value = None
        database.return_value = db
        settings.return_value = {"training": {"enabled": True}}
        context.side_effect = UnsupportedBacktestStrategyMode(
            f"RETIRED_BACKTEST_STRATEGY_MODE: {OLD_MODE}"
        )
        with self.assertRaises(HTTPException) as raised:
            queue_backtest_job(start_thread=False)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn(OLD_MODE, str(raised.exception.detail))
        self.assertFalse(db.__getitem__.return_value.insert_one.called)


if __name__ == "__main__":
    unittest.main()
