from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from market_cycle_trader_api.schemas.paper_market import StartNextSessionRequest, StopPaperRobotRequest
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings
from market_cycle_trader_api.services.paper_market_scheduler import (
    PREMARKET_ANALYSIS_POLICY,
    _ensure_continuous_run,
    _prepared_run_has_valid_premarket_analysis,
    _rearm_prepared_run_for_premarket_analysis,
    arm_next_session,
    paper_market_robot_status,
)


def _paper_settings() -> dict:
    return {
        "enabled": True,
        "client_order_id_prefix": "mct-xgb-paper",
        "premarket_analysis_minutes": 90,
        "market_open_delay_seconds": 60,
        "market_execution_window_seconds": 900,
        "order_fill_timeout_seconds": 180,
        "order_poll_interval_seconds": 2.0,
        "cash_reserve_dollars": 0.0,
    }


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

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        target = None
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                target = document
                break
        if target is None and upsert:
            target = dict(query)
            target.update(update.get("$setOnInsert", {}))
            self.documents.append(target)
        if target is not None:
            target.update(update.get("$set", {}))
            for key in update.get("$unset", {}):
                target.pop(key, None)
        return _InsertResult()


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
        settings = PaperTradingSettings.model_validate(_paper_settings())
        self.assertEqual(settings.market_execution_window_seconds, 900)
        self.assertEqual(settings.premarket_analysis_minutes, 90)

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
            "settings": _paper_settings(),
            "strategy_cash": 10_000.0,
            "managed_symbol": None,
        }
        with patch(
            "market_cycle_trader_api.services.paper_market_scheduler.paper_market_readiness",
            return_value=readiness,
        ):
            run = arm_next_session(db)

        self.assertEqual(run["status"], "armed")
        self.assertEqual(run["phase"], "waiting_for_premarket_analysis")
        self.assertEqual(run["analysis_policy"], PREMARKET_ANALYSIS_POLICY)
        self.assertEqual(run["premarket_analysis_at"], datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(run["execution_session"], "2026-08-03")
        self.assertEqual(run["strategy_cash"], 10_000.0)
        self.assertEqual(len(db["paper_market_runs"].documents), 1)
        self.assertTrue(run["automation_enabled"])
        controller = db["paper_market_automation"].find_one({"_id": "default"})
        self.assertTrue(controller["enabled"])

    def test_stop_request_requires_confirmation(self) -> None:
        self.assertTrue(StopPaperRobotRequest(confirm_stop=True).confirm_stop)
        with self.assertRaises(ValueError):
            StopPaperRobotRequest(confirm_stop=False)

    def test_robot_status_reports_continuous_mode(self) -> None:
        db = _Database()
        db["paper_market_automation"].documents.append({
            "_id": "default",
            "enabled": True,
            "status": "active",
            "phase": "waiting_for_next_market_open",
        })
        status = paper_market_robot_status(db, public=True)
        self.assertTrue(status["enabled"])
        self.assertEqual(status["mode"], "continuous_regular_sessions")

    def test_continuous_controller_arms_following_session_after_terminal_run(self) -> None:
        db = _Database()
        db["paper_market_automation"].documents.append({
            "_id": "default",
            "enabled": True,
            "status": "active",
            "phase": "completed",
        })
        db["paper_market_runs"].documents.append({
            "run_id": "previous",
            "status": "completed",
            "phase": "completed",
            "created_at": datetime(2026, 8, 4, 13, 31, tzinfo=timezone.utc),
        })
        readiness = {
            "clock": {
                "timestamp": datetime(2026, 8, 4, 13, 32, tzinfo=timezone.utc),
                "is_open": True,
                "next_open": datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc),
                "next_close": datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc),
            },
            "settings": _paper_settings(),
            "strategy_cash": 10_000.0,
            "managed_symbol": None,
        }
        with patch(
            "market_cycle_trader_api.services.paper_market_scheduler.paper_market_readiness",
            return_value=readiness,
        ):
            active = _ensure_continuous_run(db)

        self.assertIsNotNone(active)
        self.assertEqual(active["status"], "armed")
        self.assertEqual(active["execution_session"], "2026-08-05")

    def test_upgrade_adopts_existing_prepared_run_as_continuous(self) -> None:
        db = _Database()
        db["paper_market_runs"].documents.append({
            "run_id": "prepared-run",
            "active_key": "alpaca-paper-next-session",
            "status": "prepared",
            "phase": "waiting_for_next_market_open",
            "requested_at": datetime(2026, 8, 3, 13, 36, tzinfo=timezone.utc),
            "expected_market_open": datetime(2026, 8, 4, 13, 30, tzinfo=timezone.utc),
            "execution_session": "2026-08-04",
        })
        active = _ensure_continuous_run(db)
        controller = db["paper_market_automation"].find_one({"_id": "default"})

        self.assertEqual(active["run_id"], "prepared-run")
        self.assertTrue(controller["enabled"])
        self.assertTrue(controller["adopted_existing_run"])

    def test_valid_premarket_analysis_must_complete_inside_window(self) -> None:
        run = {
            "analysis_policy": PREMARKET_ANALYSIS_POLICY,
            "premarket_analysis_at": datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
            "premarket_analysis_completed_at": datetime(2026, 8, 5, 12, 15, tzinfo=timezone.utc),
            "expected_market_open": datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc),
        }
        self.assertTrue(_prepared_run_has_valid_premarket_analysis(run))
        run["premarket_analysis_completed_at"] = datetime(2026, 8, 5, 11, 59, tzinfo=timezone.utc)
        self.assertFalse(_prepared_run_has_valid_premarket_analysis(run))

    def test_legacy_prepared_plan_is_rearmed_without_touching_position_state(self) -> None:
        db = _Database()
        db["paper_market_runs"].documents.append({
            "run_id": "legacy-plan-run",
            "active_key": "alpaca-paper-next-session",
            "status": "prepared",
            "phase": "waiting_for_next_market_open",
            "plan_id": "legacy-plan",
            "action": "hold",
            "target_asset": "AMZN",
            "managed_symbol": "AMZN",
            "expected_market_open": datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc),
            "execution_session": "2026-08-05",
        })
        db["paper_trade_plans"].documents.append({
            "plan_id": "legacy-plan",
            "status": "prepared",
        })

        refreshed = _rearm_prepared_run_for_premarket_analysis(
            db,
            db["paper_market_runs"].find_one({"run_id": "legacy-plan-run"}),
        )

        self.assertEqual(refreshed["status"], "armed")
        self.assertEqual(refreshed["phase"], "waiting_for_premarket_analysis")
        self.assertEqual(refreshed["managed_symbol"], "AMZN")
        self.assertNotIn("plan_id", refreshed)
        plan = db["paper_trade_plans"].find_one({"plan_id": "legacy-plan"})
        self.assertEqual(plan["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
