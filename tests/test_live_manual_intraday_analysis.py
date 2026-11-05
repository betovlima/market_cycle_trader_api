"""Live current-session Winner analysis is separate from D-1 execution.

All Alpaca responses and model calculations are mocks. Tests send no orders,
do not write Paper collections and do not need user credentials.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import BackgroundTasks
from market_cycle_trader_api.services import live_manual_intraday_analysis as subject

ACCOUNT = "account-live"
SESSION = "2026-11-05"
PRIOR = "2026-11-04"
NOW = datetime(2026, 11, 5, 15, 30, tzinfo=timezone.utc)
PREFIX = "market_cycle_trader_api.services.live_manual_intraday_analysis."


def live_broker(now=NOW, market_open=True):
    broker = MagicMock()
    broker.get_account.return_value = SimpleNamespace(id=ACCOUNT)
    broker.get_clock.return_value = SimpleNamespace(
        timestamp=now, is_open=market_open,
        next_close=datetime(2026, 11, 5, 21, tzinfo=timezone.utc),
        next_open=datetime(2026, 11, 6, 14, 30, tzinfo=timezone.utc),
    )
    return broker


class ManualLiveIntradayAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.collection = self.db[subject.MANUAL_ANALYSES]
        self.collection.find_one.return_value = None
        self.broker = live_broker()

    def test_market_open_uses_alpaca_and_exchange_calendar(self):
        account, session, stamp = subject._verified_current_session(self.broker)
        self.assertEqual(account, ACCOUNT)
        self.assertEqual(session, SESSION)
        self.assertEqual(stamp, NOW)
        with self.assertRaisesRegex(RuntimeError, "regular market"):
            subject._verified_current_session(live_broker(market_open=False))

    def test_analysis_available_even_if_the_live_session_has_already_executed(self):
        self.db["live_winner_session_runs"].find_one.return_value = {
            "status": "completed", "target_asset": "CAPR",
        }
        with patch(PREFIX + "create_live_readonly_client", return_value=self.broker):
            status = subject.manual_intraday_status(self.db)
        self.assertEqual(status["status"], "idle")
        self.assertTrue(status["can_analyze"])
        self.assertFalse(status["order_submission_enabled"])
        self.db["live_winner_session_runs"].find_one.assert_not_called()
        self.broker.submit_order.assert_not_called()

    def test_paused_live_still_allows_readonly_reanalysis(self):
        with (
            patch(PREFIX + "create_live_readonly_client", return_value=self.broker),
            patch(PREFIX + "uuid4", return_value=SimpleNamespace(hex="a" * 32)),
        ):
            self.collection.find_one_and_update.return_value = {
                "_id": f"{ACCOUNT}:{SESSION}", "status": "running",
                "analysis_id": "a" * 32,
            }
            result = subject.request_manual_intraday_analysis(self.db, actor="admin")
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["queued_new"])
        self.assertEqual(result["analysis_id"], "a" * 32)
        self.assertEqual(result["_account_id"], ACCOUNT)
        self.broker.submit_order.assert_not_called()
        # A plan, session, operational control or brokerage order is never changed.
        self.db["live_winner_session_runs"].update_one.assert_not_called()
        self.db["live_trader_control"].update_one.assert_not_called()

    def test_duplicate_running_job_is_not_started_twice(self):
        self.collection.find_one.return_value = {
            "_id": f"{ACCOUNT}:{SESSION}", "status": "running",
            "analysis_id": "old",
        }
        with patch(PREFIX + "create_live_readonly_client", return_value=self.broker):
            result = subject.request_manual_intraday_analysis(self.db, actor="admin")
        self.assertFalse(result["queued_new"])
        self.assertEqual(result["analysis_id"], "old")
        self.collection.find_one_and_update.assert_not_called()

    def test_analysis_is_not_available_after_market_close(self):
        late = live_broker(
            now=datetime(2026, 11, 5, 22, tzinfo=timezone.utc),
            market_open=False,
        )
        with patch(PREFIX + "create_live_readonly_client", return_value=late):
            result = subject.manual_intraday_status(self.db)
        self.assertFalse(result["can_analyze"])

    def test_background_worker_builds_advisory_result_without_editing_live_run(self):
        token = "a" * 32
        self.collection.update_one.return_value = SimpleNamespace(matched_count=1)
        cfg = MagicMock()
        cfg.strategy_mode = subject.TCC_U67_CONTROL_OPERATIONAL_MODE
        cfg.assets = ["CAPR", "AMZN"]
        cfg.model_copy.return_value = cfg
        prior = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2026-11-04"], utc=True))
        snap = SimpleNamespace(
            frames={"CAPR": prior, "AMZN": prior},
            manifest={"snapshot_sha256": "audit-hash"},
        )
        decision = SimpleNamespace(
            decision_date=pd.Timestamp(SESSION),
            target_asset="AMZN", raw_best_asset="AMZN",
            selected_utility=0.25, utilities={"CAPR": .1, "AMZN": .25},
        )
        self.db["live_winner_session_runs"].find_one.return_value = {
            "status": "completed", "target_asset": "CAPR",
        }
        with (
            patch(PREFIX + "database", return_value=self.db),
            patch(PREFIX + "create_live_readonly_client", return_value=self.broker),
            patch(PREFIX + "verified_live_state", return_value={
                "managed_symbol": "CAPR", "managed_quantity": 5.0,
                "holding_sessions": 3, "revision": 7,
                "last_execution_session": SESSION,
            }),
            patch(PREFIX + "get_trader_winner_context", return_value=(
                cfg, {"id": "w1", "revision": 2, "configuration_hash": "h1"},
            )),
            patch(PREFIX + "get_trader_winner_model_snapshot", return_value={
                "family": "lightgbm_utility", "settings_snapshot": {},
            }),
            patch(PREFIX + "apply_training_runtime_settings", return_value=cfg),
            patch(PREFIX + "apply_execution_profile", return_value=cfg),
            patch(PREFIX + "assert_tcc_u67_operational_contract"),
            patch(PREFIX + "assert_tcc_u67_model_snapshot"),
            patch(PREFIX + "latest_safe_completed_xnys_session", return_value=pd.Timestamp(PRIOR)),
            patch(PREFIX + "download_operational_market_snapshot", return_value=snap) as download,
            patch(PREFIX + "append_current_session_intraday", return_value=(
                {"CAPR": prior, "AMZN": prior},
                {"analysis_mode": "current_session_intraday",
                 "live_session": SESSION,
                 "analysis_timestamp_utc": NOW.isoformat(),
                 "intraday_market_data_cutoff_utc": NOW.isoformat(),
                 "intraday_data_delay_minutes": 0, "live_feed": "iex"},
            )) as intraday,
            patch(PREFIX + "build_live_model_decision", return_value=decision),
        ):
            subject.run_manual_intraday_analysis(ACCOUNT, SESSION, token)
        self.assertGreaterEqual(self.collection.update_one.call_count, 2)
        saved = [
            call.args[1]["$set"]
            for call in self.collection.update_one.call_args_list
            if call.args[1]["$set"].get("status") == "prepared"
        ]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["action"], "rotate")
        self.assertEqual(saved[0]["target_asset"], "AMZN")
        self.assertEqual(saved[0]["original_signal_status"], "completed")
        self.assertFalse(saved[0]["order_submission_enabled"])
        self.assertEqual(saved[0]["historical_cutoff"], PRIOR)
        download.assert_called_once()
        intraday.assert_called_once()
        self.db["live_winner_session_runs"].update_one.assert_not_called()
        self.broker.submit_order.assert_not_called()

    def test_invalid_worker_analysis_fails_without_exposing_provider_text(self):
        self.collection.update_one.return_value = SimpleNamespace(matched_count=1)
        with (
            patch(PREFIX + "database", return_value=self.db),
            patch(PREFIX + "create_live_readonly_client",
                  side_effect=RuntimeError("Authorization: Bearer SECRET-TOKEN")),
        ):
            subject.run_manual_intraday_analysis(ACCOUNT, SESSION, "a" * 32)
        saved = [
            call.args[1]["$set"]
            for call in self.collection.update_one.call_args_list
        ]
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertEqual(saved[-1]["failure_type"], "RuntimeError")
        self.assertNotIn("SECRET-TOKEN", str(saved[-1]))
        self.db["live_winner_session_runs"].update_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
