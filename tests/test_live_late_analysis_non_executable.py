"""Late Live recalibration never transitions into executable session state."""
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pandas as pd

from market_cycle_trader_api.services.live_winner_recalibration import (
    recalibrate_live_cash_winner_signal,
)
from market_cycle_trader_api.core.config import TCC_U77_FIXED_CHALLENGER_MODE


class LateLiveRecalibrationTests(unittest.TestCase):
    def test_late_analysis_stays_non_executable_and_updates_candidates(self):
        run = {
            "_id": "acct:2026-10-16", "live_account_id": "acct",
            "execution_session": "2026-10-16",
            "status": "analysis_only_requested",
            "expected_market_open": datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc),
            "winner_strategy_id": "winner", "winner_revision": 2,
            "winner_configuration_hash": "hash",
        }
        db = MagicMock()
        collection = MagicMock()
        db.__getitem__.return_value = collection
        collection.find_one.return_value = run
        db["live_winner_session_runs"].find_one_and_update.side_effect = [
            {**run, "status": "analysis_only_recalibrating"},
            {**run, "status": "analysis_only_completed", "action": "buy"},
        ]
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="acct")
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 16, 15, 5, tzinfo=timezone.utc),
        )
        broker.get_orders.return_value = []
        cfg = MagicMock()
        cfg.strategy_mode = TCC_U77_FIXED_CHALLENGER_MODE
        cfg.assets = ["ADEA", "NVDA"]
        cfg.model_copy.return_value = cfg
        snapshot = SimpleNamespace(frames={"ADEA": pd.DataFrame({"close": [1]})})
        decision = SimpleNamespace(decision_date=pd.Timestamp("2026-10-16"), target_asset="ADEA")
        prefix = "market_cycle_trader_api.services.live_winner_recalibration."
        with ExitStack() as stack:
            patches = {
                "order_submission_opted_in": False,
                "create_live_readonly_client": broker,
                "verified_live_state": {
                    "managed_symbol": "CASH", "managed_quantity": 0.0, "revision": 0,
                    "holding_sessions": 0,
                },
                "get_paper_trading_settings": {},
                "get_trader_winner_context": (cfg, {
                    "id": "winner", "revision": 2, "configuration_hash": "hash",
                }),
                "get_trader_winner_model_snapshot": {"family": "lightgbm_utility"},
                "latest_safe_completed_xnys_session": pd.Timestamp("2026-10-15"),
                "download_u67_operational_snapshot": snapshot,
                "append_u67_current_session_intraday": (snapshot.frames, {}),
                "build_live_model_decision": decision,
                "winner_candidate_fields": {"utilities": {"CASH": 0.0, "ADEA": 1.0},
                                            "candidate_count": 1},
            }
            for name, value in patches.items():
                stack.enter_context(patch(prefix + name, return_value=value))
            stack.enter_context(patch(prefix + "PaperTradingSettings.model_validate",
                                     return_value=SimpleNamespace(market_execution_window_seconds=1800)))
            stack.enter_context(patch(prefix + "apply_execution_profile", return_value=cfg))
            stack.enter_context(patch(prefix + "apply_training_runtime_settings", return_value=cfg))
            for name in ("assert_tcc_u77_contract", "assert_tcc_u67_model_snapshot"):
                stack.enter_context(patch(prefix + name))
            result = recalibrate_live_cash_winner_signal(db, "2026-10-16")
        self.assertEqual(result["status"], "analysis_only_completed")
        self.assertFalse(result["order_submission_enabled"])
        update = db["live_winner_session_runs"].find_one_and_update.call_args.args[1]["$set"]
        self.assertEqual(update["status"], "analysis_only_completed")
        self.assertEqual(update["candidate_count"], 1)
        self.assertFalse(update["order_submission_enabled"])

    def test_analytical_completed_session_is_never_reexecuted(self):
        db = MagicMock()
        db.__getitem__.return_value = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "analysis_only_completed", "target_asset": "ADEA"
        }
        result = recalibrate_live_cash_winner_signal(db, "2026-10-16")
        self.assertEqual(result["status"], "analysis_only_completed")
        self.assertFalse(result["order_submission_enabled"])
