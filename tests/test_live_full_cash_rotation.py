"""No broker orders are submitted: isolated Live Winner rotation regression tests."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from market_cycle_trader_api.services.live_order_execution import (
    execute_recalibrated_live_winner_session,
)


class LiveWinnerRotationTests(TestCase):
    def test_rotation_sells_entire_position_then_uses_updated_alpaca_cash(self):
        session = "2026-10-19"
        run = {
            "_id": "acct:" + session, "execution_session": session,
            "status": "recalibrated_no_orders", "action": "rotate",
            "target_asset": "ADEA", "current_asset": "NVDA",
            "winner_signal_prepared": True, "post_close_signal_ready": True,
            "decision_date": "2026-10-16", "snapshot_sha256": "frozen-sha",
            "winner_strategy_id": "winner",
            "winner_configuration_hash": "hash", "winner_revision": 2,
            "live_account_id": "acct", "live_state_revision": 7,
            "expected_market_open": datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc),
        }
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = run
        db["live_winner_session_runs"].find_one_and_update.return_value = run
        broker = MagicMock()
        first_account = SimpleNamespace(
            id="acct", status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False,
            cash="0.00", equity="153.45",
        )
        second_account = SimpleNamespace(
            id="acct", status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False,
            cash="153.45", equity="153.45",
        )
        broker.get_account.side_effect = [first_account, second_account]
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 10, 19, 13, 40, tzinfo=timezone.utc),
        )
        broker.get_orders.return_value = []
        state = {
            "account_id": "acct", "managed_symbol": "NVDA",
            "managed_quantity": 2.125, "revision": 7,
        }
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client", return_value=broker),
            patch(prefix + "verified_live_state", return_value=state),
            patch(prefix + "asset_snapshot", return_value={
                "tradable": True, "fractionable": True, "status": "active",
            }),
            patch(prefix + "_execute_leg", side_effect=[
                SimpleNamespace(filled_quantity=2.125),
                SimpleNamespace(filled_quantity=13.0),
            ]) as execute_leg,
            patch(prefix + "confirm_live_fill_state") as confirm_state,
        ):
            result = execute_recalibrated_live_winner_session(db, session)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(execute_leg.call_count, 2)
        sell, buy = execute_leg.call_args_list
        self.assertEqual(sell.kwargs["side"], "sell")
        self.assertEqual(sell.kwargs["symbol"], "NVDA")
        self.assertEqual(sell.kwargs["quantity"], 2.125)
        self.assertEqual(buy.kwargs["side"], "buy")
        self.assertEqual(buy.kwargs["symbol"], "ADEA")
        self.assertEqual(buy.kwargs["notional"], 153.45)
        self.assertEqual(confirm_state.call_count, 2)
        self.assertEqual(broker.get_account.call_count, 2)
        broker.submit_order.assert_not_called()

    def test_closed_market_never_executes(self):
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = {
            "status": "recalibrated_no_orders",
            "winner_strategy_id": "winner", "winner_revision": 2,
            "winner_configuration_hash": "hash",
            "winner_signal_prepared": True, "post_close_signal_ready": True,
            "snapshot_sha256": "frozen-sha", "decision_date": "2026-10-16",
            "target_asset": "ADEA", "action": "buy", "live_account_id": "acct",
        }
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id="acct", status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False,
        )
        broker.get_clock.return_value = SimpleNamespace(is_open=False)
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with (
            patch(prefix + "live_control_status", return_value={"control_mode": "active"}),
            patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }),
            patch(prefix + "create_live_execution_client", return_value=broker),
            patch(prefix + "_execute_leg") as execute_leg,
        ):
            result = execute_recalibrated_live_winner_session(db, "2026-10-19")
        self.assertEqual(result["status"], "market_closed")
        execute_leg.assert_not_called()
        broker.submit_order.assert_not_called()


class LivePaperDependencyTests(TestCase):
    def test_live_execution_state_and_recalibration_do_not_import_paper(self):
        from market_cycle_trader_api import services
        source_dir = Path(services.__file__).parent
        paths = [
            source_dir / "live_order_execution.py",
            source_dir / "live_trader_state.py",
            source_dir / "live_winner_recalibration.py",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                imports = [node.module for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom)]
                self.assertFalse(any(module and ("alpaca_paper" in module
                                                 or module.endswith("schemas.paper_trading")
                                                 or "mongo_repository" in module)
                                     for module in imports))
                self.assertNotIn("get_paper_trading_settings", source)
                self.assertNotIn("cash_reserve_dollars", source)
