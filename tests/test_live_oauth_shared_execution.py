"""OAuth Winner execution with two accounts: fully mocked, zero live broker orders."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import live_oauth_execution as executor
from market_cycle_trader_api.services.live_order_execution import _unique_order_id
from market_cycle_trader_api.services.live_shared_winner_signal import SharedWinnerSignal


class OAuthExecutionTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.user_a = SimpleNamespace(subject="trader-A")
        self.user_b = SimpleNamespace(subject="trader-B")
        self.signal = SharedWinnerSignal(
            execution_session="2026-11-05",
            decision_date="2026-11-04",
            target_asset="CAPR", winner_strategy_id="single-winner",
            winner_revision=5, winner_configuration_hash="global-hash",
            snapshot_sha256="immutable-snapshot",
            source_account_id="primary-broker-account",
        )
        self.clock = SimpleNamespace(
            is_open=True,
            timestamp=datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc),
            next_close=datetime(2026, 11, 5, 21, 0, tzinfo=timezone.utc),
        )
        self.client = MagicMock()
        self.client.get_clock.return_value = self.clock
        self.client.get_account.return_value = SimpleNamespace(
            id="trader-a-account", status="ACTIVE",
            account_blocked=False, trading_blocked=False,
            trade_suspended_by_user=False, cash="110.10",
        )
        self.client.get_all_positions.return_value = []
        self.client.get_orders.return_value = []

    def _run(self, user=None, mode="active"):
        user = user or self.user_a
        with (patch.object(executor, "live_control_status", return_value={
                  "control_mode": mode,
              }) as control,
              patch.object(executor, "connected_trading_client",
                           return_value=self.client) as oauth_client,
              patch.object(executor, "read_shared_winner_signal",
                           return_value=self.signal) as shared,
              patch.object(executor, "_assert_live_order_window") as clock_check,
              patch.object(executor, "asset_snapshot", return_value={
                  "tradable": True, "fractionable": True, "status": "active",
              }),
              patch.object(executor, "_execute_leg") as execute):
            result = executor.execute_shared_winner_for_oauth_account(
                self.db, user, session="2026-11-05",
            )
        return result, execute, oauth_client, shared, clock_check, control

    def test_stopped_local_live_cannot_access_oauth_or_send_orders(self):
        result, exec_leg, oauth, signal, *_ = self._run(mode="stopped")
        self.assertEqual(result["status"], "blocked_by_live_mode")
        exec_leg.assert_not_called()
        oauth.assert_not_called()
        signal.assert_not_called()

    def test_shared_target_buys_with_only_broker_reported_cash(self):
        result, execute, oauth, shared, _, _ = self._run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["action"], "buy")
        self.assertTrue(result["order_submission_enabled"])
        oauth.assert_called_once_with(self.db, self.user_a)
        shared.assert_called_once_with(self.db, session="2026-11-05")
        call = execute.call_args
        self.assertEqual(call.kwargs["account"], "trader-a-account")
        self.assertEqual(call.kwargs["symbol"], "CAPR")
        self.assertEqual(call.kwargs["side"], "buy")
        self.assertEqual(call.kwargs["notional"], 110.1)
        self.assertEqual(call.kwargs["winner_hash"], "global-hash")
        self.assertNotIn("initial_capital", call.kwargs)

    def test_same_winner_target_is_hold_for_already_invested_user(self):
        self.client.get_all_positions.return_value = [
            SimpleNamespace(symbol="CAPR", qty="4.25", side="long"),
        ]
        result, execute, *_ = self._run(user=self.user_b)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["action"], "hold")
        execute.assert_not_called()
        self.client.submit_order.assert_not_called()

    def test_other_position_rotates_but_only_account_specific_orders(self):
        self.client.get_account.return_value = SimpleNamespace(
            id="trader-b-account", status="ACTIVE",
            account_blocked=False, trading_blocked=False,
            trade_suspended_by_user=False, cash="199.27",
        )
        self.client.get_all_positions.side_effect = [
            [SimpleNamespace(symbol="MAN", qty="12.5", side="long")],
            [],
        ]
        result, execute, oauth, shared, *_ = self._run(user=self.user_b)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["action"], "rotate")
        self.assertEqual(execute.call_count, 2)
        sell, buy = execute.call_args_list
        self.assertEqual(sell.kwargs["account"], "trader-b-account")
        self.assertEqual(sell.kwargs["side"], "sell")
        self.assertEqual(sell.kwargs["symbol"], "MAN")
        self.assertEqual(sell.kwargs["quantity"], 12.5)
        self.assertEqual(buy.kwargs["account"], "trader-b-account")
        self.assertEqual(buy.kwargs["side"], "buy")
        self.assertEqual(buy.kwargs["symbol"], "CAPR")
        self.assertEqual(buy.kwargs["notional"], 199.27)
        self.assertEqual(sell.kwargs["winner_hash"], buy.kwargs["winner_hash"])

    def test_exit_only_does_not_buy_or_rotate(self):
        result, execute, *_ = self._run(mode="exit_only")
        self.assertEqual(result["status"], "blocked_by_exit_only_mode")
        execute.assert_not_called()

    def test_missing_shared_signal_has_no_order(self):
        with (patch.object(executor, "live_control_status",
                           return_value={"control_mode": "active"}),
              patch.object(executor, "connected_trading_client",
                           return_value=self.client),
              patch.object(executor, "read_shared_winner_signal",
                           return_value=None),
              patch.object(executor, "_execute_leg") as execute):
            result = executor.execute_shared_winner_for_oauth_account(
                self.db, self.user_a, session="2026-11-05",
            )
        self.assertEqual(result["status"], "shared_winner_signal_pending")
        execute.assert_not_called()

    def test_order_identifiers_are_already_isolated_by_account(self):
        self.assertNotEqual(
            _unique_order_id("trader-a-account", "2026-11-05", "buy", "CAPR"),
            _unique_order_id("trader-b-account", "2026-11-05", "buy", "CAPR"),
        )


if __name__ == "__main__":
    unittest.main()
