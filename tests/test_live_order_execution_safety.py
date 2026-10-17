import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from pymongo.errors import DuplicateKeyError
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_order_execution import (
    execute_recalibrated_live_winner_session, _unique_order_id, _execute_leg, _full_live_cash_notional,
)


class LiveOrderSafetyTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_order_execution.live_control_status", return_value={"control_mode": "stopped"})
    @patch("market_cycle_trader_api.services.live_order_execution.create_live_execution_client")
    def test_stopped_does_not_trade(self, broker_factory, control):
        result = execute_recalibrated_live_winner_session(MagicMock(), "2026-10-16")
        self.assertEqual(result["status"], "blocked_by_live_mode")
        broker_factory.assert_not_called()

    def test_mocked_live_buy_uses_all_broker_cash_and_persists_state(self):
        run = {
            "_id": "acct:2026-10-16", "execution_session": "2026-10-16",
            "status": "recalibrated_no_orders", "action": "buy",
            "target_asset": "NVDA", "current_asset": "CASH",
            "winner_signal_prepared": True, "winner_strategy_id": "winner",
            "winner_configuration_hash": "hash", "winner_revision": 2,
            "live_account_id": "acct", "live_state_revision": 0,
            "session_equity_baseline": 108.25,
            "expected_market_open": datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc),
        }
        db = MagicMock()
        db["live_winner_session_runs"].find_one.return_value = run
        db["live_winner_session_runs"].find_one_and_update.return_value = run
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id="acct", status="active", account_blocked=False,
            trading_blocked=False, trade_suspended_by_user=False,
            equity=108.25, cash=108.25, buying_power=108.25,
        )
        broker.get_clock.return_value = SimpleNamespace(
            is_open=True, timestamp=datetime(2026, 10, 16, 13, 40, tzinfo=timezone.utc),
        )
        broker.get_orders.return_value = []
        state = {
            "account_id": "acct", "managed_symbol": "CASH",
            "managed_quantity": 0, "revision": 0,
        }
        prefix = "market_cycle_trader_api.services.live_order_execution."
        with ExitStack() as stack:
            stack.enter_context(patch(prefix + "live_control_status", return_value={"control_mode": "active"}))
            stack.enter_context(patch(prefix + "create_live_execution_client", return_value=broker))
            stack.enter_context(patch(prefix + "get_trader_winner_summary", return_value={
                "id": "winner", "revision": 2, "configuration_hash": "hash",
            }))
            stack.enter_context(patch(prefix + "asset_snapshot", return_value={
                "symbol": "NVDA", "tradable": True,
                "fractionable": True, "status": "active",
            }))
            stack.enter_context(patch(prefix + "verified_live_state", return_value=state))
            submit = stack.enter_context(patch(prefix + "_execute_leg", return_value=SimpleNamespace(filled_quantity=0.2)))
            persist = stack.enter_context(patch(prefix + "confirm_live_fill_state"))
            result = execute_recalibrated_live_winner_session(db, "2026-10-16")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(submit.call_args.kwargs["notional"], 108.25)
        persist.assert_called_once()
        broker.submit_order.assert_not_called()

    def test_duplicate_persisted_intent_blocks_broker(self):
        db = MagicMock()
        db["live_winner_broker_orders"].insert_one.side_effect = DuplicateKeyError("duplicate")
        broker = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            _execute_leg(
                db, broker, account="acct", session="2026-10-16",
                side="buy", symbol="NVDA", quantity=None, notional=25.0,
                winner_hash="hash",
            )
        broker.submit_order.assert_not_called()

    def test_cash_notional_uses_full_broker_amount(self):
        self.assertEqual(_full_live_cash_notional(SimpleNamespace(cash="108.259")), 108.25)
        self.assertEqual(_full_live_cash_notional(SimpleNamespace(cash="12000.99")), 12000.99)
        self.assertEqual(_full_live_cash_notional(SimpleNamespace(cash="1.00")), 1.0)

    def test_buy_rejects_unavailable_or_insufficient_broker_cash(self):
        for balance in ("0", "0.99", "-3", "NaN", None):
            with self.subTest(balance=balance), self.assertRaises(RuntimeError):
                _full_live_cash_notional(SimpleNamespace(cash=balance))

    def test_deterministic_client_order_identity(self):
        a = _unique_order_id("a", "2026-10-16", "buy", "NVDA")
        self.assertEqual(a, _unique_order_id("a", "2026-10-16", "buy", "NVDA"))
        self.assertNotEqual(a, _unique_order_id("a", "2026-10-17", "buy", "NVDA"))
