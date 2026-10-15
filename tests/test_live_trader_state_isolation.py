import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_trader_state import verified_live_state


class LiveStateIsolationTests(unittest.TestCase):
    def test_new_flat_live_account_initializes_without_paper(self):
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="live-a")
        broker.get_all_positions.return_value = []
        db = MagicMock()
        db["live_trader_strategy_state"].find_one.side_effect = [
            None, {"managed_symbol": "CASH", "managed_quantity": 0,
                   "revision": 0, "entry_session": None},
        ]
        state = verified_live_state(db, broker, session="2026-10-16")
        self.assertEqual(state["managed_symbol"], "CASH")
        self.assertEqual(state["holding_sessions"], 0)
        db["live_trader_strategy_state"].update_one.assert_called_once()

    def test_unmanaged_existing_position_blocks_automation(self):
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="live-a")
        broker.get_all_positions.return_value = [
            SimpleNamespace(symbol="NVDA", qty="0.5", side="long")
        ]
        db = MagicMock()
        db["live_trader_strategy_state"].find_one.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Unmanaged Live position"):
            verified_live_state(db, broker, session="2026-10-16")
        broker.submit_order.assert_not_called()

    def test_changed_quantity_blocks_automation(self):
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="live-a")
        broker.get_all_positions.return_value = [
            SimpleNamespace(symbol="NVDA", qty="2.0", side="long")
        ]
        db = MagicMock()
        db["live_trader_strategy_state"].find_one.return_value = {
            "managed_symbol": "NVDA", "managed_quantity": 1.0,
            "entry_session": "2026-10-15", "revision": 2,
        }
        with self.assertRaisesRegex(RuntimeError, "differ"):
            verified_live_state(db, broker, session="2026-10-16")
        broker.submit_order.assert_not_called()
