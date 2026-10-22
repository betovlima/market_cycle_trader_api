"""Evidence-based Live order provenance, day-before decisions and no-order HOLD.

All tests use mocks; no Alpaca orders, Mongo writes, or requests.
"""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_portfolio import (
    _order, _verified_order_origins, _last_verified_live_execution, LIVE_RUNS, ORDER_LEDGER,
)
from market_cycle_trader_api.services.live_trader_state import LIVE_STATE
from market_cycle_trader_api.services.live_winner_candidates import (
    latest_live_winner_decision,
)


def broker_order(order_id, client_id):
    return SimpleNamespace(
        id=order_id, client_order_id=client_id, symbol="CAPR",
        qty="16.89", filled_qty="16.89", side="buy", status="filled",
        filled_avg_price="6.75",
    )


class LiveOrderProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.tables = {name: MagicMock() for name in (LIVE_RUNS, ORDER_LEDGER)}
        self.db.__getitem__.side_effect = self.tables.__getitem__
        self.orders = [
            broker_order("manual-b1", "manual-123"),
            broker_order("auto-b1", "mct-live-buy-123"),
            broker_order("unknown-b1", "external-123"),
            broker_order("unknown-mct", "mct-live-buy-not-in-ledger"),
        ]
        self.db[LIVE_RUNS].find.return_value = [{
            "manual_reconciliation": {"origin": "manual_alpaca_fills",
                                      "order_ids": ["manual-b1", "manual-s1"]},
        }]
        self.db[ORDER_LEDGER].find.return_value = [{"_id": "mct-live-buy-123"}]

    def test_exact_receipts_distinguish_manual_automatic_unknown(self):
        verified = _verified_order_origins(self.db, "acct", self.orders)
        expected = {
            "manual-b1": "manual_alpaca_fills",
            "auto-b1": "live_automated_verified",
        }
        self.assertEqual(verified, expected)
        self.assertEqual(
            _order(self.orders[0], execution_origin=verified["manual-b1"])["execution_origin"],
            "manual_alpaca_fills",
        )
        self.assertEqual(_order(self.orders[2])["execution_origin"], "live_broker_unclassified")
        self.assertEqual(_order(self.orders[3])["execution_origin"], "live_broker_unclassified")
        self.db[LIVE_RUNS].find.assert_called_once()
        self.db[ORDER_LEDGER].find.assert_called_once()
        self.assertEqual(
            self.db[LIVE_RUNS].find.call_args.args[0]["live_account_id"], "acct",
        )
        self.assertEqual(
            self.db[ORDER_LEDGER].find.call_args.args[0]["broker_account_id"], "acct",
        )
        self.db[LIVE_RUNS].update_one.assert_not_called()

    def test_contradictory_manual_and_automated_receipts_are_unclassified(self):
        self.db[LIVE_RUNS].find.return_value = [{
            "manual_reconciliation": {"order_ids": ["auto-b1"]},
        }]
        verified = _verified_order_origins(self.db, "acct", self.orders)
        self.assertEqual(verified.get("auto-b1"), "live_broker_unclassified")

    def test_unknown_orders_never_assumed_manual(self):
        self.db[LIVE_RUNS].find.return_value = []
        self.db[ORDER_LEDGER].find.return_value = []
        verified = _verified_order_origins(self.db, "acct", self.orders)
        self.assertEqual(verified, {})


class LiveDecisionOriginTests(unittest.TestCase):
    def decision(self, status, action, manual=None):
        db = MagicMock()
        db[LIVE_RUNS].find_one.return_value = {
            "_id": "acct:2026-10-22",
            "execution_session": "2026-10-22", "decision_date": "2026-10-21",
            "winner_signal_prepared": True, "status": status, "action": action,
            "current_asset": "CAPR", "target_asset": "CAPR",
            "raw_best_asset": "CAPR", "utilities": {"CAPR": 1.1, "CASH": 0.1},
            "cash_edges": {}, "manual_reconciliation": manual,
        }
        return latest_live_winner_decision(db, account_id="acct")

    def test_hold_completes_without_broker_order_origin(self):
        audit = self.decision("completed", "hold")
        self.assertEqual(audit["execution_origin"], "live_no_orders")
        self.assertEqual(audit["decision_origin"], "live_winner_model")

    def test_pending_decision_is_not_executed(self):
        self.assertEqual(
            self.decision("recalibrated_no_orders", "hold")["execution_origin"],
            "live_not_executed",
        )

    def test_manually_adopted_fills_keep_manual_provenance(self):
        audit = self.decision("manual_reconciled", "rotate", {
            "origin": "manual_alpaca_fills",
        })
        self.assertEqual(audit["execution_origin"], "manual_alpaca_fills")

    def test_filled_automatic_plan_keeps_broker_origin(self):
        self.assertEqual(
            self.decision("completed", "buy")["execution_origin"], "live_broker",
        )



class LastVerifiedLiveExecutionTests(unittest.TestCase):
    """Pure mocked Mongo snapshot checks: no broker requests or financial writes."""

    def setUp(self):
        self.db = MagicMock()
        self.broker_time = datetime(2026, 10, 22, 16, 40, tzinfo=timezone.utc)
        self.position = [{"symbol": "CAPR", "quantity": 16.89037037}]
        self.state = {
            "managed_symbol": "CAPR", "managed_quantity": 16.89037037,
            "last_execution_session": "2026-10-20",
            "last_reconciliation_origin": "manual_alpaca_fills",
            "last_reconciliation_session": "2026-10-20",
        }
        self.db[LIVE_STATE].find_one.return_value = self.state

    def read(self, positions=None):
        return _last_verified_live_execution(
            self.db, account_id="acct-1", positions=self.position if positions is None else positions,
            recorded_now=self.broker_time,
        )

    def test_verified_manual_adoption_has_date_and_manual_source(self):
        self.assertEqual(self.read(), {
            "session": "2026-10-20", "origin": "manual_alpaca_fills",
        })
        self.db[LIVE_STATE].find_one.assert_called_once()
        self.assertEqual(self.db[LIVE_STATE].find_one.call_args.args[0], {"_id": "acct-1"})
        self.db[LIVE_STATE].update_one.assert_not_called()

    def test_position_divergence_never_shows_stale_execution(self):
        self.assertEqual(
            self.read([{"symbol": "MAN", "quantity": 2.668639053}]),
            {"session": None, "origin": None},
        )

    def test_broker_quantity_divergence_never_shows_stale_execution(self):
        self.state["managed_quantity"] = 14.0
        self.assertEqual(self.read(), {"session": None, "origin": None})

    def test_later_automatic_trade_does_not_inherit_manual_origin(self):
        self.state["last_execution_session"] = "2026-10-21"
        self.assertEqual(self.read(), {
            "session": "2026-10-21", "origin": None,
        })

    def test_missing_state_or_invalid_session_has_no_inferred_trade(self):
        self.db[LIVE_STATE].find_one.return_value = None
        self.assertEqual(self.read(), {"session": None, "origin": None})
        self.db[LIVE_STATE].find_one.return_value = self.state
        self.state["last_execution_session"] = "2026-10-23"
        self.assertEqual(self.read(), {"session": None, "origin": None})
        self.state["last_execution_session"] = "2026-10-20junk"
        self.assertEqual(self.read(), {"session": None, "origin": None})

    def test_cash_state_can_preserve_last_sale_without_manual_inference(self):
        self.state.update({
            "managed_symbol": "CASH", "managed_quantity": 0.0,
            "last_execution_session": "2026-10-21",
        })
        self.assertEqual(self.read([]), {
            "session": "2026-10-21", "origin": None,
        })


if __name__ == "__main__":
    unittest.main()
