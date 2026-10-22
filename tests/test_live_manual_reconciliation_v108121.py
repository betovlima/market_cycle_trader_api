"""Manual Alpaca rotation reconciliation; all broker operations are read-only."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from alpaca.trading.enums import QueryOrderStatus

from market_cycle_trader_api.services.live_manual_reconciliation import (
    manual_reconciliation_preview, reconcile_manual_live_rotation,
    ManualReconciliationBlocked, LIVE_RUNS, LIVE_STATE, LEDGER, AUDIT,
)

SESSION = "2026-10-20"
PRIOR = "2026-10-19"
PREPARED = datetime(2026, 10, 20, 15, 11, tzinfo=timezone.utc)


def _order(symbol, side, qty, order_id, minute):
    return SimpleNamespace(
        id=order_id, client_order_id="manual-" + order_id,
        symbol=symbol, side=side, filled_qty=str(qty), status="filled",
        filled_at=datetime(2026, 10, 20, 15, minute, tzinfo=timezone.utc),
    )


class ManualReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.tables = {name: MagicMock() for name in (LIVE_RUNS, LIVE_STATE, LEDGER, AUDIT)}
        self.db.__getitem__.side_effect = self.tables.__getitem__
        self.run = {
            "_id": "acct:" + SESSION, "live_account_id": "acct",
            "execution_session": SESSION, "status": "recalibrated_no_orders",
            "action": "rotate", "target_asset": "CAPR", "current_asset": "MAN",
            "current_quantity": 2.668639053,
            "decision_date": PRIOR,
            "historical_cutoff": PRIOR,
            "analysis_mode": "previous_completed_session_daily",
            "post_close_signal_ready": True,
            "winner_signal_prepared": True,
            "snapshot_sha256": "verified-daily-snapshot",
            "winner_strategy_id": "winner", "winner_revision": 2,
            "winner_configuration_hash": "hash",
        }
        self.state = {
            "_id": "acct", "managed_symbol": "MAN",
            "managed_quantity": 2.668639053, "revision": 3,
            "entry_session": "2026-10-19",
        }
        self.tables[LIVE_RUNS].find_one.return_value = self.run
        self.tables[LIVE_STATE].find_one.return_value = self.state
        self.tables[LEDGER].count_documents.return_value = 0
        self.tables[LIVE_RUNS].find_one_and_update.return_value = {**self.run, "status": "manual_reconciliation_in_progress"}
        self.tables[LIVE_STATE].find_one_and_update.return_value = {**self.state, "managed_symbol": "CAPR", "revision": 4}
        self.tables[LIVE_RUNS].update_one.return_value = SimpleNamespace(matched_count=1)
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(
            id="acct", trading_blocked=False, account_blocked=False,
        )
        self.broker.get_all_positions.return_value = [SimpleNamespace(
            symbol="CAPR", qty="16.890370370",
        )]
        self.orders = [
            _order("MAN", "sell", 1, "s1", 38),
            _order("MAN", "sell", 1.002106248, "s2", 40),
            _order("MAN", "sell", 0.666532805, "s3", 42),
            _order("CAPR", "buy", 16.890370370, "b1", 43),
        ]

        def orders(filter):
            return [] if filter.status == QueryOrderStatus.OPEN else self.orders
        self.broker.get_orders.side_effect = orders

        self.patches = [
            patch("market_cycle_trader_api.services.live_manual_reconciliation.create_live_readonly_client",
                  return_value=self.broker),
            patch("market_cycle_trader_api.services.live_manual_reconciliation.live_control_status",
                  return_value={"control_mode": "paused"}),
            patch("market_cycle_trader_api.services.live_manual_reconciliation.get_trader_winner_summary",
                  return_value={"id": "winner", "revision": 2, "configuration_hash": "hash"}),
        ]
        for ctx in self.patches:
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_preview_is_readonly_and_reports_fills(self):
        result = manual_reconciliation_preview(self.db, execution_session=SESSION)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["source"], "MAN")
        self.assertEqual(result["target"], "CAPR")
        self.assertEqual(result["sell_order_count"], 3)
        self.assertEqual(result["buy_order_count"], 1)
        self.assertAlmostEqual(result["broker_quantity"], 16.890370370)
        self.assertEqual(len(result["confirmation_token"]), 64)
        self.assertEqual(result["decision_date"], PRIOR)
        self.assertEqual(result["signal_contract"], "previous_completed_session_daily")
        self.tables[LIVE_STATE].find_one_and_update.assert_not_called()
        self.tables[LIVE_RUNS].find_one_and_update.assert_not_called()
        self.broker.submit_order.assert_not_called()
        self.broker.cancel_order_by_id.assert_not_called()

    def test_confirmed_reconciliation_retires_plan_with_no_orders(self):
        preview = manual_reconciliation_preview(self.db, execution_session=SESSION)
        result = reconcile_manual_live_rotation(
            self.db, execution_session=SESSION,
            confirmation_token=preview["confirmation_token"], actor="admin",
        )
        self.assertEqual(result["status"], "reconciled")
        claim = self.tables[LIVE_RUNS].find_one_and_update.call_args.args[1]["$set"]
        self.assertEqual(claim["status"], "manual_reconciliation_in_progress")
        updated = self.tables[LIVE_STATE].find_one_and_update.call_args.args[1]
        self.assertEqual(updated["$set"]["managed_symbol"], "CAPR")
        self.assertEqual(updated["$set"]["entry_session"], SESSION)
        self.assertEqual(updated["$set"]["last_reconciliation_origin"], "manual_alpaca_fills")
        self.assertEqual(updated["$inc"]["revision"], 1)
        finished = self.tables[LIVE_RUNS].update_one.call_args.args[1]["$set"]
        self.assertEqual(finished["status"], "manual_reconciled")
        self.assertEqual(finished["execution_origin"], "manual_alpaca_fills")
        self.assertFalse(finished["order_submission_enabled"])
        self.assertEqual(len(finished["manual_reconciliation"]["order_ids"]), 4)
        self.assertEqual(finished["manual_reconciliation"]["decision_date"], PRIOR)
        self.assertEqual(finished["manual_reconciliation"]["signal_contract"], "previous_completed_session_daily")
        self.broker.submit_order.assert_not_called()
        self.broker.cancel_order_by_id.assert_not_called()

    def test_recheck_blocks_changes_to_broker_after_claim(self):
        self.broker.get_all_positions.side_effect = [
            [SimpleNamespace(symbol="CAPR", qty="16.890370370")],
            [SimpleNamespace(symbol="CAPR", qty="16.890370370")],
            [SimpleNamespace(symbol="CAPR", qty="16.5")],
        ]
        preview = manual_reconciliation_preview(self.db, execution_session=SESSION)
        with self.assertRaises(ManualReconciliationBlocked):
            reconcile_manual_live_rotation(
                self.db, execution_session=SESSION,
                confirmation_token=preview["confirmation_token"], actor="admin",
            )
        self.tables[LIVE_STATE].find_one_and_update.assert_not_called()
        self.assertEqual(
            self.tables[LIVE_RUNS].update_one.call_args.args[1]["$set"]["status"],
            "review_required",
        )
        self.broker.submit_order.assert_not_called()

    def test_daily_signal_without_explicit_cutoff_compatibility(self):
        # v10.8.124 post-close records did not carry analysis_mode/cutoff.
        self.run.pop("historical_cutoff")
        self.run.pop("analysis_mode")
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "ready",
        )

    def test_old_intraday_plan_cannot_be_adopted_as_new_manual_reconciliation(self):
        self.run["decision_date"] = SESSION
        self.run["analysis_mode"] = "live_current_session_intraday"
        self.run["market_open_calibration_completed_at"] = PREPARED
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "blocked",
        )
        self.tables[LIVE_RUNS].find_one_and_update.assert_not_called()

    def test_stale_daily_cutoff_blocks_reconciliation(self):
        self.run["historical_cutoff"] = "2026-10-16"
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "blocked",
        )

    def test_missing_provenance_sha_blocks_reconciliation(self):
        self.run["snapshot_sha256"] = None
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "blocked",
        )

    def test_wrong_token_never_claims_session(self):
        with self.assertRaisesRegex(ManualReconciliationBlocked, "evidencia mudou"):
            reconcile_manual_live_rotation(
                self.db, execution_session=SESSION, confirmation_token="0" * 64, actor="admin",
            )
        self.tables[LIVE_RUNS].find_one_and_update.assert_not_called()

    def test_missing_broker_fills_blocks_with_no_changes(self):
        self.orders = self.orders[:-1]
        result = manual_reconciliation_preview(self.db, execution_session=SESSION)
        self.assertEqual(result["status"], "blocked")
        self.tables[LIVE_STATE].find_one_and_update.assert_not_called()

    def test_other_day_trade_blocks(self):
        self.orders.append(_order("AAPL", "buy", 1, "other", 41))
        result = manual_reconciliation_preview(self.db, execution_session=SESSION)
        self.assertEqual(result["status"], "blocked")

    def test_open_orders_block(self):
        self.broker.get_orders.side_effect = lambda filter: [SimpleNamespace(id="pending")] if filter.status == QueryOrderStatus.OPEN else self.orders
        result = manual_reconciliation_preview(self.db, execution_session=SESSION)
        self.assertEqual(result["status"], "blocked")

    def test_existing_mct_order_intent_blocks(self):
        self.tables[LEDGER].count_documents.return_value = 1
        result = manual_reconciliation_preview(self.db, execution_session=SESSION)
        self.assertEqual(result["status"], "blocked")

    def test_active_robot_requires_pause_without_changes(self):
        with patch("market_cycle_trader_api.services.live_manual_reconciliation.live_control_status",
                   return_value={"control_mode": "active"}):
            result = manual_reconciliation_preview(self.db, execution_session=SESSION)
            self.assertEqual(result["status"], "pause_required")
            with self.assertRaises(ManualReconciliationBlocked):
                reconcile_manual_live_rotation(
                    self.db, execution_session=SESSION,
                    confirmation_token=result["confirmation_token"], actor="admin",
                )
        self.tables[LIVE_STATE].find_one_and_update.assert_not_called()
        self.broker.submit_order.assert_not_called()

    def test_concurrent_claim_refuses_mutation(self):
        self.tables[LIVE_RUNS].find_one_and_update.return_value = None
        preview = manual_reconciliation_preview(self.db, execution_session=SESSION)
        with self.assertRaisesRegex(ManualReconciliationBlocked, "alterada"):
            reconcile_manual_live_rotation(
                self.db, execution_session=SESSION,
                confirmation_token=preview["confirmation_token"], actor="admin",
            )
        self.tables[LIVE_STATE].find_one_and_update.assert_not_called()

    def test_already_reconciled_rechecks_broker_state(self):
        self.run["status"] = "manual_reconciled"
        self.run["manual_reconciliation"] = {"target_symbol": "CAPR"}
        self.state["managed_symbol"] = "CAPR"
        self.state["managed_quantity"] = 16.890370370
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "already_reconciled",
        )
        self.broker.get_all_positions.return_value = [SimpleNamespace(symbol="MAN", qty="2")]
        self.assertEqual(
            manual_reconciliation_preview(self.db, execution_session=SESSION)["status"],
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()
