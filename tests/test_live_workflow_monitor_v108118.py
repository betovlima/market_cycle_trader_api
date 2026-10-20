"""Read-only Live supervision and per-observation Tenacity integration."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_workflow_monitor import (
    LIVE_ATTEMPTS, LIVE_EVENTS, LIVE_RUNS, ORDER_LEDGER,
    LiveStageMachine, execute_monitored_live_stage, live_workflow_monitor,
)

SESSION = "2026-10-20"
PREFIX = "market_cycle_trader_api.services.live_workflow_monitor."


class LiveWorkflowMonitorTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        # Different collections must have independent mock cursors.
        self.collections = {
            name: MagicMock() for name in (
                LIVE_RUNS, LIVE_ATTEMPTS, LIVE_EVENTS, ORDER_LEDGER
            )
        }
        self.db.__getitem__.side_effect = self.collections.__getitem__
        self.db[LIVE_RUNS].find_one.return_value = {
            "_id": "acct:2026-10-20",
            "live_account_id": "acct",
            "execution_session": SESSION,
            "status": "waiting_for_winner_signal",
            "phase": "post_close_signal_retry_wait",
            "preparation_attempts": 3,
        }

    def test_machine_prohibits_skipping_stages(self):
        machine = LiveStageMachine()
        self.assertEqual(machine.state, "pending")
        machine.start()
        machine.retry_from_start()
        self.assertEqual(machine.state, "retry_pending")

    def test_failed_calculation_is_invoked_once_per_observation_and_recorded(self):
        # The next scheduler invocation runs the complete calculation again.
        operation = MagicMock(side_effect=RuntimeError("data unavailable"))
        with self.assertRaisesRegex(RuntimeError, "data unavailable"):
            execute_monitored_live_stage(
                self.db, session=SESSION,
                stage="post_close_calculation", operation=operation,
            )
        operation.assert_called_once()
        record = self.db[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(record["status"], "retry_pending")
        self.assertEqual(record["error_type"], "RuntimeError")

        operation = MagicMock(return_value={"status": "recalibrated_no_orders"})
        output = execute_monitored_live_stage(
            self.db, session=SESSION,
            stage="post_close_calculation", operation=operation,
        )
        self.assertEqual(output["status"], "recalibrated_no_orders")
        operation.assert_called_once()
        record = self.db[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(self.db[LIVE_ATTEMPTS].insert_one.call_count, 2)

    def test_broker_execution_is_never_automatically_retried(self):
        operation = MagicMock(side_effect=RuntimeError("broker response unknown"))
        with self.assertRaisesRegex(RuntimeError, "broker response unknown"):
            execute_monitored_live_stage(
                self.db, session=SESSION, stage="broker_execution",
                operation=operation, brokerage=True,
            )
        operation.assert_called_once()
        record = self.db[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(record["status"], "reconciliation_required")

    def test_waiting_data_is_not_reported_as_failed(self):
        result = execute_monitored_live_stage(
            self.db, session=SESSION, stage="post_close_calculation",
            operation=lambda: {"status": "waiting_for_completed_daily_bar"},
        )
        self.assertEqual(result["status"], "waiting_for_completed_daily_bar")
        updated = self.db[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(updated["status"], "waiting")

    @patch(PREFIX + "create_live_readonly_client")
    def test_readonly_monitor_is_scoped_to_current_broker_account(self, factory):
        broker = factory.return_value
        broker.get_account.return_value = SimpleNamespace(id="acct")
        now = datetime(2026, 10, 20, 20, 30, tzinfo=timezone.utc)
        self.db[LIVE_ATTEMPTS].find.return_value.sort.return_value.limit.return_value = [
            {
                "attempt_id": "id-1", "status": "retry_pending",
                "stage": "post_close_calculation", "started_at": now,
                "error_type": "RuntimeError", "brokerage": False,
            }
        ]
        self.db[LIVE_EVENTS].find.return_value.sort.return_value.limit.return_value = [
            {"recorded_at": now, "stage": "signal_preparation_failed", "level": "error"}
        ]
        self.db[ORDER_LEDGER].count_documents.return_value = 1
        payload = live_workflow_monitor(self.db, session=SESSION)
        self.assertEqual(payload["execution_session"], SESSION)
        self.assertEqual(payload["recovery"], "broker_reconciliation_required")
        self.assertEqual(payload["attempts"][0]["error_type"], "RuntimeError")
        self.assertEqual(payload["events"][0]["stage"], "signal_preparation_failed")
        self.assertEqual(payload["orders_requiring_reconciliation"], 1)
        self.db[LIVE_RUNS].find_one.assert_called_with({
            "live_account_id": "acct", "execution_session": SESSION,
        })
        attempts_query = self.db[LIVE_ATTEMPTS].find.call_args.args[0]
        self.assertEqual(attempts_query["account_id"], "acct")
        broker.submit_order.assert_not_called()

    @patch(PREFIX + "create_live_readonly_client")
    def test_monitor_with_no_session_is_read_only(self, factory):
        factory.return_value.get_account.return_value = SimpleNamespace(id="acct")
        self.db[LIVE_RUNS].find_one.return_value = None
        payload = live_workflow_monitor(self.db)
        self.assertEqual(payload["status"], "no_live_session")
        self.db[LIVE_ATTEMPTS].insert_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
