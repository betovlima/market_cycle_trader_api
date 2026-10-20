"""Broker-execution diagnostics must be useful, bounded and strictly read-only."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import live_order_execution
from market_cycle_trader_api.services.live_workflow_monitor import (
    LIVE_ATTEMPTS, LIVE_EVENTS, LIVE_RUNS, ORDER_LEDGER,
    _safe_broker_failure, execute_monitored_live_stage, live_workflow_monitor,
)

SESSION = "2026-10-20"
PREFIX = "market_cycle_trader_api.services.live_workflow_monitor."


class LiveBrokerFailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.collections = {
            name: MagicMock() for name in (
                LIVE_ATTEMPTS, LIVE_EVENTS, LIVE_RUNS, ORDER_LEDGER
            )
        }
        self.db.__getitem__.side_effect = self.collections.__getitem__
        self.collections[LIVE_RUNS].find_one.return_value = {
            "execution_session": SESSION, "status": "recalibrated_no_orders",
            "live_account_id": "acct",
            "winner_strategy_id": "winner-1", "winner_revision": 2,
            "winner_configuration_hash": "h",
        }

    def test_winner_identity_preflight_reports_exact_safe_guard_and_no_order(self):
        submitted = MagicMock()
        with (
            patch.object(live_order_execution, "live_control_status",
                         return_value={"control_mode": "active"}),
            patch.object(live_order_execution, "get_trader_winner_summary",
                         return_value={"id": "winner-2", "revision": 2,
                                       "configuration_hash": "h"}),
            patch.object(live_order_execution, "create_live_execution_client",
                         submitted),
            patch(PREFIX + "append_live_event") as event,
        ):
            with self.assertRaisesRegex(RuntimeError, "Winner changed since"):
                execute_monitored_live_stage(
                    self.db, session=SESSION, stage="broker_execution",
                    brokerage=True,
                    operation=lambda: live_order_execution.execute_prepared_live_winner_session(
                        self.db, SESSION
                    ),
                )
        self.assertEqual(
            self.collections[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]["status"],
            "reconciliation_required",
        )
        fields = self.collections[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(fields["error_code"], "winner_identity_mismatch")
        self.assertRegex(fields["error_location"], r"^live_order_execution\.py:\d+$")
        self.assertNotIn("/", fields["error_location"])
        self.assertNotIn("\\", fields["error_location"])
        self.assertEqual(event.call_args.kwargs["details"]["error_code"],
                         "winner_identity_mismatch")
        submitted.assert_not_called()
        self.collections[ORDER_LEDGER].insert_one.assert_not_called()
        self.collections[LIVE_RUNS].update_one.assert_not_called()

    def test_unknown_provider_message_is_not_leaked_or_retried(self):
        message = "private broker account=123 secret=abc https://broker.invalid/"
        operation = MagicMock(side_effect=RuntimeError(message))
        with patch(PREFIX + "append_live_event") as event:
            with self.assertRaisesRegex(RuntimeError, "private broker"):
                execute_monitored_live_stage(
                    self.db, session=SESSION, stage="broker_execution",
                    brokerage=True, operation=operation,
                )
        operation.assert_called_once()
        fields = self.collections[LIVE_ATTEMPTS].update_one.call_args.args[1]["$set"]
        self.assertEqual(fields["error_code"], "broker_execution_unclassified")
        self.assertEqual(fields["error_location"], "unclassified")
        recorded = event.call_args.kwargs
        self.assertEqual(recorded["details"]["error_code"], "broker_execution_unclassified")
        self.assertNotIn("private", str(recorded))
        self.assertNotIn("secret", str(recorded))
        self.assertNotIn("broker.invalid", str(recorded))
        self.collections[ORDER_LEDGER].insert_one.assert_not_called()

    def test_readonly_monitor_exposes_failure_fields_from_existing_attempts(self):
        self.collections[LIVE_ATTEMPTS].find.return_value.sort.return_value.limit.return_value = [{
            "attempt_id": "attempt-x", "stage": "broker_execution",
            "status": "reconciliation_required", "brokerage": True,
            "error_type": "RuntimeError", "error_code": "winner_identity_mismatch",
            "error_location": "live_order_execution.py:145",
            "started_at": datetime(2026, 10, 20, 15, 11, tzinfo=timezone.utc),
        }]
        self.collections[LIVE_EVENTS].find.return_value.sort.return_value.limit.return_value = []
        self.collections[ORDER_LEDGER].count_documents.return_value = 0
        with patch(PREFIX + "create_live_readonly_client") as factory:
            factory.return_value.get_account.return_value = SimpleNamespace(id="acct")
            result = live_workflow_monitor(self.db, session=SESSION)
        attempt = result["attempts"][0]
        self.assertEqual(attempt["error_code"], "winner_identity_mismatch")
        self.assertEqual(attempt["error_location"], "live_order_execution.py:145")
        self.assertEqual(result["orders_requiring_reconciliation"], 0)
        self.collections[ORDER_LEDGER].insert_one.assert_not_called()

    def test_unknown_error_without_trusted_stack_has_no_private_path(self):
        failure = RuntimeError("provider token abc")
        self.assertEqual(_safe_broker_failure(failure), {
            "error_code": "broker_execution_unclassified",
            "error_location": "unclassified",
        })


if __name__ == "__main__":
    unittest.main()
