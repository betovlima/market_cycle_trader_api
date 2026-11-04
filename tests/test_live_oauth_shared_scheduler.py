"""One Winner/market signal, multiple OAuth accounts; all calls are mocked."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import (
    live_observation_scheduler,
    live_oauth_scheduler,
)
from market_cycle_trader_api.services.live_shared_winner_signal import SharedWinnerSignal


class OAuthSharedSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.grants = MagicMock()
        self.db[live_oauth_scheduler.CONNECTIONS] = self.grants

    def _signal(self):
        return SharedWinnerSignal(
            execution_session="2026-11-05", decision_date="2026-11-04",
            target_asset="CAPR", winner_strategy_id="one-Winner",
            winner_revision=5, winner_configuration_hash="same-config",
            snapshot_sha256="one-snapshot",
            source_account_id="primary-legacy-account",
        )

    def test_stopped_local_live_never_queries_tokens_or_sends_orders(self):
        with (patch.object(live_oauth_scheduler, "live_control_status",
                           return_value={"control_mode": "stopped"}),
              patch.object(live_oauth_scheduler, "read_shared_winner_signal") as signal,
              patch.object(live_oauth_scheduler, "execute_shared_winner_for_oauth_account") as orders):
            result = live_oauth_scheduler.execute_oauth_accounts_for_shared_winner(
                self.db, session="2026-11-05",
            )
        self.assertEqual(result, {"status": "live_stopped", "processed": 0})
        signal.assert_not_called()
        orders.assert_not_called()

    def test_canonical_account_excluded_and_each_authorized_user_is_independent(self):
        self.db[live_oauth_scheduler.CONNECTIONS].find.return_value = [
            {"subject": "legacy-admin", "role": "admin",
             "account_id": "primary-legacy-account"},
            {"subject": "google:trader-A", "role": "trader",
             "account_id": "alpaca-A"},
            {"subject": "google:trader-B", "role": "trader",
             "account_id": "alpaca-B"},
        ]
        def fake_execute(db, identity, session):
            self.assertEqual(session, "2026-11-05")
            self.assertIn(identity.subject, {"google:trader-A", "google:trader-B"})
            if identity.subject == "google:trader-A":
                raise RuntimeError("Account A token revoked")
            return {"status": "completed", "action": "hold"}
        with (patch.object(live_oauth_scheduler, "live_control_status",
                           return_value={"control_mode": "active"}),
              patch.object(live_oauth_scheduler, "read_shared_winner_signal",
                           return_value=self._signal()) as shared,
              patch.object(live_oauth_scheduler, "execute_shared_winner_for_oauth_account",
                           side_effect=fake_execute) as execute):
            result = live_oauth_scheduler.execute_oauth_accounts_for_shared_winner(
                self.db, session="2026-11-05",
            )
        self.assertEqual(result, {
            "status": "processed", "processed": 2, "completed": 1, "failed": 1,
        })
        shared.assert_called_once_with(self.db, session="2026-11-05")
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            {call.args[1].subject for call in execute.call_args_list},
            {"google:trader-A", "google:trader-B"},
        )
        self.db[live_oauth_scheduler.CONNECTIONS].find.assert_called_once()

    def test_no_shared_signal_means_no_user_execution(self):
        with (patch.object(live_oauth_scheduler, "live_control_status",
                           return_value={"control_mode": "active"}),
              patch.object(live_oauth_scheduler, "read_shared_winner_signal",
                           return_value=None),
              patch.object(live_oauth_scheduler, "execute_shared_winner_for_oauth_account") as orders):
            result = live_oauth_scheduler.execute_oauth_accounts_for_shared_winner(
                self.db, session="2026-11-05",
            )
        self.assertEqual(result["status"], "shared_signal_pending")
        orders.assert_not_called()

    def test_existing_live_scheduler_retains_legacy_and_adds_shared_fanout(self):
        source = inspect.getsource(live_observation_scheduler.LiveObservationScheduler._run)
        self.assertIn("prepare_live_winner_signal(", source)
        self.assertIn("execute_prepared_live_winner_session(", source)
        self.assertIn("execute_oauth_accounts_for_shared_winner(", source)


if __name__ == "__main__":
    unittest.main()
