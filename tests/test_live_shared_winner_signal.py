"""Shared Winner target is computed once and never sourced from another user's money."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services import (
    live_shared_winner_signal as shared,
    live_oauth_winner,
)


def canonical(session="2026-11-05", *, status="completed", target="CAPR"):
    return {
        "_id": "primary-account:" + session,
        "live_account_id": "primary-account",
        "execution_session": session,
        "decision_date": "2026-11-04",
        "historical_cutoff": "2026-11-04",
        "analysis_mode": "previous_completed_session_daily",
        "status": status,
        "winner_signal_prepared": True,
        "post_close_signal_ready": True,
        "snapshot_sha256": "sha-final-daily",
        "action": "hold",
        "target_asset": target,
        "winner_strategy_id": "winner-1",
        "winner_revision": 7,
        "winner_configuration_hash": "sha-strategy",
    }


def winner():
    return {
        "id": "winner-1",
        "revision": 7,
        "configuration_hash": "sha-strategy",
        "name": "Shared Winner",
    }


class SharedWinnerSignalTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()

    def _read(self, session="2026-11-05"):
        with (patch.object(shared, "_canonical_account_id", return_value="primary-account"),
              patch("market_cycle_trader_api.infrastructure.trading.canonical_live_oauth.canonical_live_auth_mode",
                    return_value="api_keys"),
              patch.object(shared, "get_trader_winner_summary", return_value=winner())):
            return shared.read_shared_winner_signal(self.db, session=session)

    def test_reuses_persisted_immutable_daily_signal_without_training(self):
        self.db[shared.LIVE_RUNS].find_one.return_value = canonical()
        actual = self._read()
        self.assertEqual(actual.target_asset, "CAPR")
        self.assertEqual(actual.decision_date, "2026-11-04")
        self.assertEqual(actual.winner_configuration_hash, "sha-strategy")
        self.db[shared.LIVE_RUNS].find_one.assert_called_once_with(
            {"_id": "primary-account:2026-11-05"}
        )
        self.db[shared.LIVE_RUNS].update_one.assert_not_called()
        self.db[shared.LIVE_RUNS].insert_one.assert_not_called()

    def test_source_can_execute_before_other_accounts_without_losing_signal(self):
        for status in ("recalibrated_no_orders", "executing", "completed"):
            with self.subTest(status=status):
                self.db[shared.LIVE_RUNS].find_one.return_value = canonical(status=status)
                self.assertEqual(self._read().target_asset, "CAPR")

    def test_pending_signal_is_not_reused(self):
        for status in ("waiting_for_winner_signal", "preparing_signal", "review_required"):
            with self.subTest(status=status):
                self.db[shared.LIVE_RUNS].find_one.return_value = canonical(status=status)
                self.assertIsNone(self._read())

    def test_date_snapshot_or_winner_change_cannot_be_used_as_shared_signal(self):
        failures = (
            {"decision_date": "2026-11-03"},
            {"snapshot_sha256": ""},
            {"winner_signal_prepared": False},
            {"historical_cutoff": "2026-11-03"},
            {"live_account_id": "other-account"},
            {"target_asset": ""},
            {"analysis_mode": "intraday"},
        )
        for bad in failures:
            with self.subTest(bad=bad):
                self.db[shared.LIVE_RUNS].find_one.return_value = {
                    **canonical(), **bad,
                }
                with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                    self._read()

        self.db[shared.LIVE_RUNS].find_one.return_value = canonical()
        with (patch.object(shared, "_canonical_account_id", return_value="primary-account"),
              patch.object(shared, "get_trader_winner_summary", return_value={
                  **winner(), "revision": 8,
              })):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                shared.read_shared_winner_signal(self.db, session="2026-11-05")

    def test_non_market_date_is_never_a_live_signal(self):
        with self.assertRaises(ValueError):
            self._read("2026-11-08")


class AccountPreviewWithSharedWinnerTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.connections = MagicMock()
        self.sessions = MagicMock()
        self.orders = MagicMock()
        collections = {
            "alpaca_oauth_connections": self.connections,
            "live_winner_session_runs": self.sessions,
            "live_winner_broker_orders": self.orders,
        }
        self.db.__getitem__.side_effect = lambda name: collections[name]
        self.identity = SimpleNamespace(subject="trader-one")
        self.db["alpaca_oauth_connections"].find_one.return_value = {
            "subject": "trader-one", "environment": "live",
            "account_id": "their-own-account",
            "trading_authorized": True, "oauth_scope": "trading",
            "read_only": False,
        }
        self.broker = MagicMock()
        self.broker.get_account.return_value = SimpleNamespace(
            id="their-own-account", status="ACTIVE",
            cash="100.00", equity="500.00",
        )
        from datetime import datetime, timezone
        self.broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc),
            is_open=True, next_open=None,
        )

    def _preview(self, positions):
        self.broker.get_all_positions.return_value = positions
        signal = shared.SharedWinnerSignal(
            execution_session="2026-11-05", decision_date="2026-11-04",
            target_asset="CAPR", winner_strategy_id="winner-1",
            winner_revision=7, winner_configuration_hash="sha-strategy",
            snapshot_sha256="sha-daily", source_account_id="primary-account",
        )
        with (patch.object(live_oauth_winner, "connected_trading_client",
                           return_value=self.broker),
              patch.object(live_oauth_winner, "read_shared_winner_signal",
                           return_value=signal),
              patch.object(live_oauth_winner, "_oauth_winner_operational",
                           return_value=False),
              patch.object(live_oauth_winner, "get_trader_winner_summary",
                           return_value=winner())):
            return live_oauth_winner.oauth_winner_readiness(self.db, self.identity)

    def test_same_signal_becomes_buy_hold_or_rotate_by_alpaca_position(self):
        cases = (
            ([], "buy", "CASH"),
            ([SimpleNamespace(symbol="CAPR", qty="5")], "hold", "CAPR"),
            ([SimpleNamespace(symbol="MAN", qty="7")], "rotate", "MAN"),
        )
        for positions, action, asset in cases:
            with self.subTest(action=action):
                preview = self._preview(positions)
                self.assertEqual(preview["shared_target_asset"], "CAPR")
                self.assertTrue(preview["shared_signal_prepared"])
                self.assertEqual(preview["current_asset"], asset)
                self.assertEqual(preview["account_action"], action)
                self.assertFalse(preview["order_submission_enabled"])
        self.broker.submit_order.assert_not_called()
        self.db["live_winner_session_runs"].find_one.assert_not_called()
        self.db["live_winner_broker_orders"].insert_one.assert_not_called()

    def test_more_than_one_position_is_reported_without_inventing_action(self):
        positions = [
            SimpleNamespace(symbol="CAPR", qty="3"),
            SimpleNamespace(symbol="MAN", qty="2"),
        ]
        result = self._preview(positions)
        self.assertIsNone(result["account_action"])
        self.assertEqual(result["position_count"], 2)
        self.assertFalse(result["order_submission_enabled"])
        self.broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
