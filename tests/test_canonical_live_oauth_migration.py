"""Canonical Alpaca OAuth is configured by the administrator in Live, not .env.

All broker operations are mocked. No real orders, API keys or tokens are sent.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from market_cycle_trader_api.infrastructure.trading import canonical_live_oauth as auth
from market_cycle_trader_api.infrastructure.trading import alpaca_live
from market_cycle_trader_api.api.routers import live_market
from market_cycle_trader_api.services import live_control


class CanonicalLiveOAuthTests(unittest.TestCase):
    def setUp(self):
        self.account_id = "verified-canonical-live"
        self.subject = "google:administrator"
        self.key = Fernet.generate_key().decode("ascii")
        self.token = "only-used-by-mocks"
        self.state = {
            "_id": "default", "control_mode": "stopped", "revision": 7,
            "broker_auth_mode": "oauth", "oauth_orders_allowed": False,
            "oauth_subject": self.subject, "oauth_account_id": self.account_id,
        }
        self.grant = {
            "_id": auth._owner_key(self.subject),
            "subject": self.subject, "role": "admin", "environment": "live",
            "account_id": self.account_id, "trading_authorized": True,
            "read_only": False, "oauth_scope": "trading",
            "encrypted_token": Fernet(self.key.encode()).encrypt(
                self.token.encode()
            ).decode(),
        }
        self.db = MagicMock()
        self.control = MagicMock()
        self.connections = MagicMock()
        self.history = MagicMock()
        self.control.find_one.side_effect = lambda *args, **kw: dict(self.state)
        self.control.find_one_and_update.side_effect = self._save
        self.connections.find_one.return_value = self.grant
        self.db.__getitem__.side_effect = lambda key: {
            "live_trader_control": self.control,
            "live_trader_control_history": self.history,
            "alpaca_oauth_connections": self.connections,
        }.get(key, MagicMock())
        self.client = MagicMock()
        self.client.get_account.return_value = SimpleNamespace(id=self.account_id)
        self.client.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 11, 6, 15, tzinfo=timezone.utc),
            is_open=False,
        )
        self.identity = SimpleNamespace(
            is_admin=True, role="admin", subject=self.subject, email="admin@example.test",
        )
        self.env = patch.dict("os.environ", {
            "ALPACA_OAUTH_ENCRYPTION_KEY": self.key,
            "MCT_CANONICAL_LIVE_AUTH_MODE": "api_keys",
            "MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS": "1",
            "MCT_CANONICAL_LIVE_OAUTH_SUBJECT": "another-user",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def _save(self, query, update, **kwargs):
        if query.get("revision") != self.state["revision"]:
            return None
        self.state.update(update["$set"])
        self.state["revision"] += update["$inc"]["revision"]
        return dict(self.state)

    def test_old_environment_flags_have_no_effect_on_persisted_policy(self):
        self.assertEqual(auth.canonical_live_auth_mode(self.db), "oauth")
        self.assertFalse(auth.canonical_oauth_orders_enabled(self.db))
        self.assertEqual(auth.canonical_live_account_id(self.db), self.account_id)

    def test_canonical_broker_reads_by_persisted_verified_oauth_without_api_keys(self):
        with (
            patch("alpaca.trading.client.TradingClient", return_value=self.client) as factory,
            patch("market_cycle_trader_api.core.runtime.database", return_value=self.db),
            patch.object(alpaca_live, "live_credentials") as keys,
        ):
            read_client = alpaca_live.create_live_readonly_client()
            self.assertIs(read_client, self.client)
            with self.assertRaisesRegex(RuntimeError, "order submission is disabled"):
                alpaca_live.create_live_execution_client()
        factory.assert_called_once_with(oauth_token=self.token, paper=False)
        keys.assert_not_called()
        self.client.submit_order.assert_not_called()

    def test_enable_orders_requires_admin_explicit_second_action_and_robot_stopped(self):
        with patch("alpaca.trading.client.TradingClient", return_value=self.client):
            with self.assertRaisesRegex(live_control.LiveControlConflict, "First bind"):
                # Already bound, but an enable must use a separate action.
                self.state["broker_auth_mode"] = "api_keys"
                live_control.update_canonical_live_broker(
                    self.db, identity=self.identity, expected_revision=7,
                    enable_orders=True, confirmation="ENABLE_LIVE_REAL_ORDERS",
                )
            self.state["broker_auth_mode"] = "oauth"
            with patch("market_cycle_trader_api.api.routers.alpaca_oauth.connected_trading_client",
                       return_value=self.client):
                status = live_control.update_canonical_live_broker(
                    self.db, identity=self.identity, expected_revision=7,
                    enable_orders=True, confirmation="ENABLE_LIVE_REAL_ORDERS",
                )
        self.assertTrue(status["oauth_orders_allowed"])
        self.assertEqual(status["control_mode"], "stopped")
        self.assertFalse(status["order_submission_enabled"])
        self.assertEqual(self.state["revision"], 8)
        self.history.insert_one.assert_called_once()
        self.client.submit_order.assert_not_called()

    def test_bind_current_admin_oauth_from_legacy_starts_disarmed(self):
        self.state.update(broker_auth_mode="api_keys", oauth_orders_allowed=False,
                          oauth_subject=None, oauth_account_id=None)
        with patch("market_cycle_trader_api.api.routers.alpaca_oauth.connected_trading_client",
                   return_value=self.client):
            updated = live_control.update_canonical_live_broker(
                self.db, identity=self.identity, expected_revision=7,
                enable_orders=False, confirmation="BIND_VERIFIED_OAUTH_NO_ORDERS",
            )
        self.assertEqual(updated["mode"], "oauth")
        self.assertFalse(updated["oauth_orders_allowed"])
        self.assertEqual(updated["oauth_account_suffix"], self.account_id[-6:])
        self.assertEqual(self.state["oauth_subject"], self.subject)
        self.assertEqual(self.state["revision"], 8)

    def test_bind_rejects_another_user_or_non_admin_grant(self):
        self.state.update(broker_auth_mode="api_keys", oauth_account_id=None)
        self.grant["role"] = "trader"
        with self.assertRaisesRegex(live_control.LiveControlConflict, "administrator OAuth"):
            live_control.update_canonical_live_broker(
                self.db, identity=self.identity, expected_revision=7,
                enable_orders=False, confirmation="BIND_VERIFIED_OAUTH_NO_ORDERS",
            )
        self.control.find_one_and_update.assert_not_called()

    def test_orders_can_be_disarmed_while_active_without_reconnecting_broker(self):
        self.state.update(control_mode="active", oauth_orders_allowed=True)
        with patch("market_cycle_trader_api.api.routers.alpaca_oauth.connected_trading_client") as broker:
            result = live_control.update_canonical_live_broker(
                self.db, identity=self.identity, expected_revision=7,
                enable_orders=False, confirmation="DISABLE_LIVE_REAL_ORDERS",
            )
        self.assertFalse(result["order_submission_enabled"])
        broker.assert_not_called()
        self.assertFalse(self.state["oauth_orders_allowed"])

    def test_running_robot_cannot_enable_or_rebind_oauth(self):
        self.state.update(control_mode="active")
        with self.assertRaisesRegex(live_control.LiveControlConflict, "Stop the Live robot"):
            live_control.update_canonical_live_broker(
                self.db, identity=self.identity, expected_revision=7,
                enable_orders=True, confirmation="ENABLE_LIVE_REAL_ORDERS",
            )
        self.control.find_one_and_update.assert_not_called()

    def test_stale_revision_does_not_write_or_arm(self):
        with self.assertRaisesRegex(live_control.LiveControlConflict, "Refresh"):
            live_control.update_canonical_live_broker(
                self.db, identity=self.identity, expected_revision=6,
                enable_orders=True, confirmation="ENABLE_LIVE_REAL_ORDERS",
            )
        self.history.insert_one.assert_not_called()

    def test_broker_mismatch_rejected_with_no_order(self):
        self.client.get_account.return_value = SimpleNamespace(id="a-different-account")
        with patch("alpaca.trading.client.TradingClient", return_value=self.client):
            with self.assertRaisesRegex(RuntimeError, "broker account mismatch"):
                auth.create_canonical_live_oauth_client(self.db)
        self.client.submit_order.assert_not_called()

    def test_revoked_token_is_not_replaced_with_api_keys(self):
        self.grant["encrypted_token"] = "cannot-be-decrypted"
        with patch.object(alpaca_live, "live_credentials") as keys:
            with self.assertRaisesRegex(RuntimeError, "cannot be decrypted"):
                auth.create_canonical_live_oauth_client(self.db)
        keys.assert_not_called()

    def test_safe_diagnostic_current_admin_and_broker_checks(self):
        with patch("alpaca.trading.client.TradingClient", return_value=self.client):
            result = auth.canonical_live_oauth_diagnostic(self.db, self.identity)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["orders_enabled"])
        self.assertNotIn(self.account_id, repr(result))
        self.assertNotIn(self.token, repr(result))
        self.client.submit_order.assert_not_called()

    def test_admin_portfolio_and_export_share_canonical_broker(self):
        with (
            patch.object(live_market, "database", return_value=self.db),
            patch.object(live_market, "create_canonical_live_oauth_client",
                         return_value=self.client) as factory,
            patch.object(live_market, "connected_readonly_client") as legacy,
            patch.object(live_market, "live_portfolio_snapshot",
                         return_value={"status": "ready"}),
            patch.object(live_market, "build_live_portfolio_export",
                         return_value=b"test-zip"),
        ):
            self.assertEqual(live_market.portfolio(self.identity)["status"], "ready")
            self.assertEqual(live_market.export_portfolio(self.identity, None, None).body, b"test-zip")
        self.assertEqual(factory.call_count, 2)
        legacy.assert_not_called()

    def test_settings_endpoints_are_admin_only_and_no_order_submission(self):
        for path in ("/api/live-market/broker/settings",
                     "/api/live-market/oauth-diagnostic"):
            routes = [r for r in live_market.router.routes if r.path == path]
            self.assertTrue(routes)
            self.assertTrue(all(route.dependant.dependencies for route in routes))
        self.client.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
