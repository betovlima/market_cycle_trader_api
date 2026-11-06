"""Dev-only migration tests: canonical Live OAuth, never actual broker orders."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from market_cycle_trader_api.infrastructure.trading import canonical_live_oauth as auth
from market_cycle_trader_api.infrastructure.trading import alpaca_live
from market_cycle_trader_api.api.routers import alpaca_oauth, live_market
from market_cycle_trader_api.services import live_control


class CanonicalLiveOAuthTests(unittest.TestCase):
    def setUp(self):
        self.account_id = "canonical-live-account"
        self.subject = "google:canonical-admin"
        self.key = Fernet.generate_key().decode("ascii")
        self.token = "private-bearer-for-mocks-only"
        self.grant = {
            "_id": auth._owner_key(self.subject),
            "subject": self.subject, "role": "admin",
            "environment": "live", "account_id": self.account_id,
            "trading_authorized": True, "oauth_scope": "trading",
            "read_only": False,
            "encrypted_token": Fernet(self.key.encode()).encrypt(self.token.encode()).decode(),
        }
        self.db = MagicMock()
        self.db[auth._CONNECTIONS].find.return_value.limit.return_value = [self.grant]
        self.db[auth._CONNECTIONS].find_one.return_value = self.grant
        self.env = patch.dict(os.environ, {
            "MCT_CANONICAL_LIVE_AUTH_MODE": "oauth",
            "ALPACA_LIVE_ACCOUNT_ID": self.account_id,
            "ALPACA_OAUTH_ENCRYPTION_KEY": self.key,
            "MCT_CANONICAL_LIVE_OAUTH_SUBJECT": "",
            "MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = MagicMock()
        self.client.get_account.return_value = SimpleNamespace(id=self.account_id)
        self.client.get_clock.return_value = SimpleNamespace(is_open=False)

    def test_read_and_order_clients_both_use_verified_oauth_and_no_api_keys(self):
        with (
            patch("alpaca.trading.client.TradingClient", return_value=self.client) as make,
            patch("market_cycle_trader_api.core.runtime.database", return_value=self.db),
            patch.object(alpaca_live, "live_credentials") as old_keys,
        ):
            read_client = alpaca_live.create_live_readonly_client()
            execute_client = alpaca_live.create_live_execution_client()
        self.assertIs(read_client, self.client)
        self.assertIs(execute_client, self.client)
        old_keys.assert_not_called()
        self.assertEqual(make.call_count, 2)
        for call in make.call_args_list:
            self.assertEqual(call.kwargs, {"oauth_token": self.token, "paper": False})
        self.client.submit_order.assert_not_called()

    def test_dev_mode_defaults_to_no_order_client_even_with_valid_grant(self):
        with patch.dict(os.environ, {"MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS": ""}):
            with (
                patch("market_cycle_trader_api.core.runtime.database", return_value=self.db),
                patch("alpaca.trading.client.TradingClient") as make,
                patch.object(alpaca_live, "live_credentials") as keys,
            ):
                with self.assertRaisesRegex(RuntimeError, "order submission is disabled"):
                    alpaca_live.create_live_execution_client()
                # Reads and portfolio can still authenticate in advisory mode.
                self.assertFalse(auth.canonical_oauth_orders_enabled())
            make.assert_not_called()
            keys.assert_not_called()

    def test_no_grant_fails_closed_even_if_keys_exist(self):
        self.db[auth._CONNECTIONS].find.return_value.limit.return_value = []
        with (
            patch("market_cycle_trader_api.core.runtime.database", return_value=self.db),
            patch.object(alpaca_live, "live_credentials") as old_keys,
            patch("alpaca.trading.client.TradingClient") as make,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                alpaca_live.create_live_execution_client()
        old_keys.assert_not_called()
        make.assert_not_called()

    def test_two_admin_grants_same_account_are_ambiguous(self):
        self.db[auth._CONNECTIONS].find.return_value.limit.return_value = [
            self.grant, {**self.grant, "subject": "another-admin"},
        ]
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            auth.create_canonical_live_oauth_client(self.db)

    def test_explicit_owner_disambiguates_but_cannot_cross_account(self):
        with patch.dict(os.environ, {"MCT_CANONICAL_LIVE_OAUTH_SUBJECT": self.subject}):
            with patch("alpaca.trading.client.TradingClient", return_value=self.client):
                self.assertIs(auth.create_canonical_live_oauth_client(self.db), self.client)
            query = self.db[auth._CONNECTIONS].find_one.call_args.args[0]
            self.assertEqual(query["account_id"], self.account_id)
            self.assertEqual(query["subject"], self.subject)
            self.assertEqual(query["_id"], auth._owner_key(self.subject))
            self.assertEqual(query["role"], "admin")

    def test_invalid_owner_binding_does_not_connect(self):
        self.db[auth._CONNECTIONS].find.return_value.limit.return_value = [{
            **self.grant, "_id": "arbitrary-id",
        }]
        with patch("alpaca.trading.client.TradingClient") as make:
            with self.assertRaisesRegex(RuntimeError, "owner binding"):
                auth.create_canonical_live_oauth_client(self.db)
        make.assert_not_called()

    def test_revoked_or_unreadable_token_blocks_execution(self):
        for encrypted in ("garbage", "", None):
            with self.subTest(encrypted=encrypted):
                self.db[auth._CONNECTIONS].find.return_value.limit.return_value = [{
                    **self.grant, "encrypted_token": encrypted,
                }]
                with patch("alpaca.trading.client.TradingClient") as make:
                    with self.assertRaisesRegex(RuntimeError, "cannot be decrypted"):
                        auth.create_canonical_live_oauth_client(self.db)
                make.assert_not_called()

    def test_oauth_token_rejected_by_broker_does_not_fallback(self):
        with (
            patch("alpaca.trading.client.TradingClient", return_value=self.client),
            patch.object(alpaca_live, "live_credentials") as old_keys,
        ):
            self.client.get_account.side_effect = RuntimeError("authorization revoked")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                auth.create_canonical_live_oauth_client(self.db)
        old_keys.assert_not_called()

    def test_mismatched_broker_account_fails_before_any_order(self):
        self.client.get_account.return_value = SimpleNamespace(id="different-broker-account")
        with patch("alpaca.trading.client.TradingClient", return_value=self.client):
            with self.assertRaisesRegex(RuntimeError, "broker account mismatch"):
                auth.create_canonical_live_oauth_client(self.db)
        self.client.submit_order.assert_not_called()

    def test_mode_invalid_fails_before_any_credential_use(self):
        with patch.dict(os.environ, {"MCT_CANONICAL_LIVE_AUTH_MODE": "maybe"}):
            with self.assertRaisesRegex(RuntimeError, "Invalid canonical"):
                alpaca_live.create_live_execution_client()

    def test_live_controls_validate_oauth_without_reading_legacy_keys(self):
        # Prevent activating a Live control mode without valid OAuth.
        db = MagicMock()
        db[live_control.LIVE_CONTROL_COLLECTION].find_one.return_value = {
            "_id": "default", "control_mode": "stopped", "revision": 0,
        }
        with (
            patch.object(live_control, "create_canonical_live_oauth_client",
                         side_effect=RuntimeError("OAuth revoked")) as check,
            patch.object(live_control, "live_credentials") as old_keys,
        ):
            with self.assertRaisesRegex(RuntimeError, "OAuth revoked"):
                live_control.set_live_control_mode(
                    db, mode="active", expected_revision=0,
                    actor_email="admin@example.com", reason="dev test",
                )
        check.assert_called_once_with(db)
        old_keys.assert_not_called()
        db[live_control.LIVE_CONTROL_COLLECTION].find_one_and_update.assert_not_called()

    def test_admin_portfolio_and_export_use_same_canonical_oauth_not_fallback(self):
        identity = SimpleNamespace(subject=self.subject, role="admin")
        with (
            patch.object(live_market, "database", return_value=self.db),
            patch.object(live_market, "create_canonical_live_oauth_client",
                         return_value=self.client) as maker,
            patch.object(live_market, "connected_readonly_client") as legacy,
            patch.object(live_market, "live_portfolio_snapshot",
                         return_value={"status": "ready"}) as snapshot,
            patch.object(live_market, "build_live_portfolio_export",
                         return_value=b"test-zip") as exporter,
        ):
            self.assertEqual(live_market.portfolio(identity)["status"], "ready")
            response = live_market.export_portfolio(identity, None, None)
        self.assertEqual(maker.call_count, 2)
        legacy.assert_not_called()
        snapshot.assert_called_once_with(
            self.db, broker_client=self.client, broker_authentication="oauth",
        )
        self.assertEqual(exporter.call_args.kwargs["broker_client"], self.client)
        self.assertEqual(exporter.call_args.kwargs["broker_authentication"], "oauth")
        self.assertEqual(response.body, b"test-zip")
        self.client.submit_order.assert_not_called()

    def test_canonical_oauth_label_enabled_only_in_explicit_mode_and_live_active(self):
        saved = {"trading_authorized": True, "account_id": self.account_id}
        with (
            patch("market_cycle_trader_api.services.live_shared_winner_signal._canonical_account_id",
                  return_value=self.account_id),
            patch("market_cycle_trader_api.services.live_control.live_control_status",
                  return_value={"control_mode": "active"}),
        ):
            self.assertTrue(alpaca_oauth._oauth_winner_operational(self.db, saved))
            with patch.dict(os.environ, {"MCT_CANONICAL_LIVE_AUTH_MODE": "api_keys"}):
                self.assertFalse(alpaca_oauth._oauth_winner_operational(self.db, saved))

    def test_live_status_reports_oauth_readonly_when_order_flag_disabled(self):
        db = MagicMock()
        db[live_control.LIVE_CONTROL_COLLECTION].find_one.return_value = {
            "_id": "default", "control_mode": "active", "revision": 7,
        }
        with patch.dict(os.environ, {"MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS": ""}):
            result = live_control.live_control_status(db)
        self.assertEqual(result["control_mode"], "active")
        self.assertEqual(result["canonical_broker_authentication"], "oauth")
        self.assertFalse(result["execution_enabled"])
        self.assertFalse(result["order_submission_enabled"])

    def test_oauth_readonly_status_never_submits_trades(self):
        self.assertEqual(auth.canonical_live_auth_mode(), "oauth")
        self.client.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
