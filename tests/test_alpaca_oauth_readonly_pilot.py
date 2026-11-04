"""Alpaca Connect pilot: no network, no broker orders, no real tokens."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

from cryptography.fernet import Fernet
from market_cycle_trader_api.api.routers import alpaca_oauth as oauth


class AlpacaOauthPilotTests(unittest.TestCase):
    def setUp(self):
        self.identity = SimpleNamespace(subject="google:current-admin")
        self.cfg = {
            "client": "test-client", "secret": "private-test-secret",
            "redirect": "https://api.example.com/api/alpaca-oauth/callback",
            "frontend": "https://frontend.example.com/",
            "expected": "expected-live-account",
            "key": Fernet.generate_key().decode("ascii"),
        }
        self.db = MagicMock()
        self.origin = SimpleNamespace(headers={"origin": "https://frontend.example.com"})

    def test_default_authorization_scope_read_only_and_live(self):
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db)):
            result = oauth.start(self.origin, self.identity)
        query = parse_qs(urlsplit(result["authorization_url"]).query)
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["env"], ["live"])
        self.assertNotIn("scope", query)
        self.assertNotIn("client_secret", query)
        self.assertNotIn(self.cfg["secret"], result["authorization_url"])
        inserted = self.db[oauth.PENDING].insert_one.call_args.args[0]
        self.assertEqual(inserted["subject"], self.identity.subject)
        self.assertNotEqual(inserted["_id"], query["state"][0])
        self.assertGreater(inserted["expires"], datetime.now(timezone.utc))

    def test_cross_origin_start_is_denied(self):
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db)):
            with self.assertRaisesRegex(Exception, "origin"):
                oauth.start(SimpleNamespace(headers={"origin": "https://attacker.example"}), self.identity)
        self.db[oauth.PENDING].insert_one.assert_not_called()

    def test_invalid_state_never_exchanges_token(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = None
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post") as post):
            result = oauth.callback(code="short-code",
                                    state="unknown-state", error=None)
        self.assertIn("invalid_state", result.headers["location"])
        post.assert_not_called()
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_trading_permission_is_rejected_without_account_lookup(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.identity.subject,
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "access_token": "test-token", "token_type": "bearer", "scope": "trading"
        }
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=response),
              patch.object(oauth.requests, "get") as get):
            result = oauth.callback(code="test-code",
                                    state="test-state", error=None)
        self.assertIn("permission_failed", result.headers["location"])
        get.assert_not_called()
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_only_expected_live_account_is_saved(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.identity.subject,
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        exchange = MagicMock(status_code=200)
        exchange.json.return_value = {
            "access_token": "sensitive-test-token", "token_type": "bearer", "scope": ""
        }
        account = MagicMock(status_code=200)
        account.json.return_value = {"id": self.cfg["expected"], "status": "ACTIVE"}
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=exchange),
              patch.object(oauth.requests, "get", return_value=account)):
            result = oauth.callback(code="code",
                                    state="state", error=None)
        self.assertIn("connected", result.headers["location"])
        self.assertEqual(self.db[oauth.CONNECTIONS].update_one.call_args.args[0]["_id"],
                         oauth._key(self.identity.subject))
        saved = self.db[oauth.CONNECTIONS].update_one.call_args.args[1]["$set"]
        self.assertEqual(saved["subject"], self.identity.subject)
        self.assertNotEqual(saved["encrypted_token"], "sensitive-test-token")
        self.assertEqual(
            Fernet(self.cfg["key"].encode()).decrypt(
                saved["encrypted_token"].encode()).decode(),
            "sensitive-test-token"
        )
        self.assertTrue(saved["read_only"])

    def test_different_account_rejected(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.identity.subject,
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        exchange = MagicMock(status_code=200)
        exchange.json.return_value = {"access_token": "test-token", "token_type": "bearer"}
        account = MagicMock(status_code=200)
        account.json.return_value = {"id": "another-person-account", "status": "ACTIVE"}
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=exchange),
              patch.object(oauth.requests, "get", return_value=account)):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("account_mismatch", result.headers["location"])
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_status_never_contains_token_or_full_account_id(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = {
            "account_id": "sensitive-account-123456",
            "account_status": "ACTIVE",
            "encrypted_token": "sensitive-encrypted-token",
        }
        result = oauth._status(self.db, self.identity)
        self.assertEqual(result["account_id_suffix"], "123456")
        self.assertNotIn("account_id", result)
        self.assertNotIn("encrypted_token", result)
        self.assertFalse(result["trading_enabled"])


    def test_callback_redirects_without_session_but_others_still_require_admin(self):
        # FastAPI's route dependency graph can be checked without TestClient
        # (the CI environment intentionally does not install httpx2).
        from market_cycle_trader_api.auth.security import require_portfolio_session

        with patch.object(oauth, "_config", return_value=self.cfg):
            response = oauth.callback(code=None, state=None, error=None)
        self.assertEqual(response.status_code, 303)
        self.assertIn("alpaca_oauth=invalid_state", response.headers["location"])

        for route in oauth.router.routes:
            dependencies = [item.call for item in route.dependant.dependencies]
            if route.path.endswith("/callback"):
                self.assertNotIn(require_portfolio_session, dependencies)
            else:
                self.assertIn(require_portfolio_session, dependencies, route.path)

    def test_state_without_a_subject_cannot_connect_an_account(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": "",
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post") as post):
            response = oauth.callback(code="code", state="state", error=None)
        self.assertIn("invalid_state", response.headers["location"])
        post.assert_not_called()
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()


    def test_readonly_client_uses_token_of_authenticated_admin(self):
        grant = "token-visible-only-in-mock"
        self.db[oauth.CONNECTIONS].find_one.return_value = {
            "subject": self.identity.subject,
            "environment": "live",
            "read_only": True,
            "account_id": self.cfg["expected"],
            "encrypted_token": Fernet(self.cfg["key"].encode()).encrypt(grant.encode()).decode(),
        }
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id=self.cfg["expected"])
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch("alpaca.trading.client.TradingClient", return_value=broker) as create):
            result = oauth.connected_readonly_client(self.db, self.identity)
        self.assertIs(result, broker)
        create.assert_called_once_with(oauth_token=grant, paper=False)
        self.db[oauth.CONNECTIONS].find_one.assert_called_once_with(
            {"_id": oauth._key(self.identity.subject)}
        )

    def test_disconnected_account_does_not_create_oauth_broker(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = None
        with (patch.object(oauth, "_config") as config,
              patch("alpaca.trading.client.TradingClient") as create):
            self.assertIsNone(oauth.connected_readonly_client(self.db, self.identity))
        config.assert_not_called()
        create.assert_not_called()

    def test_invalid_binding_fails_closed_without_creating_client(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = {
            "subject": "google:another-user",
            "environment": "live",
            "read_only": True,
            "account_id": self.cfg["expected"],
            "encrypted_token": "invalid-token",
        }
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch("alpaca.trading.client.TradingClient") as create):
            with self.assertRaisesRegex(RuntimeError, "binding is invalid"):
                oauth.connected_readonly_client(self.db, self.identity)
        create.assert_not_called()

    def test_broker_account_mismatch_fails_closed(self):
        grant = Fernet(self.cfg["key"].encode()).encrypt(b"test-token").decode()
        self.db[oauth.CONNECTIONS].find_one.return_value = {
            "subject": self.identity.subject, "environment": "live", "read_only": True,
            "account_id": self.cfg["expected"], "encrypted_token": grant,
        }
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(id="other-account")
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch("alpaca.trading.client.TradingClient", return_value=broker)):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                oauth.connected_readonly_client(self.db, self.identity)

    def test_live_portfolio_prefers_verified_oauth_and_never_sends_orders(self):
        from market_cycle_trader_api.api.routers import live_market
        broker = MagicMock()
        with (patch.object(live_market, "database", return_value=self.db),
              patch.object(live_market, "connected_readonly_client", return_value=broker),
              patch.object(live_market, "live_portfolio_snapshot", return_value={
                  "status": "ready", "broker_authentication": "oauth",
              }) as snapshot):
            response = live_market.portfolio(self.identity)
        self.assertEqual(response["broker_authentication"], "oauth")
        snapshot.assert_called_once_with(
            self.db, broker_client=broker, broker_authentication="oauth",
        )
        broker.submit_order.assert_not_called()

    def test_live_portfolio_without_oauth_retains_existing_api_key_reads(self):
        from market_cycle_trader_api.api.routers import live_market
        with (patch.object(live_market, "database", return_value=self.db),
              patch.object(live_market, "connected_readonly_client", return_value=None),
              patch.object(live_market, "live_portfolio_snapshot", return_value={
                  "status": "ready", "broker_authentication": "api_keys",
              }) as snapshot):
            response = live_market.portfolio(self.identity)
        self.assertEqual(response["broker_authentication"], "api_keys")
        snapshot.assert_called_once_with(
            self.db, broker_client=None, broker_authentication="api_keys",
        )

    def test_live_portfolio_must_not_fallback_if_oauth_fails(self):
        from fastapi import HTTPException
        from market_cycle_trader_api.api.routers import live_market
        with (patch.object(live_market, "database", return_value=self.db),
              patch.object(live_market, "connected_readonly_client",
                           side_effect=RuntimeError("OAuth token expired")),
              patch.object(live_market, "live_portfolio_snapshot") as snapshot):
            with self.assertRaises(HTTPException) as failure:
                live_market.portfolio(self.identity)
        self.assertEqual(failure.exception.status_code, 503)
        snapshot.assert_not_called()

    def test_live_portfolio_export_uses_same_oauth_broker(self):
        from market_cycle_trader_api.api.routers import live_market
        broker = MagicMock()
        with (patch.object(live_market, "database", return_value=self.db),
              patch.object(live_market, "connected_readonly_client", return_value=broker),
              patch.object(live_market, "build_live_portfolio_export",
                           return_value=b"PK\\x03\\x04") as export):
            response = live_market.export_portfolio(
                self.identity, start_date=None, end_date=None,
            )
        export.assert_called_once_with(
            self.db, start_date=None, end_date=None, broker_client=broker,
            broker_authentication="oauth",
        )
        self.assertEqual(response.media_type, "application/zip")


    def test_snapshot_reads_authorized_broker_without_api_key_client(self):
        from market_cycle_trader_api.services import live_portfolio
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id=self.cfg["expected"], cash="1500", equity="2000",
            buying_power="1500", non_marginable_buying_power="1500",
            status="ACTIVE", trading_blocked=False,
        )
        broker.get_all_positions.return_value = []
        broker.get_orders.return_value = []
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime.now(timezone.utc), is_open=False,
            next_open=None, next_close=None,
        )
        with (patch.object(live_portfolio, "create_live_readonly_client") as legacy,
              patch.object(live_portfolio, "_store_observation", return_value=[]),
              patch.object(live_portfolio, "latest_live_winner_decision", return_value=None),
              patch.object(live_portfolio, "_verified_order_origins", return_value={}),
              patch.object(live_portfolio, "_last_verified_live_execution",
                           return_value={"session": None, "origin": None})):
            result = live_portfolio.live_portfolio_snapshot(
                self.db, broker_client=broker, broker_authentication="oauth",
            )
        legacy.assert_not_called()
        self.assertEqual(result["portfolio_value"], 2000)
        self.assertEqual(result["strategy_cash"], 1500)
        self.assertEqual(result["positions"], [])
        self.assertEqual(result["recent_orders"], [])
        self.assertEqual(result["broker_authentication"], "oauth")
        broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
