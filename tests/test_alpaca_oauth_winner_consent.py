"""OAuth Winner trading permission: explicit consent, no live order execution."""
from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

from cryptography.fernet import Fernet
from fastapi import HTTPException

from market_cycle_trader_api.api.routers import alpaca_oauth as oauth
from market_cycle_trader_api.services import live_oauth_winner


class OAuthWinnerConsentTests(unittest.TestCase):
    def setUp(self):
        self.subject = "google:admin-for-tests"
        self.identity = SimpleNamespace(subject=self.subject)
        self.origin = SimpleNamespace(headers={"origin": "https://frontend.example.com"})
        self.config = {
            "client": "test-client", "secret": "not-a-real-secret",
            "redirect": "https://api.example.com/api/alpaca-oauth/callback",
            "frontend": "https://frontend.example.com/",
            "expected": "verified-live-account",
            "key": Fernet.generate_key().decode("ascii"),
        }
        self.db = MagicMock()
        self.linked = {
            "subject": self.subject,
            "environment": "live",
            "account_id": self.config["expected"],
            "read_only": True,
            "encrypted_token": Fernet(self.config["key"].encode()).encrypt(
                b"read-only-token"
            ).decode(),
        }

    def test_trading_consent_defaults_off(self):
        with patch.dict(os.environ, {"ALPACA_OAUTH_TRADING_CONSENT_ENABLED": ""}):
            self.assertFalse(oauth._trading_consent_enabled())

    def test_trading_consent_disabled_fails_before_db_write(self):
        with patch.object(oauth, "_trading_consent_enabled", return_value=False):
            with self.assertRaises(HTTPException) as err:
                oauth.start_trading(self.origin, self.identity)
        self.assertEqual(err.exception.status_code, 409)
        self.db[oauth.PENDING].insert_one.assert_not_called()

    def test_trading_start_requires_existing_verified_account(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = None
        with (patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "database", return_value=self.db)):
            with self.assertRaises(HTTPException) as err:
                oauth.start_trading(self.origin, self.identity)
        self.assertEqual(err.exception.status_code, 409)
        self.db[oauth.PENDING].insert_one.assert_not_called()

    def test_trading_start_requires_same_origin(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        bad_origin = SimpleNamespace(headers={"origin": "https://attacker.example"})
        with (patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "database", return_value=self.db)):
            with self.assertRaises(HTTPException) as err:
                oauth.start_trading(bad_origin, self.identity)
        self.assertEqual(err.exception.status_code, 403)
        self.db[oauth.PENDING].insert_one.assert_not_called()

    def test_trading_start_requests_exact_scope_and_is_bound_to_user(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        with (patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "database", return_value=self.db)):
            result = oauth.start_trading(self.origin, self.identity)
        query = parse_qs(urlsplit(result["authorization_url"]).query)
        self.assertEqual(query["scope"], ["trading"])
        self.assertEqual(query["env"], ["live"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertFalse(result["trading_enabled"])
        self.assertNotIn("client_secret", query)
        saved = self.db[oauth.PENDING].insert_one.call_args.args[0]
        self.assertEqual(saved["subject"], self.subject)
        self.assertEqual(saved["requested_scope"], "trading")
        self.assertEqual(saved["linked_token_hash"], oauth._digest(self.linked["encrypted_token"]))
        self.assertNotEqual(saved["_id"], query["state"][0])

    def test_trade_scope_cannot_be_granted_from_readonly_state(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "read",
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "access_token": "token", "token_type": "bearer", "scope": "trading"
        }
        with (patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=response),
              patch.object(oauth, "_read_account") as read_account):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("permission_failed", result.headers["location"])
        read_account.assert_not_called()
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_requested_trading_scope_must_be_returned_by_alpaca(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "trading",
            "linked_token_hash": oauth._digest(self.linked["encrypted_token"]),
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "access_token": "token", "token_type": "bearer", "scope": ""
        }
        with (patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth.requests, "post", return_value=response),
              patch.object(oauth, "_read_account") as read_account):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("trading_permission_failed", result.headers["location"])
        read_account.assert_not_called()

    def test_trading_grant_is_encrypted_but_does_not_enable_orders(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        self.db[oauth.CONNECTIONS].update_one.return_value.matched_count = 1
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "trading",
            "linked_token_hash": oauth._digest(self.linked["encrypted_token"]),
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "access_token": "granted-token", "token_type": "bearer", "scope": "trading"
        }
        with (patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=response),
              patch.object(oauth, "_read_account", return_value={
                  "id": self.config["expected"], "status": "ACTIVE",
              })):
            result = oauth.callback(code="code", state="state", error=None)
            self.assertIn("trading_connected", result.headers["location"])
        self.assertFalse(self.db[oauth.CONNECTIONS].update_one.call_args.kwargs["upsert"])
        self.assertEqual(
            self.db[oauth.CONNECTIONS].update_one.call_args.args[0]["encrypted_token"],
            self.linked["encrypted_token"],
        )
        stored = self.db[oauth.CONNECTIONS].update_one.call_args.args[1]["$set"]
        self.assertTrue(stored["trading_authorized"])
        self.assertFalse(stored["read_only"])
        self.assertEqual(stored["oauth_scope"], "trading")
        self.assertNotEqual(stored["encrypted_token"], "granted-token")
        plaintext = Fernet(self.config["key"].encode()).decrypt(
            stored["encrypted_token"].encode()
        ).decode()
        self.assertEqual(plaintext, "granted-token")
        self.db[oauth.CONNECTIONS].find_one.return_value = stored
        status = oauth._status(self.db, self.identity)
        self.assertTrue(status["trading_authorized"])
        self.assertFalse(status["trading_enabled"])
        self.assertNotIn("encrypted_token", status)

    def test_callback_cas_failure_never_reconnects_revoked_grant(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        self.db[oauth.CONNECTIONS].update_one.return_value.matched_count = 0
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "trading",
            "linked_token_hash": oauth._digest(self.linked["encrypted_token"]),
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "access_token": "token", "token_type": "bearer", "scope": "trading",
        }
        with (patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "_trading_consent_enabled", return_value=True),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=response),
              patch.object(oauth, "_read_account",
                           return_value={"id": self.config["expected"]})):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("invalid_state", result.headers["location"])
        self.assertFalse(self.db[oauth.CONNECTIONS].update_one.call_args.kwargs["upsert"])

    def test_disconnected_or_replaced_oauth_grant_invalidates_pending_trading_state(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "trading",
            "linked_token_hash": oauth._digest(self.linked["encrypted_token"]),
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        for current in (None, {**self.linked, "encrypted_token": "other-grant"}):
            with self.subTest(current=current):
                self.db[oauth.CONNECTIONS].find_one.return_value = current
                with (patch.object(oauth, "_config", return_value=self.config),
                      patch.object(oauth, "_trading_consent_enabled", return_value=True),
                      patch.object(oauth, "database", return_value=self.db),
                      patch.object(oauth.requests, "post") as token_exchange):
                    result = oauth.callback(code="code", state="state", error=None)
                self.assertIn("invalid_state", result.headers["location"])
                token_exchange.assert_not_called()
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_disable_flag_while_authorizing_rejects_callback(self):
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.subject, "requested_scope": "trading",
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        with (patch.object(oauth, "_config", return_value=self.config),
              patch.object(oauth, "_trading_consent_enabled", return_value=False),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post") as post):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("trading_unavailable", result.headers["location"])
        post.assert_not_called()

    def test_preflight_with_readonly_grant_never_creates_broker(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        with patch.object(live_oauth_winner, "connected_trading_client") as client:
            result = live_oauth_winner.oauth_winner_readiness(self.db, self.identity)
        self.assertEqual(result["status"], "trading_consent_required")
        self.assertFalse(result["order_submission_enabled"])
        client.assert_not_called()

    def test_trading_client_requires_explicit_trade_grant(self):
        self.db[oauth.CONNECTIONS].find_one.return_value = self.linked
        with patch.object(oauth, "connected_readonly_client") as client:
            with self.assertRaisesRegex(RuntimeError, "not been authorized"):
                oauth.connected_trading_client(self.db, self.identity)
        client.assert_not_called()

    def test_preflight_uses_account_qualified_session_and_never_trades(self):
        connections = MagicMock()
        runs = MagicMock()
        self.db.__getitem__.side_effect = lambda name: {
            oauth.CONNECTIONS: connections,
            live_oauth_winner.LIVE_RUNS: runs,
        }[name]
        connections.find_one.return_value = {
            "subject": self.subject, "environment": "live",
            "account_id": self.config["expected"],
            "trading_authorized": True, "oauth_scope": "trading",
            "read_only": False,
        }
        client = MagicMock()
        client.get_account.return_value = SimpleNamespace(
            id=self.config["expected"], status="ACTIVE",
        )
        client.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc),
            next_open=datetime(2026, 11, 4, 14, 30, tzinfo=timezone.utc),
            is_open=False,
        )
        runs.find_one.return_value = {
            "_id": f"{self.config['expected']}:2026-11-04",
            "winner_strategy_id": "winner-1",
            "winner_configuration_hash": "sha",
            "winner_revision": 1,
            "status": "recalibrated_no_orders",
            "winner_signal_prepared": True,
            "post_close_signal_ready": True,
            "snapshot_sha256": "sha256",
            "target_asset": "CAPR",
        }
        with (patch.object(live_oauth_winner, "connected_trading_client",
                           return_value=client),
              patch.object(live_oauth_winner, "get_trader_winner_summary",
                           return_value={
                               "id": "winner-1", "configuration_hash": "sha",
                               "revision": 1, "name": "Winner",
                           })):
            result = live_oauth_winner.oauth_winner_readiness(self.db, self.identity)
        self.assertEqual(result["status"], "trading_authorized_execution_disabled")
        self.assertTrue(result["winner_signal_prepared"])
        self.assertFalse(result["order_submission_enabled"])
        self.db[live_oauth_winner.LIVE_RUNS].find_one.assert_called_once_with(
            {"_id": f"{self.config['expected']}:2026-11-04"}
        )
        client.submit_order.assert_not_called()
        client.cancel_order_by_id.assert_not_called()
        self.db[live_oauth_winner.LIVE_RUNS].update_one.assert_not_called()

    def test_existing_live_executor_and_scheduler_remain_api_key_based(self):
        from market_cycle_trader_api.services import (
            live_observation_scheduler, live_order_execution
        )
        import inspect
        scheduler = inspect.getsource(live_observation_scheduler.LiveObservationScheduler._run)
        executor = inspect.getsource(live_order_execution.execute_prepared_live_winner_session)
        self.assertIn("execute_prepared_live_winner_session(", scheduler)
        self.assertIn("create_live_execution_client()", executor)
        self.assertNotIn("connected_trading_client", scheduler)
        self.assertNotIn("connected_trading_client", executor)


if __name__ == "__main__":
    unittest.main()
