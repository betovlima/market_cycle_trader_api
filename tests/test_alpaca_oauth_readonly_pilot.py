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
            result = oauth.callback(self.identity, code="short-code",
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
            result = oauth.callback(self.identity, code="test-code",
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
            result = oauth.callback(self.identity, code="code",
                                    state="state", error=None)
        self.assertIn("connected", result.headers["location"])
        saved = self.db[oauth.CONNECTIONS].update_one.call_args.args[1]["$set"]
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
            result = oauth.callback(self.identity, code="code", state="state", error=None)
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


if __name__ == "__main__":
    unittest.main()
