"""Individual Alpaca OAuth accounts, with one Winner and no MCT cash ledger."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
from fastapi import HTTPException

from market_cycle_trader_api.api.routers import alpaca_oauth as oauth
from market_cycle_trader_api.services.live_personal_portfolio import (
    personal_live_portfolio_snapshot,
)


class MultiuserOAuthTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.trader = SimpleNamespace(subject="google:trader-A", role="trader")
        self.other = SimpleNamespace(subject="google:trader-B", role="trader")
        self.admin = SimpleNamespace(subject="trader-admin", role="admin")
        self.cfg = {
            "client": "app-id", "secret": "fake-secret",
            "redirect": "https://api.example.org/api/alpaca-oauth/callback",
            "frontend": "https://ui.example.org/",
            "expected": "admin-primary-account",
            "key": Fernet.generate_key().decode(),
        }
        self.origin = SimpleNamespace(headers={"origin": "https://ui.example.org"})

    def test_trader_does_not_inherit_primary_admin_account_id(self):
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db)):
            oauth.start(self.origin, self.trader)
        pending = self.db[oauth.PENDING].insert_one.call_args.args[0]
        self.assertEqual(pending["subject"], self.trader.subject)
        self.assertEqual(pending["expected_account_id"], "")
        self.assertEqual(
            oauth._expected_account_id(self.cfg, self.admin),
            "admin-primary-account",
        )

    def test_trader_can_authorize_different_live_account_but_admin_still_cannot(self):
        exchange = MagicMock(status_code=200)
        exchange.json.return_value = {
            "access_token": "opaque-access-token", "token_type": "bearer",
            "scope": "",
        }
        account = {"id": "alpaca-account-A", "status": "ACTIVE"}
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.trader.subject, "requested_scope": "read",
            "expected_account_id": "",
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=exchange),
              patch.object(oauth, "_read_account", return_value=account)):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("connected", result.headers["location"])
        kwargs = self.db[oauth.CONNECTIONS].update_one.call_args
        self.assertEqual(kwargs.args[0], {"_id": oauth._key(self.trader.subject)})
        self.assertEqual(kwargs.args[1]["$set"]["account_id"], account["id"])

        self.db[oauth.CONNECTIONS].update_one.reset_mock()
        self.db[oauth.PENDING].find_one_and_delete.return_value = {
            "subject": self.admin.subject, "requested_scope": "read",
            "expected_account_id": "admin-primary-account",
            "expires": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth.requests, "post", return_value=exchange),
              patch.object(oauth, "_read_account", return_value=account)):
            result = oauth.callback(code="code", state="state", error=None)
        self.assertIn("account_mismatch", result.headers["location"])
        self.db[oauth.CONNECTIONS].update_one.assert_not_called()

    def test_two_users_never_receive_each_others_oauth_token(self):
        def saved_for(subject):
            return {
                "subject": subject, "environment": "live", "read_only": True,
                "account_id": "alpaca-" + subject,
                "encrypted_token": Fernet(self.cfg["key"].encode()).encrypt(
                    ("token-" + subject).encode()
                ).decode(),
            }

        def fake_lookup(query):
            return saved_for(self.trader.subject if query["_id"] ==
                             oauth._key(self.trader.subject) else self.other.subject)

        self.db[oauth.CONNECTIONS].find_one.side_effect = fake_lookup
        observed = []

        def client_factory(**kwargs):
            token = kwargs["oauth_token"]
            observed.append(token)
            account_id = "alpaca-" + token.removeprefix("token-")
            client = MagicMock()
            client.get_account.return_value = SimpleNamespace(id=account_id)
            return client

        with (patch.object(oauth, "_config", return_value=self.cfg),
              patch("alpaca.trading.client.TradingClient",
                    side_effect=client_factory)):
            a = oauth.connected_readonly_client(self.db, self.trader)
            b = oauth.connected_readonly_client(self.db, self.other)
        self.assertIsNot(a, b)
        self.assertEqual(observed, [
            "token-" + self.trader.subject,
            "token-" + self.other.subject,
        ])

    def test_trader_own_account_portfolio_has_no_api_key_fallback(self):
        broker = MagicMock()
        with (patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth, "connected_readonly_client", return_value=broker),
              patch("market_cycle_trader_api.services.live_personal_portfolio.personal_live_portfolio_snapshot",
                    return_value={"portfolio_value": 135.3}) as snapshot):
            result = oauth.personal_portfolio(self.trader)
        self.assertEqual(result["portfolio_value"], 135.3)
        snapshot.assert_called_once_with(broker)

        with (patch.object(oauth, "database", return_value=self.db),
              patch.object(oauth, "connected_readonly_client", return_value=None)):
            with self.assertRaises(HTTPException) as raised:
                oauth.personal_portfolio(self.other)
        self.assertEqual(raised.exception.status_code, 404)

    def test_personal_snapshot_is_realtime_and_never_persists_money(self):
        broker = MagicMock()
        broker.get_account.return_value = SimpleNamespace(
            id="alpaca-user-A", cash="42.12", equity="110.22",
            buying_power="42.12", non_marginable_buying_power="42.12",
            status="ACTIVE",
        )
        broker.get_clock.return_value = SimpleNamespace(
            timestamp=datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc),
            is_open=True, next_open=None, next_close=None,
        )
        broker.get_all_positions.return_value = [
            SimpleNamespace(
                symbol="CAPR", qty="4", avg_entry_price="10", current_price="17.025",
                market_value="68.10", unrealized_pl="28.10",
                unrealized_plpc="0.7025",
            )
        ]
        broker.get_orders.return_value = []
        with patch("market_cycle_trader_api.services.live_personal_portfolio.create_live_readonly_client",
                   create=True) as legacy_client:
            result = personal_live_portfolio_snapshot(broker)
        self.assertEqual(result["account_id_suffix"], "user-A")
        self.assertEqual(result["portfolio_value"], 110.22)
        self.assertEqual(result["strategy_cash"], 42.12)
        self.assertEqual(result["positions"][0]["symbol"], "CAPR")
        self.assertEqual(result["unrealized_pnl"], 28.1)
        self.assertEqual(result["broker_authentication"], "oauth")
        self.assertNotIn("account_id", result)
        self.assertNotIn("encrypted_token", result)
        self.assertNotIn("initial_capital", result)
        legacy_client.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_all_oauth_endpoints_except_callback_require_portfolio_access(self):
        from market_cycle_trader_api.auth.security import require_portfolio_session
        for route in oauth.router.routes:
            dependencies = [dependency.call for dependency in route.dependant.dependencies]
            if route.path.endswith("/callback"):
                self.assertNotIn(require_portfolio_session, dependencies)
            else:
                self.assertIn(require_portfolio_session, dependencies)


if __name__ == "__main__":
    unittest.main()
