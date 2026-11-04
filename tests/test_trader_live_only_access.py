"""Trader accounts may access only their own OAuth Live workspace."""
from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.auth.security import SessionIdentity
from market_cycle_trader_api.auth.trader_live import (
    trader_live_denied,
    trader_live_request_allowed,
)


class TraderLiveOnlyAccessTests(unittest.TestCase):
    def test_trader_has_only_portfolio_capability(self):
        trader = capabilities_for_role("trader")
        self.assertEqual(
            {name for name, enabled in trader.items() if enabled},
            {"portfolio.view"},
        )
        for name in (
            "dashboard.view", "backtest.view", "backtest.start",
            "annual_replay.view", "tuning.view", "research_models.view",
            "temporal_intelligence.view", "decision_science.view",
            "analytics.view", "administration.view", "settings.view",
            "research.manage", "admin.manage",
        ):
            self.assertFalse(trader[name])
        self.assertTrue(all(capabilities_for_role("admin").values()))
        self.assertTrue(capabilities_for_role("viewer")["dashboard.view"])
        self.assertTrue(capabilities_for_role("reviewer")["annual_replay.view"])

    def test_trader_can_use_alpaca_oauth_but_not_other_panels(self):
        allowed = [
            ("GET", "/api/auth/session"),
            ("POST", "/api/auth/logout"),
            ("GET", "/api/alpaca-oauth/status"),
            ("GET", "/api/alpaca-oauth/portfolio"),
            ("GET", "/api/alpaca-oauth/verify"),
            ("GET", "/api/alpaca-oauth/winner/readiness"),
            ("GET", "/api/alpaca-oauth/callback"),
            ("POST", "/api/alpaca-oauth/start"),
            ("POST", "/api/alpaca-oauth/start-trading"),
            ("POST", "/api/alpaca-oauth/disconnect"),
            ("OPTIONS", "/api/alpaca-oauth/start-trading"),
        ]
        for method, path in allowed:
            with self.subTest(method=method, path=path):
                self.assertTrue(trader_live_request_allowed(method, path))
                self.assertFalse(trader_live_denied("trader", method, path))

        blocked = [
            ("GET", "/api/dashboard/summary"),
            ("GET", "/api/dashboard/strategy-intelligence"),
            ("GET", "/api/analytics/backtests"),
            ("GET", "/api/analytics/portfolio"),
            ("GET", "/api/jobs"),
            ("POST", "/api/jobs"),
            ("GET", "/api/admin/strategies"),
            ("GET", "/api/live-market/portfolio"),
            ("GET", "/api/public-paper-portfolio"),
            ("GET", "/api/auth/admin/login"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
            ("POST", "/api/alpaca-oauth/verify"),
            ("DELETE", "/api/alpaca-oauth/disconnect"),
        ]
        for method, path in blocked:
            with self.subTest(method=method, path=path):
                self.assertFalse(trader_live_request_allowed(method, path))
                self.assertTrue(trader_live_denied("trader", method, path))
                self.assertFalse(trader_live_denied("admin", method, path))

    def test_existing_live_and_research_http_requests_are_denied_before_handlers(self):
        # Exercises the actual HTTP middleware, not just the helper.
        from market_cycle_trader_api.main import create_app

        identity = SessionIdentity(
            subject="google:invited-trader",
            role="trader",
            scope="trader:read portfolio:read",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        with patch("market_cycle_trader_api.main.get_session_manager") as factory:
            factory.return_value.decode_session_token.return_value = identity
            client = TestClient(create_app())
            for path in (
                "/api/dashboard/summary",
                "/api/analytics/backtests",
                "/api/admin/strategies",
                "/api/live-market/portfolio",
            ):
                with self.subTest(path=path):
                    response = client.get(
                        path, cookies={"market_cycle_trader_session": "mock-cookie"},
                    )
                    self.assertEqual(response.status_code, 403)
                    self.assertIn("Only your Alpaca Live", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
