"""Trader receives only their own OAuth position, prices and compact orders."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.api.routers import alpaca_oauth
from market_cycle_trader_api.services.live_personal_portfolio import (
    trader_personal_live_projection,
)


class TraderLiveMinimalProjectionTests(unittest.TestCase):
    def test_projection_strips_shared_or_sensitive_fields(self):
        snapshot = {
            "status": "ready", "environment": "live",
            "account_id_suffix": "7ee6b7",
            "account_status": "ACTIVE", "recorded_at": "2026-11-06T21:00:00Z",
            "portfolio_value": 123.4, "strategy_cash": 12,
            "unrealized_pnl": 4.5, "buying_power": 987,
            "winner_performance": {"return_pct": 0.3},
            "shared_target_asset": "AMZN",
            "positions": [{
                "symbol": "CAPR", "quantity": 3, "current_price": 14,
                "unrealized_return": 0.2, "average_entry_price": 12,
                "market_value": 42, "unrealized_pnl": 6,
                "internal_decision": "SECRET",
            }],
            "recent_orders": [{
                "symbol": "CAPR", "created_at": "2026-11-06T14:00:00Z",
                "side": "buy", "status": "filled", "filled_quantity": 3,
                "filled_average_price": 12, "client_order_id": "internal-id",
                "decision_audit": {"candidates": ["AMZN"]},
                "execution_origin": "live_winner_model",
            }],
            "market_clock": {"is_open": True, "next_close": "secret"},
        }
        result = trader_personal_live_projection(snapshot)
        self.assertEqual(result["positions"][0]["current_price"], 14)
        self.assertEqual(result["positions"][0]["unrealized_return"], 0.2)
        self.assertEqual(result["recent_orders"][0]["symbol"], "CAPR")
        self.assertEqual(result["market_clock"], {"is_open": True})
        for forbidden in ("winner_performance", "shared_target_asset",
                          "buying_power", "account_status"):
            self.assertNotIn(forbidden, result)
        self.assertNotIn("decision_audit", result["recent_orders"][0])
        self.assertNotIn("client_order_id", result["recent_orders"][0])
        self.assertNotIn("internal_decision", result["positions"][0])

    def test_trader_status_never_contains_admin_broker_configuration(self):
        db = MagicMock()
        db[alpaca_oauth.CONNECTIONS].find_one.return_value = {
            "account_id": "1237ee6b7", "trading_authorized": True,
        }
        identity = SimpleNamespace(subject="google:traderA", role="trader")
        with (patch.object(alpaca_oauth, "_trading_consent_enabled", return_value=True),
              patch.object(alpaca_oauth, "canonical_live_auth_mode",
                           side_effect=AssertionError("admin settings queried")),
              patch.object(alpaca_oauth, "_oauth_winner_operational",
                           side_effect=AssertionError("shared control queried"))):
            status = alpaca_oauth._status(db, identity)
        self.assertEqual(status["account_id_suffix"], "7ee6b7")
        self.assertTrue(status["trading_authorized"])
        self.assertNotIn("canonical_live_auth_mode", status)
        self.assertNotIn("trading_enabled", status)

    def test_trader_portfolio_api_projects_broker_snapshot(self):
        db = MagicMock()
        trader = SimpleNamespace(subject="google:traderA", role="trader")
        snapshot = {
            "portfolio_value": 200,
            "positions": [{"symbol": "CAPR", "current_price": 7}],
            "recent_orders": [{"symbol": "CAPR", "side": "buy",
                               "decision_audit": {"target_asset": "AMZN"}}],
        }
        with (patch.object(alpaca_oauth, "database", return_value=db),
              patch.object(alpaca_oauth, "connected_readonly_client",
                           return_value=MagicMock()),
              patch("market_cycle_trader_api.services.live_personal_portfolio.personal_live_portfolio_snapshot",
                    return_value=snapshot)):
            result = alpaca_oauth.personal_portfolio(trader)
        self.assertEqual(result["portfolio_value"], 200)
        self.assertEqual(result["recent_orders"], [{"symbol": "CAPR", "side": "buy"}])
        self.assertEqual(result["positions"], [{"symbol": "CAPR", "current_price": 7}])


if __name__ == "__main__":
    unittest.main()
