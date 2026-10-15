"""Live/Paper isolation guardrails — stdlib unittest, no Alpaca network."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from market_cycle_trader_api.infrastructure.trading.alpaca_live import live_credentials
from market_cycle_trader_api.services import live_portfolio


class LivePaperIsolationTests(unittest.TestCase):
    def test_live_creds_are_never_taken_from_paper_environment(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY_ID": "paper-key",
            "ALPACA_SECRET_KEY": "paper-secret",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALPACA_LIVE"):
                live_credentials()

    def test_live_creds_must_differ_from_paper(self):
        with patch.dict(os.environ, {
            "ALPACA_LIVE_API_KEY_ID": "paper-key",
            "ALPACA_LIVE_SECRET_KEY": "live-secret",
            "ALPACA_API_KEY_ID": "paper-key",
            "ALPACA_SECRET_KEY": "paper-secret",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "different"):
                live_credentials()

    def test_live_broker_fill_never_implies_decision_or_realized_pnl(self):
        order = SimpleNamespace(
            client_order_id="live-order-1", symbol="CAPR", side="buy",
            status="filled", qty="3", filled_qty="3", filled_avg_price="6.5",
            submitted_at=None, created_at=None, filled_at=None,
        )
        result = live_portfolio._order(order)
        self.assertEqual(result["filled_value"], 19.5)
        self.assertIsNone(result["decision_audit"])
        self.assertIsNone(result["realized_pnl"])
        self.assertEqual(result["execution_origin"], "live_broker_history")

    def test_live_robot_is_explicitly_readonly(self):
        with (
            patch.object(
                live_portfolio, "get_trader_winner_summary",
                return_value={"name": "Strategy #30"},
            ),
            patch.object(live_portfolio, "latest_live_session", return_value=None),
            patch.object(
                live_portfolio, "live_control_status",
                return_value={
                    "control_mode": "stopped",
                    "status": "stopped",
                    "execution_enabled": False,
                    "scheduler_alive": False,
                },
            ),
        ):
            response = live_portfolio.live_robot_status(object())
        self.assertFalse(response["execution_enabled"])
        self.assertEqual(response["status"], "stopped")
        self.assertFalse(response["scheduler_alive"])
        self.assertEqual(response["trader_winner"]["name"], "Strategy #30")
        self.assertIsNone(response["active_run"])


if __name__ == "__main__":
    unittest.main()
