"""A provider failure must never echo Live credentials through HTTP responses.

No real broker calls, credentials, database writes, or trading orders.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from market_cycle_trader_api.api.routers import live_execution, live_market
from market_cycle_trader_api.infrastructure.trading.broker_errors import (
    public_live_broker_error,
)


class LiveBrokerHttpSecretRedactionTests(unittest.TestCase):
    def setUp(self):
        self.provider_error = RuntimeError(
            "Broker failure APCA-API-KEY-ID=LIVE-ID-EXAMPLE "
            "APCA-API-SECRET-KEY=VERY-PRIVATE "
            "Authorization: Bearer TOKEN-PRIVATE "
            "account=PRIVATE-ACCOUNT"
        )

    def assert_redacted(self, exception: HTTPException):
        self.assertEqual(exception.detail, public_live_broker_error())
        for value in ("LIVE-ID-EXAMPLE", "VERY-PRIVATE", "TOKEN-PRIVATE",
                      "PRIVATE-ACCOUNT", "Authorization", "APCA-API-SECRET-KEY"):
            self.assertNotIn(value, str(exception.detail))

    def test_portfolio_read_failure_is_redacted(self):
        with (
            patch.object(live_market, "database", return_value=MagicMock()),
            patch.object(live_market, "connected_readonly_client",
                         side_effect=self.provider_error),
        ):
            with self.assertRaises(HTTPException) as caught:
                live_market.portfolio(SimpleNamespace(subject="admin"))
        self.assertEqual(caught.exception.status_code, 503)
        self.assert_redacted(caught.exception)

    def test_audit_export_broker_failure_is_redacted(self):
        with (
            patch.object(live_market, "database", return_value=MagicMock()),
            patch.object(live_market, "connected_readonly_client",
                         side_effect=self.provider_error),
        ):
            with self.assertRaises(HTTPException) as caught:
                live_market.export_portfolio(
                    identity=SimpleNamespace(subject="admin"),
                    start_date=None, end_date=None,
                )
        self.assertEqual(caught.exception.status_code, 503)
        self.assert_redacted(caught.exception)

    def test_manual_live_execution_preflight_failure_is_redacted(self):
        db = MagicMock()
        with (
            patch.object(live_execution, "database", return_value=db),
            patch.object(live_execution, "live_control_status",
                         return_value={"control_mode": "active"}),
            patch.object(live_execution, "create_live_readonly_client",
                         side_effect=self.provider_error),
            patch.object(live_execution, "execute_prepared_live_winner_session") as submit,
        ):
            with self.assertRaises(HTTPException) as caught:
                live_execution.execute_current_winner_live(
                    identity=SimpleNamespace(subject="admin"),
                )
            submit.assert_not_called()
        self.assertEqual(caught.exception.status_code, 409)
        self.assert_redacted(caught.exception)


if __name__ == "__main__":
    unittest.main()
