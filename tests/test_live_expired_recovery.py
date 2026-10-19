"""Expired Live sessions never retrain the model intraday."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_expired_recovery import queue_expired_live_analysis


class ExpiredLiveRecoveryTests(unittest.TestCase):
    def test_expired_intraday_reanalysis_is_disabled(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayReanalysisDisabled"):
            queue_expired_live_analysis(
                db, execution_session="2026-10-19",
                actor_email="operator@example.com",
            )
        db.__getitem__.assert_not_called()

    def test_authentication_still_required(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "authenticated operator"):
            queue_expired_live_analysis(db, execution_session="2026-10-19", actor_email="")
        db.__getitem__.assert_not_called()


class LiveRecoveryRouteIdentityTests(unittest.TestCase):
    def test_password_admin_without_email_uses_authenticated_subject(self):
        from market_cycle_trader_api.api.routers.live_market import (
            LiveExpiredAnalysisRequest,
            recover_expired_live_analysis,
        )
        identity = SimpleNamespace(email=None, subject="trader-admin")
        payload = LiveExpiredAnalysisRequest(
            execution_session="2026-10-16",
            confirm="LIVE_ANALYSIS_ONLY_NO_ORDERS",
        )
        with patch("market_cycle_trader_api.api.routers.live_market.database") as db, \
             patch("market_cycle_trader_api.api.routers.live_market.queue_expired_live_analysis") as recover:
            recover.return_value = {"status": "analysis_only_requested"}
            response = recover_expired_live_analysis(payload, identity)
            self.assertEqual(response["status"], "analysis_only_requested")
            recover.assert_called_once_with(
                db.return_value,
                execution_session="2026-10-16",
                actor_email="trader-admin",
            )

    def test_email_is_used_if_available(self):
        from market_cycle_trader_api.api.routers.live_market import (
            LiveExpiredAnalysisRequest,
            recover_expired_live_analysis,
        )
        identity = SimpleNamespace(email="operator@example.com", subject="google:1234")
        payload = LiveExpiredAnalysisRequest(
            execution_session="2026-10-16",
            confirm="LIVE_ANALYSIS_ONLY_NO_ORDERS",
        )
        with patch("market_cycle_trader_api.api.routers.live_market.database"), \
             patch("market_cycle_trader_api.api.routers.live_market.queue_expired_live_analysis") as recover:
            recover.return_value = {"status": "analysis_only_requested"}
            recover_expired_live_analysis(payload, identity)
            self.assertEqual(recover.call_args.kwargs["actor_email"], "operator@example.com")
