"""Live observer isolation and stop semantics."""
import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_observation_scheduler import observe_live_cycle


class LiveObservationTests(unittest.TestCase):
    @patch("market_cycle_trader_api.services.live_observation_scheduler.live_control_status")
    @patch("market_cycle_trader_api.services.live_observation_scheduler.live_portfolio_snapshot")
    def test_stopped_skips_broker_and_db(self, portfolio, control):
        control.return_value = {"control_mode": "stopped"}
        db = MagicMock()
        result = observe_live_cycle(db)
        self.assertEqual(result["status"], "skipped")
        portfolio.assert_not_called()
        db.__getitem__.assert_not_called()

    @patch("market_cycle_trader_api.services.live_observation_scheduler.get_trader_winner_summary")
    @patch("market_cycle_trader_api.services.live_observation_scheduler.live_portfolio_snapshot")
    @patch("market_cycle_trader_api.services.live_observation_scheduler.live_control_status")
    def test_active_records_only_live_observation(self, control, portfolio, winner):
        control.return_value = {"control_mode": "active"}
        portfolio.return_value = {
            "account_status": "active", "portfolio_value": 0.0, "strategy_cash": 0.0,
            "positions": [], "market_clock": {"is_open": False},
        }
        winner.return_value = {"name": "Winner"}
        db = MagicMock()
        result = observe_live_cycle(db)
        self.assertEqual(result["phase"], "read_only_observation")
        self.assertFalse(result["winner_signal_prepared"])
        self.assertFalse(result["order_submission_enabled"])
        db.__getitem__.assert_called_once_with("live_trader_session_observations")
        db["live_trader_session_observations"].insert_one.assert_called_once()
