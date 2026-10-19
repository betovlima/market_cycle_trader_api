"""Regression: retired intraday recalibration may never retrain or submit Live orders."""
import unittest
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_winner_recalibration import (
    recalibrate_live_cash_winner_signal,
)


class LiveRecalibrationSafetyTests(unittest.TestCase):
    def test_all_intraday_recalibration_is_disabled(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayRecalibrationDisabled"):
            recalibrate_live_cash_winner_signal(db, "2026-10-19")
        db.__getitem__.assert_not_called()
