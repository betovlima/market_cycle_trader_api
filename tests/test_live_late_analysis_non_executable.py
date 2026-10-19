"""No second LightGBM training even in analysis-only mode during Live session."""
import unittest
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_winner_recalibration import (
    recalibrate_live_cash_winner_signal,
)


class LateLiveRecalibrationTests(unittest.TestCase):
    def test_late_intraday_analysis_is_explicitly_disabled(self):
        db = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "LiveIntradayRecalibrationDisabled"):
            recalibrate_live_cash_winner_signal(db, "2026-10-19")
        db.__getitem__.assert_not_called()
