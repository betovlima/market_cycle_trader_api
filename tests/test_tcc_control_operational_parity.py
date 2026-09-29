"""Control decision parity between frozen TCC v1.0.6 and the live policy.

These are unit-level decision tests, not end-to-end market-data, model-training
or live execution parity. In particular, the scientific replay requires a
subsequent OHLCV row for an executable next-open trade, which is unavailable
when a real live decision is prepared.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

# Keep this standalone regression runnable from Spyder/unittest on Windows,
# even when the editable API package has not been installed in the interpreter.
API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES as LIVE_FEATURES,
)
from market_cycle_trader_api.engine.live_policy import build_live_rotation_policy
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES as TCC_FEATURES,
    _utility_policy as scientific_utility_policy,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    CONFIG as TCC_CONFIG,
    build_control_config,
)


class ConstantUtilityModel:
    def __init__(self, utility: float) -> None:
        self.utility = float(utility)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.utility, dtype=np.float64)


class ControlOperationalDecisionParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_control_config(TCC_CONFIG)
        self.symbols = ["AAPL", "NVDA"]
        self.date = pd.Timestamp("2026-09-15T04:00:00Z")
        dates = pd.DatetimeIndex([
            self.date,
            pd.Timestamp("2026-09-16T04:00:00Z"),
        ])
        columns = set(LIVE_FEATURES) | set(TCC_FEATURES)
        self.frames = {
            symbol: pd.DataFrame(
                {
                    **{feature: [1.0, 1.0] for feature in columns},
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [1000.0, 1000.0],
                },
                index=dates,
            )
            for symbol in self.symbols
        }

    def _policies(self, aapl: float, nvda: float, margin: float = 0.0):
        models = {
            "AAPL": ConstantUtilityModel(aapl),
            "NVDA": ConstantUtilityModel(nvda),
        }
        scientific = scientific_utility_policy(
            models, self.frames, self.symbols, self.config, margin
        )
        live = build_live_rotation_policy(
            models, self.frames, self.symbols, self.config, margin
        )
        return scientific, live

    def test_control_matches_live_decisions_with_executable_replay_prices(self):
        # Position 0=CASH, 1=AAPL, 2=NVDA.
        cases = [
            ("enter", 0.01, 0.004, 0, 0, 0.0, 1),
            ("cash_edge", 0.0005, 0.0003, 0, 0, 0.0, 0),
            ("minimum_hold", 0.01, 0.02, 1, 1, 0.0, 1),
            ("margin_blocks_rotation", 0.01, 0.02, 1, 2, 0.02, 1),
            ("rotate", 0.01, 0.02, 1, 2, 0.005, 2),
            ("exit_to_cash", -0.01, -0.02, 1, 2, 0.0, 0),
            ("hold_best", 0.02, 0.01, 1, 2, 0.0, 1),
        ]
        for name, aapl, nvda, current, holding, margin, expected in cases:
            with self.subTest(name=name):
                scientific, live = self._policies(aapl, nvda, margin)
                scientific_decision = scientific(self.date, current, holding)
                live_decision = live(self.date, current, holding)
                self.assertEqual(scientific_decision, live_decision)
                self.assertEqual(scientific_decision[0], expected)

    def test_next_open_availability_requires_a_distinct_live_data_contract(self):
        # Scientific simulation validates the next OHLCV row. A real premarket
        # decision must not depend on a future candle that does not yet exist.
        self.frames = {
            symbol: frame.iloc[:1].copy()
            for symbol, frame in self.frames.items()
        }
        scientific, live = self._policies(0.01, 0.004)
        self.assertEqual(scientific(self.date, 0, 0)[0], 0)
        self.assertEqual(live(self.date, 0, 0)[0], 1)


if __name__ == "__main__":
    unittest.main()
