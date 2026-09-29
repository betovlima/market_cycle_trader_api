"""Regression guards for the isolated Control operational data contract."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine import operational_control_contract as contract
from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES as LIVE_FEATURES,
    build_rotation_frame as live_frame,
    prepare_rotation_panel as live_panel,
)
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES as SCIENTIFIC_FEATURES,
    build_rotation_frame as scientific_frame,
)
from market_cycle_trader_api.tcc_v106_reference.config import CONFIG, build_control_config


def _bars(rows: int) -> pd.DataFrame:
    i = np.arange(rows, dtype=float)
    close = 100.0 + 0.05 * i + 3.0 * np.sin(i / 5.0) + 2.0 * np.sin(i / 13.0)
    return pd.DataFrame(
        {
            "open": close * (1.0 + 0.002 * np.sin(i / 11.0)),
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 1_000_000.0 + 100_000.0 * np.sin(i / 8.0)
            + 50_000.0 * np.cos(i / 19.0),
        },
        index=pd.bdate_range("2022-01-03", periods=rows, tz="UTC")
        + pd.Timedelta(hours=4),
    )


class ControlDataContractTests(TestCase):
    def test_feature_names_and_values_match_on_identical_completed_ohlcv(self):
        bars = _bars(265)
        control = build_control_config(CONFIG)
        self.assertEqual(list(SCIENTIFIC_FEATURES), list(LIVE_FEATURES))
        original = scientific_frame(bars, control)
        operational = live_frame(bars, control)
        pd.testing.assert_index_equal(original.index, operational.index)
        pd.testing.assert_frame_equal(
            original.loc[:, SCIENTIFIC_FEATURES],
            operational.loc[:, LIVE_FEATURES],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_operational_panel_preserves_frozen_anchor_calendar_not_longest_asset(self):
        full = _bars(960)
        # This fixture limits the otherwise full 25-reference-asset set to the
        # two anchors needed to demonstrate the exact calendar distinction.
        bars = {"AAPL": full, "NVDA": full.iloc[10:].copy()}
        session = full.index[-1].date().isoformat()
        with patch.object(contract, "REFERENCE_ASSETS", ("AAPL", "NVDA")):
            _, scientific_dates, audit = contract.prepare_operational_control_panel(
                bars, completed_session=session
            )
        _, existing_live_dates = live_panel(bars, build_control_config(CONFIG))
        self.assertLess(len(scientific_dates), len(existing_live_dates))
        self.assertEqual(scientific_dates[-1], existing_live_dates[-1])
        self.assertEqual(audit.last_session, session)
        self.assertEqual(audit.required_reference_assets, ("AAPL", "NVDA"))

    def test_missing_reference_asset_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing assets:"):
            contract.prepare_operational_control_panel(
                {"AAPL": _bars(3)}, completed_session="2022-01-05"
            )

    def test_future_candle_is_rejected_before_any_model_training(self):
        bars = _bars(3)
        with patch.object(contract, "REFERENCE_ASSETS", ("AAPL",)):
            with self.assertRaisesRegex(ValueError, "Future daily bars"):
                contract.prepare_operational_control_panel(
                    {"AAPL": bars},
                    completed_session=bars.index[-2].date().isoformat(),
                )


if __name__ == "__main__":
    import unittest
    unittest.main()
