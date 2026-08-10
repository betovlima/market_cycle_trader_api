from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _build_walk_forward_folds,
    _xgb_utilities,
    prepare_rotation_panel,
)


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.value], dtype=float)


class CandidateCalendarAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["calendar_anchor_assets"] = ["CORE_A", "CORE_B"]
        self.config = SimpleNamespace(**payload)

    @staticmethod
    def bars(index: pd.DatetimeIndex, start_price: float) -> pd.DataFrame:
        steps = np.arange(len(index), dtype=float)
        close = start_price * (
            1.0
            + 0.00025 * steps
            + 0.035 * np.sin(steps / 7.0)
            + 0.015 * np.sin(steps / 23.0)
        )
        open_price = close * (1.0 + 0.0015 * np.sin(steps / 5.0))
        volume = 2_000_000.0 * (1.0 + 0.12 * np.sin(steps / 11.0) + 0.05 * np.cos(steps / 29.0))
        return pd.DataFrame(
            {
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.01,
                "low": np.minimum(open_price, close) * 0.99,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

    def test_younger_candidate_does_not_move_anchor_calendar_or_folds(self) -> None:
        full_index = pd.bdate_range("2016-01-04", periods=2700, tz="UTC")
        candidate_index = full_index[900:]

        core = {
            "CORE_A": self.bars(full_index, 100.0),
            "CORE_B": self.bars(full_index, 120.0),
        }
        baseline_frames, baseline_dates = prepare_rotation_panel(core, self.config)
        baseline_folds = _build_walk_forward_folds(baseline_dates, self.config)

        expanded = {
            **core,
            "YOUNG": self.bars(candidate_index, 80.0),
        }
        expanded_frames, expanded_dates = prepare_rotation_panel(expanded, self.config)
        expanded_folds = _build_walk_forward_folds(expanded_dates, self.config)

        self.assertTrue(baseline_dates.equals(expanded_dates))
        self.assertEqual(
            [(fold["test_start"], fold["test_end"]) for fold in baseline_folds],
            [(fold["test_start"], fold["test_end"]) for fold in expanded_folds],
        )
        self.assertEqual(len(expanded_frames["YOUNG"]), len(expanded_dates))
        self.assertTrue(expanded_frames["YOUNG"].iloc[0].isna().all())

    def test_candidate_without_model_is_unavailable_not_a_failure(self) -> None:
        full_index = pd.bdate_range("2016-01-04", periods=2700, tz="UTC")
        candidate_index = full_index[900:]
        frames, dates = prepare_rotation_panel(
            {
                "CORE_A": self.bars(full_index, 100.0),
                "CORE_B": self.bars(full_index, 120.0),
                "YOUNG": self.bars(candidate_index, 80.0),
            },
            self.config,
        )
        timestamp = dates[1000]
        utilities = _xgb_utilities(
            {
                "CORE_A": _ConstantModel(0.1),
                "CORE_B": _ConstantModel(0.2),
            },
            frames,
            ["CORE_A", "CORE_B", "YOUNG"],
            timestamp,
            self.config,
        )
        self.assertEqual(utilities.shape, (4,))
        self.assertTrue(np.isneginf(utilities[-1]))


if __name__ == "__main__":
    unittest.main()
