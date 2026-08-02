from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _analysis_decision_dates,
    _build_walk_forward_folds,
)


class ChampionAnalysisWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.common_dates = pd.bdate_range("2019-01-01", periods=2000, tz="UTC")
        self.base = {
            "rotation_purge_days": 60,
            "rotation_horizon_days": 40,
            "rotation_target_horizons": [5, 10, 20, 40, 60],
            "rotation_walk_forward_calibration_days": 126,
            "rotation_walk_forward_test_days": 504,
            "rotation_walk_forward_min_test_days": 126,
            "rotation_minimum_training_rows": 700,
            "start_date": "2019-01-01",
        }

    def config(self, analysis_start_date: str) -> SimpleNamespace:
        return SimpleNamespace(
            **self.base,
            analysis_start_date=analysis_start_date,
        )

    def test_public_start_does_not_move_champion_fold_boundaries(self) -> None:
        early = _build_walk_forward_folds(
            self.common_dates,
            self.config("2024-01-01"),
        )
        late = _build_walk_forward_folds(
            self.common_dates,
            self.config("2026-01-01"),
        )

        boundary_fields = (
            "train_end_index",
            "calibration_start_index",
            "calibration_end_index",
            "test_start_index",
            "test_end_index",
        )
        early_boundaries = [tuple(fold[field] for field in boundary_fields) for fold in early]
        late_boundaries = [tuple(fold[field] for field in boundary_fields) for fold in late]

        self.assertEqual(early_boundaries, late_boundaries)

    def test_public_start_only_selects_execution_window(self) -> None:
        folds = _build_walk_forward_folds(
            self.common_dates,
            self.config("2024-01-01"),
        )

        early_dates = _analysis_decision_dates(
            self.common_dates,
            folds,
            self.config("2024-01-01"),
        )
        late_dates = _analysis_decision_dates(
            self.common_dates,
            folds,
            self.config("2026-01-01"),
        )

        self.assertEqual(early_dates[1], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(late_dates[1], pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertGreater(len(early_dates), len(late_dates))


if __name__ == "__main__":
    unittest.main()
