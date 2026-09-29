"""No-order Control shadow inference: scientific fit windows and live action."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine import operational_control_preview as preview
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import ROTATION_FEATURES


class FixedModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.value, dtype=float)


def _fixture(count: int = 1000):
    dates = pd.bdate_range("2022-01-03", periods=count, tz="UTC") + pd.Timedelta(hours=4)
    frames = {}
    for symbol in ("AAPL", "NVDA"):
        rows = {key: [0.2] * count for key in ROTATION_FEATURES}
        rows.update({
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
            "volume": [1000.0] * count,
            "forward_risk_adjusted_utility": [0.01] * count,
        })
        frames[symbol] = pd.DataFrame(rows, index=dates)
    audit = SimpleNamespace(
        requested_assets=56,
        available_assets=2,
        calendar_sessions=count,
        first_session=dates[0].date().isoformat(),
        last_session=dates[-1].date().isoformat(),
        required_reference_assets=("AAPL", "NVDA"),
    )
    return frames, dates, audit


class ControlShadowInferenceTests(TestCase):
    def test_one_shadow_decision_uses_scientific_fit_and_correct_temporal_cutoff(self):
        frames, dates, audit = _fixture()
        models = {"AAPL": FixedModel(0.008), "NVDA": FixedModel(0.02)}
        with (
            patch.object(preview, "prepare_operational_control_panel",
                         return_value=(frames, dates, audit)) as panel,
            patch.object(preview, "scientific_fit_models",
                         return_value=models) as fitter,
            patch.object(preview, "scientific_policy_growth",
                         side_effect=[0.1, 0.4, 0.2, 0.0]) as growth,
        ):
            result = preview.build_control_shadow_decision(
                {"AAPL": frames["AAPL"], "NVDA": frames["NVDA"]},
                completed_session=audit.last_session,
                current_asset="AAPL",
                holding_sessions=3,
            )
        self.assertEqual(result["status"], "shadow_only")
        self.assertIs(result["order_eligible"], False)
        self.assertEqual(result["order_submission"], "never")
        self.assertEqual(result["current_asset"], "AAPL")
        self.assertEqual(result["target_asset"], "NVDA")
        self.assertEqual(result["decision_date"], audit.last_session)
        self.assertEqual(result["calibrated_candidate_margin"], 0.0025)
        self.assertEqual(result["effective_switch_margin"], 0.0025)
        self.assertEqual(result["calibration_score"], 0.4)
        self.assertEqual(result["utilities"]["NVDA"], 0.02)
        self.assertEqual(result["input_audit"]["calendar_sessions"], 1000)
        panel.assert_called_once()
        self.assertEqual(fitter.call_count, 2)
        # N=1000, purge=60, calibration=126; the next-open execution index
        # is N, not N-1. Exact protocol: train_end=754, final_fit_end=940.
        self.assertEqual(len(fitter.call_args_list[0].args[2]), 754)
        self.assertEqual(len(fitter.call_args_list[1].args[2]), 940)
        self.assertEqual(result["training_end"], dates[753].date().isoformat())
        self.assertEqual(result["calibration_start"], dates[814].date().isoformat())
        self.assertEqual(result["calibration_end"], dates[939 - 66].date().isoformat())
        self.assertEqual(result["final_fit_end"], dates[939].date().isoformat())
        self.assertEqual(growth.call_count, 4)

    def test_state_must_be_consistent_and_no_unknown_assets(self):
        frames, dates, audit = _fixture()
        with self.assertRaisesRegex(ValueError, "CASH state"):
            with patch.object(preview, "prepare_operational_control_panel",
                              return_value=(frames, dates, audit)):
                preview.build_control_shadow_decision(
                    frames, completed_session=audit.last_session,
                    current_asset=None, holding_sessions=1,
                )
        with self.assertRaisesRegex(ValueError, "outside frozen Control universe"):
            preview.build_control_shadow_decision(
                {"INVALID_ASSET": frames["AAPL"]},
                completed_session=audit.last_session,
                current_asset=None,
                holding_sessions=0,
            )

    def test_empty_calibration_does_not_create_a_plan(self):
        frames, dates, audit = _fixture()
        with (
            patch.object(preview, "prepare_operational_control_panel",
                         return_value=(frames, dates, audit)),
            patch.object(preview, "scientific_fit_models",
                         return_value={"AAPL": FixedModel(0.01)}),
            patch.object(preview, "scientific_policy_growth",
                         return_value=float("nan")),
        ):
            with self.assertRaisesRegex(ValueError, "no finite score"):
                preview.build_control_shadow_decision(
                    frames,
                    completed_session=audit.last_session,
                    current_asset="CASH",
                    holding_sessions=0,
                )


if __name__ == "__main__":
    import unittest
    unittest.main()
