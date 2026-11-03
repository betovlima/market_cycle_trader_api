"""Native LightGBM TreeSHAP audit must never change original Live scores."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.live_feature_attribution import (
    explain_live_utilities,
)
from market_cycle_trader_api.tcc_u67_v1210_reference.rotacao import (
    ROTATION_FEATURES,
)


class FakeLightGBM:
    def __init__(self, *, fail=False, mismatch=False):
        self.fail = fail
        self.mismatch = mismatch
        self.calls = []

    def predict(self, row, pred_contrib=False):
        self.calls.append(bool(pred_contrib))
        if self.fail:
            raise RuntimeError("SHAP audit unavailable")
        if not pred_contrib:
            return np.asarray([0.94])
        values = np.full(len(ROTATION_FEATURES) + 1, 0.01, dtype=float)
        values[-1] = 0.94 - 0.01 * len(ROTATION_FEATURES)
        if self.mismatch:
            values[-1] += 0.1
        return values.reshape(1, -1)


class LiveAttributionTests(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2026-11-02", tz="UTC")
        frame = pd.DataFrame(
            {name: [float(index)] for index, name in enumerate(ROTATION_FEATURES)},
            index=pd.DatetimeIndex([self.date]),
        )
        self.frames = {"CAPR": frame}

    def test_52_features_reconstruct_exact_existing_utility(self):
        model = FakeLightGBM()
        scores = np.asarray([0.0, 0.94], dtype=float)
        before = scores.copy()
        result = explain_live_utilities(
            {"CAPR": model}, self.frames, ["CAPR"], self.date,
            scores, ROTATION_FEATURES,
        )
        self.assertTrue(np.array_equal(scores, before))
        self.assertEqual(model.calls, [True])
        self.assertEqual(set(result), {"CAPR"})
        details = result["CAPR"]
        self.assertEqual(len(details["feature_values"]), 52)
        self.assertEqual(len(details["contributions"]), 52)
        self.assertAlmostEqual(
            details["base_value"] + sum(details["contributions"].values()),
            scores[1],
        )
        self.assertEqual(details["feature_values"]["return_1"], 0.0)

    def test_failed_audit_does_not_change_scores(self):
        scores = np.asarray([0.0, 0.94], dtype=float)
        for model in (FakeLightGBM(fail=True), FakeLightGBM(mismatch=True)):
            with self.subTest(model=model):
                result = explain_live_utilities(
                    {"CAPR": model}, self.frames, ["CAPR"], self.date,
                    scores, ROTATION_FEATURES,
                )
                self.assertEqual(result, {})
                self.assertEqual(scores[1], 0.94)

    def test_missing_asset_is_not_fabricated(self):
        result = explain_live_utilities(
            {"CAPR": FakeLightGBM()}, self.frames, ["CAPR", "DNN"],
            self.date, np.asarray([0.0, 0.94, float("-inf")]),
            ROTATION_FEATURES,
        )
        self.assertEqual(list(result), ["CAPR"])


if __name__ == "__main__":
    unittest.main()
