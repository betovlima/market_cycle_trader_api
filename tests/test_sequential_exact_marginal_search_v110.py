from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_sequential_exact_marginal_search_v110 as accel  # noqa: E402


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return (
            pd.to_numeric(frame[accel.rotation.ROTATION_FEATURES[0]], errors="coerce").to_numpy(dtype=float)
            * 0.25
            + 0.125
        )


def frame_for(index: pd.DatetimeIndex, offset: float = 0.0) -> pd.DataFrame:
    data = {
        name: np.linspace(1.0 + offset, 2.0 + offset, len(index))
        for name in accel.rotation.ROTATION_FEATURES
    }
    data.update(
        {
            "open": np.linspace(100.0, 104.0, len(index)),
            "high": np.linspace(101.0, 105.0, len(index)),
            "low": np.linspace(99.0, 103.0, len(index)),
            "close": np.linspace(100.5, 104.5, len(index)),
            "volume": np.linspace(1000.0, 1400.0, len(index)),
        }
    )
    return pd.DataFrame(data, index=index)


class SequentialExactMarginalSearchV110Tests(unittest.TestCase):
    def tearDown(self) -> None:
        accel._THREAD.state = None

    def test_batched_utilities_match_original_for_every_timestamp(self) -> None:
        index = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
        model_a = DummyModel()
        model_b = DummyModel()
        models = {"AAA": model_a, "BBB": model_b}
        frames = {"AAA": frame_for(index), "BBB": frame_for(index, 0.5)}
        symbols = ["AAA", "BBB"]
        config = object()

        expected = [
            accel._ORIGINAL_MODEL_UTILITIES(models, frames, symbols, ts, config)
            for ts in index
        ]

        accel._THREAD.state = accel._new_state()
        actual = [
            accel._accelerated_model_utilities(models, frames, symbols, ts, config)
            for ts in index
        ]

        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_prediction_cache_batches_once_per_model_set(self) -> None:
        index = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
        model = DummyModel()
        models = {"AAA": model}
        frames = {"AAA": frame_for(index)}
        accel._THREAD.state = accel._new_state()

        for ts in index[:-1]:
            accel._accelerated_model_utilities(models, frames, ["AAA"], ts, object())

        state = accel._THREAD.state
        self.assertEqual(model.calls, 1)
        self.assertEqual(state["batch_predict_calls"], 1)
        self.assertEqual(state["cache_builds"], 1)
        self.assertGreaterEqual(state["cache_hits"], len(index[:-1]) - 1)

    def test_parity_comparator_accepts_equal_nested_metrics(self) -> None:
        sessions = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        left = ({"ending_capital": 123.0, "folds": [1.0, {"x": 2.0}]}, sessions)
        right = ({"ending_capital": 123.0, "folds": [1.0, {"x": 2.0}]}, sessions.copy())

        ok, diff, error = accel._compare_replay_results(left, right)

        self.assertTrue(ok)
        self.assertEqual(diff, 0.0)
        self.assertIsNone(error)

    def test_parity_comparator_rejects_material_numeric_change(self) -> None:
        sessions = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
        left = ({"ending_capital": 123.0}, sessions)
        right = ({"ending_capital": 123.0001}, sessions)

        ok, diff, error = accel._compare_replay_results(left, right)

        self.assertFalse(ok)
        self.assertGreater(diff, accel._PARITY_ATOL)
        self.assertIn("ending_capital", str(error))


if __name__ == "__main__":
    unittest.main()
