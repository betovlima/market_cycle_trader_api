from __future__ import annotations

import unittest

import numpy as np

from market_cycle_trader_api.services.model_tuning_probability import (
    PROBABILITY_MODEL,
    _botorch_posterior_stats,
    _fit_botorch_surrogates,
    _probabilistic_acquisition,
)


class BotorchTuningParityTests(unittest.TestCase):
    def test_probability_model_uses_botorch_backend(self) -> None:
        self.assertIn("botorch_single_task_gp", PROBABILITY_MODEL)

    def test_botorch_surrogate_returns_one_model_per_metric(self) -> None:
        x_train = np.asarray(
            [
                [0.05, 0.10],
                [0.20, 0.30],
                [0.40, 0.35],
                [0.55, 0.65],
                [0.75, 0.80],
                [0.95, 0.90],
            ],
            dtype=float,
        )
        y_train = np.asarray(
            [
                [100.0, 1.00, -0.30, 0.10],
                [110.0, 1.05, -0.28, 0.12],
                [130.0, 1.10, -0.27, 0.15],
                [150.0, 1.15, -0.25, 0.18],
                [170.0, 1.20, -0.24, 0.20],
                [190.0, 1.25, -0.22, 0.22],
            ],
            dtype=float,
        )

        models, device = _fit_botorch_surrogates(
            x_train,
            y_train,
            seed=42,
        )

        self.assertEqual(len(models), 4)
        self.assertIn(device, {"cpu", "cuda"})

        mean, std = _botorch_posterior_stats(
            models[0],
            np.asarray([[0.50, 0.50], [0.85, 0.85]], dtype=float),
        )
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue((std > 0.0).all())

    def test_acquisition_uses_multiplicative_exploration_or_probability_fallback(self) -> None:
        means = np.asarray(
            [
                [120.0, 1.20, -0.20, 0.15],
                [90.0, 1.20, -0.20, 0.15],
            ],
            dtype=float,
        )
        stds = np.asarray(
            [
                [2.0, 0.02, 0.01, 0.01],
                [40.0, 0.02, 0.01, 0.01],
            ],
            dtype=float,
        )
        thresholds = {
            "capital": 105.0,
            "sharpe": 1.00,
            "maximum_drawdown": -0.30,
            "worst_fold_return": 0.0,
            "baseline_capital": 100.0,
        }
        y_train = np.asarray(
            [
                [100.0, 1.00, -0.30, 0.05],
                [110.0, 1.10, -0.28, 0.10],
                [115.0, 1.15, -0.25, 0.12],
                [120.0, 1.20, -0.22, 0.15],
            ],
            dtype=float,
        )

        probability, expected_improvement, acquisition, mode = (
            _probabilistic_acquisition(
                means,
                stds,
                thresholds=thresholds,
                y_train=y_train,
                seed=42,
                exploration_weight=0.15,
            )
        )

        self.assertIn(
            mode,
            {"multiplicative_constrained_ei", "probability_fallback"},
        )
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(np.isfinite(expected_improvement).all())
        self.assertTrue(np.isfinite(acquisition).all())
        self.assertGreaterEqual(float(acquisition[0]), 0.0)
        self.assertGreaterEqual(float(acquisition[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
