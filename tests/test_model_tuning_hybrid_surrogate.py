from __future__ import annotations

import math

import numpy as np

from market_cycle_trader_api.services.model_tuning_probability import (
    PROBABILITY_MODEL,
    _empirical_champion_pass_prior,
    _reliability_adjusted_acquisition,
    _surrogate_cross_validated_reliability,
)


def test_hybrid_probability_model_is_versioned() -> None:
    assert "hybrid_gp_extra_trees" in PROBABILITY_MODEL
    assert PROBABILITY_MODEL.endswith("_v5")


def test_empirical_champion_prior_uses_beta_smoothing() -> None:
    document = {
        "prior_observations": [],
        "candidates": [
            {
                "candidate_id": 1,
                "status": "completed",
                "settings": {"x": 0.1},
                "metrics": {
                    "ending_capital": 100.0,
                    "sharpe": 1.0,
                    "maximum_drawdown": -0.2,
                    "worst_fold_return": 0.1,
                },
                "champion_gate_passed": False,
            },
            {
                "candidate_id": 2,
                "status": "completed",
                "settings": {"x": 0.2},
                "metrics": {
                    "ending_capital": 120.0,
                    "sharpe": 1.1,
                    "maximum_drawdown": -0.2,
                    "worst_fold_return": 0.2,
                },
                "champion_gate_passed": True,
            },
        ],
    }

    prior, successes, trials = _empirical_champion_pass_prior(document)

    assert successes == 1
    assert trials == 2
    assert math.isclose(prior, 0.5)


def test_low_reliability_shrinks_probability_to_empirical_prior() -> None:
    raw_probability = np.asarray([0.90, 0.70], dtype=float)
    raw_expected_improvement = np.asarray([0.50, 0.25], dtype=float)
    stds = np.asarray(
        [
            [20.0, 0.1, 0.1, 0.1],
            [10.0, 0.1, 0.1, 0.1],
        ],
        dtype=float,
    )
    thresholds = {
        "baseline_capital": 100.0,
        "capital": 100.0,
        "sharpe": 1.0,
        "maximum_drawdown": -0.3,
        "worst_fold_return": 0.0,
    }

    probability, expected_improvement, acquisition, effective_exploration = (
        _reliability_adjusted_acquisition(
            raw_probability,
            raw_expected_improvement,
            stds,
            thresholds=thresholds,
            exploration_weight=0.15,
            surrogate_reliability=0.0,
            empirical_probability=0.10,
        )
    )

    assert np.allclose(probability, 0.10)
    assert np.allclose(expected_improvement, raw_expected_improvement * 0.25)
    assert effective_exploration > 0.15
    assert np.all(np.isfinite(acquisition))


def test_cross_validated_hybrid_weights_are_finite_and_normalized() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(16, 4))
    y = np.column_stack(
        [
            np.where(x[:, 0] > 0.5, 10.0, 1.0) + rng.normal(0.0, 0.2, size=16),
            x[:, 1] + rng.normal(0.0, 0.05, size=16),
            -np.abs(x[:, 2] - 0.5),
            x[:, 3] + rng.normal(0.0, 0.05, size=16),
        ]
    )

    diagnostics = _surrogate_cross_validated_reliability(x, y, seed=123)

    gp_weight = np.asarray(diagnostics["gp_weight"], dtype=float)
    tree_weight = np.asarray(diagnostics["extra_trees_weight"], dtype=float)
    reliability = np.asarray(diagnostics["metric_reliability"], dtype=float)

    assert gp_weight.shape == (4,)
    assert tree_weight.shape == (4,)
    assert reliability.shape == (4,)
    assert np.all(np.isfinite(gp_weight))
    assert np.all(np.isfinite(tree_weight))
    assert np.allclose(gp_weight + tree_weight, 1.0)
    assert np.all((reliability >= 0.0) & (reliability <= 1.0))
    # The first synthetic metric is deliberately discontinuous. Unless GP
    # proves material OOF superiority, the discontinuity guard keeps trees
    # dominant in the central surrogate estimate.
    assert float(tree_weight[0]) >= 0.80
