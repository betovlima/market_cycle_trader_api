from __future__ import annotations

import math

from market_cycle_trader_api.services.model_tuning_optuna import (
    ask_optuna_candidate,
    constraint_violations,
    create_optuna_tpe_study,
    default_startup_trials,
    fixed_control_constraints,
    optuna_distributions,
    tell_optuna_candidate,
    _trial_constraints,
)


SEARCH_SPACE = [
    {
        "name": "n_estimators",
        "type": "integer",
        "min": 220,
        "max": 380,
    },
    {
        "name": "min_child_weight",
        "type": "number",
        "min": 0.001,
        "max": 10.0,
        "scale": "log",
    },
    {
        "name": "subsample",
        "type": "number",
        "min": 0.65,
        "max": 1.0,
    },
]

BASE = {
    "n_estimators": 329,
    "min_child_weight": 5.0,
    "subsample": 0.85,
}

BASELINE = {
    "ending_capital": 5_551_143.96,
    "sharpe": 1.9468,
    "maximum_drawdown": -0.3665,
    "worst_fold_return": 1.2399,
}


def test_optuna_distributions_preserve_log_child_weight() -> None:
    distributions = optuna_distributions(SEARCH_SPACE)

    child = distributions["min_child_weight"]
    assert child.log is True
    assert math.isclose(float(child.low), 0.001)
    assert math.isclose(float(child.high), 10.0)


def test_default_startup_trials_scales_with_dimension() -> None:
    assert default_startup_trials(SEARCH_SPACE) == 10
    assert default_startup_trials(
        SEARCH_SPACE
        + [
            {"name": f"x{index}", "type": "number", "min": 0.0, "max": 1.0}
            for index in range(10)
        ]
    ) == 14


def test_control_constraints_match_mct_robustness_tolerances() -> None:
    thresholds = fixed_control_constraints(BASELINE)

    assert math.isclose(thresholds["sharpe"], BASELINE["sharpe"] - 0.05)
    assert math.isclose(
        thresholds["maximum_drawdown"],
        BASELINE["maximum_drawdown"] - 0.03,
    )
    assert thresholds["worst_fold_return"] == 0.0

    violations = constraint_violations(BASELINE, thresholds)
    assert all(float(value) <= 0.0 for value in violations.values())


def test_seeded_tpe_ask_is_reproducible() -> None:
    study_a, distributions_a, _ = create_optuna_tpe_study(
        search_space=SEARCH_SPACE,
        base_tuning_values=BASE,
        baseline_metrics=BASELINE,
        seed=42,
    )
    study_b, distributions_b, _ = create_optuna_tpe_study(
        search_space=SEARCH_SPACE,
        base_tuning_values=BASE,
        baseline_metrics=BASELINE,
        seed=42,
    )

    trial_a, params_a = ask_optuna_candidate(study_a, distributions_a)
    trial_b, params_b = ask_optuna_candidate(study_b, distributions_b)

    assert trial_a.number == 1
    assert trial_b.number == 1
    assert params_a == params_b


def test_control_is_seeded_and_candidate_can_be_told() -> None:
    study, distributions, thresholds = create_optuna_tpe_study(
        search_space=SEARCH_SPACE,
        base_tuning_values=BASE,
        baseline_metrics=BASELINE,
        seed=42,
    )

    assert len(study.trials) == 1
    control = study.trials[0]
    assert control.number == 0
    assert control.params == BASE
    assert math.isclose(float(control.value), BASELINE["ending_capital"])
    assert all(
        float(value) <= 0.0
        for value in _trial_constraints(control).values()
    )

    trial, _ = ask_optuna_candidate(study, distributions)
    candidate_metrics = {
        "ending_capital": 4_000_000.0,
        "sharpe": 1.8,
        "maximum_drawdown": -0.40,
        "worst_fold_return": 0.5,
    }
    violations = tell_optuna_candidate(
        study,
        trial,
        candidate_metrics,
        thresholds=thresholds,
        champion_gate_passed=False,
    )

    assert len(study.trials) == 2
    assert math.isclose(float(study.trials[1].value), 4_000_000.0)
    assert study.trials[1].user_attrs["champion_gate_passed"] is False
    assert violations["sharpe"] > 0.0



def test_dynamic_num_leaves_respects_max_depth() -> None:
    search_space = [
        {
            "name": "max_depth",
            "type": "integer",
            "min": 2,
            "max": 4,
        },
        {
            "name": "num_leaves",
            "type": "integer",
            "min": 4,
            "max": 12,
        },
    ]
    base = {
        "max_depth": 3,
        "num_leaves": 6,
    }
    study, active_space, _ = create_optuna_tpe_study(
        search_space=search_space,
        base_tuning_values=base,
        baseline_metrics=BASELINE,
        seed=7,
    )

    for _ in range(6):
        trial, settings = ask_optuna_candidate(study, active_space)
        assert int(settings["num_leaves"]) <= 2 ** int(settings["max_depth"])
        metrics = {
            "ending_capital": 1_000_000.0 + float(trial.number),
            "sharpe": 2.0,
            "maximum_drawdown": -0.30,
            "worst_fold_return": 0.5,
        }
        tell_optuna_candidate(
            study,
            trial,
            metrics,
            thresholds=fixed_control_constraints(BASELINE),
            champion_gate_passed=False,
        )
