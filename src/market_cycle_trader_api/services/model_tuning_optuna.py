from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.study import Study
from optuna.trial import Trial


OPTUNA_TPE_MODEL = "optuna_tpe_multivariate_constrained_v1"
DEFAULT_SHARPE_TOLERANCE = 0.05
DEFAULT_DRAWDOWN_TOLERANCE = 0.03
DEFAULT_MIN_WORST_FOLD_RETURN = 0.0


def default_startup_trials(
    search_space: Sequence[dict[str, Any]],
) -> int:
    return max(10, min(24, len(list(search_space)) + 1))


def optuna_distributions(
    search_space: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        low = spec["min"]
        high = spec["max"]
        scale = str(spec.get("scale") or "linear").strip().lower()
        if str(spec["type"]) == "integer":
            distributions[name] = IntDistribution(
                low=int(low),
                high=int(high),
                log=(scale == "log"),
            )
        else:
            distributions[name] = FloatDistribution(
                low=float(low),
                high=float(high),
                log=(scale == "log"),
            )
    return distributions


def fixed_control_constraints(
    baseline_metrics: dict[str, Any],
    *,
    sharpe_tolerance: float = DEFAULT_SHARPE_TOLERANCE,
    drawdown_tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE,
    min_worst_fold_return: float = DEFAULT_MIN_WORST_FOLD_RETURN,
) -> dict[str, float]:
    return {
        "sharpe": float(baseline_metrics["sharpe"]) - float(sharpe_tolerance),
        "maximum_drawdown": (
            float(baseline_metrics["maximum_drawdown"])
            - float(drawdown_tolerance)
        ),
        "worst_fold_return": float(min_worst_fold_return),
    }


def constraint_violations(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, float]:
    worst_fold = metrics.get("worst_fold_return")
    return {
        # Optuna constraint semantics: <= 0 is feasible.
        "sharpe": float(thresholds["sharpe"]) - float(metrics["sharpe"]),
        "maximum_drawdown": (
            float(thresholds["maximum_drawdown"])
            - float(metrics["maximum_drawdown"])
        ),
        "worst_fold_return": (
            float(thresholds["worst_fold_return"])
            - float(worst_fold if worst_fold is not None else float("-inf"))
        ),
    }


def create_optuna_tpe_study(
    *,
    search_space: Sequence[dict[str, Any]],
    base_tuning_values: dict[str, Any],
    baseline_metrics: dict[str, Any],
    seed: int,
    startup_trials: int | None = None,
) -> tuple[Study, dict[str, Any], dict[str, float]]:
    distributions = optuna_distributions(search_space)
    resolved_startup_trials = int(
        startup_trials
        if startup_trials is not None
        else default_startup_trials(search_space)
    )

    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=resolved_startup_trials,
        multivariate=True,
        constant_liar=False,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="mct_raw_split_optuna_tpe",
    )

    thresholds = fixed_control_constraints(baseline_metrics)
    control_params = {
        name: deepcopy(base_tuning_values[name])
        for name in distributions
    }
    control_violations = constraint_violations(
        baseline_metrics,
        thresholds,
    )
    control_trial = optuna.trial.create_trial(
        params=control_params,
        distributions=distributions,
        value=float(baseline_metrics["ending_capital"]),
        user_attrs={
            "kind": "control",
            "is_control": True,
            "metrics": deepcopy(baseline_metrics),
        },
        constraints=control_violations,
    )
    study.add_trial(control_trial)
    return study, distributions, thresholds


def ask_optuna_candidate(
    study: Study,
    distributions: dict[str, Any],
) -> tuple[Trial, dict[str, Any]]:
    trial = study.ask(fixed_distributions=distributions)
    return trial, dict(trial.params)


def tell_optuna_candidate(
    study: Study,
    trial: Trial,
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, float],
    champion_gate_passed: bool,
) -> dict[str, float]:
    violations = constraint_violations(metrics, thresholds)
    for name, value in violations.items():
        trial.set_constraint(name, float(value))
    trial.set_user_attr("kind", "optuna_tpe")
    trial.set_user_attr("is_control", False)
    trial.set_user_attr("champion_gate_passed", bool(champion_gate_passed))
    trial.set_user_attr("metrics", deepcopy(metrics))
    study.tell(trial, float(metrics["ending_capital"]))
    return violations


def optuna_study_diagnostics(study: Study) -> dict[str, Any]:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    feasible = [
        trial
        for trial in completed
        if all(float(value) <= 0.0 for value in trial.constraints.values())
    ]
    best = max(
        completed,
        key=lambda trial: float(trial.value or float("-inf")),
        default=None,
    )
    best_feasible = max(
        feasible,
        key=lambda trial: float(trial.value or float("-inf")),
        default=None,
    )
    return {
        "sampler": OPTUNA_TPE_MODEL,
        "completed_trial_count": len(completed),
        "feasible_trial_count": len(feasible),
        "best_trial_number": (best.number if best is not None else None),
        "best_trial_value": (float(best.value) if best is not None and best.value is not None else None),
        "best_feasible_trial_number": (
            best_feasible.number if best_feasible is not None else None
        ),
        "best_feasible_trial_value": (
            float(best_feasible.value)
            if best_feasible is not None and best_feasible.value is not None
            else None
        ),
    }
