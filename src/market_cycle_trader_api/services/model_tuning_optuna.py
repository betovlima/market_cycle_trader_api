from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence
import warnings

import numpy as np
import optuna
from scipy.stats import qmc
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.study import Study
from optuna.trial import Trial

from .model_tuning_space import settings_from_unit_point, unit_value_for_setting


OPTUNA_TPE_MODEL = "optuna_tpe_multivariate_constrained_v1"
DEFAULT_SHARPE_TOLERANCE = 0.05
DEFAULT_DRAWDOWN_TOLERANCE = 0.03
DEFAULT_MIN_WORST_FOLD_RETURN = 0.0
_CONSTRAINT_ATTR = "_mct_optimizer_constraints"
_CONSTRAINT_ORDER = (
    "sharpe",
    "maximum_drawdown",
    "worst_fold_return",
)


def _native_constraint_api_available() -> bool:
    return hasattr(optuna.trial.Trial, "set_constraint")


def _legacy_constraints_func(
    frozen_trial: optuna.trial.FrozenTrial,
) -> list[float]:
    values = frozen_trial.user_attrs.get(_CONSTRAINT_ATTR) or {}
    return [
        float(values.get(name, 0.0))
        for name in _CONSTRAINT_ORDER
    ]


def _trial_constraints(
    frozen_trial: optuna.trial.FrozenTrial,
) -> dict[str, float]:
    native = getattr(frozen_trial, "constraints", None)
    if isinstance(native, dict):
        return {
            str(key): float(value)
            for key, value in native.items()
        }
    values = frozen_trial.user_attrs.get(_CONSTRAINT_ATTR) or {}
    return {
        str(key): float(value)
        for key, value in values.items()
    }


def default_startup_trials(
    search_space: Sequence[dict[str, Any]],
) -> int:
    return max(10, min(24, len(list(search_space)) + 1))


def _distribution_for_spec(
    spec: dict[str, Any],
    *,
    current_settings: dict[str, Any] | None = None,
) -> Any:
    name = str(spec["name"])
    low = spec["min"]
    high = spec["max"]
    scale = str(spec.get("scale") or "linear").strip().lower()

    if (
        name == "num_leaves"
        and current_settings is not None
        and current_settings.get("max_depth") is not None
    ):
        high = min(int(high), 2 ** int(current_settings["max_depth"]))

    if str(spec["type"]) == "integer":
        return IntDistribution(
            low=int(low),
            high=int(high),
            log=(scale == "log"),
        )
    return FloatDistribution(
        low=float(low),
        high=float(high),
        log=(scale == "log"),
    )


def optuna_distributions(
    search_space: Sequence[dict[str, Any]],
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    working = dict(settings or {})
    for spec in search_space:
        name = str(spec["name"])
        distributions[name] = _distribution_for_spec(
            spec,
            current_settings=working,
        )
    return distributions


def _suggest_optuna_settings(
    trial: Trial,
    search_space: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        distribution = _distribution_for_spec(
            spec,
            current_settings=settings,
        )
        if isinstance(distribution, IntDistribution):
            settings[name] = trial.suggest_int(
                name,
                int(distribution.low),
                int(distribution.high),
                log=bool(distribution.log),
            )
        elif isinstance(distribution, FloatDistribution):
            settings[name] = trial.suggest_float(
                name,
                float(distribution.low),
                float(distribution.high),
                log=bool(distribution.log),
            )
        else:
            raise TypeError(
                f"Unsupported Optuna distribution for {name}: "
                f"{type(distribution).__name__}"
            )
    return settings


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
            + 1e-12
        ),
    }


def _apply_trial_constraints(
    trial: Trial,
    violations: dict[str, float],
) -> None:
    trial.set_user_attr(
        _CONSTRAINT_ATTR,
        deepcopy(violations),
    )
    if _native_constraint_api_available():
        for name, value in violations.items():
            trial.set_constraint(name, float(value))


def control_centered_warm_start_settings(
    search_space: Sequence[dict[str, Any]],
    base_tuning_values: dict[str, Any],
    *,
    seed: int,
    count: int = 6,
    radius: float = 0.06,
) -> list[dict[str, Any]]:
    active_space = [dict(item) for item in search_space]
    if count <= 0:
        return []
    bounded_radius = max(0.01, min(float(radius), 0.25))
    anchor = np.asarray(
        [
            unit_value_for_setting(
                spec,
                base_tuning_values[str(spec["name"])],
            )
            for spec in active_space
        ],
        dtype=float,
    )
    design = qmc.LatinHypercube(
        d=len(active_space),
        seed=int(seed) + 17011,
    ).random(n=int(count))
    offsets = (design - 0.5) * (2.0 * bounded_radius)
    points = np.clip(anchor[None, :] + offsets, 0.0, 1.0)

    warm_settings: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = {
        tuple(sorted(base_tuning_values.items()))
    }
    for point in points:
        settings = settings_from_unit_point(
            base_tuning_values,
            active_space,
            point,
        )
        key = tuple(sorted(settings.items()))
        if key in seen:
            continue
        seen.add(key)
        warm_settings.append(settings)
    return warm_settings


def create_optuna_tpe_study(
    *,
    search_space: Sequence[dict[str, Any]],
    base_tuning_values: dict[str, Any],
    baseline_metrics: dict[str, Any],
    seed: int,
    startup_trials: int | None = None,
    warm_start_count: int = 6,
    warm_start_radius: float = 0.06,
) -> tuple[Study, list[dict[str, Any]], dict[str, float]]:
    active_space = [dict(item) for item in search_space]
    warm_settings = control_centered_warm_start_settings(
        active_space,
        base_tuning_values,
        seed=int(seed),
        count=int(warm_start_count),
        radius=float(warm_start_radius),
    )
    resolved_startup_trials = int(
        startup_trials
        if startup_trials is not None
        else max(1, len(warm_settings) + 1)
    )
    resolved_startup_trials = max(
        resolved_startup_trials,
        len(warm_settings) + 1,
    )

    sampler_kwargs: dict[str, Any] = {
        "seed": int(seed),
        "n_startup_trials": resolved_startup_trials,
        "multivariate": True,
        "group": True,
        "constant_liar": False,
        "warn_independent_sampling": False,
    }
    if not _native_constraint_api_available():
        sampler_kwargs["constraints_func"] = _legacy_constraints_func

    with warnings.catch_warnings():
        experimental_warning = getattr(
            optuna.exceptions,
            "ExperimentalWarning",
            Warning,
        )
        warnings.simplefilter(
            "ignore",
            experimental_warning,
        )
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="mct_raw_split_optuna_tpe",
    )

    thresholds = fixed_control_constraints(baseline_metrics)
    control_params = {
        str(spec["name"]): deepcopy(
            base_tuning_values[str(spec["name"])]
        )
        for spec in active_space
    }
    control_violations = constraint_violations(
        baseline_metrics,
        thresholds,
    )

    # Seed through Optuna's own trial lifecycle rather than add_trial().
    # This guarantees that constraints are materialized by both the legacy
    # 4.8 constraints callback and the native 5.x Trial.set_constraint API.
    study.enqueue_trial(
        control_params,
        user_attrs={
            "kind": "control",
            "is_control": True,
        },
    )
    control_trial = study.ask()
    effective_control = _suggest_optuna_settings(
        control_trial,
        active_space,
    )
    if effective_control != control_params:
        raise RuntimeError(
            "Optuna did not materialize the frozen MCT Control exactly."
        )
    _apply_trial_constraints(
        control_trial,
        control_violations,
    )
    control_trial.set_user_attr(
        "metrics",
        deepcopy(baseline_metrics),
    )
    study.tell(
        control_trial,
        float(baseline_metrics["ending_capital"]),
    )

    for warm_index, warm_params in enumerate(warm_settings, start=1):
        study.enqueue_trial(
            warm_params,
            user_attrs={
                "kind": "control_local_warm_start",
                "is_control": False,
                "warm_start_index": int(warm_index),
                "warm_start_radius": float(warm_start_radius),
            },
        )

    return study, active_space, thresholds


def ask_optuna_candidate(
    study: Study,
    search_space: Sequence[dict[str, Any]],
) -> tuple[Trial, dict[str, Any]]:
    trial = study.ask()
    settings = _suggest_optuna_settings(trial, search_space)
    return trial, settings


def tell_optuna_candidate(
    study: Study,
    trial: Trial,
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, float],
    champion_gate_passed: bool,
) -> dict[str, float]:
    violations = constraint_violations(metrics, thresholds)
    _apply_trial_constraints(
        trial,
        violations,
    )
    if not trial.user_attrs.get("kind"):
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
        if all(
            float(value) <= 0.0
            for value in _trial_constraints(trial).values()
        )
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
