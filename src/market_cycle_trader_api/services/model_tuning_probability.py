from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import warnings
from typing import Any

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

PROBABILITY_MODEL = "gaussian_process_constrained_ei_v1"
_METRIC_COUNT = 4
_MONTE_CARLO_SCENARIOS = 512
_GLOBAL_POOL_FRACTION = 0.35
_ANCHOR_LOCAL_FRACTION = 0.35
_LOCAL_SCALE = 0.12
_TOP_REGION_SCALE = 0.15


def _settings_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_value(spec: dict[str, Any], unit_value: float) -> Any:
    low = float(spec["min"])
    high = float(spec["max"])
    value = low + float(unit_value) * (high - low)
    if spec["type"] == "integer":
        return int(round(value))
    return round(value, int(spec.get("precision") or 8))


def _settings_from_unit_point(
    base_values: dict[str, Any],
    search_space: list[dict[str, Any]],
    point: np.ndarray,
) -> dict[str, Any]:
    values = deepcopy(base_values)
    for spec, unit_value in zip(search_space, point, strict=True):
        values[spec["name"]] = _sample_value(spec, float(unit_value))
    depth = int(values.get("max_depth") or 0)
    if depth > 0 and "num_leaves" in values:
        values["num_leaves"] = min(int(values["num_leaves"]), 2 ** depth)
        values["num_leaves"] = max(2, int(values["num_leaves"]))
    return values


def _normalized_vector(settings: dict[str, Any], search_space: list[dict[str, Any]]) -> list[float]:
    vector: list[float] = []
    for spec in search_space:
        low = float(spec["min"])
        high = float(spec["max"])
        value = float(settings.get(spec["name"], low))
        vector.append(0.0 if high <= low else max(0.0, min(1.0, (value - low) / (high - low))))
    return vector


def _all_completed_observations(document: dict[str, Any]) -> list[dict[str, Any]]:
    observations = list(document.get("prior_observations") or []) + list(document.get("candidates") or [])
    result: list[dict[str, Any]] = []
    for candidate in observations:
        if candidate.get("status") != "completed":
            continue
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else None
        settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else None
        if not metrics or not settings:
            continue
        required = (
            metrics.get("ending_capital"),
            metrics.get("sharpe"),
            metrics.get("maximum_drawdown"),
            metrics.get("worst_fold_return"),
        )
        if any(value is None for value in required):
            continue
        result.append(candidate)
    return result


def _completed_observations(document: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    search_space = list(document.get("search_space") or [])
    observations = _all_completed_observations(document)
    x_rows = [_normalized_vector(dict(item["settings"]), search_space) for item in observations]
    y_rows = [
        [
            float(item["metrics"]["ending_capital"]),
            float(item["metrics"]["sharpe"]),
            float(item["metrics"]["maximum_drawdown"]),
            float(item["metrics"]["worst_fold_return"]),
        ]
        for item in observations
    ]
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def _baseline_thresholds(document: dict[str, Any]) -> dict[str, float]:
    config = dict(document.get("probability_config") or {})
    anchor = document.get("probability_anchor") if isinstance(document.get("probability_anchor"), dict) else {}
    if isinstance(anchor.get("metrics"), dict):
        metrics = anchor["metrics"]
    else:
        baseline = document.get("baseline_execution") if isinstance(document.get("baseline_execution"), dict) else {}
        metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    ending_capital = float(metrics.get("ending_capital") or 0.0)
    if ending_capital <= 0:
        raise RuntimeError("A positive baseline ending capital is required for probabilistic tuning.")
    return {
        "capital": ending_capital * (1.0 + float(config.get("min_capital_improvement") or 0.0)),
        "sharpe": float(metrics.get("sharpe") or 0.0) - float(config.get("sharpe_tolerance") or 0.0),
        "maximum_drawdown": float(metrics.get("maximum_drawdown") or 0.0) - float(config.get("drawdown_tolerance") or 0.0),
        "worst_fold_return": float(config.get("min_worst_fold_return") or 0.0),
        "baseline_capital": ending_capital,
    }


def champion_gate_evaluation(document: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an observed candidate against the configured Champion-relative gate."""
    thresholds = _baseline_thresholds(document)
    ending_capital = float(metrics.get("ending_capital") or 0.0)
    sharpe = float(metrics.get("sharpe") or 0.0)
    maximum_drawdown = float(metrics.get("maximum_drawdown") or 0.0)
    worst_fold_return = metrics.get("worst_fold_return")
    worst_fold = float(worst_fold_return) if worst_fold_return is not None else float("-inf")
    passed = bool(
        ending_capital >= thresholds["capital"]
        and sharpe >= thresholds["sharpe"]
        and maximum_drawdown >= thresholds["maximum_drawdown"]
        and worst_fold > thresholds["worst_fold_return"]
    )
    return {
        "passed": passed,
        "thresholds": thresholds,
        "observed": {
            "ending_capital": ending_capital,
            "sharpe": sharpe,
            "maximum_drawdown": maximum_drawdown,
            "worst_fold_return": worst_fold_return,
        },
    }


def _top_observation_centers(
    document: dict[str, Any],
    search_space: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> np.ndarray:
    observations = _all_completed_observations(document)
    eligible = [item for item in observations if bool((item.get("metrics") or {}).get("eligible", True))]
    ranked = eligible or observations
    ranked.sort(
        key=lambda item: (
            float((item.get("metrics") or {}).get("risk_adjusted_compound_score") or float("-inf")),
            float((item.get("metrics") or {}).get("ending_capital") or float("-inf")),
        ),
        reverse=True,
    )
    centers = [_normalized_vector(dict(item["settings"]), search_space) for item in ranked[: max(1, limit)]]
    return np.asarray(centers, dtype=float)


def _candidate_unit_pool(
    document: dict[str, Any],
    *,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    search_space = list(document.get("search_space") or [])
    dimensions = len(search_space)
    rng = np.random.default_rng(seed)
    global_count = max(1, int(round(pool_size * _GLOBAL_POOL_FRACTION)))
    anchor_count = max(1, int(round(pool_size * _ANCHOR_LOCAL_FRACTION)))
    top_count = max(1, pool_size - global_count - anchor_count)

    global_points = qmc.LatinHypercube(d=dimensions, seed=seed + 17).random(n=global_count)
    anchor = document.get("probability_anchor") if isinstance(document.get("probability_anchor"), dict) else {}
    anchor_settings = anchor.get("settings") if isinstance(anchor.get("settings"), dict) else document.get("base_model_values")
    anchor_vector = np.asarray(_normalized_vector(dict(anchor_settings or {}), search_space), dtype=float)
    anchor_points = np.clip(
        anchor_vector + rng.normal(0.0, _LOCAL_SCALE, size=(anchor_count, dimensions)),
        0.0,
        1.0,
    )

    centers = _top_observation_centers(document, search_space)
    center_indices = rng.integers(0, len(centers), size=top_count)
    top_points = np.clip(
        centers[center_indices] + rng.normal(0.0, _TOP_REGION_SCALE, size=(top_count, dimensions)),
        0.0,
        1.0,
    )
    return np.vstack([global_points, anchor_points, top_points])


def _fit_gaussian_processes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> list[GaussianProcessRegressor]:
    dimensions = int(x_train.shape[1])
    models: list[GaussianProcessRegressor] = []
    for metric_index in range(_METRIC_COUNT):
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(
                length_scale=np.full(dimensions, 0.35, dtype=float),
                length_scale_bounds=(0.05, 5.0),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 0.2))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed + metric_index * 1543,
            n_restarts_optimizer=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train[:, metric_index])
        models.append(model)
    return models


def _outcome_correlation(y_train: np.ndarray) -> np.ndarray:
    if len(y_train) < 3:
        return np.eye(_METRIC_COUNT, dtype=float)
    correlation = np.corrcoef(y_train[:, :_METRIC_COUNT], rowvar=False)
    if correlation.shape != (_METRIC_COUNT, _METRIC_COUNT) or not np.all(np.isfinite(correlation)):
        return np.eye(_METRIC_COUNT, dtype=float)
    # Shrink the empirical correlation toward independence because the sample of
    # expensive backtests is intentionally small. Then project to a positive-
    # definite matrix so deterministic Monte Carlo sampling is numerically stable.
    correlation = 0.5 * correlation + 0.5 * np.eye(_METRIC_COUNT, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(projected))
    return projected / np.outer(diagonal, diagonal)


def _probabilistic_acquisition(
    means: np.ndarray,
    stds: np.ndarray,
    *,
    thresholds: dict[str, float],
    y_train: np.ndarray,
    seed: int,
    exploration_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    correlation = _outcome_correlation(y_train)
    try:
        cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        cholesky = np.eye(_METRIC_COUNT, dtype=float)
    z = rng.normal(size=(_MONTE_CARLO_SCENARIOS, _METRIC_COUNT)) @ cholesky.T

    probability = np.zeros(len(means), dtype=float)
    expected_improvement = np.zeros(len(means), dtype=float)
    chunk_size = 1024
    for start in range(0, len(means), chunk_size):
        stop = min(len(means), start + chunk_size)
        scenarios = means[None, start:stop, :] + stds[None, start:stop, :] * z[:, None, :]
        capital = scenarios[:, :, 0]
        risk_ok = (
            (scenarios[:, :, 1] >= thresholds["sharpe"])
            & (scenarios[:, :, 2] >= thresholds["maximum_drawdown"])
            & (scenarios[:, :, 3] > thresholds["worst_fold_return"])
        )
        beats = risk_ok & (capital >= thresholds["capital"])
        probability[start:stop] = beats.mean(axis=0)
        positive = np.maximum(capital - thresholds["capital"], 0.0) / thresholds["baseline_capital"]
        expected_improvement[start:stop] = (positive * risk_ok).mean(axis=0)

    normalized_uncertainty = stds[:, 0] / thresholds["baseline_capital"]
    # Constrained Expected Improvement already rewards useful uncertainty. The extra
    # exploration term is feasibility-weighted so a wildly uncertain but clearly
    # unsafe region cannot dominate a plausible Champion-beating neighborhood.
    feasibility_weight = 0.25 + 0.75 * probability
    acquisition = (
        expected_improvement
        + exploration_weight * normalized_uncertainty * feasibility_weight
        + 0.02 * probability
    )
    return probability, expected_improvement, acquisition


def propose_champion_probability_candidate(document: dict[str, Any]) -> dict[str, Any]:
    """Propose one adaptive CARO candidate using Champion-anchored Gaussian processes.

    The probability is a surrogate research estimate under the frozen validation
    protocol. It is deliberately not presented as a probability of future profit.
    """
    search_space = list(document.get("search_space") or [])
    base_values = dict(document.get("base_model_values") or {})
    if not search_space or not base_values:
        raise RuntimeError("Probabilistic tuning requires a search space and base model values.")

    x_train, y_train = _completed_observations(document)
    if len(x_train) < 4:
        raise RuntimeError("At least four completed observations are required for probabilistic refinement.")

    config = dict(document.get("probability_config") or {})
    seed = int(document.get("seed") or 42)
    all_observations = list(document.get("prior_observations") or []) + list(document.get("candidates") or [])
    existing_ids = [int(item.get("candidate_id") or 0) for item in all_observations]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    pool_size = max(256, min(int(config.get("candidate_pool_size") or 2048), 16_384))
    exploration_weight = max(0.0, min(float(config.get("exploration_weight") or 0.15), 2.0))

    seen_hashes = {
        str(item.get("settings_hash") or "")
        for item in all_observations
        if item.get("settings_hash")
    }
    raw_points = _candidate_unit_pool(document, pool_size=pool_size, seed=seed + next_id * 7919)
    proposal_settings: list[dict[str, Any]] = []
    proposal_vectors: list[list[float]] = []
    proposal_hashes: list[str] = []
    for point in raw_points:
        values = _settings_from_unit_point(base_values, search_space, point)
        fingerprint = _settings_hash(values)
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        proposal_settings.append(values)
        proposal_vectors.append(_normalized_vector(values, search_space))
        proposal_hashes.append(fingerprint)
    if len(proposal_settings) < 32:
        raise RuntimeError("Unable to generate a sufficiently diverse probabilistic candidate pool.")

    x_pool = np.asarray(proposal_vectors, dtype=float)
    models = _fit_gaussian_processes(x_train, y_train, seed=seed + next_id * 104729)
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for model in models:
        mean, std = model.predict(x_pool, return_std=True)
        means.append(mean)
        stds.append(np.maximum(std, 1e-12))
    mean_matrix = np.stack(means, axis=1)
    std_matrix = np.stack(stds, axis=1)

    thresholds = _baseline_thresholds(document)
    probability, constrained_expected_improvement, acquisition = _probabilistic_acquisition(
        mean_matrix,
        std_matrix,
        thresholds=thresholds,
        y_train=y_train,
        seed=seed + next_id * 65537,
        exploration_weight=exploration_weight,
    )
    selected_index = int(np.argmax(acquisition))
    selected_settings = proposal_settings[selected_index]

    top_count = max(12, min(64, max(1, len(proposal_settings) // 40)))
    top_indices = np.argsort(acquisition)[-top_count:]
    promising_region: dict[str, dict[str, float | int]] = {}
    for spec in search_space:
        values = np.asarray([float(proposal_settings[int(index)][spec["name"]]) for index in top_indices], dtype=float)
        low = float(np.quantile(values, 0.10))
        high = float(np.quantile(values, 0.90))
        if spec["type"] == "integer":
            promising_region[spec["name"]] = {"low": int(round(low)), "high": int(round(high))}
        else:
            precision = int(spec.get("precision") or 8)
            promising_region[spec["name"]] = {"low": round(low, precision), "high": round(high, precision)}

    return {
        "candidate_id": next_id,
        "kind": "champion_probability",
        "is_control": False,
        "settings": selected_settings,
        "settings_hash": proposal_hashes[selected_index],
        "status": "pending",
        "proposal": {
            "probability_model": PROBABILITY_MODEL,
            "observation_count": int(len(x_train)),
            "candidate_pool_size": int(len(proposal_settings)),
            "estimated_probability_beats_champion": float(probability[selected_index]),
            "estimated_expected_improvement": float(constrained_expected_improvement[selected_index]),
            "estimated_ending_capital_mean": float(mean_matrix[selected_index, 0]),
            "estimated_ending_capital_std": float(std_matrix[selected_index, 0]),
            "estimated_sharpe_mean": float(mean_matrix[selected_index, 1]),
            "estimated_maximum_drawdown_mean": float(mean_matrix[selected_index, 2]),
            "estimated_worst_fold_mean": float(mean_matrix[selected_index, 3]),
            "acquisition_score": float(acquisition[selected_index]),
            "exploration_weight": exploration_weight,
            "monte_carlo_scenarios": _MONTE_CARLO_SCENARIOS,
            "pool_composition": {
                "global_fraction": _GLOBAL_POOL_FRACTION,
                "anchor_local_fraction": _ANCHOR_LOCAL_FRACTION,
                "top_regions_fraction": round(1.0 - _GLOBAL_POOL_FRACTION - _ANCHOR_LOCAL_FRACTION, 6),
            },
            "promising_region": promising_region,
            "promising_region_probability_mean": float(probability[top_indices].mean()),
            "promising_region_expected_improvement_mean": float(constrained_expected_improvement[top_indices].mean()),
            "thresholds": thresholds,
            "interpretation": "research_surrogate_probability_not_future_profit_probability",
        },
    }
