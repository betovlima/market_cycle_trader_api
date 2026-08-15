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

from .model_tuning_space import settings_from_unit_point as _settings_from_unit_point
from .model_tuning_ranking import candidate_economic_sort_key

PROBABILITY_MODEL = "gaussian_process_unified_exploration_trust_region_cei_v4"
_METRIC_COUNT = 4
_MONTE_CARLO_SCENARIOS = 512
_TRUST_REGION_INITIAL = 0.20
_TRUST_REGION_MIN = 0.04
_TRUST_REGION_MAX = 0.40
_TRUST_REGION_SUCCESS_EXPANSION = 1.25
_TRUST_REGION_FAILURE_CONTRACTION = 0.70
_TRUST_REGION_FAILURE_TOLERANCE = 3
_GLOBAL_POOL_FRACTION_INITIAL = 0.30
_GLOBAL_POOL_FRACTION_MIN = 0.15
_ANCHOR_LOCAL_FRACTION = 0.50
_UNIFIED_SPACE_FILLING_POOL_SIZE = 1024
_UNIFIED_INITIAL_EXPLORATION_FRACTION = 0.45
_UNIFIED_MIN_EXPLORATION_FRACTION = 0.20
_UNIFIED_STAGNATION_RECOVERY_TRIALS = 4
_UNIFIED_RECOVERY_COOLDOWN_TRIALS = 2
_UNIFIED_READINESS_MEAN_SPAN_MIN = 0.50
_UNIFIED_READINESS_BROAD_DIMENSION_FRACTION = 0.70
_UNIFIED_READINESS_DIMENSION_SPAN_MIN = 0.45


def _settings_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    champion_aware = any(item.get("champion_gate_passed") is not None for item in ranked)
    ranked.sort(
        key=lambda item: candidate_economic_sort_key(item, champion_aware=champion_aware),
        reverse=True,
    )
    centers = [_normalized_vector(dict(item["settings"]), search_space) for item in ranked[: max(1, limit)]]
    return np.asarray(centers, dtype=float)


def _probability_state(document: dict[str, Any]) -> dict[str, Any]:
    raw = document.get("probability_state") if isinstance(document.get("probability_state"), dict) else {}
    radius = float(raw.get("trust_region_radius") or _TRUST_REGION_INITIAL)
    return {
        "trust_region_radius": max(_TRUST_REGION_MIN, min(radius, _TRUST_REGION_MAX)),
        "success_streak": max(0, int(raw.get("success_streak") or 0)),
        "failure_streak": max(0, int(raw.get("failure_streak") or 0)),
        "no_improvement_streak": max(0, int(raw.get("no_improvement_streak") or 0)),
        "recovery_cooldown_remaining": max(0, int(raw.get("recovery_cooldown_remaining") or 0)),
        "stagnation_recoveries": max(0, int(raw.get("stagnation_recoveries") or 0)),
        "adaptive_trials_completed": max(0, int(raw.get("adaptive_trials_completed") or 0)),
        "exploration_trials_completed": max(0, int(raw.get("exploration_trials_completed") or 0)),
        "proposal_mode_switches": max(0, int(raw.get("proposal_mode_switches") or 0)),
        "last_proposal_mode": str(raw.get("last_proposal_mode") or ""),
        "champion_revision": max(0, int(raw.get("champion_revision") or 0)),
        "last_champion_candidate_id": raw.get("last_champion_candidate_id"),
    }


def initial_probability_state(observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    state = _probability_state({})
    for item in observations or []:
        if item.get("status") != "completed":
            continue
        kind = str(item.get("kind") or "")
        if kind == "champion_probability":
            state["adaptive_trials_completed"] += 1
        elif kind in {"unified_exploration", "probability_startup", "latin_hypercube", "prior_latin_hypercube_observation"}:
            state["exploration_trials_completed"] += 1
    return state


def evolve_probability_search(
    document: dict[str, Any],
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    champion_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    






    state = _probability_state(document)
    kind = str(candidate.get("kind") or "")
    adaptive = kind == "champion_probability"
    exploration = kind in {"unified_exploration", "probability_startup", "latin_hypercube", "prior_latin_hypercube_observation"}
    proposal_mode = "adaptive_probability" if adaptive else ("space_filling" if exploration else kind or "other")
    if state.get("last_proposal_mode") and state.get("last_proposal_mode") != proposal_mode:
        state["proposal_mode_switches"] += 1
    state["last_proposal_mode"] = proposal_mode
    if exploration:
        state["exploration_trials_completed"] += 1
        selection_reason = str((candidate.get("proposal") or {}).get("selection_reason") or "")
        if selection_reason == "stagnation_recovery":
            # A recovery exploration opens a new learning cycle. Reset the stale
            # failure evidence and require a short adaptive cooldown before another
            # stagnation-triggered recovery can be requested.
            state["no_improvement_streak"] = 0
            state["failure_streak"] = 0
            state["success_streak"] = 0
            state["recovery_cooldown_remaining"] = _UNIFIED_RECOVERY_COOLDOWN_TRIALS
            state["stagnation_recoveries"] += 1
    promoted = bool(champion_gate and champion_gate.get("passed"))
    next_anchor = None

    if promoted:
        next_anchor = {
            "source": "adaptive_candidate" if adaptive else "startup_candidate",
            "candidate_id": int(candidate.get("candidate_id") or 0),
            "job_id": candidate.get("job_id"),
            "settings_hash": str(candidate.get("settings_hash") or ""),
            "settings": deepcopy(candidate.get("settings") or {}),
            "metrics": deepcopy(metrics),
        }
        state["champion_revision"] += 1
        state["last_champion_candidate_id"] = int(candidate.get("candidate_id") or 0)
        state["no_improvement_streak"] = 0

    if adaptive:
        state["adaptive_trials_completed"] += 1
        if state["recovery_cooldown_remaining"] > 0:
            state["recovery_cooldown_remaining"] -= 1
        if promoted:
            state["success_streak"] += 1
            state["failure_streak"] = 0
            state["trust_region_radius"] = min(
                _TRUST_REGION_MAX,
                float(state["trust_region_radius"]) * _TRUST_REGION_SUCCESS_EXPANSION,
            )
        else:
            state["success_streak"] = 0
            state["failure_streak"] += 1
            state["no_improvement_streak"] += 1
            if state["failure_streak"] >= _TRUST_REGION_FAILURE_TOLERANCE:
                state["trust_region_radius"] = max(
                    _TRUST_REGION_MIN,
                    float(state["trust_region_radius"]) * _TRUST_REGION_FAILURE_CONTRACTION,
                )
                state["failure_streak"] = 0

    return {
        "state": state,
        "champion_promoted": promoted,
        "probability_anchor": next_anchor,
    }


def _candidate_unit_pool(
    document: dict[str, Any],
    *,
    pool_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    search_space = list(document.get("search_space") or [])
    dimensions = len(search_space)
    rng = np.random.default_rng(seed)
    state = _probability_state(document)
    radius = float(state["trust_region_radius"])
    adaptive_trials = int(state["adaptive_trials_completed"])
    global_fraction = max(
        _GLOBAL_POOL_FRACTION_MIN,
        _GLOBAL_POOL_FRACTION_INITIAL - 0.01 * adaptive_trials,
    )
    anchor_fraction = _ANCHOR_LOCAL_FRACTION
    if global_fraction + anchor_fraction > 0.90:
        anchor_fraction = 0.90 - global_fraction
    top_fraction = 1.0 - global_fraction - anchor_fraction

    global_count = max(1, int(round(pool_size * global_fraction)))
    anchor_count = max(1, int(round(pool_size * anchor_fraction)))
    top_count = max(1, pool_size - global_count - anchor_count)

    global_points = qmc.LatinHypercube(d=dimensions, seed=seed + 17).random(n=global_count)
    anchor = document.get("probability_anchor") if isinstance(document.get("probability_anchor"), dict) else {}
    anchor_settings = anchor.get("settings") if isinstance(anchor.get("settings"), dict) else document.get("base_tuning_values") or document.get("base_model_values")
    anchor_vector = np.asarray(_normalized_vector(dict(anchor_settings or {}), search_space), dtype=float)
    anchor_points = np.clip(
        anchor_vector + rng.uniform(-radius, radius, size=(anchor_count, dimensions)),
        0.0,
        1.0,
    )

    centers = _top_observation_centers(document, search_space)
    center_indices = rng.integers(0, len(centers), size=top_count)
    top_radius = min(_TRUST_REGION_MAX, radius * 1.35)
    top_points = np.clip(
        centers[center_indices] + rng.uniform(-top_radius, top_radius, size=(top_count, dimensions)),
        0.0,
        1.0,
    )
    metadata = {
        "trust_region_radius": radius,
        "global_fraction": global_fraction,
        "anchor_fraction": anchor_fraction,
        "top_regions_fraction": top_fraction,
    }
    return np.vstack([global_points, anchor_points, top_points]), metadata



def _automatic_minimum_exploration_trials(document: dict[str, Any]) -> int:
    search_space = list(document.get("search_space") or [])
    dimensions = max(1, len(search_space))
    config = dict(document.get("probability_config") or {})
    configured = config.get("minimum_exploration_trials")
    if configured is None:
        configured = config.get("startup_trials")  # v3.10 backward-compatible override.
    if configured is not None:
        return max(4, min(24, int(configured)))
    # A 10D campaign now starts at 12 observations instead of six. The rule
    # scales with dimensionality and is only a floor; readiness checks below can
    # request additional space-filling observations when coverage is still weak.
    return max(4, min(24, dimensions + 2))


def _surrogate_readiness(document: dict[str, Any]) -> dict[str, Any]:
    search_space = list(document.get("search_space") or [])
    dimensions = max(1, len(search_space))
    minimum = _automatic_minimum_exploration_trials(document)
    observations = [item for item in _all_completed_observations(document) if not bool(item.get("is_control"))]
    count = len(observations)
    max_initial = max(minimum, min(24, dimensions + 6))

    if not observations:
        return {
            "ready": False, "observation_count": 0, "minimum_observations": minimum,
            "design_rank": 0, "required_rank": dimensions, "mean_dimension_span": 0.0,
            "broad_dimension_fraction": 0.0, "forced_after": max_initial,
            "reason": "insufficient_observations",
        }

    x = np.asarray([_normalized_vector(dict(item["settings"]), search_space) for item in observations], dtype=float)
    spans = np.ptp(x, axis=0) if len(x) > 1 else np.zeros(dimensions, dtype=float)
    mean_span = float(np.mean(spans)) if len(spans) else 0.0
    broad_fraction = float(np.mean(spans >= _UNIFIED_READINESS_DIMENSION_SPAN_MIN)) if len(spans) else 0.0
    centered = x - np.mean(x, axis=0, keepdims=True)
    design_rank = int(np.linalg.matrix_rank(centered, tol=1e-9)) if len(x) > 1 else 0
    required_rank = min(dimensions, max(1, count - 1))
    rank_ok = design_rank >= required_rank
    coverage_ok = (
        mean_span >= _UNIFIED_READINESS_MEAN_SPAN_MIN
        and broad_fraction >= _UNIFIED_READINESS_BROAD_DIMENSION_FRACTION
    )
    enough = count >= minimum
    forced = count >= max_initial
    ready = bool(forced or (enough and rank_ok and coverage_ok))
    if not enough:
        reason = "insufficient_observations"
    elif not rank_ok:
        reason = "insufficient_design_rank"
    elif not coverage_ok:
        reason = "insufficient_space_coverage"
    else:
        reason = "ready"
    if forced and reason != "ready":
        reason = "forced_ready_after_exploration_cap"

    return {
        "ready": ready,
        "observation_count": count,
        "minimum_observations": minimum,
        "design_rank": design_rank,
        "required_rank": required_rank,
        "mean_dimension_span": mean_span,
        "broad_dimension_fraction": broad_fraction,
        "forced_after": max_initial,
        "reason": reason,
    }


def _unified_observation_kind_counts(document: dict[str, Any]) -> dict[str, int]:
    observations = _all_completed_observations(document)
    counts = {"control": 0, "exploration": 0, "adaptive": 0, "other": 0}
    for item in observations:
        if bool(item.get("is_control")):
            counts["control"] += 1
            continue
        kind = str(item.get("kind") or "")
        if kind == "champion_probability":
            counts["adaptive"] += 1
        elif kind in {"unified_exploration", "probability_startup", "latin_hypercube", "prior_latin_hypercube_observation"}:
            counts["exploration"] += 1
        else:
            counts["other"] += 1
    return counts


def unified_caro_next_mode(document: dict[str, Any]) -> dict[str, Any]:
    """Choose the next research action without a fixed LHS→CARO boundary.

    Unified CARO starts with enough space-filling evidence for the surrogate to be
    usable, then continuously mixes adaptive proposals with deterministic
    space-filling recovery. The global exploration path therefore never disappears.
    """
    config = dict(document.get("probability_config") or {})
    counts = _unified_observation_kind_counts(document)
    readiness = _surrogate_readiness(document)
    minimum_exploration = int(readiness["minimum_observations"])
    initial_fraction = max(
        0.10,
        min(float(config.get("initial_exploration_fraction") or _UNIFIED_INITIAL_EXPLORATION_FRACTION), 0.90),
    )
    minimum_fraction = max(
        0.05,
        min(float(config.get("minimum_exploration_fraction") or _UNIFIED_MIN_EXPLORATION_FRACTION), initial_fraction),
    )
    stagnation_trials = max(
        2,
        min(int(config.get("stagnation_recovery_trials") or _UNIFIED_STAGNATION_RECOVERY_TRIALS), 12),
    )
    state = _probability_state(document)
    local_candidates = [item for item in document.get("candidates") or [] if not bool(item.get("is_control"))]
    generated_local = len(local_candidates)
    budget = max(1, int(document.get("candidate_count") or document.get("total_candidates") or 1))
    progress = max(0.0, min(1.0, generated_local / budget))
    target_fraction = initial_fraction + (minimum_fraction - initial_fraction) * progress
    observed_noncontrol = counts["exploration"] + counts["adaptive"] + counts["other"]
    actual_fraction = counts["exploration"] / max(1, observed_noncontrol)
    last_mode = str(state.get("last_proposal_mode") or "")

    if not bool(readiness.get("ready")):
        mode = "space_filling"
        reason = str(readiness.get("reason") or "surrogate_not_ready")
    elif (
        int(state.get("no_improvement_streak") or 0) >= stagnation_trials
        and int(state.get("recovery_cooldown_remaining") or 0) == 0
        and last_mode != "space_filling"
    ):
        mode = "space_filling"
        reason = "stagnation_recovery"
    elif actual_fraction + 1e-12 < target_fraction and last_mode != "space_filling":
        mode = "space_filling"
        reason = "dynamic_exploration_balance"
    else:
        mode = "adaptive_probability"
        reason = "surrogate_refinement"

    return {
        "mode": mode,
        "reason": reason,
        "minimum_exploration_trials": minimum_exploration,
        "exploration_observations": counts["exploration"],
        "adaptive_observations": counts["adaptive"],
        "other_observations": counts["other"],
        "target_exploration_fraction": float(target_fraction),
        "actual_exploration_fraction": float(actual_fraction),
        "progress": float(progress),
        "no_improvement_streak": int(state.get("no_improvement_streak") or 0),
        "recovery_cooldown_remaining": int(state.get("recovery_cooldown_remaining") or 0),
        "surrogate_readiness": readiness,
    }


def propose_unified_space_filling_candidate(document: dict[str, Any]) -> dict[str, Any]:
    search_space = list(document.get("search_space") or [])
    base_values = dict(document.get("base_tuning_values") or document.get("base_model_values") or {})
    if not search_space or not base_values:
        raise RuntimeError("Unified CARO requires a search space and baseline tuning values.")

    observations = list(document.get("prior_observations") or []) + list(document.get("candidates") or [])
    existing_ids = [int(item.get("candidate_id") or 0) for item in observations]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    seen_hashes = {str(item.get("settings_hash") or "") for item in observations if item.get("settings_hash")}
    completed = [
        item for item in observations
        if item.get("status") == "completed" and isinstance(item.get("settings"), dict)
    ]
    existing_vectors = np.asarray(
        [_normalized_vector(dict(item["settings"]), search_space) for item in completed],
        dtype=float,
    ) if completed else np.empty((0, len(search_space)), dtype=float)

    config = dict(document.get("probability_config") or {})
    pool_size = max(256, min(int(config.get("space_filling_pool_size") or _UNIFIED_SPACE_FILLING_POOL_SIZE), 8192))
    seed = int(document.get("seed") or 42)
    points = qmc.LatinHypercube(d=len(search_space), seed=seed + next_id * 3571).random(n=pool_size)
    proposals: list[tuple[float, dict[str, Any], str]] = []
    dimension_scale = max(1.0, float(np.sqrt(len(search_space))))
    for point in points:
        values = _settings_from_unit_point(base_values, search_space, point)
        fingerprint = _settings_hash(values)
        if fingerprint in seen_hashes:
            continue
        vector = np.asarray(_normalized_vector(values, search_space), dtype=float)
        if len(existing_vectors):
            minimum_distance = float(np.linalg.norm(existing_vectors - vector, axis=1).min() / dimension_scale)
        else:
            minimum_distance = 1.0
        proposals.append((minimum_distance, values, fingerprint))

    if not proposals:
        raise RuntimeError("Unable to generate a unique Unified CARO space-filling proposal.")
    proposals.sort(key=lambda item: (item[0], item[2]), reverse=True)
    distance, settings, fingerprint = proposals[0]
    policy = unified_caro_next_mode(document)
    return {
        "candidate_id": next_id,
        "kind": "unified_exploration",
        "is_control": False,
        "settings": settings,
        "settings_hash": fingerprint,
        "status": "pending",
        "proposal": {
            "proposal_mode": "space_filling",
            "selection_reason": policy["reason"],
            "observation_count": len(completed),
            "candidate_pool_size": len(proposals),
            "maximin_normalized_distance": float(distance),
            "target_exploration_fraction": policy["target_exploration_fraction"],
            "actual_exploration_fraction": policy["actual_exploration_fraction"],
            "minimum_exploration_trials": policy["minimum_exploration_trials"],
            "surrogate_readiness": deepcopy(policy.get("surrogate_readiness") or {}),
            "interpretation": "internal_latin_hypercube_space_filling_not_outcome_prediction",
        },
    }


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
    
    
    
    feasibility_weight = 0.25 + 0.75 * probability
    acquisition = (
        expected_improvement
        + exploration_weight * normalized_uncertainty * feasibility_weight
        + 0.02 * probability
    )
    return probability, expected_improvement, acquisition


def propose_champion_probability_candidate(document: dict[str, Any]) -> dict[str, Any]:
    




    search_space = list(document.get("search_space") or [])
    base_values = dict(document.get("base_tuning_values") or document.get("base_model_values") or {})
    if not search_space or not base_values:
        raise RuntimeError("Probabilistic tuning requires a search space and baseline tuning values.")

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
    raw_points, pool_metadata = _candidate_unit_pool(document, pool_size=pool_size, seed=seed + next_id * 7919)
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
            "proposal_mode": "adaptive_probability",
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
                "global_fraction": float(pool_metadata["global_fraction"]),
                "anchor_local_fraction": float(pool_metadata["anchor_fraction"]),
                "top_regions_fraction": float(pool_metadata["top_regions_fraction"]),
            },
            "trust_region_radius": float(pool_metadata["trust_region_radius"]),
            "champion_revision": int(_probability_state(document)["champion_revision"]),
            "champion_candidate_id": (document.get("probability_anchor") or {}).get("candidate_id"),
            "promising_region": promising_region,
            "promising_region_probability_mean": float(probability[top_indices].mean()),
            "promising_region_expected_improvement_mean": float(constrained_expected_improvement[top_indices].mean()),
            "thresholds": thresholds,
            "interpretation": "research_surrogate_probability_not_future_profit_probability",
        },
    }
