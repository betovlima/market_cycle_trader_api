from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable

from scipy.stats import qmc

from ..model_tuning_space import settings_from_unit_point
from .search_space import TEMPORAL_POLICY_SEARCH_SPACE, normalize_settings


def settings_hash(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_latin_hypercube(
    base_settings: dict[str, Any],
    *,
    candidate_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if candidate_count < 4:
        raise ValueError("Latin Hypercube requires at least four candidate trials.")
    active_space = [dict(item) for item in TEMPORAL_POLICY_SEARCH_SPACE]
    control = normalize_settings(base_settings)
    candidates: list[dict[str, Any]] = [{
        "candidate_id": 0,
        "kind": "control",
        "is_control": True,
        "settings": control,
        "settings_hash": settings_hash(control),
        "status": "pending",
    }]
    seen = {candidates[0]["settings_hash"]}

    def add_points(points: Any) -> None:
        for point in points:
            if len(candidates) >= candidate_count + 1:
                return
            values = normalize_settings(settings_from_unit_point(control, active_space, point))
            fingerprint = settings_hash(values)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append({
                "candidate_id": len(candidates),
                "kind": "latin_hypercube",
                "is_control": False,
                "settings": values,
                "settings_hash": fingerprint,
                "status": "pending",
            })

    add_points(qmc.LatinHypercube(d=len(active_space), seed=seed).random(n=candidate_count))
    attempt = 1
    while len(candidates) < candidate_count + 1 and attempt <= 12:
        missing = candidate_count + 1 - len(candidates)
        add_points(qmc.LatinHypercube(d=len(active_space), seed=seed + attempt).random(n=max(8, missing * 2)))
        attempt += 1
    if len(candidates) != candidate_count + 1:
        raise RuntimeError("Unable to generate the requested number of unique Temporal policy candidates.")
    return candidates


def candidate_summary(candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metrics = deepcopy(result.get("metrics") or {})
    return {
        "candidate_id": int(candidate.get("candidate_id") or 0),
        "kind": str(candidate.get("kind") or "latin_hypercube"),
        "is_control": bool(candidate.get("is_control")),
        "settings": deepcopy(candidate.get("settings") or {}),
        "settings_hash": str(candidate.get("settings_hash") or ""),
        "status": "completed",
        "metrics": metrics,
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    eligible = 1.0 if bool(metrics.get("eligible")) else 0.0
    utility = metrics.get("search_utility")
    return (
        eligible,
        float(utility) if utility is not None else float("-inf"),
        float(metrics.get("ending_capital") or 0.0),
        float(metrics.get("sharpe") or 0.0),
    )


def evaluate_latin_hypercube(
    base_settings: dict[str, Any],
    *,
    candidate_count: int,
    seed: int,
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    candidates = generate_latin_hypercube(base_settings, candidate_count=candidate_count, seed=seed)
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        result = evaluate(dict(candidate["settings"]))
        evaluated.append(candidate_summary(candidate, result))
    ranked = sorted(evaluated, key=candidate_sort_key, reverse=True)
    champion = ranked[0]
    return {
        "candidate_count": int(candidate_count),
        "evaluated_count": len(evaluated),
        "seed": int(seed),
        "candidates": evaluated,
        "champion": deepcopy(champion),
    }
