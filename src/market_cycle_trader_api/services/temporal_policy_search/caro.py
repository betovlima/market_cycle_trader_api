from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from ..model_tuning_probability import (
    champion_gate_evaluation,
    evolve_probability_search,
    initial_probability_state,
    propose_champion_probability_candidate,
)
from .sampling import candidate_sort_key
from .search_space import TEMPORAL_POLICY_SEARCH_SPACE


def _anchor_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "latin_hypercube_champion" if candidate.get("kind") == "latin_hypercube" else "control",
        "candidate_id": int(candidate.get("candidate_id") or 0),
        "job_id": None,
        "settings_hash": str(candidate.get("settings_hash") or ""),
        "settings": deepcopy(candidate.get("settings") or {}),
        "metrics": deepcopy(candidate.get("metrics") or {}),
    }


def run_caro_refinement(
    sampling_result: dict[str, Any],
    *,
    base_settings: dict[str, Any],
    trial_count: int,
    seed: int,
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if trial_count < 1:
        raise ValueError("CARO requires at least one adaptive trial.")
    lhs_candidates = [deepcopy(item) for item in sampling_result.get("candidates") or []]
    if len(lhs_candidates) < 4:
        raise ValueError("CARO requires completed Latin Hypercube observations first.")
    initial_champion = max(lhs_candidates, key=candidate_sort_key)
    document: dict[str, Any] = {
        "seed": int(seed),
        "search_space": [dict(item) for item in TEMPORAL_POLICY_SEARCH_SPACE],
        "base_tuning_values": deepcopy(base_settings),
        "baseline_execution": {"metrics": deepcopy(next((item.get("metrics") for item in lhs_candidates if item.get("is_control")), initial_champion.get("metrics") or {}))},
        "candidates": lhs_candidates,
        "prior_observations": [],
        "probability_anchor": _anchor_from_candidate(initial_champion),
        "probability_state": initial_probability_state(lhs_candidates),
        "probability_config": {
            "min_capital_improvement": 0.0,
            "sharpe_tolerance": 0.05,
            "drawdown_tolerance": 0.03,
            "min_worst_fold_return": 0.0,
            "candidate_pool_size": 1024,
            "exploration_weight": 0.15,
            "minimum_exploration_trials": min(24, max(4, len(TEMPORAL_POLICY_SEARCH_SPACE) + 2)),
        },
    }
    trials: list[dict[str, Any]] = []
    for _ in range(int(trial_count)):
        proposal = propose_champion_probability_candidate(document)
        result = evaluate(dict(proposal["settings"]))
        proposal["status"] = "completed"
        proposal["metrics"] = deepcopy(result.get("metrics") or {})
        gate = champion_gate_evaluation(document, proposal["metrics"])
        proposal["champion_gate_passed"] = bool(gate.get("passed"))
        proposal["champion_gate"] = deepcopy(gate)
        evolution = evolve_probability_search(document, proposal, proposal["metrics"], gate)
        document["candidates"].append(deepcopy(proposal))
        document["probability_state"] = deepcopy(evolution.get("state") or document.get("probability_state") or {})
        if evolution.get("probability_anchor"):
            document["probability_anchor"] = deepcopy(evolution["probability_anchor"])
        trials.append(deepcopy(proposal))

    all_candidates = list(document.get("candidates") or [])
    champion = max(all_candidates, key=candidate_sort_key)
    adaptive_champion = document.get("probability_anchor") if isinstance(document.get("probability_anchor"), dict) else _anchor_from_candidate(champion)
    last_proposal = trials[-1].get("proposal") if trials and isinstance(trials[-1].get("proposal"), dict) else None
    return {
        "trial_count": int(trial_count),
        "completed_count": len(trials),
        "trials": trials,
        "champion": deepcopy(champion),
        "adaptive_anchor": deepcopy(adaptive_champion),
        "probability_state": deepcopy(document.get("probability_state") or {}),
        "promising_region": deepcopy((last_proposal or {}).get("promising_region") or {}),
        "last_proposal": deepcopy(last_proposal),
    }
