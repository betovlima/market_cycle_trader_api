from __future__ import annotations

from copy import deepcopy
from typing import Any


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    left = candidate.get(field)
    right = baseline.get(field)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def controlled_comparison(validation: dict[str, Any], winner_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    baseline = validation.get("baseline_aggregate") if isinstance(validation.get("baseline_aggregate"), dict) else {}
    candidate = validation.get("candidate_aggregate") if isinstance(validation.get("candidate_aggregate"), dict) else {}
    supported = bool(validation.get("supported"))
    reasons: list[str] = []
    criteria = validation.get("criteria") if isinstance(validation.get("criteria"), dict) else {}
    if not criteria.get("ending_capital_improved"):
        reasons.append("ending_capital_not_improved")
    if not criteria.get("sharpe_preserved"):
        reasons.append("sharpe_not_preserved")
    if not criteria.get("maximum_drawdown_preserved"):
        reasons.append("maximum_drawdown_not_preserved")
    if not criteria.get("positive_outer_folds"):
        reasons.append("non_positive_outer_fold")
    if int(criteria.get("folds_improved") or 0) < int(criteria.get("minimum_folds_improved") or 0):
        reasons.append("insufficient_outer_fold_improvement")
    return {
        "outcome": "supported" if supported else "rejected",
        "decision": "candidate_search_procedure_supported" if supported else "no_robust_candidate",
        "baseline": deepcopy(baseline),
        "candidate": deepcopy(candidate),
        "winner_reference": deepcopy(winner_reference or {}),
        "deltas": {
            "ending_capital": _delta(candidate, baseline, "ending_capital"),
            "strategy_return": _delta(candidate, baseline, "strategy_return"),
            "cagr": _delta(candidate, baseline, "cagr"),
            "sharpe": _delta(candidate, baseline, "sharpe"),
            "maximum_drawdown": _delta(candidate, baseline, "maximum_drawdown"),
            "capital_rotations": _delta(candidate, baseline, "capital_rotations"),
            "cash_days": _delta(candidate, baseline, "cash_days"),
        },
        "criteria": deepcopy(criteria),
        "rejection_reasons": reasons,
        "interpretation": "Nested outer-fold evidence is the acceptance criterion; full-period metrics are not used to select the fold-specific candidates.",
    }
