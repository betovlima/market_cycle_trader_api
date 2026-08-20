from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .policy import metrics_from_interval_returns


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], name: str) -> float | None:
    candidate_value = candidate.get(name)
    baseline_value = baseline.get(name)
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def nested_validation(
    outer_fold_payloads: list[dict[str, Any]],
    *,
    initial_capital: float,
    winner_fold_returns: dict[int, float],
    evaluate_outer: Callable[[int, dict[str, Any]], dict[str, Any]],
    evaluate_baseline_outer: Callable[[int], dict[str, Any]],
) -> dict[str, Any]:
    outer_results: list[dict[str, Any]] = []
    candidate_intervals: list[dict[str, Any]] = []
    baseline_intervals: list[dict[str, Any]] = []

    for fold_payload in outer_fold_payloads:
        outer_fold_id = int(fold_payload["outer_fold_id"])
        champion = fold_payload.get("champion") if isinstance(fold_payload.get("champion"), dict) else {}
        settings = dict(champion.get("settings") or {})
        candidate = evaluate_outer(outer_fold_id, settings)
        baseline = evaluate_baseline_outer(outer_fold_id)
        candidate_intervals.extend(deepcopy(candidate.get("intervals") or []))
        baseline_intervals.extend(deepcopy(baseline.get("intervals") or []))
        candidate_metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
        outer_results.append({
            "outer_fold_id": outer_fold_id,
            "inner_fold_ids": list(fold_payload.get("inner_fold_ids") or []),
            "selected_candidate_id": champion.get("candidate_id"),
            "selected_candidate_kind": champion.get("kind"),
            "settings": deepcopy(settings),
            "inner_metrics": deepcopy(champion.get("metrics") or {}),
            "baseline_outer_metrics": deepcopy(baseline_metrics),
            "candidate_outer_metrics": deepcopy(candidate_metrics),
            "return_delta": _metric_delta(candidate_metrics, baseline_metrics, "strategy_return"),
            "sharpe_delta": _metric_delta(candidate_metrics, baseline_metrics, "sharpe"),
            "maximum_drawdown_delta": _metric_delta(candidate_metrics, baseline_metrics, "maximum_drawdown"),
            "late_exit_cash_guard_count": int(candidate_metrics.get("late_exit_cash_guard_count") or 0),
        })

    baseline_aggregate = metrics_from_interval_returns(
        baseline_intervals,
        initial_capital=initial_capital,
        winner_fold_returns=winner_fold_returns,
    )
    candidate_aggregate = metrics_from_interval_returns(
        candidate_intervals,
        initial_capital=initial_capital,
        winner_fold_returns=winner_fold_returns,
    )
    folds_improved = sum(float(item.get("return_delta") or 0.0) > 0.0 for item in outer_results)
    outer_count = len(outer_results)
    criteria = {
        "ending_capital_improved": float(candidate_aggregate.get("ending_capital") or 0.0) > float(baseline_aggregate.get("ending_capital") or 0.0),
        "sharpe_preserved": float(candidate_aggregate.get("sharpe") or 0.0) >= float(baseline_aggregate.get("sharpe") or 0.0) - 0.05,
        "maximum_drawdown_preserved": float(candidate_aggregate.get("maximum_drawdown") or -1.0) >= float(baseline_aggregate.get("maximum_drawdown") or -1.0) - 0.03,
        "positive_outer_folds": all(float((item.get("candidate_outer_metrics") or {}).get("strategy_return") or 0.0) > 0.0 for item in outer_results),
        "folds_improved": int(folds_improved),
        "minimum_folds_improved": max(1, (2 * outer_count + 2) // 3),
    }
    supported = bool(
        outer_count >= 2
        and criteria["ending_capital_improved"]
        and criteria["sharpe_preserved"]
        and criteria["maximum_drawdown_preserved"]
        and criteria["positive_outer_folds"]
        and criteria["folds_improved"] >= criteria["minimum_folds_improved"]
    )
    return {
        "method": "nested_leave_one_temporal_fold_out",
        "outer_fold_count": outer_count,
        "outer_results": outer_results,
        "baseline_aggregate": baseline_aggregate,
        "candidate_aggregate": candidate_aggregate,
        "criteria": criteria,
        "supported": supported,
    }
