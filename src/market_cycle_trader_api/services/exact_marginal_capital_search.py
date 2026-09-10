from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any, Iterable

DEFAULT_PRESELECTOR_CUTOFFS = (5, 10, 20, 50, 100)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def annotate_preselector_ranks(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach descending preselector ranks without filtering any candidate.

    Ranking is diagnostic only. Candidates without a finite raw score remain in the
    returned collection with ``preselector_rank=None`` so the benchmark can account
    for every attempted asset instead of silently dropping difficult cases.
    """
    result = [dict(row) for row in rows]
    ranked_indexes = [
        index
        for index, row in enumerate(result)
        if finite_float(row.get("preselector_raw_score")) is not None
    ]
    ranked_indexes.sort(
        key=lambda index: (
            -float(result[index]["preselector_raw_score"]),
            str(result[index].get("symbol") or ""),
        )
    )
    count = len(ranked_indexes)
    for row in result:
        row["preselector_rank"] = None
        row["preselector_rank_score"] = None
    for rank, index in enumerate(ranked_indexes, start=1):
        result[index]["preselector_rank"] = rank
        result[index]["preselector_rank_score"] = (
            1.0 if count <= 1 else float(1.0 - ((rank - 1) / (count - 1)))
        )
    return result


def economic_outcome(row: dict[str, Any]) -> str:
    status = str(row.get("evaluation_status") or "").strip().lower()
    if status != "completed":
        return status or "unknown"
    delta = finite_float(row.get("ending_capital_delta_rate"))
    if delta is None:
        return "invalid_delta"
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "negative"
    return "neutral"


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def build_preselector_recall(
    rows: Iterable[dict[str, Any]],
    cutoffs: Iterable[int] = DEFAULT_PRESELECTOR_CUTOFFS,
) -> list[dict[str, Any]]:
    """Measure how many exact positive candidates are recovered by each rank cutoff."""
    values = [dict(row) for row in rows]
    completed = [
        row
        for row in values
        if str(row.get("evaluation_status") or "").strip().lower() == "completed"
        and finite_float(row.get("ending_capital_delta_rate")) is not None
    ]
    positives = [row for row in completed if float(row["ending_capital_delta_rate"]) > 0.0]
    total_positive = len(positives)
    rankable_completed = [
        row
        for row in completed
        if isinstance(row.get("preselector_rank"), int)
        and int(row["preselector_rank"]) > 0
    ]
    rankable_count = len(rankable_completed)
    unranked_positive = sum(
        1 for row in positives if not isinstance(row.get("preselector_rank"), int)
    )

    normalized_cutoffs = sorted({max(1, int(value)) for value in cutoffs})
    if rankable_count and rankable_count not in normalized_cutoffs:
        normalized_cutoffs.append(rankable_count)
        normalized_cutoffs.sort()

    result: list[dict[str, Any]] = []
    for requested in normalized_cutoffs:
        cutoff = min(requested, rankable_count) if rankable_count else 0
        top = [
            row
            for row in rankable_completed
            if int(row.get("preselector_rank") or 0) <= cutoff
        ]
        top_positive = [row for row in top if float(row["ending_capital_delta_rate"]) > 0.0]
        best_delta = max(
            (float(row["ending_capital_delta_rate"]) for row in top),
            default=None,
        )
        result.append(
            {
                "requested_cutoff": requested,
                "effective_cutoff": cutoff,
                "completed_candidates": len(completed),
                "rankable_completed_candidates": rankable_count,
                "total_exact_positive_candidates": total_positive,
                "unranked_exact_positive_candidates": unranked_positive,
                "evaluations_in_top_k": len(top),
                "positive_candidates_in_top_k": len(top_positive),
                "positive_recall": (
                    float(len(top_positive) / total_positive) if total_positive else None
                ),
                "positive_precision": (
                    float(len(top_positive) / len(top)) if top else None
                ),
                "evaluation_fraction_of_completed": (
                    float(len(top) / len(completed)) if completed else None
                ),
                "best_ending_capital_delta_rate_in_top_k": best_delta,
            }
        )
    return result


def summarize_exact_search(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    outcomes: dict[str, int] = {}
    exact_seconds: list[float] = []
    deltas: list[float] = []
    for row in values:
        outcome = economic_outcome(row)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        seconds = finite_float(row.get("exact_replay_seconds"))
        if seconds is not None and seconds >= 0.0:
            exact_seconds.append(seconds)
        delta = finite_float(row.get("ending_capital_delta_rate"))
        if str(row.get("evaluation_status") or "").strip().lower() == "completed" and delta is not None:
            deltas.append(delta)

    positives = [value for value in deltas if value > 0.0]
    return {
        "candidate_count": len(values),
        "outcome_counts": outcomes,
        "completed_count": len(deltas),
        "positive_count": len(positives),
        "positive_rate": float(len(positives) / len(deltas)) if deltas else None,
        "best_ending_capital_delta_rate": max(deltas) if deltas else None,
        "median_ending_capital_delta_rate": float(median(deltas)) if deltas else None,
        "total_exact_replay_seconds": float(sum(exact_seconds)),
        "median_exact_replay_seconds": float(median(exact_seconds)) if exact_seconds else None,
        "p90_exact_replay_seconds": _percentile(exact_seconds, 0.90),
        "p95_exact_replay_seconds": _percentile(exact_seconds, 0.95),
        "max_exact_replay_seconds": max(exact_seconds) if exact_seconds else None,
    }
