from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

TOLERANCE = 1e-12
_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def display_date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def humanize_dates(text: str) -> str:
    return _DATE_PATTERN.sub(lambda m: f"{m.group(3)}/{m.group(2)}/{m.group(1)}", str(text))


def console_log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S")
    print(f"[{stamp}] {humanize_dates(str(message))}", flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)) or float(seconds) < 0:
        return "?"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def cross_universe_rank_mean(dataset: pd.DataFrame, universe_count: int) -> float | None:
    correlations: list[float] = []
    for _, group in dataset.groupby("decision_date", sort=True):
        if group["universe_name"].nunique() != universe_count:
            continue
        pivot = group.pivot_table(
            index="candidate",
            columns="universe_name",
            values="action_advantage_log",
            aggfunc="first",
        ).dropna()
        if len(pivot) < 3 or len(pivot.columns) != universe_count:
            continue
        value = pivot.iloc[:, 0].rank().corr(pivot.iloc[:, 1].rank())
        if pd.notna(value):
            correlations.append(float(value))
    return float(np.mean(correlations)) if correlations else None


def partial_analysis(dataset: pd.DataFrame, universe_count: int) -> dict[str, Any]:
    if dataset.empty:
        return {
            "rows": 0,
            "positive": 0,
            "negative": 0,
            "zero": 0,
            "mean_action_advantage_log": None,
            "median_action_advantage_log": None,
            "completed_contexts": 0,
            "intervention_preferred_contexts": 0,
            "normal_policy_preferred_contexts": 0,
            "oracle_mean_context_advantage_log": None,
            "candidate_mean_leader": None,
            "candidate_mean_leader_value": None,
            "cross_universe_rank_spearman_mean": None,
        }

    values = pd.to_numeric(dataset["action_advantage_log"], errors="coerce")
    positive = int((values > TOLERANCE).sum())
    negative = int((values < -TOLERANCE).sum())
    zero = int(len(values) - positive - negative)
    context_best = dataset.groupby(
        ["decision_date", "universe_name"], sort=True
    )["action_advantage_log"].max().astype(float)
    intervention = int((context_best > TOLERANCE).sum())
    normal = int((context_best <= TOLERANCE).sum())
    oracle = context_best.clip(lower=0.0)
    candidate_means = dataset.groupby("candidate", sort=True)["action_advantage_log"].mean().sort_values(ascending=False)
    leader = str(candidate_means.index[0]) if len(candidate_means) else None
    leader_value = float(candidate_means.iloc[0]) if len(candidate_means) else None

    return {
        "rows": int(len(dataset)),
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "mean_action_advantage_log": float(values.mean()),
        "median_action_advantage_log": float(values.median()),
        "completed_contexts": int(len(context_best)),
        "intervention_preferred_contexts": intervention,
        "normal_policy_preferred_contexts": normal,
        "oracle_mean_context_advantage_log": float(oracle.mean()) if len(oracle) else None,
        "candidate_mean_leader": leader,
        "candidate_mean_leader_value": leader_value,
        "cross_universe_rank_spearman_mean": cross_universe_rank_mean(dataset, universe_count),
    }


def readiness(dataset: pd.DataFrame, candidate_count: int, universe_count: int) -> dict[str, Any]:
    stats = partial_analysis(dataset, universe_count)
    nonzero = dataset.loc[dataset["action_advantage_log"].abs() > TOLERANCE].copy()
    years = int(pd.to_datetime(dataset["decision_date"], errors="coerce").dt.year.nunique())
    checks = {
        "rows": int(len(dataset)) >= 280,
        "temporal_states": int(dataset["decision_date"].nunique()) >= 20,
        "candidate_diversity": int(nonzero["candidate"].nunique()) >= candidate_count,
        "year_diversity": years >= 7,
        "both_signs": stats["positive"] > 0 and stats["negative"] > 0,
        "intervention_and_abstention": (
            stats["intervention_preferred_contexts"] >= 3
            and stats["normal_policy_preferred_contexts"] >= 3
        ),
    }
    return {
        **checks,
        "ready": bool(all(checks.values())),
        "rows_total": int(len(dataset)),
        "nonzero_action_advantage_rows": int(len(nonzero)),
        "positive_action_advantage_rows": int(stats["positive"]),
        "negative_action_advantage_rows": int(stats["negative"]),
        "zero_action_advantage_rows": int(stats["zero"]),
        "temporal_states_total": int(dataset["decision_date"].nunique()),
        "years_total": years,
        "nonzero_candidates": int(nonzero["candidate"].nunique()),
        "intervention_preferred_contexts": int(stats["intervention_preferred_contexts"]),
        "normal_policy_preferred_contexts": int(stats["normal_policy_preferred_contexts"]),
    }
