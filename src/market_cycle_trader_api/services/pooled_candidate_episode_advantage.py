from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .pooled_candidate_marginal_advantage import (
    PooledCandidateMarginalModel,
    fit_pooled_candidate_marginal_model,
    score_candidate_samples,
)

TARGET_COLUMN = "marginal_episode_net_log_return"


def _holding_days(selected: pd.Series) -> pd.Series:
    values = selected.astype(str).tolist()
    result: list[int] = []
    previous: str | None = None
    holding = 0
    for value in values:
        if value == "CASH":
            holding = 0
        elif value == previous:
            holding += 1
        else:
            holding = 1
        result.append(int(holding))
        previous = value
    return pd.Series(result, index=selected.index, dtype=int)


def extract_candidate_divergence_episodes(
    *,
    baseline_daily: pd.DataFrame,
    expanded_daily: pd.DataFrame,
    candidate: str,
    fold_id: int,
) -> pd.DataFrame:
    required = {"timestamp", "selected_asset", "net_log_return"}
    for label, frame in (("baseline", baseline_daily), ("expanded", expanded_daily)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(
                f"{label} replay is missing required columns: " + ", ".join(missing)
            )

    baseline = baseline_daily.reset_index(drop=True).copy()
    expanded = expanded_daily.reset_index(drop=True).copy()
    if len(baseline) != len(expanded):
        raise RuntimeError("Baseline and expanded replay lengths differ.")

    baseline["timestamp"] = pd.to_datetime(
        baseline["timestamp"], utc=True, format="mixed", errors="raise"
    )
    expanded["timestamp"] = pd.to_datetime(
        expanded["timestamp"], utc=True, format="mixed", errors="raise"
    )
    if not baseline["timestamp"].equals(expanded["timestamp"]):
        raise RuntimeError("Baseline and expanded replay calendars differ.")

    baseline_asset = baseline["selected_asset"].astype(str).str.upper()
    expanded_asset = expanded["selected_asset"].astype(str).str.upper()
    baseline_holding = _holding_days(baseline_asset)
    expanded_holding = _holding_days(expanded_asset)

    baseline_log = pd.to_numeric(baseline["net_log_return"], errors="coerce")
    expanded_log = pd.to_numeric(expanded["net_log_return"], errors="coerce")
    if baseline_log.isna().any() or expanded_log.isna().any():
        raise RuntimeError("Replay contains non-finite daily log returns.")

    delta = expanded_log - baseline_log
    # Policy-state reconvergence is defined by the observable state that controls
    # future decisions: selected asset plus holding-days state. The account-level
    # log-return delta on the reconvergence session is still included in the target.
    # We deliberately do not impose an economic tolerance/gate here.
    state_equal = (
        baseline_asset.eq(expanded_asset)
        & baseline_holding.eq(expanded_holding)
    )
    direct_candidate_divergence = (
        expanded_asset.eq(str(candidate).upper())
        & ~baseline_asset.eq(expanded_asset)
    )

    rows: list[dict[str, Any]] = []
    i = 0
    episode_id = 0
    while i < len(baseline):
        if not bool(direct_candidate_divergence.iloc[i]):
            i += 1
            continue

        start = i
        j = start + 1
        while j < len(baseline) and not bool(state_equal.iloc[j]):
            j += 1

        right_censored = j >= len(baseline)
        end = len(baseline) - 1 if right_censored else j
        window = slice(start, end + 1)
        episode_id += 1

        rows.append(
            {
                "fold": int(fold_id),
                "candidate": str(candidate).upper(),
                "episode_id": int(episode_id),
                "episode_start": baseline["timestamp"].iloc[start],
                "episode_end": baseline["timestamp"].iloc[end],
                "episode_sessions": int(end - start + 1),
                "right_censored": bool(right_censored),
                "baseline_asset_at_start": baseline_asset.iloc[start],
                "expanded_asset_at_start": expanded_asset.iloc[start],
                "baseline_holding_days_at_start": int(baseline_holding.iloc[start]),
                "expanded_holding_days_at_start": int(expanded_holding.iloc[start]),
                TARGET_COLUMN: float(delta.iloc[window].sum()),
                "positive_episode": bool(float(delta.iloc[window].sum()) > 0.0),
                "changed_position_sessions": int(
                    (~baseline_asset.iloc[window].eq(expanded_asset.iloc[window])).sum()
                ),
            }
        )

        i = end + 1

    return pd.DataFrame(rows)


def build_episode_samples(
    *,
    daily_features: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    features = daily_features.copy()
    features["timestamp"] = pd.to_datetime(
        features["timestamp"], utc=True, format="mixed", errors="raise"
    )
    ep = episodes.loc[~episodes["right_censored"].astype(bool)].copy()
    if ep.empty:
        return pd.DataFrame()
    ep["episode_start"] = pd.to_datetime(
        ep["episode_start"], utc=True, format="mixed", errors="raise"
    )

    merged = ep.merge(
        features,
        left_on=["fold", "episode_start", "candidate"],
        right_on=["fold", "timestamp", "candidate"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_daily"),
    )
    if len(merged) != len(ep):
        missing = ep.merge(
            features[["fold", "timestamp", "candidate"]],
            left_on=["fold", "episode_start", "candidate"],
            right_on=["fold", "timestamp", "candidate"],
            how="left",
            indicator=True,
        )
        missing = missing.loc[
            missing["_merge"] != "both",
            ["fold", "episode_start", "candidate"],
        ]
        raise RuntimeError(
            "Episode starts are missing decision-time features: "
            + missing.head(10).to_dict(orient="records").__repr__()
        )

    drop_columns = [
        "marginal_next_session_net_log_return",
        "baseline_next_session_net_log_return",
        "candidate_next_session_net_log_return",
    ]
    merged = merged.drop(columns=[c for c in drop_columns if c in merged.columns])
    return merged.sort_values(["fold", "episode_start", "candidate"]).reset_index(drop=True)


def fit_episode_model(
    samples: pd.DataFrame,
    feature_names: Iterable[str],
) -> PooledCandidateMarginalModel:
    return fit_pooled_candidate_marginal_model(
        samples,
        feature_names,
        target_column=TARGET_COLUMN,
    )


def score_episode_samples(
    model: PooledCandidateMarginalModel,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    return score_candidate_samples(
        model,
        samples,
        target_column=TARGET_COLUMN,
    )


def choose_non_overlapping_episode_overrides(
    scored_samples: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "fold",
        "candidate",
        "episode_start",
        "episode_end",
        TARGET_COLUMN,
        "predicted_marginal_advantage",
        "predicted_marginal_std",
    }
    missing = sorted(required.difference(scored_samples.columns))
    if missing:
        raise RuntimeError(
            "Scored episode samples are missing required columns: "
            + ", ".join(missing)
        )

    frame = scored_samples.copy()
    frame["episode_start"] = pd.to_datetime(
        frame["episode_start"], utc=True, format="mixed", errors="raise"
    )
    frame["episode_end"] = pd.to_datetime(
        frame["episode_end"], utc=True, format="mixed", errors="raise"
    )

    rows: list[dict[str, Any]] = []
    blocked_until: pd.Timestamp | None = None

    for start, group in frame.groupby("episode_start", sort=True):
        start = pd.Timestamp(start)
        if blocked_until is not None and start <= blocked_until:
            continue

        ordered = group.sort_values(
            ["predicted_marginal_advantage", "candidate"],
            ascending=[False, True],
        )
        best = ordered.iloc[0]
        predicted = float(best["predicted_marginal_advantage"])
        override = bool(np.isfinite(predicted) and predicted > 0.0)

        if override:
            chosen_candidate = str(best["candidate"])
            realized = float(best[TARGET_COLUMN])
            end = pd.Timestamp(best["episode_end"])
            blocked_until = end
        else:
            chosen_candidate = "BASELINE"
            realized = 0.0
            end = start

        rows.append(
            {
                "fold": int(best["fold"]),
                "episode_start": start,
                "episode_end": end,
                "chosen_candidate": chosen_candidate,
                "override_baseline": override,
                "predicted_marginal_advantage": predicted,
                "predicted_marginal_std": float(best["predicted_marginal_std"]),
                "realized_marginal_episode_net_log_return": realized,
                "candidate_count_at_start": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values(["fold", "episode_start"]).reset_index(drop=True)


def summarize_episode_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "decision_episode_starts": 0,
            "override_count": 0,
            "override_rate": 0.0,
            "realized_marginal_log_sum": 0.0,
            "realized_marginal_log_mean_when_overridden": None,
            "positive_realized_override_rate": None,
        }

    override = decisions["override_baseline"].astype(bool)
    realized = pd.to_numeric(
        decisions["realized_marginal_episode_net_log_return"], errors="coerce"
    ).fillna(0.0)
    chosen = realized.loc[override]
    return {
        "decision_episode_starts": int(len(decisions)),
        "override_count": int(override.sum()),
        "override_rate": float(override.mean()),
        "realized_marginal_log_sum": float(realized.sum()),
        "realized_marginal_log_mean_when_overridden": (
            float(chosen.mean()) if not chosen.empty else None
        ),
        "positive_realized_override_rate": (
            float((chosen > 0.0).mean()) if not chosen.empty else None
        ),
    }
