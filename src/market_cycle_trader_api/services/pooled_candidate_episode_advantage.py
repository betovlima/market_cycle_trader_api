from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .pooled_candidate_marginal_advantage import (
    PooledCandidateMarginalModel,
    fit_pooled_candidate_marginal_model,
)

TARGET_COLUMN = "marginal_episode_net_log_return"
EPISODE_DEFINITION_VERSION = "pcea-2.0.4-policy-sufficient-holding-state"


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


def _replay_min_holding_days(
    baseline_daily: pd.DataFrame,
    expanded_daily: pd.DataFrame,
) -> int:
    baseline_value = baseline_daily.attrs.get("rotation_min_holding_days")
    expanded_value = expanded_daily.attrs.get("rotation_min_holding_days")
    if baseline_value is None or expanded_value is None:
        raise RuntimeError(
            "PCEA replay is missing rotation_min_holding_days metadata. "
            "Recompute from the current cached-score replay; do not reuse older artifacts."
        )
    baseline_min = int(baseline_value)
    expanded_min = int(expanded_value)
    if baseline_min != expanded_min:
        raise RuntimeError(
            "Baseline and expanded replays disagree on rotation_min_holding_days."
        )
    if baseline_min < 0:
        raise RuntimeError("rotation_min_holding_days cannot be negative.")
    return baseline_min


def _policy_holding_state(holding: pd.Series, min_holding_days: int) -> pd.Series:
    """Collapse exact holding age to the state actually observed by the policy.

    The Utility policy only asks whether holding < rotation_min_holding_days. Once
    the threshold has been reached, 2, 3, 20, ... holding days are behaviorally
    equivalent for future decisions. Keeping the exact age would falsely prolong
    divergence episodes after both paths have already returned to the same policy
    state.
    """
    threshold = int(min_holding_days)
    if threshold <= 0:
        return pd.Series(0, index=holding.index, dtype=int)
    return holding.astype(int).clip(upper=threshold)


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

    min_holding_days = _replay_min_holding_days(baseline_daily, expanded_daily)
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
    baseline_policy_holding = _policy_holding_state(
        baseline_holding, min_holding_days
    )
    expanded_policy_holding = _policy_holding_state(
        expanded_holding, min_holding_days
    )

    baseline_log = pd.to_numeric(baseline["net_log_return"], errors="coerce")
    expanded_log = pd.to_numeric(expanded["net_log_return"], errors="coerce")
    if baseline_log.isna().any() or expanded_log.isna().any():
        raise RuntimeError("Replay contains non-finite daily log returns.")

    delta = expanded_log - baseline_log
    # Reconvergence must use the minimal sufficient state of the actual policy.
    # Exact holding age only matters while it is below the minimum-holding guard;
    # after the threshold, both paths are behaviorally identical if the selected
    # asset is the same. The reconvergence session itself remains in the target so
    # switch-back costs are accounted for.
    state_equal = (
        baseline_asset.eq(expanded_asset)
        & baseline_policy_holding.eq(expanded_policy_holding)
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
        marginal = float(delta.iloc[window].sum())

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
                "policy_min_holding_days": int(min_holding_days),
                "episode_definition_version": EPISODE_DEFINITION_VERSION,
                TARGET_COLUMN: marginal,
                "positive_episode": bool(marginal > 0.0),
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
    """Score PCEA samples without discarding stateful episode metadata.

    The generic PCMA scorer intentionally narrows its frame to model columns. PCEA
    needs episode_start/episode_end after scoring so the non-overlap selector can
    enforce one realizable chronological path. Keep the original episode columns,
    filter only invalid model rows, then attach posterior predictions in place.
    """
    feature_names = list(model.feature_names)
    required = {
        "fold",
        "timestamp",
        "candidate",
        "episode_start",
        "episode_end",
        TARGET_COLUMN,
        *feature_names,
    }
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise RuntimeError(
            "PCEA episode samples are missing required scoring columns: "
            + ", ".join(missing)
        )

    frame = samples.copy()
    for column in ("timestamp", "episode_start", "episode_end"):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, format="mixed", errors="coerce"
        )

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    finite = np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=float))
    finite &= frame["timestamp"].notna().to_numpy(dtype=bool)
    finite &= frame["episode_start"].notna().to_numpy(dtype=bool)
    finite &= frame["episode_end"].notna().to_numpy(dtype=bool)
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))

    frame = frame.loc[finite].copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No finite PCEA episode samples remain for scoring.")
    if (frame["episode_end"] < frame["episode_start"]).any():
        raise RuntimeError("PCEA episode_end precedes episode_start.")

    mean, std = model.predict(frame)
    frame["predicted_marginal_advantage"] = np.asarray(mean, dtype=float)
    frame["predicted_marginal_std"] = np.asarray(std, dtype=float)
    return frame


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
