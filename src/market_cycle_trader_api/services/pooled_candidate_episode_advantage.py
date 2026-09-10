from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .pooled_candidate_marginal_advantage import (
    PooledCandidateMarginalModel,
    fit_pooled_candidate_marginal_model,
)

TARGET_COLUMN = "marginal_episode_net_log_return"
OBSERVED_TARGET_COLUMN = "observed_marginal_episode_net_log_return"
LABEL_AVAILABLE_COLUMN = "label_available_at"
EPISODE_DEFINITION_VERSION = "pcea-2.1.0-censor-aware-evaluation"
EPISODE_COLUMNS = [
    "fold", "candidate", "episode_id", "episode_start", "episode_end",
    "episode_observed_through", LABEL_AVAILABLE_COLUMN, "episode_sessions",
    "right_censored", "baseline_asset_at_start", "expanded_asset_at_start",
    "baseline_holding_days_at_start", "expanded_holding_days_at_start",
    "policy_min_holding_days", "episode_definition_version", OBSERVED_TARGET_COLUMN,
    TARGET_COLUMN, "positive_episode", "changed_position_sessions",
]


def _censored(frame: pd.DataFrame) -> pd.Series:
    if "right_censored" not in frame:
        raise RuntimeError("Episode censoring metadata is missing; recompute the samples.")
    values = frame["right_censored"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if values.isna().any():
        raise RuntimeError("Invalid right_censored flag in episode samples.")
    return values.astype(bool)


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
    required = {"timestamp", "execution_timestamp", "selected_asset", "net_log_return"}
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
    for label, frame in (("baseline", baseline), ("expanded", expanded)):
        frame["execution_timestamp"] = pd.to_datetime(
            frame["execution_timestamp"], utc=True, format="mixed", errors="raise"
        )
        if (
            frame[["timestamp", "execution_timestamp"]].isna().any().any()
            or frame["timestamp"].duplicated().any()
            or not frame["timestamp"].is_monotonic_increasing
            or (frame["execution_timestamp"] <= frame["timestamp"]).any()
        ):
            raise RuntimeError(f"Invalid {label} replay decision/execution calendar.")
    if not baseline["execution_timestamp"].equals(expanded["execution_timestamp"]):
        raise RuntimeError("Baseline and expanded execution calendars differ.")

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
    if not np.isfinite(baseline_log).all() or not np.isfinite(expanded_log).all():
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
                "episode_observed_through": baseline["execution_timestamp"].iloc[end],
                LABEL_AVAILABLE_COLUMN: (
                    pd.NaT if right_censored else baseline["execution_timestamp"].iloc[end]
                ),
                "episode_sessions": int(end - start + 1),
                "right_censored": bool(right_censored),
                "baseline_asset_at_start": baseline_asset.iloc[start],
                "expanded_asset_at_start": expanded_asset.iloc[start],
                "baseline_holding_days_at_start": int(baseline_holding.iloc[start]),
                "expanded_holding_days_at_start": int(expanded_holding.iloc[start]),
                "policy_min_holding_days": int(min_holding_days),
                "episode_definition_version": EPISODE_DEFINITION_VERSION,
                OBSERVED_TARGET_COLUMN: marginal,
                TARGET_COLUMN: np.nan if right_censored else marginal,
                "positive_episode": None if right_censored else bool(marginal > 0.0),
                "changed_position_sessions": int(
                    (~baseline_asset.iloc[window].eq(expanded_asset.iloc[window])).sum()
                ),
            }
        )

        i = end + 1

    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def build_episode_samples(
    *,
    daily_features: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=list(dict.fromkeys([
            *EPISODE_COLUMNS, *daily_features.columns, "timestamp",
        ])))

    features = daily_features.copy()
    features["timestamp"] = pd.to_datetime(
        features["timestamp"], utc=True, format="mixed", errors="raise"
    )
    # Censoring is an outcome, not information available at admission time.
    # Retain every start here; only the training path may exclude open labels.
    ep = episodes.copy()
    censored = _censored(ep)
    if OBSERVED_TARGET_COLUMN not in ep:
        raise RuntimeError("Observed episode outcomes are missing; recompute the samples.")
    ep.loc[censored, TARGET_COLUMN] = np.nan
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


def training_episode_samples(
    samples: pd.DataFrame,
    *,
    available_before: Any,
) -> pd.DataFrame:
    """Select completed labels observed strictly before the scoring session.

    Fold numbers alone do not establish label maturity. Daily replay timestamps
    refer to decisions; the associated return is observed on the next execution
    session. Open episodes never supply a completed-episode target.
    """
    if LABEL_AVAILABLE_COLUMN not in samples:
        raise RuntimeError("Episode label availability is missing; recompute the samples.")
    cutoff = pd.to_datetime(available_before, utc=True, errors="raise")
    if pd.isna(cutoff):
        raise RuntimeError("A finite training cutoff is required.")
    censored = _censored(samples)
    available = pd.to_datetime(
        samples[LABEL_AVAILABLE_COLUMN], utc=True, format="mixed", errors="raise"
    )
    if available.loc[~censored].isna().any():
        raise RuntimeError("Completed episodes must declare when their labels became available.")
    return samples.loc[~censored & available.lt(cutoff)].copy()


def fit_episode_model(
    samples: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    available_before: Any,
) -> PooledCandidateMarginalModel:
    training = training_episode_samples(samples, available_before=available_before)
    if training.empty:
        raise RuntimeError("No completed episode labels are available before the scoring session.")
    return fit_pooled_candidate_marginal_model(
        training,
        feature_names,
        target_column=TARGET_COLUMN,
    )


def score_episode_samples(
    model: PooledCandidateMarginalModel,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Predict using decision-time columns only, preserving optional audit metadata.

    Neither episode completion nor an observed future return is required to make
    a prediction. Those columns must never decide which candidates get scored.
    """
    if samples.empty:
        return samples.assign(
            predicted_marginal_advantage=pd.Series(dtype=float),
            predicted_marginal_std=pd.Series(dtype=float),
        )
    feature_names = list(model.feature_names)
    required = {
        "fold",
        "timestamp",
        "candidate",
        "episode_start",
        *feature_names,
    }
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise RuntimeError(
            "PCEA episode samples are missing required scoring columns: "
            + ", ".join(missing)
        )

    frame = samples.copy()
    for column in ("timestamp", "episode_start"):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, format="mixed", errors="coerce"
        )

    finite = frame["timestamp"].notna().to_numpy(dtype=bool)
    finite &= frame["episode_start"].notna().to_numpy(dtype=bool)
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))

    frame = frame.loc[finite].copy().reset_index(drop=True)
    if frame.empty:
        return frame.assign(
            predicted_marginal_advantage=pd.Series(dtype=float),
            predicted_marginal_std=pd.Series(dtype=float),
        )

    mean, std = model.predict(frame)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("PCEA model produced non-finite posterior predictions.")
    frame["predicted_marginal_advantage"] = np.asarray(mean, dtype=float)
    frame["predicted_marginal_std"] = np.asarray(std, dtype=float)
    return frame


def choose_non_overlapping_episode_overrides(
    scored_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Retrospective episode diagnostic, not a full portfolio execution engine.

    Rank exclusively by predictions. Inspect the chosen outcome afterwards to
    advance the retrospective clock. A still-open chosen episode occupies the
    rest of its fold; its observed prefix must not become a completed target.
    """
    columns = [
        "fold", "episode_start", "episode_end", "chosen_candidate",
        "override_baseline", "predicted_marginal_advantage", "predicted_marginal_std",
        "right_censored", "realized_marginal_episode_net_log_return",
        OBSERVED_TARGET_COLUMN, "candidate_count_at_start",
    ]
    if scored_samples.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "fold",
        "candidate",
        "episode_start",
        "episode_end",
        "right_censored",
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
    frame["right_censored"] = _censored(frame)
    if (
        frame[["episode_start", "episode_end"]].isna().any().any()
        or (frame["episode_end"] < frame["episode_start"]).any()
    ):
        raise RuntimeError("Retrospective episode bounds are invalid.")
    if frame.duplicated(["fold", "episode_start", "candidate"]).any():
        raise RuntimeError("Duplicate candidate at an episode start.")

    rows: list[dict[str, Any]] = []
    for fold_id, fold in frame.groupby("fold", sort=True):
        blocked_until: pd.Timestamp | None = None
        open_episode = False
        for start, group in fold.groupby("episode_start", sort=True):
            start = pd.Timestamp(start)
            if open_episode or (blocked_until is not None and start <= blocked_until):
                continue
            ordered = group.sort_values(
                ["predicted_marginal_advantage", "candidate"],
                ascending=[False, True],
            )
            best = ordered.iloc[0]
            predicted = float(best["predicted_marginal_advantage"])
            override = bool(np.isfinite(predicted) and predicted > 0.0)
            censored = bool(override and best["right_censored"])
            if override:
                chosen_candidate = str(best["candidate"])
                realized = np.nan if censored else float(best.get(TARGET_COLUMN, np.nan))
                observed = float(best.get(OBSERVED_TARGET_COLUMN, realized))
                end = pd.Timestamp(best["episode_end"])
                blocked_until = end
                open_episode = censored
            else:
                chosen_candidate, realized, observed, end = "BASELINE", 0.0, 0.0, start
            rows.append(
                {
                    "fold": int(fold_id),
                    "episode_start": start,
                    "episode_end": end,
                    "chosen_candidate": chosen_candidate,
                    "override_baseline": override,
                    "predicted_marginal_advantage": predicted,
                    "predicted_marginal_std": float(best["predicted_marginal_std"]),
                    "right_censored": censored,
                    "realized_marginal_episode_net_log_return": realized,
                    OBSERVED_TARGET_COLUMN: observed,
                    "candidate_count_at_start": int(len(group)),
                }
            )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["fold", "episode_start"]
    ).reset_index(drop=True)


def summarize_episode_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    chosen = decisions.loc[decisions["override_baseline"].astype(bool)] if not decisions.empty else decisions
    if chosen.empty:
        censored = pd.Series(dtype=bool)
        realized = observed = pd.Series(dtype=float)
    else:
        censored = _censored(chosen)
        realized = pd.to_numeric(
            chosen["realized_marginal_episode_net_log_return"], errors="coerce"
        )
        observed = pd.to_numeric(chosen[OBSERVED_TARGET_COLUMN], errors="coerce")
    complete = realized.loc[~censored]
    complete_finite = complete.loc[np.isfinite(complete)]
    missing_complete = int((~np.isfinite(complete)).sum())
    missing_observed = int((~np.isfinite(observed)).sum())
    has_pending = bool(censored.any() or missing_complete)
    return {
        "decision_episode_starts": int(len(decisions)),
        "override_count": int(len(chosen)),
        "override_rate": float(len(chosen) / len(decisions)) if len(decisions) else 0.0,
        "completed_override_count": int(len(complete_finite)),
        "right_censored_override_count": int(censored.sum()),
        "missing_completed_outcome_count": missing_complete,
        "missing_observed_outcome_count": missing_observed,
        "completed_override_marginal_log_sum": float(complete_finite.sum()),
        "observed_marginal_log_sum": None if missing_observed else float(observed.sum()),
        # A missing or still-open outcome is never silently treated as zero.
        "realized_marginal_log_sum": None if has_pending else float(complete.sum()),
        "realized_marginal_log_mean_when_overridden": (
            float(complete_finite.mean()) if not complete_finite.empty else None
        ),
        "positive_realized_override_rate": (
            float((complete_finite > 0.0).mean()) if not complete_finite.empty else None
        ),
        "realized_rate_basis": "completed overrides only; open outcomes reported separately",
        "is_portfolio_capital_return": False,
    }
