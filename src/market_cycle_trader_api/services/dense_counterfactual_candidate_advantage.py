from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import math
import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import _execute_buy
from market_cycle_trader_api.engine.compound_rotation_backtest import (
    apply_slippage,
    calculate_reference_fees,
)
from market_cycle_trader_api.services.asset_marginal_score_replay import (
    ReplayResult,
    ScoreReplayPanel,
    replay_policy_settings,
)
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (
    TARGET_COLUMN,
    extract_candidate_divergence_episodes,
)

DCCA_EPISODE_DEFINITION_VERSION = "dcca-1.0.0-forced-candidate-policy-sufficient-state"


@dataclass
class ForcedCandidateReplay:
    replay: ReplayResult
    intervention_applied: bool
    intervention_timestamp: pd.Timestamp
    candidate: str
    blocked_reason: str | None


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _daily_frame_with_attrs(
    panel: ScoreReplayPanel,
    *,
    equities: list[float],
    positions: list[str],
    reasons: list[str],
    initial_capital: float,
    min_holding_days: int,
) -> pd.DataFrame:
    daily = panel.calendar.reset_index().copy()
    daily["selected_asset"] = positions
    daily["decision_reason"] = reasons
    daily["strategy_equity"] = equities
    daily["net_log_return"] = np.diff(
        np.log(np.r_[float(initial_capital), np.asarray(equities, dtype=float)])
    )
    daily.attrs["rotation_min_holding_days"] = int(min_holding_days)
    return daily


def replay_forced_candidate_intervention(
    *,
    panel: ScoreReplayPanel,
    candidate: str,
    settings: dict[str, Any],
    intervention_timestamp: Any,
) -> ForcedCandidateReplay:
    """Replay one candidate intervention from the exact baseline prefix state.

    Before the intervention session only the immutable baseline assets may influence
    decisions. At the intervention session the candidate is forced only when the
    production minimum-holding guard is already satisfied. After a successful
    intervention the normal Utility policy runs on baseline + candidate. This makes
    the label answer "what if this candidate were chosen now?" without allowing the
    candidate to alter the pre-intervention path.
    """
    config = SimpleNamespace(**settings)
    replay_policy_settings(config)

    symbol = str(candidate).strip().upper()
    if not symbol or symbol in set(panel.baseline):
        raise ValueError("Forced candidate must be one external symbol.")
    if not panel.complete(symbol):
        raise ValueError(f"Candidate {symbol} does not cover the replay calendar.")

    timestamp = _normalized_timestamp(intervention_timestamp)
    try:
        location = panel.calendar.index.get_loc(timestamp)
    except KeyError as exc:
        raise ValueError(
            f"Intervention timestamp is outside the replay calendar: {timestamp.isoformat()}"
        ) from exc
    if not isinstance(location, (int, np.integer)):
        raise ValueError("Intervention timestamp is not unique in the replay calendar.")
    activation = int(location)

    symbols = sorted(set([*panel.baseline, symbol]))
    symbol_to_index = {name: index for index, name in enumerate(symbols)}
    candidate_index = symbol_to_index[symbol]
    baseline_set = set(panel.baseline)
    baseline_indices = np.asarray(
        [index for index, name in enumerate(symbols) if name in baseline_set],
        dtype=int,
    )
    if baseline_indices.size != len(panel.baseline):
        raise RuntimeError("Immutable baseline mapping is inconsistent in forced replay.")

    scores = panel.scores[symbols].to_numpy(dtype=float)
    opens = panel.opens[symbols].to_numpy(dtype=float)
    closes = panel.closes[symbols].to_numpy(dtype=float)

    cash = float(config.initial_capital)
    quantity = 0.0
    position = -1
    holding = 0
    fees_total = 0.0
    rotations = 0
    equities: list[float] = []
    positions: list[str] = []
    reasons: list[str] = []
    intervention_applied = False
    blocked_reason: str | None = None
    min_holding_days = int(config.rotation_min_holding_days)

    for i, utilities in enumerate(scores):
        candidate_enabled = bool(intervention_applied and i > activation)

        if i < activation or not candidate_enabled:
            available = np.zeros(len(symbols), dtype=bool)
            available[baseline_indices] = True
        else:
            available = np.ones(len(symbols), dtype=bool)

        masked = np.asarray(utilities, dtype=float).copy()
        masked[~available] = -np.inf
        values = np.r_[0.0, masked]
        best = int(np.argmax(values)) - 1
        best_value = float(values[best + 1])
        current_value = (
            float(utilities[position])
            if position >= 0 and np.isfinite(utilities[position])
            else 0.0
        )

        if i == activation:
            if position >= 0 and holding < min_holding_days:
                blocked_reason = "MIN_HOLD_GUARD"
                target = position
                reason = "MIN_HOLD_GUARD"
            else:
                target = candidate_index
                reason = "FORCED_CANDIDATE_INTERVENTION"
                intervention_applied = True
        elif position >= 0 and holding < min_holding_days:
            target, reason = position, "MIN_HOLD_GUARD"
        elif best == -1 or best_value <= float(config.rotation_cash_threshold):
            target, reason = -1, "CASH_THRESHOLD"
        elif position == -1:
            accepted = best_value >= (
                float(config.rotation_cash_threshold)
                + float(config.rotation_min_expected_edge)
            )
            target, reason = (
                (best, "ENTER_BEST_ASSET")
                if accepted
                else (-1, "MIN_EXPECTED_EDGE_GUARD")
            )
        elif best == position:
            target, reason = position, "HOLD_CURRENT_BEST"
        elif best_value >= current_value + float(config.rotation_switch_margin):
            target, reason = best, "ROTATE_TO_BEST_ASSET"
        else:
            target, reason = position, "SWITCH_MARGIN_GUARD"

        if target != position:
            if position >= 0:
                price = apply_slippage(float(opens[i, position]), "SELL", config)
                fee = calculate_reference_fees(
                    "SELL", quantity, price, config
                )["total_fee"]
                cash += quantity * price - fee
                fees_total += fee
            if target >= 0:
                quantity, price, buy_fees = _execute_buy(
                    cash,
                    float(opens[i, target]),
                    config,
                    calculate_reference_fees,
                    apply_slippage,
                )
                cash -= quantity * price + buy_fees["total_fee"]
                fees_total += buy_fees["total_fee"]
            else:
                quantity = 0.0
            rotations += int(position >= 0 and target >= 0)
            position = int(target)
            holding = 1 if target >= 0 else 0
        elif position >= 0:
            holding += 1

        equity = cash + (
            quantity * float(closes[i, position]) if position >= 0 else 0.0
        )
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError(
                "Forced counterfactual replay reached non-positive/non-finite equity."
            )
        equities.append(float(equity))
        positions.append(symbols[position] if position >= 0 else "CASH")
        reasons.append(str(reason))

    if position >= 0:
        price = apply_slippage(float(closes[-1, position]), "SELL", config)
        fee = calculate_reference_fees("SELL", quantity, price, config)["total_fee"]
        cash += quantity * price - fee
        fees_total += fee
        equities[-1] = float(cash)

    if not math.isfinite(cash) or cash <= 0.0:
        raise ValueError("Forced counterfactual final liquidation exhausted the account.")

    daily = _daily_frame_with_attrs(
        panel,
        equities=equities,
        positions=positions,
        reasons=reasons,
        initial_capital=float(config.initial_capital),
        min_holding_days=min_holding_days,
    )

    return ForcedCandidateReplay(
        replay=ReplayResult(
            daily=daily,
            ending_capital=float(cash),
            net_log_growth=float(math.log(cash / float(config.initial_capital))),
            rotations=int(rotations),
            fees=float(fees_total),
        ),
        intervention_applied=bool(intervention_applied),
        intervention_timestamp=timestamp,
        candidate=symbol,
        blocked_reason=blocked_reason,
    )


def build_dense_counterfactual_episode_samples(
    *,
    panel: ScoreReplayPanel,
    baseline_replay: ReplayResult,
    daily_features: pd.DataFrame,
    candidate_symbols: list[str],
    settings: dict[str, Any],
    fold_id: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate one forced, stateful episode label per legal candidate-date pair."""
    if daily_features.empty:
        raise RuntimeError("Dense counterfactual sampling received no decision-time features.")

    required = {
        "fold",
        "timestamp",
        "candidate",
        "baseline_current_asset",
        "baseline_holding_days",
    }
    missing = sorted(required.difference(daily_features.columns))
    if missing:
        raise RuntimeError(
            "Dense counterfactual decision-time features are missing columns: "
            + ", ".join(missing)
        )

    features = daily_features.copy()
    features["timestamp"] = pd.to_datetime(
        features["timestamp"], utc=True, format="mixed", errors="raise"
    )
    features["candidate"] = features["candidate"].astype(str).str.upper()
    candidates = {
        str(symbol).strip().upper()
        for symbol in candidate_symbols
        if str(symbol).strip()
    }
    features = features.loc[features["candidate"].isin(candidates)].copy()
    features = features.sort_values(["timestamp", "candidate"]).reset_index(drop=True)

    min_holding_days = int(settings["rotation_min_holding_days"])
    current = features["baseline_current_asset"].astype(str).str.upper()
    holding = pd.to_numeric(features["baseline_holding_days"], errors="coerce")
    legal = current.eq("CASH") | (holding >= min_holding_days)
    legal_features = features.loc[legal].copy().reset_index(drop=True)

    base_daily = baseline_replay.daily.copy()
    base_daily.attrs["rotation_min_holding_days"] = min_holding_days

    drop_future_columns = {
        "next_timestamp",
        "baseline_next_session_net_log_return",
        "candidate_next_session_net_log_return",
        "marginal_next_session_net_log_return",
    }

    sample_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    prefiltered_blocked_count = int((~legal).sum())
    simulator_blocked_count = 0
    no_episode_count = 0
    right_censored_count = 0
    total = len(legal_features)

    for position, row in enumerate(legal_features.to_dict(orient="records"), start=1):
        candidate = str(row["candidate"])
        timestamp = _normalized_timestamp(row["timestamp"])
        forced = replay_forced_candidate_intervention(
            panel=panel,
            candidate=candidate,
            settings=settings,
            intervention_timestamp=timestamp,
        )
        if not forced.intervention_applied:
            simulator_blocked_count += 1
            continue

        forced_daily = forced.replay.daily.copy()
        forced_daily.attrs["rotation_min_holding_days"] = min_holding_days
        episodes = extract_candidate_divergence_episodes(
            baseline_daily=base_daily,
            expanded_daily=forced_daily,
            candidate=candidate,
            fold_id=int(fold_id),
        )
        if episodes.empty:
            no_episode_count += 1
            continue

        starts = pd.to_datetime(
            episodes["episode_start"], utc=True, format="mixed", errors="raise"
        )
        exact = episodes.loc[starts.eq(timestamp)].copy()
        if len(exact) != 1:
            no_episode_count += 1
            continue

        episode = exact.iloc[0].to_dict()
        episode["episode_definition_version"] = DCCA_EPISODE_DEFINITION_VERSION
        episode["intervention_source"] = "forced_candidate_from_baseline_prefix"
        episode["raw_candidate_was_required_to_win"] = False
        episode_rows.append(dict(episode))

        if bool(episode["right_censored"]):
            right_censored_count += 1
        else:
            merged = {
                key: value
                for key, value in row.items()
                if key not in drop_future_columns
            }
            merged.update(episode)
            merged["timestamp"] = timestamp
            merged["fold"] = int(fold_id)
            merged["candidate"] = candidate
            sample_rows.append(merged)

        if progress_callback is not None and (
            position == 1 or position % 500 == 0 or position == total
        ):
            progress_callback(position, total)

    samples = pd.DataFrame(sample_rows)
    episodes = pd.DataFrame(episode_rows)
    if not samples.empty:
        samples = samples.sort_values(
            ["fold", "episode_start", "candidate"]
        ).reset_index(drop=True)
    if not episodes.empty:
        episodes = episodes.sort_values(
            ["fold", "episode_start", "candidate"]
        ).reset_index(drop=True)

    diagnostics = {
        "fold": int(fold_id),
        "candidate_date_feature_rows": int(len(features)),
        "policy_legal_candidate_date_rows": int(len(legal_features)),
        "minimum_holding_blocked_rows": int(prefiltered_blocked_count),
        "forced_replay_blocked_rows": int(simulator_blocked_count),
        "forced_episode_rows": int(len(episodes)),
        "right_censored_episode_rows": int(right_censored_count),
        "usable_episode_rows": int(len(samples)),
        "no_exact_episode_rows": int(no_episode_count),
        "candidate_count": int(features["candidate"].nunique()),
        "decision_date_count": int(features["timestamp"].nunique()),
        "policy_min_holding_days": int(min_holding_days),
        "episode_definition_version": DCCA_EPISODE_DEFINITION_VERSION,
    }
    return samples, episodes, diagnostics


def oracle_non_overlapping_summary(samples: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic upper bound using realized labels; never a tradable decision rule."""
    if samples.empty:
        return {
            "oracle_episode_count": 0,
            "oracle_realized_marginal_log_sum": 0.0,
            "oracle_incremental_factor": 0.0,
        }

    frame = samples.copy()
    frame["episode_start"] = pd.to_datetime(
        frame["episode_start"], utc=True, format="mixed", errors="raise"
    )
    frame["episode_end"] = pd.to_datetime(
        frame["episode_end"], utc=True, format="mixed", errors="raise"
    )
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    blocked_until: pd.Timestamp | None = None
    chosen: list[dict[str, Any]] = []
    for start, group in frame.groupby("episode_start", sort=True):
        start = pd.Timestamp(start)
        if blocked_until is not None and start <= blocked_until:
            continue
        ordered = group.sort_values(
            [TARGET_COLUMN, "candidate"], ascending=[False, True]
        )
        best = ordered.iloc[0]
        realized = float(best[TARGET_COLUMN])
        if not np.isfinite(realized) or realized <= 0.0:
            continue
        blocked_until = pd.Timestamp(best["episode_end"])
        chosen.append(best.to_dict())

    total = float(sum(float(row[TARGET_COLUMN]) for row in chosen))
    return {
        "oracle_episode_count": int(len(chosen)),
        "oracle_realized_marginal_log_sum": total,
        "oracle_incremental_factor": float(math.exp(total) - 1.0),
    }
