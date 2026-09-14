from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

import research_contextual_signature_protocol as protocol
import research_contextual_signature_storage as storage

TARGET = "source_direct_delta_log_capital"
PRIMARY = "Original25"
ROBUST = "Original24_MinusADM"
TOL = 1e-12
EXPERIMENT_NAME = "contextual_marginal_signature_opportunity_hole_decomposition"
COLLECTION = "research_contextual_signature_tournaments"
HOLE_HORIZONS = (5, 10, 20, 40)
PRIMARY_HOLE_HORIZON = 20

# Kept as the frozen v1.0.19 representation contract for regression tests/history.
TRAJECTORY_SOURCE_FEATURES = (
    "return_5",
    "return_20",
    "return_60",
    "vol_20",
    "ema_distance_20",
    "ema_20_vs_50",
    "rsi_14",
    "atr_pct_14",
    "trend_efficiency_20",
    "momentum_acceleration_5_20",
)
TRAJECTORY_LOOKBACKS = (5, 20, 60)

TRACE_TIME_KEYS = ("timestamp", "datetime", "date", "session", "index")
TRACE_SELECTED_KEYS = ("selected_asset", "selected_symbol", "asset", "symbol")
TRACE_EQUITY_KEYS = ("strategy_equity", "portfolio_value", "equity", "capital")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id")
    parser.add_argument("--env-file")
    parser.add_argument("--mongo-uri")
    parser.add_argument("--database")
    return parser


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _path_stats(
    series: pd.Series,
    decision: Any,
    lookback: int,
) -> tuple[float, float, float]:
    """Frozen helper from v1.0.19; never uses rows after the decision."""
    if series.empty:
        return float("nan"), float("nan"), float("nan")
    path = pd.to_numeric(series, errors="coerce").copy()
    path.index = pd.to_datetime(path.index, utc=True)
    path = path.loc[path.index <= _stamp(decision)].dropna().tail(int(lookback) + 1)
    if len(path) < 2:
        return float("nan"), float("nan"), float("nan")
    values = path.to_numpy(dtype=float)
    x = np.arange(len(values), dtype=float)
    return (
        float(values[-1] - values[0]),
        float(np.polyfit(x, values, 1)[0]),
        float(np.std(values, ddof=0)),
    )


def trajectory_status(
    trajectory_development: dict[str, Any],
    snapshot_development: dict[str, Any],
    baseline: dict[str, Any],
    holdout_metrics: dict[str, Any],
    robust_metrics: dict[str, Any],
) -> str:
    """Frozen v1.0.19 decision rule retained for reproducibility tests."""
    trajectory_gain = float(
        trajectory_development["cumulative_direct_log_gain"]
    )
    snapshot_gain = float(snapshot_development["cumulative_direct_log_gain"])
    baseline_gain = float(baseline["cumulative_direct_log_gain"])
    if trajectory_gain <= snapshot_gain + TOL:
        return "TRAJECTORY_ADDED_VALUE_NOT_FOUND"
    if trajectory_gain <= baseline_gain + TOL:
        return "TRAJECTORY_CONTEXTUAL_SIGNAL_NOT_FOUND"
    if float(holdout_metrics["cumulative_direct_log_gain"]) <= TOL:
        return "TRAJECTORY_SIGNAL_NOT_CONFIRMED"
    if float(robust_metrics["cumulative_direct_log_gain"]) <= TOL:
        return "TRAJECTORY_HOLDOUT_NOT_ROBUST"
    return "TRAJECTORY_CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST"


def _round_trip_cost_log(config: Any) -> float:
    round_trip_cost = min(
        0.25,
        2.0
        * (
            max(0.0, float(getattr(config, "slippage_bps", 0.0))) / 10000.0
            + max(0.0, float(getattr(config, "commission_rate", 0.0)))
        ),
    )
    return math.log(max(1e-12, 1.0 - round_trip_cost))


def _close_at_or_before(frame: pd.DataFrame, when: Any) -> float:
    if frame.empty or "close" not in frame.columns:
        return float("nan")
    data = frame.sort_index()
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    position = int(index.searchsorted(_stamp(when), side="right") - 1)
    if position < 0:
        return float("nan")
    value = _num(data.iloc[position]["close"])
    return value if value > 0.0 else float("nan")


def _forward_log_return(
    frame: pd.DataFrame,
    decision: Any,
    horizon_end: Any,
    cost_log: float = 0.0,
) -> float:
    """Retrospective opportunity label; bounded strictly by decision/horizon_end."""
    start = _close_at_or_before(frame, decision)
    end = _close_at_or_before(frame, horizon_end)
    if not (math.isfinite(start) and math.isfinite(end) and start > 0 and end > 0):
        return float("nan")
    return float(math.log(end / start) + float(cost_log))


def _trace_parts(trace_id: str) -> tuple[str, str, str | None, str, int] | None:
    parts = str(trace_id).rsplit(":", 5)
    if len(parts) != 6:
        return None
    _, decision_date, universe_name, candidate_token, arm, result_index_text = parts
    try:
        result_index = int(result_index_text)
    except (TypeError, ValueError):
        result_index = 0
    candidate = None if candidate_token == "BASELINE" else candidate_token
    return decision_date, universe_name, candidate, arm, result_index


def _payload_timestamp(payload: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    for key in TRACE_TIME_KEYS:
        if key not in payload:
            continue
        stamp = pd.to_datetime(payload.get(key), utc=True, errors="coerce")
        if not pd.isna(stamp):
            return pd.Timestamp(stamp)
    return pd.NaT


def _payload_selected_asset(payload: dict[str, Any]) -> str | None:
    for key in TRACE_SELECTED_KEYS:
        value = payload.get(key)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _payload_equity(payload: dict[str, Any]) -> float:
    for key in TRACE_EQUITY_KEYS:
        if key in payload:
            value = _num(payload.get(key))
            if math.isfinite(value) and value > 0.0:
                return value
    return float("nan")


def _load_prediction_traces(
    db: Any,
    run_id: str,
) -> dict[tuple[str, str, str | None, str], pd.DataFrame]:
    grouped: dict[
        tuple[str, str, str | None, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    cursor = db[protocol.TRACE_ROWS_COLLECTION].find(
        {"run_id": run_id, "row_type": "prediction"},
        {
            "_id": 0,
            "trace_id": 1,
            "row_index": 1,
            "payload": 1,
        },
    )
    for document in cursor:
        trace_id = str(document.get("trace_id") or "")
        parsed = _trace_parts(trace_id)
        if parsed is None:
            continue
        decision_date, universe_name, candidate, arm, result_index = parsed
        if arm not in {"baseline", "policy"}:
            continue
        payload = document.get("payload")
        if not isinstance(payload, dict):
            continue
        grouped[(decision_date, universe_name, candidate, arm)].append(
            {
                "timestamp": _payload_timestamp(payload),
                "selected_asset": _payload_selected_asset(payload),
                "strategy_equity": _payload_equity(payload),
                "result_index": result_index,
                "row_index": int(document.get("row_index") or 0),
            }
        )

    output: dict[tuple[str, str, str | None, str], pd.DataFrame] = {}
    for key, rows in grouped.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[key] = frame
            continue
        if frame["timestamp"].notna().any():
            frame = frame.sort_values(
                ["timestamp", "result_index", "row_index"],
                kind="stable",
            )
            frame = frame.drop_duplicates("timestamp", keep="last")
        else:
            frame = frame.sort_values(
                ["result_index", "row_index"],
                kind="stable",
            )
        output[key] = frame.reset_index(drop=True)
    return output


def _trace_until(
    frame: pd.DataFrame,
    horizon_end: Any,
    horizon_steps: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "timestamp" in frame and frame["timestamp"].notna().any():
        cutoff = _stamp(horizon_end)
        subset = frame.loc[
            frame["timestamp"].notna()
            & (pd.to_datetime(frame["timestamp"], utc=True) <= cutoff)
        ].copy()
        if not subset.empty:
            return subset.reset_index(drop=True)
    return frame.iloc[: min(len(frame), int(horizon_steps) + 1)].copy()


def _trace_equity_at_end(
    frame: pd.DataFrame,
    horizon_end: Any,
    horizon_steps: int,
) -> float:
    subset = _trace_until(frame, horizon_end, horizon_steps)
    if subset.empty or "strategy_equity" not in subset:
        return float("nan")
    equity = pd.to_numeric(subset["strategy_equity"], errors="coerce")
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    equity = equity.loc[equity > 0.0]
    return float(equity.iloc[-1]) if len(equity) else float("nan")


def _strategy_trace_log_gain(
    frame: pd.DataFrame,
    horizon_end: Any,
    horizon_steps: int,
) -> float:
    subset = _trace_until(frame, horizon_end, horizon_steps)
    if subset.empty or "strategy_equity" not in subset:
        return float("nan")
    equity = pd.to_numeric(subset["strategy_equity"], errors="coerce")
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    equity = equity.loc[equity > 0.0]
    if len(equity) < 2:
        return float("nan")
    return float(math.log(float(equity.iloc[-1]) / float(equity.iloc[0])))


def _normalized_asset(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _trace_divergence(
    baseline: pd.DataFrame,
    policy: pd.DataFrame,
    candidate: str,
) -> dict[str, Any]:
    """Detect whether policy diverged before the candidate itself was selected."""
    if baseline.empty or policy.empty:
        return {
            "candidate_ever_selected": False,
            "first_candidate_step": None,
            "first_candidate_session": None,
            "first_divergence_step": None,
            "first_divergence_session": None,
            "divergence_is_candidate_entry": False,
            "perturbation_before_candidate_selection": None,
        }

    use_time = (
        "timestamp" in baseline
        and "timestamp" in policy
        and baseline["timestamp"].notna().any()
        and policy["timestamp"].notna().any()
    )

    if use_time:
        left = baseline.loc[
            baseline["timestamp"].notna(),
            ["timestamp", "selected_asset"],
        ].copy()
        right = policy.loc[
            policy["timestamp"].notna(),
            ["timestamp", "selected_asset"],
        ].copy()
        aligned = left.merge(
            right,
            on="timestamp",
            how="inner",
            suffixes=("_baseline", "_policy"),
        )
        sessions = list(aligned["timestamp"])
        baseline_assets = [
            _normalized_asset(value)
            for value in aligned["selected_asset_baseline"]
        ]
        policy_assets = [
            _normalized_asset(value)
            for value in aligned["selected_asset_policy"]
        ]
    else:
        size = min(len(baseline), len(policy))
        sessions = [None] * size
        baseline_assets = [
            _normalized_asset(value)
            for value in baseline["selected_asset"].iloc[:size]
        ]
        policy_assets = [
            _normalized_asset(value)
            for value in policy["selected_asset"].iloc[:size]
        ]

    first_candidate_step = next(
        (
            index
            for index, selected in enumerate(policy_assets)
            if selected == candidate
        ),
        None,
    )
    first_divergence_step = next(
        (
            index
            for index, (base_asset, policy_asset) in enumerate(
                zip(baseline_assets, policy_assets)
            )
            if base_asset != policy_asset
        ),
        None,
    )

    candidate_ever_selected = first_candidate_step is not None
    divergence_is_candidate_entry = (
        first_divergence_step is not None
        and policy_assets[first_divergence_step] == candidate
    )
    if first_divergence_step is None:
        perturbation_before_candidate = False
    elif divergence_is_candidate_entry:
        perturbation_before_candidate = False
    elif first_candidate_step is None:
        perturbation_before_candidate = True
    else:
        perturbation_before_candidate = (
            first_divergence_step < first_candidate_step
        )

    def _session(step: int | None) -> str | None:
        if step is None or step >= len(sessions):
            return None
        value = sessions[step]
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).date().isoformat()

    return {
        "candidate_ever_selected": bool(candidate_ever_selected),
        "first_candidate_step": first_candidate_step,
        "first_candidate_session": _session(first_candidate_step),
        "first_divergence_step": first_divergence_step,
        "first_divergence_session": _session(first_divergence_step),
        "divergence_is_candidate_entry": bool(divergence_is_candidate_entry),
        "perturbation_before_candidate_selection": bool(
            perturbation_before_candidate
        ),
    }


def _classify_context(
    baseline_strategy_log_gain: float,
    best_incumbent_log_gain: float,
) -> str:
    if not (
        math.isfinite(baseline_strategy_log_gain)
        and math.isfinite(best_incumbent_log_gain)
    ):
        return "UNKNOWN"
    if baseline_strategy_log_gain > TOL:
        return "COVERED"
    if best_incumbent_log_gain > TOL:
        return "POLICY_MISS"
    return "UNIVERSE_HOLE"


def _universe_assets(universe_name: str) -> list[str]:
    for item in protocol.UNIVERSES:
        if str(item["name"]) == universe_name:
            return list(item["assets"])
    raise RuntimeError(f"Unknown research universe: {universe_name}")


def _context_opportunity(
    *,
    bars: dict[str, pd.DataFrame],
    universe_name: str,
    decision_date: str,
    horizon_end: str,
    horizon_sessions: int,
    baseline_trace: pd.DataFrame,
    cost_log: float,
) -> dict[str, Any]:
    returns: dict[str, float] = {}
    for symbol in _universe_assets(universe_name):
        if symbol not in bars:
            continue
        value = _forward_log_return(
            bars[symbol],
            decision_date,
            horizon_end,
            cost_log,
        )
        if math.isfinite(value):
            returns[symbol] = value

    if not returns:
        raise RuntimeError(
            f"No incumbent forward returns for {decision_date}/{universe_name}."
        )

    best_symbol = max(returns, key=returns.get)
    best_gain = float(returns[best_symbol])
    baseline_gain = _strategy_trace_log_gain(
        baseline_trace,
        horizon_end,
        horizon_sessions,
    )
    return {
        "context_class": _classify_context(baseline_gain, best_gain),
        "baseline_strategy_log_gain": baseline_gain,
        "baseline_strategy_multiplier": (
            float(math.exp(baseline_gain))
            if math.isfinite(baseline_gain)
            else None
        ),
        "best_incumbent": best_symbol,
        "best_incumbent_log_gain": best_gain,
        "best_incumbent_multiplier": float(math.exp(best_gain)),
        "positive_incumbent_count": int(
            sum(value > TOL for value in returns.values())
        ),
    }


def _research_horizon_end(
    *,
    bars: dict[str, pd.DataFrame],
    decision_date: str,
    full_horizon_end: str,
    horizon_sessions: int,
) -> str:
    if int(horizon_sessions) >= max(HOLE_HORIZONS):
        return str(full_horizon_end)
    reference = bars.get("SPY")
    if reference is None or reference.empty:
        reference = next(iter(bars.values()))
    index = pd.DatetimeIndex(pd.to_datetime(reference.index, utc=True)).sort_values()
    decision = _stamp(decision_date)
    full_end = _stamp(full_horizon_end)
    window = index[(index >= decision) & (index <= full_end)]
    if len(window) <= 1:
        return str(full_horizon_end)
    position = min(int(horizon_sessions), len(window) - 1)
    return pd.Timestamp(window[position]).date().isoformat()


def build_opportunity_hole_dataset(
    *,
    observations: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    traces: dict[tuple[str, str, str | None, str], pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    cost_log = _round_trip_cost_log(config)
    rows: list[dict[str, Any]] = []

    date_order = {
        value: index for index, value in enumerate(protocol.DECISION_DATES)
    }
    universe_order = {
        str(item["name"]): index
        for index, item in enumerate(protocol.UNIVERSES)
    }
    candidate_order = {
        value: index for index, value in enumerate(protocol.CANDIDATES)
    }

    clean_observations = observations.copy()
    clean_observations["decision_date"] = clean_observations[
        "decision_date"
    ].astype(str)
    clean_observations["universe_name"] = clean_observations[
        "universe_name"
    ].astype(str)
    clean_observations["candidate"] = clean_observations[
        "candidate"
    ].astype(str)

    for (decision_date, universe_name), group in clean_observations.groupby(
        ["decision_date", "universe_name"],
        sort=False,
    ):
        first = group.iloc[0]
        full_horizon_end = str(first["horizon_end"])
        baseline_key = (decision_date, universe_name, None, "baseline")
        baseline_trace = traces.get(baseline_key)
        if baseline_trace is None or baseline_trace.empty:
            raise RuntimeError(
                "Missing persisted baseline prediction trace for "
                f"{decision_date}/{universe_name}."
            )

        for horizon_sessions in HOLE_HORIZONS:
            horizon_end = _research_horizon_end(
                bars=bars,
                decision_date=decision_date,
                full_horizon_end=full_horizon_end,
                horizon_sessions=horizon_sessions,
            )
            context = _context_opportunity(
                bars=bars,
                universe_name=universe_name,
                decision_date=decision_date,
                horizon_end=horizon_end,
                horizon_sessions=horizon_sessions,
                baseline_trace=baseline_trace,
                cost_log=cost_log,
            )
            baseline_end_equity = _trace_equity_at_end(
                baseline_trace,
                horizon_end,
                horizon_sessions,
            )

            for observation in group.to_dict(orient="records"):
                candidate = str(observation["candidate"])
                if candidate not in bars:
                    raise RuntimeError(f"Missing market history for {candidate}.")
                policy_trace = traces.get(
                    (decision_date, universe_name, candidate, "policy")
                )
                if policy_trace is None or policy_trace.empty:
                    raise RuntimeError(
                        "Missing persisted policy prediction trace for "
                        f"{decision_date}/{universe_name}/{candidate}."
                    )

                clipped_baseline = _trace_until(
                    baseline_trace,
                    horizon_end,
                    horizon_sessions,
                )
                clipped_policy = _trace_until(
                    policy_trace,
                    horizon_end,
                    horizon_sessions,
                )
                divergence = _trace_divergence(
                    clipped_baseline,
                    clipped_policy,
                    candidate,
                )
                candidate_forward = _forward_log_return(
                    bars[candidate],
                    decision_date,
                    horizon_end,
                    cost_log,
                )
                policy_end_equity = _trace_equity_at_end(
                    policy_trace,
                    horizon_end,
                    horizon_sessions,
                )
                if (
                    math.isfinite(policy_end_equity)
                    and math.isfinite(baseline_end_equity)
                    and policy_end_equity > 0.0
                    and baseline_end_equity > 0.0
                ):
                    horizon_direct_effect = float(
                        math.log(policy_end_equity / baseline_end_equity)
                    )
                elif int(horizon_sessions) == max(HOLE_HORIZONS):
                    horizon_direct_effect = float(observation[TARGET])
                else:
                    horizon_direct_effect = float("nan")

                fills_hole = bool(
                    context["context_class"] == "UNIVERSE_HOLE"
                    and math.isfinite(candidate_forward)
                    and candidate_forward > TOL
                )
                capital_improved = bool(
                    math.isfinite(horizon_direct_effect)
                    and horizon_direct_effect > TOL
                )
                clean_coverage = bool(
                    fills_hole
                    and capital_improved
                    and divergence["candidate_ever_selected"]
                    and not divergence[
                        "perturbation_before_candidate_selection"
                    ]
                )
                covered_disruption = bool(
                    context["context_class"] == "COVERED"
                    and math.isfinite(horizon_direct_effect)
                    and horizon_direct_effect < -TOL
                )
                policy_miss_help = bool(
                    context["context_class"] == "POLICY_MISS"
                    and math.isfinite(horizon_direct_effect)
                    and horizon_direct_effect > TOL
                )

                row = {
                    "run_id": observation.get("run_id"),
                    "decision_date": decision_date,
                    "horizon_sessions": int(horizon_sessions),
                    "horizon_end": horizon_end,
                    "full_campaign_horizon_end": full_horizon_end,
                    "universe_name": universe_name,
                    "candidate": candidate,
                    TARGET: float(observation[TARGET]),
                    "horizon_direct_delta_log_capital": horizon_direct_effect,
                    **context,
                    "candidate_forward_log_gain": candidate_forward,
                    "candidate_forward_multiplier": (
                        float(math.exp(candidate_forward))
                        if math.isfinite(candidate_forward)
                        else None
                    ),
                    "fills_universe_hole": fills_hole,
                    "capital_improved": capital_improved,
                    "policy_miss_help": policy_miss_help,
                    "covered_disruption": covered_disruption,
                    "clean_coverage": clean_coverage,
                    **divergence,
                }
                rows.append(row)

    rows.sort(
        key=lambda row: (
            date_order.get(str(row["decision_date"]), 9999),
            universe_order.get(str(row["universe_name"]), 9999),
            int(row["horizon_sessions"]),
            candidate_order.get(str(row["candidate"]), 9999),
        )
    )
    return pd.DataFrame(rows)


def _candidate_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate, group in frame.groupby("candidate", sort=True):
        hole = group.loc[group["context_class"] == "UNIVERSE_HOLE"]
        covered = group.loc[group["context_class"] == "COVERED"]
        clean = hole.loc[hole["clean_coverage"]]
        disruption_values = pd.to_numeric(
            covered.loc[
                covered["covered_disruption"],
                "horizon_direct_delta_log_capital",
            ],
            errors="coerce",
        ).dropna()
        disruption_cost = (
            float(-disruption_values.sum()) if len(disruption_values) else 0.0
        )
        clean_values = pd.to_numeric(
            clean["horizon_direct_delta_log_capital"],
            errors="coerce",
        ).dropna()
        clean_gain = float(clean_values.sum()) if len(clean_values) else 0.0
        output.append(
            {
                "candidate": str(candidate),
                "universe_hole_rows": int(len(hole)),
                "hole_fill_events": int(hole["fills_universe_hole"].sum()),
                "hole_fill_with_capital_gain": int(
                    (
                        hole["fills_universe_hole"]
                        & hole["capital_improved"]
                    ).sum()
                ),
                "clean_coverage_events": int(hole["clean_coverage"].sum()),
                "covered_disruption_events": int(
                    covered["covered_disruption"].sum()
                ),
                "indirect_perturbation_events": int(
                    group["perturbation_before_candidate_selection"].sum()
                ),
                "clean_coverage_log_gain": clean_gain,
                "covered_disruption_cost_log": disruption_cost,
                "table_improvement_log": float(clean_gain - disruption_cost),
                "table_improvement_multiplier": float(
                    math.exp(clean_gain - disruption_cost)
                ),
            }
        )
    output.sort(
        key=lambda row: (
            row["clean_coverage_events"],
            row["table_improvement_log"],
        ),
        reverse=True,
    )
    return output


def _horizon_summary(
    frame: pd.DataFrame,
    horizon_sessions: int,
) -> dict[str, Any]:
    horizon = frame.loc[
        frame["horizon_sessions"] == int(horizon_sessions)
    ].copy()
    if horizon.empty:
        return {
            "horizon_sessions": int(horizon_sessions),
            "contexts": 0,
            "context_counts": {},
            "universe_holes": 0,
            "policy_misses": 0,
            "covered_contexts": 0,
            "unknown_contexts": 0,
            "holes_with_any_candidate_fill": 0,
            "holes_with_any_clean_coverage": 0,
            "candidate_summary": [],
        }

    context_columns = [
        "decision_date",
        "context_class",
        "baseline_strategy_log_gain",
        "best_incumbent",
        "best_incumbent_log_gain",
    ]
    contexts = horizon[context_columns].drop_duplicates(
        ["decision_date"],
        keep="first",
    )
    counts = {
        str(key): int(value)
        for key, value in contexts["context_class"].value_counts().items()
    }
    hole_rows = horizon.loc[
        horizon["context_class"] == "UNIVERSE_HOLE"
    ]
    fill_by_date = (
        hole_rows.groupby("decision_date")["fills_universe_hole"].any()
        if not hole_rows.empty
        else pd.Series(dtype=bool)
    )
    clean_by_date = (
        hole_rows.groupby("decision_date")["clean_coverage"].any()
        if not hole_rows.empty
        else pd.Series(dtype=bool)
    )
    indirect_contexts = horizon.groupby("decision_date")[
        "perturbation_before_candidate_selection"
    ].any()

    return {
        "horizon_sessions": int(horizon_sessions),
        "contexts": int(len(contexts)),
        "context_counts": counts,
        "universe_holes": int(counts.get("UNIVERSE_HOLE", 0)),
        "policy_misses": int(counts.get("POLICY_MISS", 0)),
        "covered_contexts": int(counts.get("COVERED", 0)),
        "unknown_contexts": int(counts.get("UNKNOWN", 0)),
        "holes_with_any_candidate_fill": int(fill_by_date.sum()),
        "holes_with_any_clean_coverage": int(clean_by_date.sum()),
        "candidate_pairs_with_indirect_perturbation": int(
            horizon["perturbation_before_candidate_selection"].sum()
        ),
        "contexts_with_any_indirect_perturbation": int(
            indirect_contexts.sum()
        ),
        "candidate_summary": _candidate_summary(horizon),
    }


def _decomposition_summary(
    dataset: pd.DataFrame,
    universe_name: str,
) -> dict[str, Any]:
    frame = dataset.loc[
        dataset["universe_name"] == universe_name
    ].copy()
    by_horizon = {
        str(horizon): _horizon_summary(frame, horizon)
        for horizon in HOLE_HORIZONS
    }
    primary = by_horizon[str(PRIMARY_HOLE_HORIZON)]
    return {
        "universe": universe_name,
        "primary_horizon_sessions": PRIMARY_HOLE_HORIZON,
        "primary_horizon": primary,
        "by_horizon": by_horizon,
    }


def opportunity_status(primary: dict[str, Any]) -> str:
    horizon = primary.get("primary_horizon", primary)
    holes = int(horizon.get("universe_holes", 0))
    if holes <= 0:
        return "NO_UNIVERSE_HOLES_FOUND"
    if int(horizon.get("holes_with_any_candidate_fill", 0)) <= 0:
        return "TESTED_CANDIDATES_DO_NOT_FILL_HOLES"
    if int(horizon.get("holes_with_any_clean_coverage", 0)) <= 0:
        return "COVERAGE_FOUND_BUT_NOT_CLEAN"
    return "CLEAN_OPPORTUNITY_COVERAGE_OBSERVED"


def _log_primary_contexts(dataset: pd.DataFrame) -> None:
    primary = dataset.loc[
        (dataset["universe_name"] == PRIMARY)
        & (dataset["horizon_sessions"] == PRIMARY_HOLE_HORIZON)
    ].copy()
    for decision_date, group in primary.groupby("decision_date", sort=True):
        first = group.iloc[0]
        protocol.live.console_log(
            f"[hole-state] {protocol.live.display_date(decision_date)} | "
            f"{PRIMARY} | horizon={PRIMARY_HOLE_HORIZON} | "
            f"class={first['context_class']} | "
            f"baseline={float(first['baseline_strategy_log_gain']):+.4f} | "
            f"best={first['best_incumbent']} "
            f"{float(first['best_incumbent_log_gain']):+.4f}"
        )
        if first["context_class"] != "UNIVERSE_HOLE":
            continue
        interesting = group.loc[
            group["fills_universe_hole"]
            | group["capital_improved"]
            | group["perturbation_before_candidate_selection"]
        ]
        for _, row in interesting.iterrows():
            protocol.live.console_log(
                f"[hole-candidate] {protocol.live.display_date(decision_date)} | "
                f"{row['candidate']} | "
                f"future={float(row['candidate_forward_log_gain']):+.4f} | "
                f"direct={float(row['horizon_direct_delta_log_capital']):+.4f} | "
                f"selected={'yes' if bool(row['candidate_ever_selected']) else 'no'} | "
                f"pre-perturb={'yes' if bool(row['perturbation_before_candidate_selection']) else 'no'} | "
                f"clean={'yes' if bool(row['clean_coverage']) else 'no'}"
            )


def _log_horizon_summaries(
    label: str,
    summary: dict[str, Any],
) -> None:
    for horizon in HOLE_HORIZONS:
        metrics = summary["by_horizon"][str(horizon)]
        protocol.live.console_log(
            f"[hole-summary] {label} | horizon={horizon} | "
            f"covered={metrics['covered_contexts']} | "
            f"policy_miss={metrics['policy_misses']} | "
            f"universe_hole={metrics['universe_holes']} | "
            f"fillable={metrics['holes_with_any_candidate_fill']} | "
            f"clean={metrics['holes_with_any_clean_coverage']}"
        )


def main(*, script_version: str) -> int:
    protocol._install_console_logging()
    args = _parser().parse_args()
    protocol.load_project_environment(args.env_file)

    from market_cycle_trader_api.engine import market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri, database_name = storage.runtime_mongo_settings(
        args,
        mongo_repository,
        protocol,
    )
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        retryWrites=False,
    )
    try:
        storage.mongo_retry(lambda: client.admin.command("ping"))
        db = client[database_name]
        strategy = protocol._strategy_document(
            db,
            args.strategy_sequence,
            args.strategy_id,
        )
        strategy_id = str(strategy["_id"])
        config = BacktestRequest.model_validate(
            protocol._configuration(strategy)
        ).model_copy(
            update={
                "end_date": protocol.SNAPSHOT_END,
                "research_market_data_mode": "database_only",
                "mongo_cache_enabled": True,
                "market_data_require_complete_history": True,
            }
        )

        expected = (
            len(protocol.DECISION_DATES)
            * len(protocol.UNIVERSES)
            * len(protocol.CANDIDATES)
        )
        source_run = db[protocol.RUNS_COLLECTION].find_one(
            {
                "experiment": protocol.EXPERIMENT_NAME,
                "strategy_id": strategy_id,
                "status": "completed",
                "completed_observations": {"$gte": expected},
            },
            sort=[("completed_utc", -1), ("updated_utc", -1)],
        )
        if not source_run:
            raise RuntimeError("No completed 23x2x7 Mongo campaign found.")

        source_run_id = str(source_run["_id"])
        observations = pd.DataFrame(
            list(
                db[protocol.OBSERVATIONS_COLLECTION].find(
                    {"run_id": source_run_id},
                    {"_id": 0},
                )
            )
        )
        required_columns = {
            "decision_date",
            "horizon_end",
            "universe_name",
            "candidate",
            TARGET,
        }
        if len(observations) != expected or not required_columns.issubset(
            observations.columns
        ):
            raise RuntimeError(
                f"Expected {expected} observations with "
                f"{sorted(required_columns)}."
            )

        protocol.live.console_log(
            f"[hole] source_run={source_run_id} | "
            f"observations={len(observations)} | "
            "loading persisted baseline/policy traces"
        )
        traces = _load_prediction_traces(db, source_run_id)
        required_trace_keys = (
            len(protocol.DECISION_DATES) * len(protocol.UNIVERSES)
            + expected
        )
        if len(traces) < required_trace_keys:
            protocol.live.console_log(
                f"[hole] trace keys loaded={len(traces)} | "
                f"minimum expected={required_trace_keys}"
            )

        bars, _ = protocol._load_market_frames(config, market_data)
        dataset = build_opportunity_hole_dataset(
            observations=observations,
            bars=bars,
            traces=traces,
            config=config,
        )

        primary_summary = _decomposition_summary(dataset, PRIMARY)
        robust_summary = _decomposition_summary(dataset, ROBUST)
        decision = opportunity_status(primary_summary)

        _log_primary_contexts(dataset)
        _log_horizon_summaries(PRIMARY, primary_summary)
        _log_horizon_summaries(ROBUST, robust_summary)
        protocol.live.console_log(f"[decision] {decision}")

        tournament_id = storage.canonical_hash(
            {
                "source": source_run_id,
                "version": script_version,
                "experiment": EXPERIMENT_NAME,
                "hole_horizons": list(HOLE_HORIZONS),
                "primary_hole_horizon": PRIMARY_HOLE_HORIZON,
                "context_rule": (
                    "COVERED if baseline strategy log gain > 0; "
                    "POLICY_MISS if baseline <= 0 and best incumbent > 0; "
                    "UNIVERSE_HOLE otherwise"
                ),
                "clean_coverage_rule": (
                    "hole + candidate forward gain > 0 + direct capital effect > 0 "
                    "+ candidate selected + no policy divergence before candidate selection"
                ),
            }
        )
        summary = {
            "_id": tournament_id,
            "schema_version": 3,
            "script_version": script_version,
            "experiment": EXPERIMENT_NAME,
            "status": "completed",
            "source_run_id": source_run_id,
            "strategy_id": strategy_id,
            "target": TARGET,
            "rows": int(len(dataset)),
            "independent_primary_temporal_states": int(
                dataset.loc[
                    dataset["universe_name"] == PRIMARY,
                    "decision_date",
                ].nunique()
            ),
            "analysis_type": "retrospective_mechanism_decomposition",
            "hole_horizons_sessions": list(HOLE_HORIZONS),
            "primary_hole_horizon_sessions": PRIMARY_HOLE_HORIZON,
            "context_definition": {
                "COVERED": (
                    "Baseline strategy compounded positively over the evaluated "
                    "5/10/20/40-session research window."
                ),
                "POLICY_MISS": (
                    "Baseline strategy did not compound positively, while at "
                    "least one incumbent asset had positive net forward return."
                ),
                "UNIVERSE_HOLE": (
                    "Baseline strategy did not compound positively and no "
                    "incumbent asset had positive net forward return."
                ),
            },
            "clean_coverage_definition": (
                "Candidate has positive net forward return in a UNIVERSE_HOLE, "
                "its insertion improves final capital, the candidate is actually "
                "selected by the policy, and the selected-asset path does not "
                "diverge from baseline before that candidate selection."
            ),
            "primary": primary_summary,
            "robustness": robust_summary,
            "decision_status": decision,
            "predictive_claim": False,
            "next_step": (
                "This experiment only tests whether the opportunity-hole / "
                "clean-coverage mechanism exists retrospectively across "
                "5/10/20/40 sessions, with 20 sessions as the primary "
                "one-trading-month horizon. If clean coverage is observed, pre-register a separate predictive "
                "experiment using only pre-decision information to forecast "
                "future holes and candidate coverage. If POLICY_MISS dominates, "
                "prioritize policy selection instead of adding assets."
            ),
            "created_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
        }
        storage.mongo_retry(
            lambda: db[COLLECTION].replace_one(
                {"_id": tournament_id},
                storage.mongo_value(summary),
                upsert=True,
            )
        )

        export_summary = {
            key: value for key, value in summary.items() if key != "_id"
        }
        export_dir, zip_path = storage.export_result(
            project_root=protocol.PROJECT_ROOT,
            run_id=tournament_id,
            script_version=script_version,
            dataset=dataset,
            summary=export_summary,
            export_folder_name=protocol.EXPORT_FOLDER_NAME,
            export_zip_name=protocol.EXPORT_ZIP_NAME,
            collections=(
                protocol.RUNS_COLLECTION,
                protocol.OBSERVATIONS_COLLECTION,
                protocol.TRACE_RUNS_COLLECTION,
                protocol.TRACE_ROWS_COLLECTION,
                COLLECTION,
            ),
        )
        protocol.live.console_log(
            f"[complete] tournament persisted | collection={COLLECTION}"
        )
        protocol.live.console_log(
            f"[complete] export folder={export_dir}"
        )
        protocol.live.console_log(f"[complete] ZIP={zip_path}")
        protocol.live.console_log(f"[complete] decision={decision}")
        return 0
    finally:
        client.close()
