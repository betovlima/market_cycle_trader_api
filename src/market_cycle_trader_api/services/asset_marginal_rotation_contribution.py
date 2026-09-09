from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarginalCandidateEvaluation:
    step: int
    candidate: str
    current_universe_size: int
    displacement_event_count: int
    displacement_fold_count: int
    marginal_forward_net_log_return_sum: float
    marginal_forward_net_log_return_mean: float
    marginal_forward_net_log_return_median: float
    marginal_positive_event_rate: float
    marginal_realized_utility_sum: float
    marginal_realized_utility_mean: float
    predicted_score_gap_mean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "candidate": str(self.candidate),
            "current_universe_size": int(self.current_universe_size),
            "displacement_event_count": int(self.displacement_event_count),
            "displacement_fold_count": int(self.displacement_fold_count),
            "marginal_forward_net_log_return_sum": float(
                self.marginal_forward_net_log_return_sum
            ),
            "marginal_forward_net_log_return_mean": float(
                self.marginal_forward_net_log_return_mean
            ),
            "marginal_forward_net_log_return_median": float(
                self.marginal_forward_net_log_return_median
            ),
            "marginal_positive_event_rate": float(self.marginal_positive_event_rate),
            "marginal_realized_utility_sum": float(self.marginal_realized_utility_sum),
            "marginal_realized_utility_mean": float(self.marginal_realized_utility_mean),
            "predicted_score_gap_mean": float(self.predicted_score_gap_mean),
        }


@dataclass
class MarginalRotationSelectionResult:
    baseline_assets: list[str]
    selected_candidates: list[str]
    final_assets: list[str]
    selected_steps: pd.DataFrame
    all_evaluations: pd.DataFrame
    selected_events: pd.DataFrame


def _parse_utc_timestamps(values: pd.Series) -> pd.Series:
    raw = values.copy()
    try:
        parsed = pd.to_datetime(raw, utc=True, format="mixed", errors="coerce")
    except TypeError:
        # Compatibility fallback for older pandas versions that predate format="mixed".
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")

    invalid = raw.notna() & parsed.isna()
    if invalid.any():
        samples = raw.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise RuntimeError(
            "Leadership predictions contain invalid timestamps after mixed-ISO parsing: "
            + ", ".join(samples)
        )
    return parsed


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold",
        "timestamp",
        "symbol",
        "predicted_utility",
        "realized_utility",
        "forward_net_log_return",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RuntimeError(
            "Leadership predictions are missing required columns: " + ", ".join(missing)
        )
    frame = predictions[list(required)].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["timestamp"] = _parse_utc_timestamps(frame["timestamp"])
    frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce").astype("Int64")
    for column in (
        "predicted_utility",
        "realized_utility",
        "forward_net_log_return",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["fold", "timestamp", "symbol", "predicted_utility"]
    )
    frame["fold"] = frame["fold"].astype(int)
    return (
        frame.sort_values(["fold", "timestamp", "symbol"])
        .drop_duplicates(subset=["fold", "timestamp", "symbol"], keep="last")
        .reset_index(drop=True)
    )


def _metric_pivots(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = ["fold", "timestamp"]
    predicted = frame.pivot(
        index=index,
        columns="symbol",
        values="predicted_utility",
    ).sort_index()
    realized_utility = frame.pivot(
        index=index,
        columns="symbol",
        values="realized_utility",
    ).reindex(predicted.index)
    forward_net = frame.pivot(
        index=index,
        columns="symbol",
        values="forward_net_log_return",
    ).reindex(predicted.index)
    return predicted, realized_utility, forward_net


def _leader_state(
    predicted: pd.DataFrame,
    realized_utility: pd.DataFrame,
    forward_net: pd.DataFrame,
    selected_assets: list[str],
) -> pd.DataFrame:
    available = [symbol for symbol in selected_assets if symbol in predicted.columns]
    if not available:
        raise RuntimeError("None of the current-universe assets have OOS predictions.")
    values = predicted[available].to_numpy(dtype=float)
    finite = np.isfinite(values)
    valid = finite.any(axis=1)
    safe = np.where(finite, values, -np.inf)
    argmax = np.argmax(safe, axis=1)
    rows = np.arange(len(predicted))
    leader_symbol = np.empty(len(predicted), dtype=object)
    leader_symbol[:] = None
    leader_predicted = np.full(len(predicted), np.nan)
    leader_utility = np.full(len(predicted), np.nan)
    leader_forward = np.full(len(predicted), np.nan)
    if valid.any():
        vr = rows[valid]
        vc = argmax[valid]
        names = np.asarray(available, dtype=object)
        leader_symbol[valid] = names[vc]
        leader_predicted[valid] = values[vr, vc]
        utility_values = realized_utility[available].to_numpy(dtype=float)
        forward_values = forward_net[available].to_numpy(dtype=float)
        leader_utility[valid] = utility_values[vr, vc]
        leader_forward[valid] = forward_values[vr, vc]
    return pd.DataFrame(
        {
            "leader_symbol": leader_symbol,
            "leader_predicted_utility": leader_predicted,
            "leader_realized_utility": leader_utility,
            "leader_forward_net_log_return": leader_forward,
        },
        index=predicted.index,
    )


def _candidate_events(
    *,
    step: int,
    candidate: str,
    selected_assets: list[str],
    predicted: pd.DataFrame,
    realized_utility: pd.DataFrame,
    forward_net: pd.DataFrame,
) -> tuple[MarginalCandidateEvaluation, pd.DataFrame]:
    leader = _leader_state(
        predicted,
        realized_utility,
        forward_net,
        selected_assets,
    )
    if candidate not in predicted.columns:
        return (
            MarginalCandidateEvaluation(
                step,
                candidate,
                len(selected_assets),
                0,
                0,
                float("-inf"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ),
            pd.DataFrame(),
        )
    candidate_predicted = pd.to_numeric(predicted[candidate], errors="coerce")
    candidate_forward = pd.to_numeric(forward_net[candidate], errors="coerce")
    candidate_utility = pd.to_numeric(realized_utility[candidate], errors="coerce")
    displaced = pd.Series(
        np.isfinite(candidate_predicted)
        & np.isfinite(leader["leader_predicted_utility"])
        & (candidate_predicted > leader["leader_predicted_utility"]),
        index=predicted.index,
        dtype=bool,
    )
    previous = displaced.groupby(level=0).shift(1, fill_value=False).astype(bool)
    event_start = displaced & ~previous
    events = leader.loc[event_start].copy()
    if events.empty:
        return (
            MarginalCandidateEvaluation(
                step,
                candidate,
                len(selected_assets),
                0,
                0,
                0.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ),
            pd.DataFrame(),
        )
    events["candidate"] = candidate
    events["candidate_predicted_utility"] = candidate_predicted.loc[event_start]
    events["candidate_realized_utility"] = candidate_utility.loc[event_start]
    events["candidate_forward_net_log_return"] = candidate_forward.loc[event_start]
    events["predicted_score_gap"] = (
        events["candidate_predicted_utility"] - events["leader_predicted_utility"]
    )
    events["marginal_realized_utility"] = (
        events["candidate_realized_utility"] - events["leader_realized_utility"]
    )
    events["marginal_forward_net_log_return"] = (
        events["candidate_forward_net_log_return"]
        - events["leader_forward_net_log_return"]
    )
    events = events.reset_index()
    finite_net = np.isfinite(
        pd.to_numeric(events["marginal_forward_net_log_return"], errors="coerce")
    )
    events = events.loc[finite_net].copy()
    if events.empty:
        return (
            MarginalCandidateEvaluation(
                step,
                candidate,
                len(selected_assets),
                0,
                0,
                0.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ),
            events,
        )
    net = pd.to_numeric(events["marginal_forward_net_log_return"], errors="coerce")
    utility = pd.to_numeric(events["marginal_realized_utility"], errors="coerce")
    gap = pd.to_numeric(events["predicted_score_gap"], errors="coerce")
    fu = utility[np.isfinite(utility)]
    fg = gap[np.isfinite(gap)]
    evaluation = MarginalCandidateEvaluation(
        step=step,
        candidate=candidate,
        current_universe_size=len(selected_assets),
        displacement_event_count=int(len(events)),
        displacement_fold_count=int(events["fold"].nunique()),
        marginal_forward_net_log_return_sum=float(net.sum()),
        marginal_forward_net_log_return_mean=float(net.mean()),
        marginal_forward_net_log_return_median=float(net.median()),
        marginal_positive_event_rate=float((net > 0.0).mean()),
        marginal_realized_utility_sum=(
            float(fu.sum()) if not fu.empty else float("nan")
        ),
        marginal_realized_utility_mean=(
            float(fu.mean()) if not fu.empty else float("nan")
        ),
        predicted_score_gap_mean=float(fg.mean()) if not fg.empty else float("nan"),
    )
    return evaluation, events


def greedy_marginal_rotation_selection(
    *,
    predictions: pd.DataFrame,
    baseline_assets: list[str],
    candidate_assets: list[str],
) -> MarginalRotationSelectionResult:
    frame = _normalize_predictions(predictions)
    predicted, realized_utility, forward_net = _metric_pivots(frame)
    baseline = list(
        dict.fromkeys(
            str(s).strip().upper()
            for s in baseline_assets
            if str(s).strip()
        )
    )
    if len(baseline) < 2:
        raise RuntimeError(
            "Marginal rotation selection requires at least two baseline assets."
        )
    missing = [s for s in baseline if s not in predicted.columns]
    if missing:
        raise RuntimeError(
            "Baseline assets are missing from leadership predictions: "
            + ", ".join(missing)
        )
    baseline_set = set(baseline)
    remaining = sorted(
        {
            str(s).strip().upper()
            for s in candidate_assets
            if str(s).strip() and str(s).strip().upper() not in baseline_set
        }
    )
    selected_assets = list(baseline)
    selected_candidates: list[str] = []
    evaluations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    selected_events: list[pd.DataFrame] = []
    step = 1
    while remaining:
        results = [
            _candidate_events(
                step=step,
                candidate=c,
                selected_assets=selected_assets,
                predicted=predicted,
                realized_utility=realized_utility,
                forward_net=forward_net,
            )
            for c in remaining
        ]
        evaluations.extend(e.as_dict() for e, _ in results)
        best_evaluation, best_events = sorted(
            results,
            key=lambda item: (
                -float(item[0].marginal_forward_net_log_return_sum),
                item[0].candidate,
            ),
        )[0]
        best_score = float(best_evaluation.marginal_forward_net_log_return_sum)
        # Zero is the economic indifference point in additive log-return space.
        if not np.isfinite(best_score) or best_score <= 0.0:
            break
        selected_candidates.append(best_evaluation.candidate)
        selected_assets.append(best_evaluation.candidate)
        steps.append(
            {
                **best_evaluation.as_dict(),
                "selected": True,
                "resulting_universe_size": len(selected_assets),
            }
        )
        if not best_events.empty:
            current = best_events.copy()
            current["selection_step"] = step
            current["selected_candidate"] = best_evaluation.candidate
            selected_events.append(current)
        remaining.remove(best_evaluation.candidate)
        step += 1
    return MarginalRotationSelectionResult(
        baseline_assets=baseline,
        selected_candidates=selected_candidates,
        final_assets=selected_assets,
        selected_steps=pd.DataFrame(steps),
        all_evaluations=pd.DataFrame(evaluations),
        selected_events=(
            pd.concat(selected_events, ignore_index=True)
            if selected_events
            else pd.DataFrame()
        ),
    )
