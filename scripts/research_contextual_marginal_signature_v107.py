from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.7"
EXPERIMENT_NAME = "contextual_marginal_signature_cluster_aware_universe_analysis"
SIGN_TOLERANCE = 1e-12
CONTEXT_CONSTANT_TOLERANCE = 1e-12

MODEL_FEATURES = [
    "relative__return_5",
    "relative__return_20",
    "relative__return_60",
    "relative__vol_20",
    "relative__vol_60",
    "relative__ema_distance_20",
    "relative__trend_efficiency_20",
    "relative__momentum_acceleration_5_20",
    "corr_20_to_universe",
    "corr_60_to_universe",
    "universe_dispersion_return_20",
    "universe_dispersion_vol_20",
    "candidate_rank_return_20",
    "candidate_rank_return_60",
    "market_return_20",
    "market_vol_20",
]

CONTEXT_FEATURES = [
    "relative__return_5",
    "relative__return_20",
    "relative__return_60",
    "relative__vol_20",
    "relative__vol_60",
    "relative__ema_distance_20",
    "relative__trend_efficiency_20",
    "relative__momentum_acceleration_5_20",
    "universe_dispersion_return_20",
    "universe_dispersion_vol_20",
    "market_return_20",
    "market_vol_20",
]

CANDIDATE_SPECIFIC_FEATURES = [
    "corr_20_to_universe",
    "corr_60_to_universe",
    "candidate_rank_return_20",
    "candidate_rank_return_60",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster-aware analysis of the v1.0.6 multi-universe counterfactual panel. "
            "No replay is executed. The analysis separates universe-date context effects "
            "from candidate-specific effects and avoids treating dependent pairwise rows "
            "as independent evidence."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to multi_universe_contextual_dataset.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=17)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _effect_sign(value: float) -> str:
    number = float(value)
    if abs(number) <= SIGN_TOLERANCE:
        return "zero"
    return "positive" if number > 0 else "negative"


def _load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "universe_name", "universe_size", "decision_date", "horizon_end",
        "candidate", "evaluation_status", "delta_log_capital", *MODEL_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Dataset is missing required columns: " + ", ".join(missing))
    completed = frame.loc[frame["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    if completed.empty:
        raise RuntimeError("Dataset contains no completed counterfactual rows.")
    completed["decision_ts"] = pd.to_datetime(completed["decision_date"], utc=True, errors="raise")
    completed["horizon_ts"] = pd.to_datetime(completed["horizon_end"], utc=True, errors="raise")
    completed["effect_sign"] = pd.to_numeric(completed["delta_log_capital"], errors="raise").map(_effect_sign)
    return completed.sort_values(["decision_ts", "universe_name", "candidate"]).reset_index(drop=True)


def build_pairwise(dataset: pd.DataFrame) -> pd.DataFrame:
    universe_order = list(dict.fromkeys(dataset["universe_name"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for (decision_date, candidate), group in dataset.groupby(["decision_date", "candidate"], sort=True):
        indexed = {str(row["universe_name"]): row for _, row in group.iterrows()}
        if any(name not in indexed for name in universe_order):
            continue
        for i, left_name in enumerate(universe_order):
            for right_name in universe_order[i + 1:]:
                left = indexed[left_name]
                right = indexed[right_name]
                left_y = float(left["delta_log_capital"])
                right_y = float(right["delta_log_capital"])
                left_sign = _effect_sign(left_y)
                right_sign = _effect_sign(right_y)
                row: dict[str, Any] = {
                    "decision_date": decision_date,
                    "candidate": candidate,
                    "universe_left": left_name,
                    "universe_right": right_name,
                    "pair_key": f"{left_name} -> {right_name}",
                    "left_delta_log_capital": left_y,
                    "right_delta_log_capital": right_y,
                    "delta_delta_log_capital": right_y - left_y,
                    "left_effect_sign": left_sign,
                    "right_effect_sign": right_sign,
                    "sign_changed": left_sign != right_sign,
                    "strict_sign_flip": {left_sign, right_sign} == {"positive", "negative"},
                    "activation_changed": (left_sign == "zero") != (right_sign == "zero"),
                }
                for feature in MODEL_FEATURES:
                    left_value = pd.to_numeric(pd.Series([left.get(feature)]), errors="coerce").iloc[0]
                    right_value = pd.to_numeric(pd.Series([right.get(feature)]), errors="coerce").iloc[0]
                    row[f"delta__{feature}"] = (
                        float(right_value - left_value)
                        if pd.notna(left_value) and pd.notna(right_value)
                        else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_context_states(dataset: pd.DataFrame) -> pd.DataFrame:
    aggregation: dict[str, Any] = {
        "mean_marginal_delta_log": ("delta_log_capital", "mean"),
        "median_marginal_delta_log": ("delta_log_capital", "median"),
        "positive_rate": ("delta_log_capital", lambda s: float((pd.to_numeric(s) > SIGN_TOLERANCE).mean())),
        "negative_rate": ("delta_log_capital", lambda s: float((pd.to_numeric(s) < -SIGN_TOLERANCE).mean())),
        "zero_rate": ("delta_log_capital", lambda s: float((pd.to_numeric(s).abs() <= SIGN_TOLERANCE).mean())),
        "candidate_count": ("candidate", "nunique"),
        "horizon_end": ("horizon_end", "first"),
    }
    for feature in CONTEXT_FEATURES:
        aggregation[f"context__{feature}"] = (feature, "mean")
    states = dataset.groupby(["decision_date", "universe_name"], as_index=False).agg(**aggregation)
    states["decision_ts"] = pd.to_datetime(states["decision_date"], utc=True)
    states["horizon_ts"] = pd.to_datetime(states["horizon_end"], utc=True)
    return states.sort_values(["decision_ts", "universe_name"]).reset_index(drop=True)


def _within_date_demean(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values - values.groupby(frame["decision_date"]).transform("mean")


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return None if pd.isna(value) else float(value)


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson")
    return None if pd.isna(value) else float(value)


def _permutation_p_within_date(
    states: pd.DataFrame,
    feature_column: str,
    *,
    permutations: int,
    random_state: int,
) -> float | None:
    x_dm = _within_date_demean(states, feature_column)
    y_dm = _within_date_demean(states, "mean_marginal_delta_log")
    pair = pd.concat([x_dm.rename("x"), y_dm.rename("y")], axis=1).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None

    indices = pair.index.to_numpy()
    x = pair["x"].to_numpy(dtype=float)
    y = pair["y"].to_numpy(dtype=float)
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    denominator = float(np.sqrt(np.sum(rx_centered ** 2) * np.sum(ry_centered ** 2)))
    if denominator <= 0:
        return None
    observed = float(np.dot(rx_centered, ry_centered) / denominator)

    n_permutations = max(1, int(permutations))
    rng = np.random.default_rng(int(random_state))
    permuted = np.tile(y, (n_permutations, 1))
    position_by_index = {index: position for position, index in enumerate(indices)}
    for _, group in states.loc[indices].groupby("decision_date", sort=True):
        positions = np.array([position_by_index[index] for index in group.index], dtype=int)
        if len(positions) <= 1:
            continue
        orders = np.argsort(rng.random((n_permutations, len(positions))), axis=1)
        permuted[:, positions] = y[positions][orders]

    ranked = rankdata(permuted, method="average", axis=1)
    ranked_centered = ranked - ranked.mean(axis=1, keepdims=True)
    numerators = ranked_centered @ rx_centered
    denominators = np.sqrt(
        np.sum(ranked_centered ** 2, axis=1) * np.sum(rx_centered ** 2)
    )
    valid = denominators > 0
    correlations = np.full(n_permutations, np.nan, dtype=float)
    correlations[valid] = numerators[valid] / denominators[valid]
    usable = correlations[np.isfinite(correlations)]
    if not len(usable):
        return None
    exceed = int(np.sum(np.abs(usable) >= abs(observed) - 1e-15))
    return float((exceed + 1) / (len(usable) + 1))


def context_feature_report(
    states: pd.DataFrame,
    *,
    permutations: int,
    random_state: int,
) -> pd.DataFrame:
    y_dm = _within_date_demean(states, "mean_marginal_delta_log")
    dates = sorted(states["decision_date"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for position, feature in enumerate(CONTEXT_FEATURES):
        column = f"context__{feature}"
        x_dm = _within_date_demean(states, column)
        leave_one_date_out: list[float] = []
        for date in dates:
            mask = states["decision_date"].astype(str) != date
            value = _spearman(x_dm.loc[mask], y_dm.loc[mask])
            if value is not None:
                leave_one_date_out.append(value)
        rows.append({
            "feature": feature,
            "universe_date_states": int(len(states)),
            "decision_dates": int(len(dates)),
            "within_date_pearson": _pearson(x_dm, y_dm),
            "within_date_spearman": _spearman(x_dm, y_dm),
            "permutation_p_two_sided": _permutation_p_within_date(
                states, column,
                permutations=permutations,
                random_state=random_state + position,
            ),
            "leave_one_date_out_min_spearman": (
                float(min(leave_one_date_out)) if leave_one_date_out else None
            ),
            "leave_one_date_out_max_spearman": (
                float(max(leave_one_date_out)) if leave_one_date_out else None
            ),
            "leave_one_date_out_mean_spearman": (
                float(np.mean(leave_one_date_out)) if leave_one_date_out else None
            ),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        "within_date_spearman",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    )


def candidate_specific_feature_report(pairwise: pd.DataFrame) -> pd.DataFrame:
    frame = pairwise.copy()
    context_keys = [frame["decision_date"], frame["pair_key"]]
    y = pd.to_numeric(frame["delta_delta_log_capital"], errors="coerce")
    y_within = y - y.groupby(context_keys).transform("mean")
    rows: list[dict[str, Any]] = []
    for feature in CANDIDATE_SPECIFIC_FEATURES:
        column = f"delta__{feature}"
        x = pd.to_numeric(frame[column], errors="coerce")
        x_within = x - x.groupby(context_keys).transform("mean")
        per_context: list[float] = []
        eligible_contexts = 0
        for _, group in frame.groupby(["decision_date", "pair_key"], sort=True):
            gx = pd.to_numeric(group[column], errors="coerce")
            gy = pd.to_numeric(group["delta_delta_log_capital"], errors="coerce")
            value = _spearman(gx, gy)
            if value is not None:
                per_context.append(value)
                eligible_contexts += 1
        rows.append({
            "feature": feature,
            "pairwise_rows": int(len(frame)),
            "eligible_contexts": int(eligible_contexts),
            "within_context_pearson": _pearson(x_within, y_within),
            "within_context_spearman": _spearman(x_within, y_within),
            "mean_context_spearman": float(np.mean(per_context)) if per_context else None,
        })
    return pd.DataFrame(rows).sort_values(
        "within_context_spearman",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    )


def pseudoreplication_report(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        column = f"delta__{feature}"
        ranges = []
        for _, group in pairwise.groupby(["decision_date", "pair_key"], sort=True):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            ranges.append(float(values.max() - values.min()))
        max_range = max(ranges) if ranges else None
        rows.append({
            "feature": feature,
            "contexts": int(len(ranges)),
            "max_within_context_candidate_range": max_range,
            "context_constant_across_candidates": (
                bool(max_range is not None and max_range <= CONTEXT_CONSTANT_TOLERANCE)
            ),
        })
    return pd.DataFrame(rows)


def pair_direction_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (left, right), group in pairwise.groupby(["universe_left", "universe_right"], sort=True):
        y = pd.to_numeric(group["delta_delta_log_capital"], errors="coerce")
        rows.append({
            "universe_left": left,
            "universe_right": right,
            "rows": int(len(group)),
            "dates": int(group["decision_date"].nunique()),
            "candidates": int(group["candidate"].nunique()),
            "improved_rows": int((y > SIGN_TOLERANCE).sum()),
            "worsened_rows": int((y < -SIGN_TOLERANCE).sum()),
            "unchanged_rows": int((y.abs() <= SIGN_TOLERANCE).sum()),
            "sign_change_rows": int(group["sign_changed"].sum()),
            "strict_flip_rows": int(group["strict_sign_flip"].sum()),
            "activation_change_rows": int(group["activation_changed"].sum()),
            "mean_delta_delta_log": float(y.mean()),
            "mean_abs_delta_delta_log": float(y.abs().mean()),
            "monotone_non_improving": bool((y <= SIGN_TOLERANCE).all()),
            "monotone_non_worsening": bool((y >= -SIGN_TOLERANCE).all()),
        })
    return pd.DataFrame(rows)


def sign_episode_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (decision_date, candidate), group in dataset.groupby(["decision_date", "candidate"], sort=True):
        signs = sorted(set(group["effect_sign"].astype(str)))
        if len(signs) <= 1:
            continue
        values = pd.to_numeric(group["delta_log_capital"], errors="coerce")
        rows.append({
            "decision_date": decision_date,
            "candidate": candidate,
            "states": "|".join(signs),
            "universe_count": int(group["universe_name"].nunique()),
            "min_delta_log": float(values.min()),
            "max_delta_log": float(values.max()),
            "range_delta_log": float(values.max() - values.min()),
            "strict_positive_negative_flip": bool("positive" in signs and "negative" in signs),
            "activation_change": bool("zero" in signs),
        })
    return pd.DataFrame(rows)


def non_overlapping_anchor_dates(dataset: pd.DataFrame) -> list[str]:
    dates = (
        dataset[["decision_date", "horizon_end"]]
        .drop_duplicates()
        .assign(
            decision_ts=lambda x: pd.to_datetime(x["decision_date"], utc=True),
            horizon_ts=lambda x: pd.to_datetime(x["horizon_end"], utc=True),
        )
        .sort_values("decision_ts")
    )
    anchors: list[str] = []
    last_horizon: pd.Timestamp | None = None
    for row in dates.itertuples(index=False):
        if last_horizon is None or row.decision_ts >= last_horizon:
            anchors.append(str(row.decision_date))
            last_horizon = row.horizon_ts
    return anchors


def main() -> int:
    args = _parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    pairwise = build_pairwise(dataset)
    states = build_context_states(dataset)
    context_report = context_feature_report(
        states,
        permutations=max(1, int(args.permutations)),
        random_state=int(args.random_state),
    )
    candidate_report = candidate_specific_feature_report(pairwise)
    pseudo = pseudoreplication_report(pairwise)
    directions = pair_direction_summary(pairwise)
    episodes = sign_episode_summary(dataset)
    anchors = non_overlapping_anchor_dates(dataset)

    _write_frame(output_dir / "cluster_aware_context_states.csv", states)
    _write_frame(output_dir / "cluster_aware_context_feature_report.csv", context_report)
    _write_frame(output_dir / "cluster_aware_candidate_specific_report.csv", candidate_report)
    _write_frame(output_dir / "cluster_aware_pseudoreplication_report.csv", pseudo)
    _write_frame(output_dir / "cluster_aware_pair_direction_summary.csv", directions)
    _write_frame(output_dir / "cluster_aware_sign_episodes.csv", episodes)

    universe_count = int(dataset["universe_name"].nunique())
    decision_dates = int(dataset["decision_date"].nunique())
    independent_contrasts_upper_bound = int(decision_dates * max(0, universe_count - 1))
    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "source_dataset": str(dataset_path),
        "completed_rows": int(len(dataset)),
        "decision_dates": decision_dates,
        "universes": universe_count,
        "candidates": int(dataset["candidate"].nunique()),
        "pairwise_rows": int(len(pairwise)),
        "universe_date_states": int(len(states)),
        "date_pair_contexts": int(
            pairwise[["decision_date", "pair_key"]].drop_duplicates().shape[0]
        ),
        "algebraically_independent_contrasts_upper_bound": independent_contrasts_upper_bound,
        "non_overlapping_anchor_dates": anchors,
        "unique_sign_change_episodes": int(len(episodes)),
        "unique_strict_flip_episodes": int(
            episodes["strict_positive_negative_flip"].sum()
        ) if not episodes.empty else 0,
        "unique_activation_change_episodes": int(
            episodes["activation_change"].sum()
        ) if not episodes.empty else 0,
        "top_context_features": context_report.head(8).to_dict(orient="records"),
        "candidate_specific_features": candidate_report.to_dict(orient="records"),
        "pair_direction_summary": directions.to_dict(orient="records"),
        "interpretation_contract": {
            "pairwise_rows_are_not_independent": True,
            "context_feature_unit": "universe x decision_date state, centered within decision_date",
            "candidate_specific_unit": "candidate residual within the same decision_date x universe-pair context",
            "overlapping_horizon_warning": (
                "Monthly decision dates with 40-session horizons overlap; repeated adjacent dates "
                "must not be treated as independent confirmations."
            ),
            "confirmation_rule": (
                "This analysis is diagnostic. A causal/portable signature requires replication "
                "on fresh dates, candidates, or independently designed universes."
            ),
        },
    }
    _write_json(output_dir / "cluster_aware_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
