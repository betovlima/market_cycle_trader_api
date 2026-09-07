from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from market_cycle_trader_api.services.asset_discovery_ranker import FEATURE_COLUMNS

RANKING_POLICY_VERSION = "existing-asset-signature-v1"
RANKING_METRICS: tuple[tuple[str, str], ...] = (
    ("oos_rank_percentile_mean", "maximize"),
    ("oos_top3_frequency", "maximize"),
    ("oos_weak_period_coverage", "maximize"),
    ("state_novelty_fold_median", "maximize"),
    ("return_corr_full_median_abs", "minimize"),
    ("return_corr_full_max_abs", "minimize"),
    ("oos_top3_fold_std", "minimize"),
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frames(collection: Any, symbols: list[str], identity: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    query = {
        "symbol": {"$in": symbols},
        **identity,
        "timestamp": {
            "$gte": start.tz_localize("UTC").to_pydatetime(),
            "$lt": (end + pd.Timedelta(days=1)).tz_localize("UTC").to_pydatetime(),
        },
    }
    projection = {
        "_id": 0, "symbol": 1, "timestamp": 1, "open": 1, "high": 1,
        "low": 1, "close": 1, "volume": 1, "vwap": 1, "trade_count": 1,
    }
    rows = list(collection.find(query, projection).sort([("symbol", 1), ("timestamp", 1)]))
    if not rows:
        return {}
    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    result: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        result[str(symbol)] = frame[~frame.index.duplicated(keep="last")]
    return result


def history_diagnostics(frames: dict[str, pd.DataFrame], symbols: list[str], expected: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            rows.append({
                "symbol": symbol, "observed_rows": 0, "expected_sessions": len(expected),
                "missing_sessions": len(expected), "actual_start": None, "actual_end": None,
                "history_complete": False, "reason": "missing_from_local_mongodb",
            })
            continue
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)).unique()
        missing = expected.difference(observed)
        rows.append({
            "symbol": symbol, "observed_rows": len(frame), "expected_sessions": len(expected),
            "missing_sessions": len(missing), "actual_start": observed.min().date().isoformat(),
            "actual_end": observed.max().date().isoformat(), "history_complete": not len(missing),
            "reason": "complete" if not len(missing) else "incomplete_strategy_history",
        })
    return pd.DataFrame(rows)


def return_correlations(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.concat(
        {symbol: pd.to_numeric(frame["close"], errors="coerce") for symbol, frame in frames.items()},
        axis=1, join="inner",
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns.corr(), returns.tail(60).corr()


def _leaf_map(model: Any, snapshot: pd.DataFrame) -> dict[str, np.ndarray]:
    if snapshot.empty:
        return {}
    ordered = snapshot.sort_values("symbol").drop_duplicates("symbol").reset_index(drop=True)
    leaves = np.asarray(model.predict(ordered[list(FEATURE_COLUMNS)], pred_leaf=True))
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    return {str(symbol): leaves[index] for index, symbol in enumerate(ordered["symbol"].astype(str))}


def _state_similarity(leaves: dict[str, np.ndarray], symbol: str, baseline: set[str]) -> dict[str, Any]:
    candidate = leaves.get(symbol)
    if candidate is None:
        return {"state_similarity": None, "state_novelty": None, "state_nearest_symbol": None}
    pairs = [
        (reference, float(np.mean(candidate == leaves[reference])))
        for reference in sorted(baseline) if reference in leaves
    ]
    if not pairs:
        return {"state_similarity": None, "state_novelty": None, "state_nearest_symbol": None}
    nearest, similarity = max(pairs, key=lambda item: item[1])
    return {"state_similarity": similarity, "state_novelty": 1.0 - similarity, "state_nearest_symbol": nearest}


def _rank_metrics(scored: pd.DataFrame, truth: pd.DataFrame, symbol: str, baseline: set[str], weak_threshold: float) -> dict[str, Any]:
    truth_lookup = {date: group.set_index("symbol")["utility"] for date, group in truth.groupby("date")}
    ranks: list[int] = []
    percentiles: list[float] = []
    weak_count = weak_wins = 0
    oracle_ranks: list[int] = []
    oracle_percentiles: list[float] = []
    for date, group in scored.groupby("date"):
        candidate = group[group["symbol"] == symbol]
        reference = group[group["symbol"].isin(baseline)]
        if candidate.empty or reference.empty:
            continue
        value = float(candidate["score"].iloc[0])
        ref = reference["score"].to_numpy(float)
        rank = 1 + int(np.sum(ref > value))
        ranks.append(rank)
        percentiles.append(float(np.mean(np.append(ref, value) <= value)))
        weak = float(np.max(ref)) <= weak_threshold
        weak_count += int(weak)
        weak_wins += int(weak and rank == 1)
        oracle = truth_lookup.get(date)
        if oracle is not None and symbol in oracle.index:
            actual = float(oracle.loc[symbol])
            ref_actual = oracle.reindex(sorted(baseline)).dropna().to_numpy(float)
            if len(ref_actual):
                oracle_ranks.append(1 + int(np.sum(ref_actual > actual)))
                oracle_percentiles.append(float(np.mean(np.append(ref_actual, actual) <= actual)))
    rank_values = np.asarray(ranks, float)
    oracle_values = np.asarray(oracle_ranks, float)
    return {
        "session_count": len(ranks),
        "rank_percentile_mean": float(np.mean(percentiles)) if percentiles else None,
        "top1_frequency": float(np.mean(rank_values <= 1)) if len(rank_values) else None,
        "top3_frequency": float(np.mean(rank_values <= 3)) if len(rank_values) else None,
        "top5_frequency": float(np.mean(rank_values <= 5)) if len(rank_values) else None,
        "weak_period_coverage": float(weak_wins / weak_count) if weak_count else None,
        "oracle_rank_percentile_mean": float(np.mean(oracle_percentiles)) if oracle_percentiles else None,
        "oracle_top1_frequency": float(np.mean(oracle_values <= 1)) if len(oracle_values) else None,
        "oracle_top3_frequency": float(np.mean(oracle_values <= 3)) if len(oracle_values) else None,
        "oracle_top5_frequency": float(np.mean(oracle_values <= 5)) if len(oracle_values) else None,
    }


def _corr_metrics(symbol: str, full: pd.DataFrame, latest: pd.DataFrame, baseline: set[str]) -> dict[str, Any]:
    def one(matrix: pd.DataFrame, prefix: str) -> dict[str, Any]:
        refs = [item for item in sorted(baseline) if item in matrix.index]
        if symbol not in matrix.columns or not refs:
            return {f"{prefix}_median_abs": None, f"{prefix}_max_abs": None, f"{prefix}_nearest_abs_symbol": None}
        values = matrix.loc[refs, symbol].dropna()
        if values.empty:
            return {f"{prefix}_median_abs": None, f"{prefix}_max_abs": None, f"{prefix}_nearest_abs_symbol": None}
        absolute = values.abs()
        return {
            f"{prefix}_median_abs": float(absolute.median()),
            f"{prefix}_max_abs": float(absolute.max()),
            f"{prefix}_nearest_abs_symbol": str(absolute.idxmax()),
        }
    return {**one(full, "return_corr_full"), **one(latest, "return_corr_60_latest")}


def build_model_payloads(calibration: Any, baseline_training: pd.DataFrame, all_features: pd.DataFrame, all_training: pd.DataFrame, baseline_assets: list[str], candidates: list[str], folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]], latest_date: pd.Timestamp, random_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = set(baseline_assets)
    symbols = baseline | set(candidates)
    relevance = calibration._relevance_for_reference(baseline_training)
    payloads: list[dict[str, Any]] = []
    for fold_number, (train_dates, validation_dates) in enumerate(folds, 1):
        calibration._log(f"Candidate ranking fold {fold_number}/{len(folds)} - fitting baseline model once.")
        train = relevance[relevance["date"].isin(set(train_dates))].copy()
        model = calibration._fit_model(train, random_state + fold_number)
        training_scores = train[["date", "symbol", *FEATURE_COLUMNS]].copy()
        training_scores["score"] = model.predict(training_scores[list(FEATURE_COLUMNS)])
        weak = float(training_scores.groupby("date")["score"].max().quantile(0.25))
        validation_set = set(validation_dates)
        scored = all_features[all_features["date"].isin(validation_set) & all_features["symbol"].isin(symbols)].copy()
        scored["score"] = model.predict(scored[list(FEATURE_COLUMNS)])
        truth = all_training[all_training["date"].isin(validation_set) & all_training["symbol"].isin(symbols)][["date", "symbol", "utility"]].copy()
        fold_end = pd.Timestamp(max(validation_dates))
        snapshot = all_features[(all_features["date"] == fold_end) & all_features["symbol"].isin(symbols)].copy()
        payloads.append({
            "fold": fold_number, "train_start": min(train_dates), "train_end": max(train_dates),
            "validation_start": min(validation_dates), "validation_end": fold_end, "weak": weak,
            "scored": scored[["date", "symbol", "score"]], "truth": truth, "leaves": _leaf_map(model, snapshot),
        })
    final_model = calibration._fit_model(relevance, random_state + 10_000)
    snapshot = all_features[(all_features["date"] == latest_date) & all_features["symbol"].isin(symbols)].copy()
    snapshot["score"] = final_model.predict(snapshot[list(FEATURE_COLUMNS)])
    return payloads, {"scored": snapshot[["date", "symbol", "score"]], "leaves": _leaf_map(final_model, snapshot)}


def analyse_candidate(symbol: str, baseline_assets: list[str], payloads: list[dict[str, Any]], latest: dict[str, Any], full_corr: pd.DataFrame, corr_60: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = set(baseline_assets)
    fold_rows: list[dict[str, Any]] = []
    for payload in payloads:
        metrics = _rank_metrics(payload["scored"], payload["truth"], symbol, baseline, payload["weak"])
        fold_rows.append({
            "symbol": symbol, "fold": payload["fold"],
            "train_start": pd.Timestamp(payload["train_start"]).date().isoformat(),
            "train_end": pd.Timestamp(payload["train_end"]).date().isoformat(),
            "validation_start": pd.Timestamp(payload["validation_start"]).date().isoformat(),
            "validation_end": pd.Timestamp(payload["validation_end"]).date().isoformat(),
            "weak_reference_threshold": payload["weak"], **metrics,
            **_state_similarity(payload["leaves"], symbol, baseline),
        })
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in fold_rows if row.get(name) is not None]
    top3 = values("top3_frequency")
    states = values("state_similarity")
    nearest = [str(row["state_nearest_symbol"]) for row in fold_rows if row.get("state_nearest_symbol")]
    latest_scored = latest["scored"]
    candidate = latest_scored[latest_scored["symbol"] == symbol]
    reference = latest_scored[latest_scored["symbol"].isin(baseline)]
    current = None if candidate.empty else float(candidate["score"].iloc[0])
    latest_rank = None if current is None or reference.empty else 1 + int(np.sum(reference["score"].to_numpy(float) > current))
    latest_percentile = None if current is None or reference.empty else float(np.mean(np.append(reference["score"].to_numpy(float), current) <= current))
    row = {
        "symbol": symbol,
        "oos_session_count": sum(int(item["session_count"]) for item in fold_rows),
        "oos_rank_percentile_mean": float(np.mean(values("rank_percentile_mean"))),
        "oos_top1_frequency": float(np.mean(values("top1_frequency"))),
        "oos_top3_frequency": float(np.mean(top3)),
        "oos_top5_frequency": float(np.mean(values("top5_frequency"))),
        "oos_top3_fold_std": float(np.std(top3, ddof=0)),
        "oos_weak_period_coverage": float(np.mean(values("weak_period_coverage"))),
        "state_similarity_fold_median": float(np.median(states)),
        "state_novelty_fold_median": 1.0 - float(np.median(states)),
        "state_nearest_symbol_fold_mode": max(set(nearest), key=nearest.count) if nearest else None,
        "latest_rank": latest_rank, "latest_rank_percentile": latest_percentile,
        **_state_similarity(latest["leaves"], symbol, baseline),
        **_corr_metrics(symbol, full_corr, corr_60, baseline),
        **{name: float(np.mean(values(name))) if values(name) else None for name in (
            "oracle_rank_percentile_mean", "oracle_top1_frequency", "oracle_top3_frequency", "oracle_top5_frequency"
        )},
    }
    return row, fold_rows


def _empirical_component(value: Any, reference: pd.Series, direction: str) -> float:
    clean = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    if value is None or not len(clean):
        return np.nan
    number = float(value)
    return float(np.mean(clean <= number)) if direction == "maximize" else float(np.mean(clean >= number))


def _pareto_layers(frame: pd.DataFrame, components: list[str]) -> pd.Series:
    values = frame[components].to_numpy(float)
    valid = np.all(np.isfinite(values), axis=1)
    layers = np.full(len(frame), np.nan)
    remaining = set(np.where(valid)[0].tolist())
    layer = 1
    while remaining:
        front = []
        for index in sorted(remaining):
            candidate = values[index]
            dominated = any(
                np.all(values[other] >= candidate) and np.any(values[other] > candidate)
                for other in remaining if other != index
            )
            if not dominated:
                front.append(index)
        if not front:
            break
        for index in front:
            layers[index] = layer
            remaining.remove(index)
        layer += 1
    return pd.Series(layers, index=frame.index)


def rank_candidates(candidates: pd.DataFrame, calibration_frame: pd.DataFrame, selection_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranked = candidates.copy()
    components: list[str] = []
    for metric, direction in RANKING_METRICS:
        component = f"component_{metric}"
        components.append(component)
        ranked[component] = [_empirical_component(value, calibration_frame[metric], direction) for value in ranked[metric]]
    ranked["ranking_complete"] = ranked[components].notna().all(axis=1)
    ranked["signature_score"] = ranked[components].median(axis=1, skipna=False)
    ranked["signature_floor"] = ranked[components].min(axis=1, skipna=False)
    ranked["pareto_layer"] = _pareto_layers(ranked, components)
    ranked = ranked.sort_values(
        ["pareto_layer", "signature_score", "signature_floor", "oos_top3_frequency", "symbol"],
        ascending=[True, False, False, False, True], na_position="last",
    ).reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    if selection_count > 0:
        ranked["selected_for_backtest"] = ranked["ranking_complete"] & (ranked["candidate_rank"] <= selection_count)
        selection_rule = f"top_{selection_count}_after_pareto_and_signature"
    else:
        ranked["selected_for_backtest"] = ranked["ranking_complete"] & (ranked["pareto_layer"] == 1)
        selection_rule = "pareto_front_layer_1"
    policy = {
        "version": RANKING_POLICY_VERSION,
        "metrics": [{"metric": metric, "direction": direction} for metric, direction in RANKING_METRICS],
        "normalization": "empirical percentile against existing Strategy assets",
        "aggregate": "median component percentile",
        "primary_order": "pareto layer, signature score, signature floor",
        "selection_rule": selection_rule,
        "backtest_metrics_used": False,
        "oracle_metrics_used": False,
    }
    return ranked, policy
