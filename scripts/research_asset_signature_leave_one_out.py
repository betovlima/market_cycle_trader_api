from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.uri_parser import parse_uri


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.core.environment import load_project_environment
from market_cycle_trader_api.services.asset_discovery_ranker import (
    FEATURE_COLUMNS,
    INITIAL_TRAIN_FRACTION,
    TARGET_HORIZON,
    VALIDATION_FOLDS,
    _future_utility,
    _group_sizes,
    _new_ranker,
    feature_frame,
)


STRATEGY_PROFILES_COLLECTION = "strategy_profiles"
STRATEGY_CONTROL_COLLECTION = "strategy_control"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
CONTROL_ID = "default"
DEFAULT_STRATEGY_SEQUENCE = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_WORKERS = 4
LOCAL_MONGO_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "mongo",
    "host.docker.internal",
}


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _stable_seed(symbol: str, base_seed: int) -> int:
    digest = hashlib.sha256(str(symbol).upper().encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 1_000_000
    return int(base_seed) + offset


def _normalize_date(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid date: {value}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _assert_local_mongo(uri: str, allow_remote: bool) -> None:
    if allow_remote:
        return
    parsed = parse_uri(uri)
    hosts = {str(host).strip().lower() for host, _port in parsed.get("nodelist") or []}
    if hosts and hosts.issubset(LOCAL_MONGO_HOSTS):
        return
    raise RuntimeError(
        "This research script is intentionally local-MongoDB only. "
        f"Resolved MongoDB hosts={sorted(hosts) or ['unknown']}. "
        "Use a local URI or pass --allow-remote-mongo explicitly."
    )


def _strategy_document(db: Any, strategy_sequence: int, strategy_id: str | None) -> dict[str, Any]:
    query = {"_id": strategy_id} if strategy_id else {"strategy_sequence": int(strategy_sequence)}
    document = db[STRATEGY_PROFILES_COLLECTION].find_one(query)
    if document is None:
        reference = strategy_id or f"Strategy #{strategy_sequence}"
        raise RuntimeError(f"Strategy not found in local MongoDB: {reference}.")
    return document


def _winner_assets(db: Any) -> set[str]:
    control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": CONTROL_ID}) or {}
    winner_id = str(control.get("trader_winner_strategy_id") or "").strip()
    if not winner_id:
        return set()
    winner = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": winner_id}) or {}
    configuration = winner.get("configuration") if isinstance(winner.get("configuration"), dict) else winner
    return {
        str(symbol).strip().upper()
        for symbol in list(configuration.get("assets") or [])
        if str(symbol).strip()
    }


def _strategy_configuration(document: dict[str, Any]) -> dict[str, Any]:
    configuration = document.get("configuration")
    if isinstance(configuration, dict):
        return dict(configuration)
    return dict(document)


def _identity(configuration: dict[str, Any]) -> dict[str, str]:
    return {
        "interval": str(configuration.get("timeframe") or "1Day"),
        "feed": str(configuration.get("alpaca_historical_feed") or "sip"),
        "adjustment": str(configuration.get("alpaca_adjustment") or "all"),
    }


def _latest_common_session(
    collection: Any,
    assets: list[str],
    identity: dict[str, str],
) -> pd.Timestamp:
    latest: list[pd.Timestamp] = []
    missing: list[str] = []
    for symbol in assets:
        document = collection.find_one(
            {"symbol": symbol, **identity},
            {"_id": 0, "timestamp": 1},
            sort=[("timestamp", -1)],
        )
        if not document or document.get("timestamp") is None:
            missing.append(symbol)
            continue
        latest.append(_normalize_date(document["timestamp"]))
    if missing:
        raise RuntimeError(
            "Local MongoDB is missing cached market bars for: " + ", ".join(missing)
        )
    if not latest:
        raise RuntimeError("No market history was found for the selected Strategy.")
    return min(latest)


def _load_frames_once(
    collection: Any,
    assets: list[str],
    identity: dict[str, str],
    start_date: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    start = start_date.tz_localize("UTC").to_pydatetime()
    end_exclusive = (snapshot_end + pd.Timedelta(days=1)).tz_localize("UTC").to_pydatetime()
    query = {
        "symbol": {"$in": assets},
        **identity,
        "timestamp": {"$gte": start, "$lt": end_exclusive},
    }
    projection = {
        "_id": 0,
        "symbol": 1,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "vwap": 1,
        "trade_count": 1,
    }
    rows = list(collection.find(query, projection).sort([("symbol", 1), ("timestamp", 1)]))
    if not rows:
        raise RuntimeError("The local MongoDB query returned no bars for the Strategy universe.")

    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frames[str(symbol)] = frame

    absent = [symbol for symbol in assets if symbol not in frames]
    if absent:
        raise RuntimeError("No in-window rows were loaded for: " + ", ".join(absent))
    return frames


def _expected_sessions(start_date: pd.Timestamp, snapshot_end: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(calendar.date_to_session(start_date, direction="next"))
    last = pd.Timestamp(calendar.date_to_session(snapshot_end, direction="previous"))
    sessions = calendar.sessions_in_range(first, last)
    sessions = pd.DatetimeIndex(sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def _validate_complete_history(
    frames: dict[str, pd.DataFrame],
    assets: list[str],
    expected: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    for symbol in assets:
        frame = frames[symbol]
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)).unique()
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        row = {
            "symbol": symbol,
            "observed_rows": int(len(frame)),
            "expected_sessions": int(len(expected)),
            "missing_sessions": int(len(missing)),
            "extra_sessions": int(len(extra)),
            "actual_start": observed.min().date().isoformat() if len(observed) else None,
            "actual_end": observed.max().date().isoformat() if len(observed) else None,
            "history_complete": bool(len(missing) == 0),
        }
        diagnostics.append(row)
        if len(missing):
            sample = ",".join(item.date().isoformat() for item in missing[:5])
            failures.append(f"{symbol} missing={len(missing)} sample={sample}")
    if failures:
        raise RuntimeError(
            "Full Strategy History validation failed before the experiment:\n- "
            + "\n- ".join(failures)
        )
    return diagnostics


def _feature_and_training_panels(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    feature_rows: list[pd.DataFrame] = []
    training_rows: list[pd.DataFrame] = []
    features_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol, frame in frames.items():
        features = feature_frame(frame).replace([np.inf, -np.inf], np.nan)
        features_by_symbol[symbol] = features

        feature_part = features.copy()
        feature_part["symbol"] = symbol
        feature_part["date"] = pd.to_datetime(feature_part.index, utc=True).normalize().tz_localize(None)
        feature_rows.append(feature_part.reset_index(drop=True))

        utility = _future_utility(frame).reindex(features.index)
        training_part = features.copy()
        training_part["utility"] = utility
        training_part["symbol"] = symbol
        training_part["date"] = pd.to_datetime(training_part.index, utc=True).normalize().tz_localize(None)
        training_rows.append(training_part.reset_index(drop=True))

    feature_panel = pd.concat(feature_rows, ignore_index=True)
    feature_panel = feature_panel.dropna(subset=[*FEATURE_COLUMNS, "date"]).copy()
    feature_panel = feature_panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    training_panel = pd.concat(training_rows, ignore_index=True)
    training_panel = training_panel.dropna(subset=[*FEATURE_COLUMNS, "utility", "date"]).copy()
    training_panel = training_panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    counts = training_panel.groupby("date")["symbol"].transform("count")
    training_panel = training_panel.loc[counts >= 3].reset_index(drop=True)
    return feature_panel, training_panel, features_by_symbol


def _relevance_for_reference(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    percentile = result.groupby("date")["utility"].rank(method="average", pct=True)
    result["relevance"] = np.minimum(np.floor(percentile * 5.0), 4).astype(int)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def _fit_reference_model(dataset: pd.DataFrame, random_state: int) -> Any:
    if dataset.empty:
        raise RuntimeError("Cannot fit a leave-one-out ranker with an empty reference dataset.")
    model = _new_ranker(int(random_state))
    model.fit(
        dataset[list(FEATURE_COLUMNS)],
        dataset["relevance"],
        group=_group_sizes(dataset),
    )
    return model


def _fold_boundaries(unique_dates: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    if len(unique_dates) < 360:
        raise RuntimeError("Leave-one-out analysis requires at least 360 labelled sessions.")
    initial_train_sessions = max(180, int(len(unique_dates) * INITIAL_TRAIN_FRACTION))
    remaining = len(unique_dates) - initial_train_sessions
    if remaining < VALIDATION_FOLDS * 20:
        raise RuntimeError("Not enough labelled sessions for the four chronological validation folds.")
    boundaries = np.linspace(initial_train_sessions, len(unique_dates), VALIDATION_FOLDS + 1, dtype=int)
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    for fold_index in range(VALIDATION_FOLDS):
        validation_start = int(boundaries[fold_index])
        validation_end = int(boundaries[fold_index + 1])
        validation_dates = list(unique_dates[validation_start:validation_end])
        purge_end = max(0, validation_start - TARGET_HORIZON)
        train_dates = list(unique_dates[:purge_end])
        if len(train_dates) < 180 or len(validation_dates) < 20:
            raise RuntimeError(f"Fold {fold_index + 1} is too short for the locked ranker policy.")
        folds.append((train_dates, validation_dates))
    return folds


def _rank_metrics_for_dates(
    scored: pd.DataFrame,
    training_truth: pd.DataFrame,
    held_out: str,
    weak_threshold: float,
) -> dict[str, Any]:
    ranks: list[int] = []
    percentiles: list[float] = []
    relative_iqrs: list[float] = []
    leader_gaps: list[float] = []
    weak_flags: list[bool] = []
    weak_wins: list[bool] = []
    oracle_ranks: list[int] = []
    oracle_percentiles: list[float] = []

    truth_by_date = {
        date: group.set_index("symbol")["utility"]
        for date, group in training_truth.groupby("date", sort=False)
    }

    for date, group in scored.groupby("date", sort=False):
        candidate = group[group["symbol"] == held_out]
        reference = group[group["symbol"] != held_out]
        if candidate.empty or reference.empty:
            continue
        candidate_score = float(candidate["score"].iloc[0])
        reference_scores = reference["score"].to_numpy(dtype=float)
        all_scores = np.concatenate([reference_scores, np.asarray([candidate_score])])
        rank = 1 + int(np.sum(reference_scores > candidate_score))
        percentile = float(np.mean(all_scores <= candidate_score))
        median = float(np.median(reference_scores))
        q25, q75 = np.quantile(reference_scores, [0.25, 0.75])
        iqr = float(q75 - q25)
        relative_iqr = float((candidate_score - median) / iqr) if abs(iqr) > 1e-12 else 0.0
        leader_gap = float(candidate_score - float(np.max(reference_scores)))
        weak = bool(float(np.max(reference_scores)) <= weak_threshold)

        ranks.append(rank)
        percentiles.append(percentile)
        relative_iqrs.append(relative_iqr)
        leader_gaps.append(leader_gap)
        weak_flags.append(weak)
        weak_wins.append(bool(weak and rank == 1))

        truth = truth_by_date.get(date)
        if truth is not None and held_out in truth.index:
            reference_truth = truth.drop(labels=[held_out], errors="ignore").dropna().to_numpy(dtype=float)
            held_truth = float(truth.loc[held_out])
            if len(reference_truth):
                all_truth = np.concatenate([reference_truth, np.asarray([held_truth])])
                oracle_ranks.append(1 + int(np.sum(reference_truth > held_truth)))
                oracle_percentiles.append(float(np.mean(all_truth <= held_truth)))

    count = len(ranks)
    weak_count = int(sum(weak_flags))
    positive_gaps = [value for value in leader_gaps if value > 0]
    return {
        "session_count": count,
        "rank_mean": float(np.mean(ranks)) if ranks else None,
        "rank_median": float(np.median(ranks)) if ranks else None,
        "rank_percentile_mean": float(np.mean(percentiles)) if percentiles else None,
        "rank_percentile_median": float(np.median(percentiles)) if percentiles else None,
        "relative_score_iqr_median": float(np.median(relative_iqrs)) if relative_iqrs else None,
        "top1_frequency": float(np.mean(np.asarray(ranks) <= 1)) if ranks else None,
        "top3_frequency": float(np.mean(np.asarray(ranks) <= 3)) if ranks else None,
        "top5_frequency": float(np.mean(np.asarray(ranks) <= 5)) if ranks else None,
        "leader_gap_mean": float(np.mean(leader_gaps)) if leader_gaps else None,
        "leader_positive_gap_mean": float(np.mean(positive_gaps)) if positive_gaps else None,
        "weak_reference_session_count": weak_count,
        "weak_period_coverage": float(sum(weak_wins) / weak_count) if weak_count else None,
        "oracle_rank_percentile_mean": float(np.mean(oracle_percentiles)) if oracle_percentiles else None,
        "oracle_top1_frequency": float(np.mean(np.asarray(oracle_ranks) <= 1)) if oracle_ranks else None,
        "oracle_top3_frequency": float(np.mean(np.asarray(oracle_ranks) <= 3)) if oracle_ranks else None,
        "oracle_top5_frequency": float(np.mean(np.asarray(oracle_ranks) <= 5)) if oracle_ranks else None,
    }


def _leaf_similarity(
    model: Any,
    snapshot: pd.DataFrame,
    held_out: str,
) -> dict[str, Any]:
    if snapshot.empty or held_out not in set(snapshot["symbol"]):
        return {
            "state_similarity": None,
            "state_novelty": None,
            "state_nearest_symbol": None,
            "state_tree_count": 0,
        }
    ordered = snapshot.sort_values("symbol").reset_index(drop=True)
    leaves = np.asarray(model.predict(ordered[list(FEATURE_COLUMNS)], pred_leaf=True))
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    candidate_index = int(ordered.index[ordered["symbol"] == held_out][0])
    candidate_leaves = leaves[candidate_index]
    reference_mask = ordered["symbol"].to_numpy() != held_out
    reference_leaves = leaves[reference_mask]
    reference_symbols = ordered.loc[reference_mask, "symbol"].astype(str).tolist()
    if not len(reference_symbols):
        return {
            "state_similarity": None,
            "state_novelty": None,
            "state_nearest_symbol": None,
            "state_tree_count": int(leaves.shape[1]),
        }
    similarities = np.mean(reference_leaves == candidate_leaves, axis=1)
    best = int(np.argmax(similarities))
    similarity = float(similarities[best])
    return {
        "state_similarity": similarity,
        "state_novelty": float(1.0 - similarity),
        "state_nearest_symbol": reference_symbols[best],
        "state_tree_count": int(leaves.shape[1]),
    }


def _score_snapshot(model: Any, snapshot: pd.DataFrame, held_out: str) -> dict[str, Any]:
    if snapshot.empty or held_out not in set(snapshot["symbol"]):
        return {
            "latest_rank": None,
            "latest_rank_percentile": None,
            "latest_relative_score_iqr": None,
            **_leaf_similarity(model, snapshot, held_out),
        }
    scored = snapshot.copy()
    scored["score"] = model.predict(scored[list(FEATURE_COLUMNS)])
    candidate = scored[scored["symbol"] == held_out]
    reference = scored[scored["symbol"] != held_out]
    candidate_score = float(candidate["score"].iloc[0])
    reference_scores = reference["score"].to_numpy(dtype=float)
    all_scores = np.concatenate([reference_scores, np.asarray([candidate_score])])
    rank = 1 + int(np.sum(reference_scores > candidate_score))
    median = float(np.median(reference_scores))
    q25, q75 = np.quantile(reference_scores, [0.25, 0.75])
    iqr = float(q75 - q25)
    return {
        "latest_rank": rank,
        "latest_rank_percentile": float(np.mean(all_scores <= candidate_score)),
        "latest_relative_score_iqr": float((candidate_score - median) / iqr) if abs(iqr) > 1e-12 else 0.0,
        **_leaf_similarity(model, snapshot, held_out),
    }


def _correlation_metrics(
    held_out: str,
    full_corr: pd.DataFrame,
    latest_60_corr: pd.DataFrame,
) -> dict[str, Any]:
    def summarize(matrix: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if held_out not in matrix.columns:
            return {}
        values = matrix[held_out].drop(labels=[held_out], errors="ignore").dropna()
        if values.empty:
            return {}
        abs_values = values.abs()
        nearest_abs = str(abs_values.idxmax())
        return {
            f"{prefix}_max": float(values.max()),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_median": float(values.median()),
            f"{prefix}_median_abs": float(abs_values.median()),
            f"{prefix}_max_abs": float(abs_values.max()),
            f"{prefix}_nearest_abs_symbol": nearest_abs,
            f"{prefix}_nearest_abs_signed": float(values.loc[nearest_abs]),
        }

    return {
        **summarize(full_corr, "return_corr_full"),
        **summarize(latest_60_corr, "return_corr_60_latest"),
    }


def _analyse_one_asset(
    held_out: str,
    assets: list[str],
    feature_panel: pd.DataFrame,
    training_panel: pd.DataFrame,
    fold_specs: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]],
    latest_date: pd.Timestamp,
    full_corr: pd.DataFrame,
    latest_60_corr: pd.DataFrame,
    winner_assets: set[str],
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    symbol_seed = _stable_seed(held_out, base_seed)
    reference_base = training_panel[training_panel["symbol"] != held_out].copy()
    reference_base = _relevance_for_reference(reference_base)
    fold_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_state_similarities: list[float] = []
    fold_state_nearest: list[str] = []

    for fold_number, (train_dates, validation_dates) in enumerate(fold_specs, start=1):
        train_set = set(train_dates)
        validation_set = set(validation_dates)
        train = reference_base[reference_base["date"].isin(train_set)].copy()
        model = _fit_reference_model(train, symbol_seed + fold_number)

        train_scores = train[["date", "symbol", *FEATURE_COLUMNS]].copy()
        train_scores["score"] = model.predict(train_scores[list(FEATURE_COLUMNS)])
        daily_reference_max = train_scores.groupby("date")["score"].max()
        weak_threshold = float(daily_reference_max.quantile(0.25))

        validation_features = feature_panel[
            feature_panel["date"].isin(validation_set)
            & feature_panel["symbol"].isin(assets)
        ].copy()
        validation_features["score"] = model.predict(validation_features[list(FEATURE_COLUMNS)])
        validation_truth = training_panel[
            training_panel["date"].isin(validation_set)
            & training_panel["symbol"].isin(assets)
        ][["date", "symbol", "utility"]].copy()
        metrics = _rank_metrics_for_dates(
            validation_features[["date", "symbol", "score"]],
            validation_truth,
            held_out,
            weak_threshold,
        )

        fold_end = max(validation_dates)
        fold_snapshot = feature_panel[
            (feature_panel["date"] == fold_end)
            & feature_panel["symbol"].isin(assets)
        ].copy()
        state = _leaf_similarity(model, fold_snapshot, held_out)
        if state.get("state_similarity") is not None:
            fold_state_similarities.append(float(state["state_similarity"]))
        if state.get("state_nearest_symbol"):
            fold_state_nearest.append(str(state["state_nearest_symbol"]))

        fold_row = {
            "symbol": held_out,
            "fold": fold_number,
            "train_start": min(train_dates).date().isoformat(),
            "train_end": max(train_dates).date().isoformat(),
            "validation_start": min(validation_dates).date().isoformat(),
            "validation_end": max(validation_dates).date().isoformat(),
            "train_sessions": len(train_dates),
            "validation_sessions": len(validation_dates),
            "weak_reference_threshold": weak_threshold,
            **metrics,
            **state,
        }
        fold_rows.append(fold_row)
        fold_metrics.append(metrics)

    all_labelled_dates = sorted(pd.unique(reference_base["date"]))
    final = reference_base[reference_base["date"].isin(set(all_labelled_dates))].copy()
    final_model = _fit_reference_model(final, symbol_seed + 10_000)
    latest_snapshot = feature_panel[
        (feature_panel["date"] == latest_date)
        & feature_panel["symbol"].isin(assets)
    ].copy()
    latest = _score_snapshot(final_model, latest_snapshot, held_out)

    def mean_metric(name: str) -> float | None:
        values = [float(row[name]) for row in fold_metrics if row.get(name) is not None]
        return float(np.mean(values)) if values else None

    def median_metric(name: str) -> float | None:
        values = [float(row[name]) for row in fold_metrics if row.get(name) is not None]
        return float(np.median(values)) if values else None

    fold_rank_percentiles = [
        float(row["rank_percentile_mean"])
        for row in fold_metrics
        if row.get("rank_percentile_mean") is not None
    ]
    fold_top3 = [float(row["top3_frequency"]) for row in fold_metrics if row.get("top3_frequency") is not None]
    nearest_mode = Counter(fold_state_nearest).most_common(1)[0][0] if fold_state_nearest else None

    result = {
        "symbol": held_out,
        "is_in_current_winner": bool(held_out in winner_assets),
        "reference_asset_count": int(len(assets) - 1),
        "oos_session_count": int(sum(int(row.get("session_count") or 0) for row in fold_metrics)),
        "oos_rank_percentile_mean": mean_metric("rank_percentile_mean"),
        "oos_rank_percentile_median_of_folds": median_metric("rank_percentile_median"),
        "oos_rank_percentile_fold_std": float(np.std(fold_rank_percentiles, ddof=0)) if fold_rank_percentiles else None,
        "oos_top1_frequency": mean_metric("top1_frequency"),
        "oos_top3_frequency": mean_metric("top3_frequency"),
        "oos_top5_frequency": mean_metric("top5_frequency"),
        "oos_top3_fold_std": float(np.std(fold_top3, ddof=0)) if fold_top3 else None,
        "oos_relative_score_iqr_median": median_metric("relative_score_iqr_median"),
        "oos_leader_gap_mean": mean_metric("leader_gap_mean"),
        "oos_leader_positive_gap_mean": mean_metric("leader_positive_gap_mean"),
        "oos_weak_period_coverage": mean_metric("weak_period_coverage"),
        "oracle_rank_percentile_mean": mean_metric("oracle_rank_percentile_mean"),
        "oracle_top1_frequency": mean_metric("oracle_top1_frequency"),
        "oracle_top3_frequency": mean_metric("oracle_top3_frequency"),
        "oracle_top5_frequency": mean_metric("oracle_top5_frequency"),
        "state_similarity_fold_median": float(np.median(fold_state_similarities)) if fold_state_similarities else None,
        "state_similarity_fold_max": float(np.max(fold_state_similarities)) if fold_state_similarities else None,
        "state_novelty_fold_median": float(1.0 - np.median(fold_state_similarities)) if fold_state_similarities else None,
        "state_nearest_symbol_fold_mode": nearest_mode,
        **latest,
        **_correlation_metrics(held_out, full_corr, latest_60_corr),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return result, fold_rows


def _distribution_summary(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = [
        "oos_rank_percentile_mean",
        "oos_top1_frequency",
        "oos_top3_frequency",
        "oos_top5_frequency",
        "oos_rank_percentile_fold_std",
        "oos_weak_period_coverage",
        "oracle_rank_percentile_mean",
        "oracle_top3_frequency",
        "state_similarity_fold_median",
        "state_novelty_fold_median",
        "state_similarity",
        "state_novelty",
        "return_corr_full_max_abs",
        "return_corr_full_median_abs",
        "return_corr_60_latest_max_abs",
        "latest_rank_percentile",
    ]
    summary: dict[str, Any] = {}
    for metric in metrics:
        if metric not in frame.columns:
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if series.empty:
            continue
        summary[metric] = {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p10": float(series.quantile(0.10)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the selected Strategy's complete market history from local MongoDB once, "
            "then run a backtest-free leave-one-out signature analysis entirely in memory."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=DEFAULT_STRATEGY_SEQUENCE)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", default=None, help="YYYY-MM-DD. Defaults to the latest common cached session.")
    parser.add_argument("--mongo-uri", default=None, help="Defaults to MONGO_URL/MONGO_URI or mongodb://localhost:27017.")
    parser.add_argument("--database", default=None, help="Defaults to MONGO_DATABASE from .env.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    load_project_environment(args.env_file)

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    _assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    workers = max(
        1,
        int(
            args.workers
            if args.workers is not None
            else os.getenv("ASSET_DISCOVERY_REPLAY_WORKERS") or DEFAULT_WORKERS
        ),
    )

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 2),
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]

    strategy = _strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = _strategy_configuration(strategy)
    assets = [
        str(symbol).strip().upper()
        for symbol in list(configuration.get("assets") or [])
        if str(symbol).strip()
    ]
    assets = list(dict.fromkeys(assets))
    if len(assets) < 3:
        raise RuntimeError("The selected Strategy needs at least three assets for this experiment.")

    strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
    strategy_name = str(strategy.get("name") or f"Strategy #{strategy_sequence}")
    strategy_id = str(strategy.get("_id") or "")
    start_date = _normalize_date(configuration.get("start_date") or "2016-01-01")
    identity = _identity(configuration)
    collection = db[ALPACA_MARKET_BARS_COLLECTION]
    snapshot_end = (
        _normalize_date(args.snapshot_end)
        if args.snapshot_end
        else _latest_common_session(collection, assets, identity)
    )
    if snapshot_end < start_date:
        raise RuntimeError("snapshot_end is earlier than the Strategy start_date.")

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_signature_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"Experiment start: {strategy_name} ({strategy_id}), assets={len(assets)}, "
        f"window={start_date.date().isoformat()} -> {snapshot_end.date().isoformat()}, workers={workers}."
    )
    _log("MongoDB is read-only for this script. No Alpaca call and no Backtest will be executed.")

    expected = _expected_sessions(start_date, snapshot_end)
    _log(f"Step 1/6 - Loading all {len(assets)} Strategy series from local MongoDB in one query...")
    load_started = time.perf_counter()
    frames = _load_frames_once(collection, assets, identity, start_date, snapshot_end)
    _log(f"Loaded {sum(len(frame) for frame in frames.values()):,} OHLCV rows in {time.perf_counter() - load_started:.1f}s.")

    _log("Step 2/6 - Validating Full Strategy History for every asset...")
    history = _validate_complete_history(frames, assets, expected)
    _write_csv(output_dir / "history_integrity.csv", history)
    _log(f"All {len(assets)} assets cover the complete {len(expected)}-session XNYS Strategy window.")

    _log("Step 3/6 - Building features and 20-session utility labels once in memory...")
    feature_panel, training_panel, _features_by_symbol = _feature_and_training_panels(frames)
    unique_dates = sorted(pd.unique(training_panel["date"]))
    fold_specs = _fold_boundaries(unique_dates)
    latest_date = max(pd.unique(feature_panel["date"]))
    if latest_date < snapshot_end:
        _log(
            f"Note: latest complete feature date is {latest_date.date().isoformat()}, "
            f"earlier than snapshot {snapshot_end.date().isoformat()}."
        )

    _log("Step 4/6 - Computing pairwise return correlations without any model/backtest...")
    close = pd.concat(
        {symbol: pd.to_numeric(frame["close"], errors="coerce") for symbol, frame in frames.items()},
        axis=1,
        join="inner",
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    full_corr = returns.corr()
    latest_60_corr = returns.tail(60).corr()
    full_corr.to_csv(output_dir / "return_correlation_full.csv")
    latest_60_corr.to_csv(output_dir / "return_correlation_latest_60.csv")

    winner_assets = _winner_assets(db)
    client.close()
    _log("MongoDB connection closed. Steps 5-6 are now entirely in RAM/CPU.")

    manifest = {
        "schema_version": 1,
        "experiment": "strategy_asset_signature_leave_one_out",
        "strategy_id": strategy_id,
        "strategy_sequence": strategy_sequence,
        "strategy_name": strategy_name,
        "strategy_revision": int(strategy.get("revision") or 0),
        "configuration_hash": strategy.get("configuration_hash"),
        "asset_count": len(assets),
        "assets": assets,
        "winner_overlap_count": int(sum(symbol in winner_assets for symbol in assets)),
        "start_date": start_date.date().isoformat(),
        "snapshot_end": snapshot_end.date().isoformat(),
        "latest_feature_date": latest_date.date().isoformat(),
        "expected_xnys_sessions": len(expected),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_horizon_sessions": TARGET_HORIZON,
        "validation_method": "leave_one_out_purged_expanding_walk_forward",
        "validation_folds": VALIDATION_FOLDS,
        "workers": workers,
        "random_state": int(args.random_state),
        "data_source": "local_mongodb_only",
        "backtest_used": False,
        "alpaca_network_used": False,
        "mongo_writes": False,
        "notes": {
            "oracle_metrics": "Evaluation-only realized future utility. Never used as a ranking input.",
            "weak_period": "Reference daily max ranker score is below the 25th percentile of the fold's training-period daily maxima.",
            "state_similarity": "LightGBM leaf agreement against the 55 remaining Strategy assets.",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)

    result_path = output_dir / "asset_signatures.csv"
    fold_path = output_dir / "asset_signature_folds.csv"
    existing_results: list[dict[str, Any]] = []
    existing_folds: list[dict[str, Any]] = []
    completed_symbols: set[str] = set()
    if not args.no_resume and result_path.exists():
        existing_frame = pd.read_csv(result_path)
        existing_results = existing_frame.to_dict(orient="records")
        completed_symbols = {
            str(symbol).upper()
            for symbol in existing_frame.get("symbol", pd.Series(dtype=str)).dropna().astype(str)
        }
        if fold_path.exists():
            existing_folds = pd.read_csv(fold_path).to_dict(orient="records")
        if completed_symbols:
            _log(f"Resume enabled: {len(completed_symbols)} assets already completed and will be skipped.")

    pending = [symbol for symbol in assets if symbol not in completed_symbols]
    results = list(existing_results)
    fold_rows = list(existing_folds)
    _log(
        f"Step 5/6 - Running leave-one-out predictive analysis for {len(pending)} assets "
        f"with {workers} worker(s). No Backtest is involved."
    )

    experiment_started = time.perf_counter()
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _analyse_one_asset,
                    symbol,
                    assets,
                    feature_panel,
                    training_panel,
                    fold_specs,
                    latest_date,
                    full_corr,
                    latest_60_corr,
                    winner_assets,
                    int(args.random_state),
                ): symbol
                for symbol in pending
            }
            completed = len(completed_symbols)
            for future in as_completed(futures):
                symbol = futures[future]
                result, symbol_folds = future.result()
                results = [row for row in results if str(row.get("symbol") or "").upper() != symbol]
                fold_rows = [row for row in fold_rows if str(row.get("symbol") or "").upper() != symbol]
                results.append(result)
                fold_rows.extend(symbol_folds)
                completed += 1
                results.sort(key=lambda row: assets.index(str(row.get("symbol") or "").upper()))
                fold_rows.sort(key=lambda row: (assets.index(str(row.get("symbol") or "").upper()), int(row.get("fold") or 0)))
                _write_csv(result_path, results)
                _write_csv(fold_path, fold_rows)
                _log(
                    f"LOO {completed}/{len(assets)} - {symbol} completed in {float(result['elapsed_seconds']):.1f}s; "
                    f"OOS Top-3={result.get('oos_top3_frequency')!s}, "
                    f"state novelty={result.get('state_novelty')!s}."
                )

    _log("Step 6/6 - Building the empirical signature distribution of the existing Strategy assets...")
    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        raise RuntimeError("No leave-one-out result was produced.")
    result_frame = result_frame.sort_values("symbol").reset_index(drop=True)
    result_frame.to_csv(result_path, index=False)
    distribution = _distribution_summary(result_frame)
    summary = {
        "schema_version": 1,
        "strategy_id": strategy_id,
        "strategy_sequence": strategy_sequence,
        "asset_count": len(assets),
        "completed_asset_count": int(len(result_frame)),
        "elapsed_seconds": float(time.perf_counter() - experiment_started),
        "metric_distributions": distribution,
        "interpretation_policy": (
            "These distributions describe the 56 assets already present in the Strategy. "
            "They are not automatic acceptance thresholds and do not use Backtest capital as a label."
        ),
    }
    _write_json(output_dir / "signature_distribution.json", summary)

    _log(f"Experiment completed. Outputs: {output_dir}")
    _log("Primary files: asset_signatures.csv, asset_signature_folds.csv, signature_distribution.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
