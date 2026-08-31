from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import ANALYSIS_VERSION, SCHEMA_VERSION


DERIVED_FEATURES = ("asset_return_1d", "asset_return_5d", "asset_volatility_10d")


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        parsed = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    return parsed if not pd.isna(parsed) else None


def _safe_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def _safe_median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else None


def _select_k(
    matrix: np.ndarray,
    *,
    min_clusters: int,
    max_clusters: int,
    sample_rows: int,
    random_state: int,
    n_init: int,
) -> tuple[int, float | None, np.ndarray, KMeans]:
    upper = min(max_clusters, max(min_clusters, len(matrix) - 1))
    if upper < min_clusters:
        model = KMeans(n_clusters=1, random_state=random_state, n_init=n_init).fit(matrix)
        return 1, None, model.labels_, model

    sample_size = min(len(matrix), max(40, sample_rows))
    sample_idx = (
        np.linspace(0, len(matrix) - 1, sample_size, dtype=int)
        if sample_size < len(matrix)
        else np.arange(len(matrix))
    )
    sample = matrix[sample_idx]
    best: tuple[float, int, np.ndarray, KMeans] | None = None
    for k in range(min_clusters, upper + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=120,
        ).fit(matrix)
        labels_sample = model.predict(sample)
        if len(set(labels_sample.tolist())) < 2:
            score = -1.0
        else:
            try:
                score = float(silhouette_score(sample, labels_sample))
            except Exception:
                score = -1.0
        candidate = (score, -k, model.labels_, model)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    score, neg_k, labels, model = best
    return -neg_k, (score if np.isfinite(score) and score >= -0.999 else None), labels, model


def _prepare_symbol_frame(rows: list[dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        ts = _timestamp(row.get("timestamp"))
        if ts is None:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        record: dict[str, Any] = {"timestamp": ts, "symbol": symbol}
        for feature in feature_names:
            if feature in DERIVED_FEATURES:
                continue
            record[feature] = _number(row.get(feature))
        record["decision_close"] = _number(row.get("decision_close"))
        record["open_to_open_return"] = _number(row.get("open_to_open_return"))
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    close = pd.to_numeric(frame["decision_close"], errors="coerce")
    frame["asset_return_1d"] = close.pct_change(1)
    frame["asset_return_5d"] = close.pct_change(5)
    frame["asset_volatility_10d"] = close.pct_change().rolling(10, min_periods=5).std(ddof=0)
    return frame


def _forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(len(returns), np.nan, dtype=float)
    if horizon <= 0 or len(returns) < horizon:
        return result
    safe = np.where(np.isfinite(returns), 1.0 + returns, np.nan)
    for start in range(0, len(returns) - horizon + 1):
        window = safe[start : start + horizon]
        if np.isfinite(window).all():
            result[start] = float(np.prod(window) - 1.0)
    return result


def _fit_state(
    feature_matrix_all: np.ndarray,
    upto: int,
    feature_names: list[str],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    matrix_history = np.asarray(feature_matrix_all[: upto + 1], dtype=float)
    if matrix_history.ndim != 2 or matrix_history.shape[0] == 0:
        return None

    if np.isfinite(matrix_history).all():
        filled = matrix_history
    else:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(matrix_history, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix_history), matrix_history, medians)

    std = np.std(filled, axis=0, ddof=0)
    valid_mask = np.isfinite(std) & (std > 1e-12)
    if int(np.sum(valid_mask)) < 2:
        return None

    valid_columns = [
        feature_names[idx] for idx in np.flatnonzero(valid_mask).tolist()
    ]
    matrix_raw = filled[:, valid_mask]
    matrix = StandardScaler().fit_transform(matrix_raw)
    k, silhouette, labels, model = _select_k(
        matrix,
        min_clusters=int(settings.get("min_clusters") or 2),
        max_clusters=int(settings.get("max_clusters") or 6),
        sample_rows=int(settings.get("silhouette_sample_rows") or 120),
        random_state=int(settings.get("random_state") or 42),
        n_init=int(settings.get("kmeans_n_init") or 5),
    )
    distances = model.transform(matrix)
    assigned_distances = distances[np.arange(len(matrix)), labels]
    novelty_quantile = float(settings.get("novelty_quantile") or 0.99)
    novelty_threshold = (
        float(np.quantile(assigned_distances, novelty_quantile))
        if len(assigned_distances) > 4
        else None
    )
    current_distance = float(assigned_distances[-1])
    current_cluster = int(labels[-1])
    return {
        "valid_features": valid_columns,
        "matrix": matrix,
        "labels": labels,
        "model": model,
        "silhouette": silhouette,
        "cluster_count": int(k),
        "current_cluster": current_cluster,
        "current_distance": current_distance,
        "novelty_threshold": novelty_threshold,
        "is_novel": bool(
            novelty_threshold is not None and current_distance > novelty_threshold
        ),
    }


def _cluster_profile_vectorized(
    forward_returns: np.ndarray,
    labels: np.ndarray,
    upto: int,
    cluster_id: int,
    *,
    horizon: int,
    severe_threshold: float,
) -> dict[str, Any]:
    matured_last_index = upto - horizon
    if matured_last_index < 0:
        values = np.asarray([], dtype=float)
    else:
        labels_slice = labels[: matured_last_index + 1]
        returns_slice = forward_returns[: matured_last_index + 1]
        mask = (labels_slice == int(cluster_id)) & np.isfinite(returns_slice)
        values = returns_slice[mask]

    if values.size == 0:
        return {
            "samples": 0,
            "mean_forward_return": None,
            "median_forward_return": None,
            "positive_rate": None,
            "severe_loss_rate": None,
        }
    return {
        "samples": int(values.size),
        "mean_forward_return": float(values.mean()),
        "median_forward_return": float(np.median(values)),
        "positive_rate": float(np.mean(values > 0.0)),
        "severe_loss_rate": float(np.mean(values <= severe_threshold)),
    }


def _latest_map(
    frame: pd.DataFrame,
    fitted: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    matrix = fitted["matrix"]
    labels = fitted["labels"]
    model = fitted["model"]
    pca = PCA(n_components=2, random_state=int(settings.get("random_state") or 42))
    coords = pca.fit_transform(matrix)
    centroid_coords = pca.transform(model.cluster_centers_)

    maximum = max(40, int(settings.get("map_sample_rows") or 120))
    sample_idx = (
        np.unique(np.linspace(0, len(coords) - 1, maximum, dtype=int))
        if len(coords) > maximum
        else np.arange(len(coords))
    )
    points: list[dict[str, Any]] = []
    last_idx = len(frame) - 1
    for idx in sample_idx.tolist():
        points.append({
            "timestamp": frame.iloc[idx]["timestamp"].isoformat(),
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "cluster_id": int(labels[idx]),
            "is_current": bool(idx == last_idx),
        })
    if not any(point["is_current"] for point in points):
        points.append({
            "timestamp": frame.iloc[last_idx]["timestamp"].isoformat(),
            "x": float(coords[last_idx, 0]),
            "y": float(coords[last_idx, 1]),
            "cluster_id": int(labels[last_idx]),
            "is_current": True,
        })

    return {
        "as_of": frame.iloc[last_idx]["timestamp"].isoformat(),
        "points": points,
        "centroids": [
            {"cluster_id": int(idx), "x": float(value[0]), "y": float(value[1])}
            for idx, value in enumerate(centroid_coords)
        ],
        "cluster_count": fitted["cluster_count"],
        "silhouette_score": fitted["silhouette"],
        "current_cluster_id": fitted["current_cluster"],
        "pca_explained_variance": [
            float(value) for value in pca.explained_variance_ratio_.tolist()
        ],
    }


def build_analysis(
    *,
    observation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    enabled = bool(settings.get("enabled", True))
    feature_names = [
        str(item) for item in (settings.get("feature_names") or []) if str(item).strip()
    ]
    if not enabled:
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "status": "completed",
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(period_start),
            "period_end": str(period_end),
            "shadow_only": True,
            "decision_effect": "none",
            "summary": {"enabled": False, "duration_seconds": 0.0},
            "asset_summaries": [],
            "latest_maps": [],
            "daily_states_by_symbol": {},
        }
    if len(feature_names) < 4:
        raise ValueError("Asset State Clustering requires at least four state features.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observation_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            grouped[symbol].append(row)

    min_history = max(30, int(settings.get("min_history_rows") or 120))
    profile_horizon = max(1, int(settings.get("profile_horizon_sessions") or 5))
    severe_threshold = float(settings.get("severe_loss_threshold") or -0.05)
    daily_states_by_symbol: dict[str, list[dict[str, Any]]] = {}
    latest_maps: list[dict[str, Any]] = []
    asset_summaries: list[dict[str, Any]] = []
    symbols = sorted(grouped)
    total_work = max(1, len(symbols))
    completed_durations: list[float] = []

    for symbol_index, symbol in enumerate(symbols):
        asset_started = perf_counter()
        frame = _prepare_symbol_frame(grouped[symbol], feature_names)
        if frame.empty or len(frame) < min_history:
            asset_summaries.append({
                "symbol": symbol,
                "status": "insufficient_history",
                "rows": int(len(frame)),
                "minimum_history_rows": min_history,
                "duration_seconds": float(perf_counter() - asset_started),
            })
            if progress_callback:
                progress_callback((symbol_index + 1) / total_work)
            continue

        returns = pd.to_numeric(frame["open_to_open_return"], errors="coerce").to_numpy(dtype=float)
        forward_returns = _forward_returns(returns, profile_horizon)
        feature_matrix_all = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        states: list[dict[str, Any]] = []
        silhouettes: list[float] = []
        novel_count = 0
        latest_fit: dict[str, Any] | None = None
        latest_profile: dict[str, Any] | None = None

        for idx in range(min_history - 1, len(frame)):
            fitted = _fit_state(feature_matrix_all, idx, feature_names, settings)
            if not fitted:
                continue
            profile = _cluster_profile_vectorized(
                forward_returns,
                fitted["labels"],
                idx,
                fitted["current_cluster"],
                horizon=profile_horizon,
                severe_threshold=severe_threshold,
            )
            if fitted["silhouette"] is not None:
                silhouettes.append(float(fitted["silhouette"]))
            novel_count += int(fitted["is_novel"])
            states.append({
                "timestamp": frame.iloc[idx]["timestamp"].isoformat(),
                "symbol": symbol,
                "cluster_id": fitted["current_cluster"],
                "cluster_count": fitted["cluster_count"],
                "silhouette_score": fitted["silhouette"],
                "nearest_distance": fitted["current_distance"],
                "novelty_threshold": fitted["novelty_threshold"],
                "is_novel": fitted["is_novel"],
                "profile_samples": profile["samples"],
                "profile_mean_forward_return": profile["mean_forward_return"],
                "profile_median_forward_return": profile["median_forward_return"],
                "profile_positive_rate": profile["positive_rate"],
                "profile_severe_loss_rate": profile["severe_loss_rate"],
            })
            latest_fit = fitted
            latest_profile = profile

        daily_states_by_symbol[symbol] = states
        if latest_fit is not None:
            latest_map = _latest_map(frame, latest_fit, settings)
            latest_map["symbol"] = symbol
            latest_map["current_profile"] = latest_profile or {}
            latest_maps.append(latest_map)
            if states:
                current_point = next(
                    (point for point in latest_map["points"] if point.get("is_current")),
                    None,
                )
                if current_point:
                    states[-1]["pca_x"] = current_point.get("x")
                    states[-1]["pca_y"] = current_point.get("y")

        asset_duration = float(perf_counter() - asset_started)
        completed_durations.append(asset_duration)
        asset_summaries.append({
            "symbol": symbol,
            "status": "completed",
            "rows": int(len(frame)),
            "states": int(len(states)),
            "mean_silhouette": _safe_mean(silhouettes),
            "median_silhouette": _safe_median(silhouettes),
            "novel_states": int(novel_count),
            "latest_cluster_id": (states[-1]["cluster_id"] if states else None),
            "latest_cluster_count": (states[-1]["cluster_count"] if states else None),
            "latest_is_novel": (states[-1]["is_novel"] if states else None),
            "latest_profile": latest_profile or {},
            "duration_seconds": asset_duration,
        })
        if progress_callback:
            progress_callback((symbol_index + 1) / total_work)

    completed_assets = [item for item in asset_summaries if item.get("status") == "completed"]
    all_states = [state for states in daily_states_by_symbol.values() for state in states]
    all_silhouettes = [
        float(state["silhouette_score"])
        for state in all_states
        if state.get("silhouette_score") is not None
    ]
    total_duration = float(perf_counter() - started)
    summary = {
        "enabled": True,
        "asset_count": int(len(symbols)),
        "completed_asset_count": int(len(completed_assets)),
        "daily_state_count": int(len(all_states)),
        "novel_state_count": int(
            sum(int(bool(state.get("is_novel"))) for state in all_states)
        ),
        "mean_silhouette": _safe_mean(all_silhouettes),
        "median_silhouette": _safe_median(all_silhouettes),
        "profile_horizon_sessions": profile_horizon,
        "severe_loss_threshold": severe_threshold,
        "refit_policy": "complete expanding unsupervised refit at every completed trading session for each asset",
        "decision_effect": "none",
        "duration_seconds": total_duration,
        "mean_completed_asset_duration_seconds": _safe_mean(completed_durations),
        "performance_revision": "10.5.1",
        "performance_optimizations": [
            "PCA computed only for the latest visualization geometry",
            "matured forward returns precomputed once per asset",
            "cluster profile aggregation vectorized per daily refit",
            "numeric feature matrix prepared once per asset",
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "shadow_only": True,
        "decision_effect": "none",
        "method": {
            "clustering": "KMeans",
            "scaling": "StandardScaler",
            "visual_projection": "PCA",
            "refit_frequency": "every_completed_session",
            "cluster_selection": "training-only silhouette",
            "future_outcomes_used_for_clustering": False,
            "profile_outcome_rule": "only matured historical forward returns are used after clustering to describe the current cluster",
            "implementation_revision": "10.5.1-performance",
        },
        "feature_names": feature_names,
        "summary": summary,
        "asset_summaries": asset_summaries,
        "latest_maps": latest_maps,
        "daily_states_by_symbol": daily_states_by_symbol,
    }
