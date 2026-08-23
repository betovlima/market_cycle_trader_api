from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import ANALYSIS_VERSION, FEATURES, MAX_CLUSTERS, MIN_CLUSTERS, RANDOM_STATE, SCHEMA_VERSION, SEVERE_MONTH_THRESHOLD


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[Any]) -> float | None:
    valid = [value for value in (_number(v) for v in values) if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _month_outcome(value: float) -> str:
    if value <= SEVERE_MONTH_THRESHOLD:
        return "severe_negative"
    if value < 0.0:
        return "negative"
    return "positive"


def _state_share(rows: list[dict[str, Any]], state: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if str(row.get("state") or "") == state) / len(rows))


def _feature_row(month: str, rows: list[dict[str, Any]], official_return: float) -> dict[str, Any]:
    def feature_mean(key: str) -> float | None:
        return _mean([((row.get("features") or {}).get(key)) for row in rows])
    return {
        "month": month,
        "official_return": float(official_return),
        "outcome": _month_outcome(float(official_return)),
        "sessions": len(rows),
        "universe_breadth_5": feature_mean("universe_breadth_5"),
        "universe_breadth_20": feature_mean("universe_breadth_20"),
        "breadth_impulse": feature_mean("breadth_impulse"),
        "spy_realized_volatility_20": feature_mean("spy_realized_volatility_20"),
        "spy_return_5": feature_mean("spy_return_5"),
        "spy_return_20": feature_mean("spy_return_20"),
        "best_vs_second_gap": feature_mean("best_vs_second_gap"),
        "position_drawdown_from_peak": feature_mean("position_drawdown_from_peak"),
        "position_return_since_entry": feature_mean("position_return_since_entry"),
        "score_change_from_entry": feature_mean("score_change_from_entry"),
        "incumbent_risk_health": feature_mean("incumbent_risk_health"),
        "all_horizon_risk_safety": feature_mean("all_horizon_risk_safety"),
        "positive_score_share": feature_mean("positive_score_share"),
        "best_score_zscore": feature_mean("best_score_zscore"),
        "short_profit_consensus": feature_mean("short_profit_consensus"),
        "long_profit_confirmation": feature_mean("long_profit_confirmation"),
        "horizon_agreement": feature_mean("horizon_agreement"),
        "recent_rotations_10": _mean([row.get("recent_rotations_10") for row in rows]),
        "healthy_leader_share": _state_share(rows, "healthy_leader"),
        "weak_relative_leader_share": _state_share(rows, "weak_relative_leader"),
        "whipsaw_leadership_share": _state_share(rows, "whipsaw_leadership"),
        "no_good_opportunity_share": _state_share(rows, "no_good_opportunity"),
    }


def _best_cluster_count(matrix: np.ndarray) -> tuple[int, float]:
    if len(matrix) < 4:
        return 2, 0.0
    candidates: list[tuple[float, int]] = []
    upper = min(MAX_CLUSTERS, len(matrix) - 1)
    for clusters in range(MIN_CLUSTERS, upper + 1):
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels))
        candidates.append((score, clusters))
    if not candidates:
        return 2, 0.0
    score, clusters = max(candidates, key=lambda item: (item[0], -item[1]))
    return int(clusters), float(score)


def build_analysis(
    leadership: dict[str, Any],
    official_monthly_returns: list[dict[str, Any]],
    *,
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    session_source = [dict(row) for row in (leadership.get("sessions") or []) if isinstance(row, dict)]
    official_map: dict[str, float] = {}
    for row in official_monthly_returns or []:
        month = str(row.get("month") or "")[:7]
        value = _number(row.get("simulation_return"))
        if month and value is not None and period_start <= month <= period_end:
            official_map[month] = value
    if not session_source or not official_map:
        raise ValueError("Regime Clustering requires Leadership Regime sessions and official Strategy monthly returns.")

    by_month: dict[str, list[dict[str, Any]]] = {}
    for item in session_source:
        month = str(item.get("month") or str(item.get("timestamp") or "")[:7])
        if month in official_map:
            row = dict(item)
            row["recent_rotations_10"] = _number(item.get("recent_rotations_10"))
            by_month.setdefault(month, []).append(row)

    monthly_rows = [_feature_row(month, rows, official_map[month]) for month, rows in sorted(by_month.items()) if rows]
    if len(monthly_rows) < 4:
        raise ValueError("Regime Clustering requires at least four monthly observations.")

    frame = pd.DataFrame(monthly_rows)
    feature_frame = frame.reindex(columns=list(FEATURES)).apply(pd.to_numeric, errors="coerce")
    medians = feature_frame.median(numeric_only=True)
    feature_frame = feature_frame.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(feature_frame.to_numpy(dtype=float))
    cluster_count, silhouette = _best_cluster_count(matrix)
    model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
    labels = model.fit_predict(matrix)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(matrix)

    frame["cluster_id"] = labels.astype(int)
    frame["pca_x"] = coords[:, 0]
    frame["pca_y"] = coords[:, 1]

    monthly: list[dict[str, Any]] = []
    matrix_rows = matrix.tolist()
    for index, row in frame.iterrows():
        distances = []
        for other_index, other in frame.iterrows():
            if index == other_index:
                continue
            distance = float(np.linalg.norm(np.asarray(matrix_rows[index]) - np.asarray(matrix_rows[other_index])))
            distances.append({
                "month": other["month"],
                "cluster_id": int(other["cluster_id"]),
                "official_return": float(other["official_return"]),
                "distance": distance,
                "same_cluster": int(other["cluster_id"]) == int(row["cluster_id"]),
            })
        distances.sort(key=lambda item: item["distance"])
        feature_values = {feature: _number(row.get(feature)) for feature in FEATURES}
        zscores = {feature: float(matrix_rows[index][position]) for position, feature in enumerate(FEATURES)}
        monthly.append({
            "month": row["month"],
            "official_return": float(row["official_return"]),
            "outcome": row["outcome"],
            "sessions": int(row["sessions"]),
            "cluster_id": int(row["cluster_id"]),
            "pca_x": float(row["pca_x"]),
            "pca_y": float(row["pca_y"]),
            "features": feature_values,
            "feature_zscores": zscores,
            "similar_months": distances[:5],
        })

    cluster_profiles: list[dict[str, Any]] = []
    for cluster_id in sorted(frame["cluster_id"].unique().tolist()):
        cluster = frame[frame["cluster_id"] == cluster_id].copy()
        zscores: dict[str, float | None] = {}
        profile: dict[str, float | None] = {}
        for feature in FEATURES:
            profile[feature] = _mean(cluster[feature].tolist())
        idxs = cluster.index.tolist()
        for position, feature in enumerate(FEATURES):
            zscores[feature] = _mean([matrix_rows[idx][position] for idx in idxs])
        positives = int((cluster["official_return"] >= 0).sum())
        negatives = int((cluster["official_return"] < 0).sum())
        cluster_profiles.append({
            "cluster_id": int(cluster_id),
            "months": int(len(cluster)),
            "average_return": _mean(cluster["official_return"].tolist()),
            "positive_months": positives,
            "negative_months": negatives,
            "negative_rate": float(negatives / len(cluster)) if len(cluster) else 0.0,
            "severe_negative_months": int((cluster["official_return"] <= SEVERE_MONTH_THRESHOLD).sum()),
            "dominant_outcome": Counter(cluster["outcome"].tolist()).most_common(1)[0][0],
            "features": profile,
            "feature_zscores": zscores,
            "months_list": cluster["month"].tolist(),
        })

    feature_importance = []
    for position, feature in enumerate(FEATURES):
        spread = float(np.std([matrix_rows[idx][position] for idx in range(len(matrix_rows))]))
        between = float(np.std([profile["feature_zscores"].get(feature) or 0.0 for profile in cluster_profiles]))
        feature_importance.append({
            "feature": feature,
            "cluster_separation": between,
            "overall_dispersion": spread,
        })
    feature_importance.sort(key=lambda row: float(row["cluster_separation"]), reverse=True)

    if silhouette >= 0.50:
        separation_quality = "strong"
    elif silhouette >= 0.25:
        separation_quality = "moderate"
    else:
        separation_quality = "weak"
    severe_distribution = []
    severe_months = [row for row in monthly if float(row.get("official_return") or 0.0) <= SEVERE_MONTH_THRESHOLD]
    for cluster_id in sorted(set(int(row["cluster_id"]) for row in monthly)):
        cluster_severe = [row for row in severe_months if int(row["cluster_id"]) == cluster_id]
        severe_distribution.append({
            "cluster_id": cluster_id,
            "severe_months": len(cluster_severe),
            "months": [row["month"] for row in cluster_severe],
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "method": {
            "name": "similar_periods_regime_clustering",
            "algorithm": "agglomerative_clustering",
            "distance_space": "standardized_feature_space",
            "cluster_count_selection": "best_silhouette_between_2_and_6",
            "projection": "pca_2d",
            "target_used_for_clustering": False,
            "features": list(FEATURES),
            "purpose": "group similar months and reveal recurring regimes before policy design",
        },
        "summary": {
            "months": len(monthly),
            "cluster_count": int(cluster_count),
            "silhouette_score": float(silhouette),
            "separation_quality": separation_quality,
            "positive_months": int(sum(1 for row in monthly if row["official_return"] >= 0)),
            "negative_months": int(sum(1 for row in monthly if row["official_return"] < 0)),
            "severe_negative_months": int(sum(1 for row in monthly if row["official_return"] <= SEVERE_MONTH_THRESHOLD)),
        },
        "readiness": {
            "status": "exploratory_diagnostic",
            "policy_ready": False,
            "requires_counterfactual_replay": False,
            "reason": "Clustering is exploratory and intended to reveal similar periods and natural regimes, not to drive policy directly.",
        },
        "monthly": monthly,
        "clusters": cluster_profiles,
        "severe_distribution": severe_distribution,
        "feature_importance": feature_importance,
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_.tolist()],
    }
    return result
