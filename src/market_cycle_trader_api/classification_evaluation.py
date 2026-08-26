from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, Any] | None:
    if threshold is None or not math.isfinite(float(threshold)) or len(labels) == 0:
        return None
    prediction = scores >= float(threshold)
    positive = labels == 1
    negative = labels == 0
    true_positive = int((prediction & positive).sum())
    false_positive = int((prediction & negative).sum())
    true_negative = int((~prediction & negative).sum())
    false_negative = int((~prediction & positive).sum())
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    return {
        "threshold": float(threshold),
        "tpr": float(true_positive / positive_count) if positive_count else None,
        "fpr": float(false_positive / negative_count) if negative_count else None,
        "specificity": float(true_negative / negative_count) if negative_count else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def roc_curve_payload(
    y_true: Any,
    scores: Any,
    *,
    operating_threshold: float | None = None,
    max_points: int = 121,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(scores, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[valid].astype(int)
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    base = {
        "auc": None,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "points": [],
        "operating_point": _operating_point(labels, probabilities, operating_threshold),
    }
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return base

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities, drop_intermediate=True)
    auc = float(roc_auc_score(labels, probabilities))
    total_points = len(false_positive_rate)
    limit = max(3, int(max_points))
    if total_points <= limit:
        selected_indices = list(range(total_points))
    else:
        selected_indices = sorted(set(np.linspace(0, total_points - 1, num=limit, dtype=int).tolist()))
        if operating_threshold is not None and math.isfinite(float(operating_threshold)):
            finite_thresholds = np.asarray([value if math.isfinite(float(value)) else np.nan for value in thresholds], dtype=float)
            finite_indices = np.flatnonzero(np.isfinite(finite_thresholds))
            if len(finite_indices):
                closest = int(finite_indices[np.argmin(np.abs(finite_thresholds[finite_indices] - float(operating_threshold)))])
                selected_indices = sorted(set([*selected_indices, closest]))

    points = []
    for index in selected_indices:
        threshold = _finite(thresholds[index])
        points.append({
            "fpr": float(false_positive_rate[index]),
            "tpr": float(true_positive_rate[index]),
            "threshold": threshold,
        })
    return {
        **base,
        "auc": auc,
        "points": points,
    }
