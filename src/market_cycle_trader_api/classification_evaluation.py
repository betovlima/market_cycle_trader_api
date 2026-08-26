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
    tpr = float(true_positive / positive_count) if positive_count else None
    fpr = float(false_positive / negative_count) if negative_count else None
    specificity = float(true_negative / negative_count) if negative_count else None
    balanced_accuracy = (float((tpr + specificity) / 2.0) if tpr is not None and specificity is not None else None)
    return {
        "threshold": float(threshold),
        "tpr": tpr,
        "fpr": fpr,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _diagnostic_best_point(false_positive_rate: np.ndarray, true_positive_rate: np.ndarray, thresholds: np.ndarray) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
        parsed_threshold = _finite(threshold)
        if parsed_threshold is None:
            continue
        specificity = float(1.0 - float(fpr))
        balanced_accuracy = float((float(tpr) + specificity) / 2.0)
        point = {
            "threshold": parsed_threshold,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "selection_rule": "max_oos_balanced_accuracy_diagnostic_only",
            "diagnostic_only": True,
        }
        rank = (balanced_accuracy, -float(fpr), float(tpr), -abs(parsed_threshold - 0.5))
        candidates.append((rank, point))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def roc_curve_payload(
    y_true: Any,
    scores: Any,
    *,
    operating_threshold: float | None = None,
    operating_point_role: str = "operating_threshold",
    threshold_origin: str | None = None,
    validation_metric_name: str | None = None,
    validation_metric_value: float | None = None,
    max_points: int = 121,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(scores, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[valid].astype(int)
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    operating = _operating_point(labels, probabilities, operating_threshold)
    if isinstance(operating, dict):
        operating["point_role"] = str(operating_point_role or "operating_threshold")
        operating["threshold_origin"] = str(threshold_origin) if threshold_origin else None
        operating["validation_metric_name"] = str(validation_metric_name) if validation_metric_name else None
        operating["validation_metric_value"] = _finite(validation_metric_value)
    base = {
        "auc": None,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "points": [],
        "operating_point": operating,
        "diagnostic_best_point": None,
    }
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return base

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities, drop_intermediate=True)
    auc = float(roc_auc_score(labels, probabilities))
    diagnostic_best = _diagnostic_best_point(false_positive_rate, true_positive_rate, thresholds)
    total_points = len(false_positive_rate)
    limit = max(3, int(max_points))
    if total_points <= limit:
        selected_indices = list(range(total_points))
    else:
        selected_indices = sorted(set(np.linspace(0, total_points - 1, num=limit, dtype=int).tolist()))
        reference_thresholds = [operating_threshold, (diagnostic_best or {}).get("threshold")]
        finite_thresholds = np.asarray([value if math.isfinite(float(value)) else np.nan for value in thresholds], dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(finite_thresholds))
        if len(finite_indices):
            for reference_threshold in reference_thresholds:
                if reference_threshold is None or not math.isfinite(float(reference_threshold)):
                    continue
                closest = int(finite_indices[np.argmin(np.abs(finite_thresholds[finite_indices] - float(reference_threshold)))])
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
        "diagnostic_best_point": diagnostic_best,
    }
