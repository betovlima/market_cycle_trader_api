from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _point_metric(metric: str, fpr: float, tpr: float) -> float:
    specificity = 1.0 - float(fpr)
    if metric == "youden_j":
        return float(tpr) - float(fpr)
    if metric == "balanced_accuracy":
        return (float(tpr) + specificity) / 2.0
    if metric == "distance_to_top_left":
        return -math.sqrt(float(fpr) ** 2 + (1.0 - float(tpr)) ** 2)
    raise ValueError(f"Unsupported ROC threshold selection metric: {metric}.")


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, *, metric: str) -> dict[str, Any]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    candidates: list[tuple[tuple[float, float, float, float, float], dict[str, Any]]] = []
    for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
        parsed = _finite(threshold)
        if parsed is None:
            continue
        score = _point_metric(metric, float(fpr), float(tpr))
        specificity = 1.0 - float(fpr)
        balanced_accuracy = (float(tpr) + specificity) / 2.0
        rank = (score, balanced_accuracy, -float(fpr), float(tpr), parsed)
        candidates.append((rank, {
            "threshold": parsed,
            "selection_metric": metric,
            "selection_score": score,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        }))
    if not candidates:
        raise ValueError("ROC calibration produced no finite threshold candidates.")
    return max(candidates, key=lambda item: item[0])[1]
