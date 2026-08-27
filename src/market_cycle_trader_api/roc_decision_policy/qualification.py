from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _confidence_z(confidence_level: float) -> float:
    confidence = float(confidence_level)
    if not 0.5 < confidence < 1.0:
        raise ValueError("ROC signal qualification confidence level must be between 0.5 and 1.0.")
    return float(NormalDist().inv_cdf(0.5 + confidence / 2.0))


def _auc_confidence_interval(auc: float, positive_count: int, negative_count: int, *, confidence_level: float) -> tuple[float, float]:
    score = float(auc)
    positives = int(positive_count)
    negatives = int(negative_count)
    if positives <= 0 or negatives <= 0:
        return (float("nan"), float("nan"))
    q1 = score / max(1e-12, 2.0 - score)
    q2 = (2.0 * score * score) / max(1e-12, 1.0 + score)
    variance = (
        score * (1.0 - score)
        + (positives - 1) * (q1 - score * score)
        + (negatives - 1) * (q2 - score * score)
    ) / max(1.0, float(positives * negatives))
    standard_error = math.sqrt(max(0.0, float(variance)))
    z_value = _confidence_z(confidence_level)
    return (
        max(0.0, score - z_value * standard_error),
        min(1.0, score + z_value * standard_error),
    )


def _mean_confidence_interval(values: np.ndarray, *, confidence_level: float) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(sample))
    if len(sample) < 2:
        return (mean, float("nan"), float("nan"))
    standard_error = float(np.std(sample, ddof=1) / math.sqrt(len(sample)))
    z_value = _confidence_z(confidence_level)
    return (mean, mean - z_value * standard_error, mean + z_value * standard_error)


def qualify_signal(
    labels: np.ndarray,
    probabilities: np.ndarray,
    relative_edges: np.ndarray,
    *,
    selected_threshold: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    method = str(settings["qualification_method"]).strip().lower()
    if method != "auc_and_net_edge_confidence":
        raise ValueError(f"Unsupported ROC signal qualification method: {method}.")

    label_values = np.asarray(labels, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    edge_values = np.asarray(relative_edges, dtype=float)
    if not (len(label_values) == len(probability_values) == len(edge_values)):
        raise ValueError("ROC signal qualification inputs must have matching lengths.")
    finite_mask = np.isfinite(probability_values) & np.isfinite(edge_values)
    label_values = label_values[finite_mask]
    probability_values = probability_values[finite_mask]
    edge_values = edge_values[finite_mask]
    if len(label_values) == 0 or len(np.unique(label_values)) < 2:
        return {
            "signal_qualified": False,
            "qualification_status": "abstain",
            "qualification_reasons": ["calibration_classes_unavailable"],
            "qualification_method": method,
        }

    confidence_level = float(settings["qualification_confidence_level"])
    minimum_action_samples = int(settings["minimum_qualification_action_samples"])
    positive_count = int((label_values == 1).sum())
    negative_count = int((label_values == 0).sum())
    auc = float(roc_auc_score(label_values, probability_values))
    auc_lower, auc_upper = _auc_confidence_interval(
        auc,
        positive_count,
        negative_count,
        confidence_level=confidence_level,
    )

    action_mask = probability_values >= float(selected_threshold)
    action_edges = edge_values[action_mask]
    action_labels = label_values[action_mask]
    action_count = int(len(action_edges))
    action_positive_rate = float(np.mean(action_labels)) if action_count else None
    edge_mean, edge_lower, edge_upper = _mean_confidence_interval(action_edges, confidence_level=confidence_level)

    reasons: list[str] = []
    if not math.isfinite(auc_lower) or auc_lower <= 0.5:
        reasons.append("auc_confidence_includes_no_skill")
    if action_count < minimum_action_samples:
        reasons.append("insufficient_selected_action_samples")
    if not math.isfinite(edge_lower) or edge_lower <= 0.0:
        reasons.append("net_edge_confidence_not_positive")

    qualified = not reasons
    return {
        "signal_qualified": bool(qualified),
        "qualification_status": "qualified" if qualified else "abstain",
        "qualification_reasons": reasons,
        "qualification_method": method,
        "qualification_confidence_level": confidence_level,
        "qualification_no_skill_auc": 0.5,
        "qualification_break_even_net_edge": 0.0,
        "calibration_auc_ci_lower": auc_lower,
        "calibration_auc_ci_upper": auc_upper,
        "qualification_action_samples": action_count,
        "qualification_action_positive_rate": action_positive_rate,
        "qualification_net_edge_mean": edge_mean if math.isfinite(edge_mean) else None,
        "qualification_net_edge_ci_lower": edge_lower if math.isfinite(edge_lower) else None,
        "qualification_net_edge_ci_upper": edge_upper if math.isfinite(edge_upper) else None,
    }
