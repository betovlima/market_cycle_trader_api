from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..classification_evaluation import roc_curve_payload
from ..engine.temporal_intelligence import (
    _fit_binary_classifier_relaxed,
    _fit_platt_calibrator,
    _prepared_xy,
)


from .threshold_selection import select_threshold

def calibrate_fold_horizon(
    training: dict[str, Any],
    config: Any,
    *,
    fold: dict[str, Any],
    horizon: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    fold_context = training["fold_contexts"][fold_id]
    prepared = {
        split_name: {
            "x": fold_context["splits"][split_name]["x"],
            "targets": fold_context["targets"][int(horizon)][split_name],
        }
        for split_name in ("train", "calibration", "final_fit")
    }
    x_train, y_train = _prepared_xy(prepared["train"], "profit_before_loss")
    x_calibration, y_calibration = _prepared_xy(prepared["calibration"], "profit_before_loss")
    labels = (np.asarray(y_calibration, dtype=float) > 0.0).astype(int)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    minimum_samples = int(settings["minimum_calibration_samples"])
    minimum_class_samples = int(settings["minimum_class_samples"])
    eligible = bool(
        len(labels) >= minimum_samples
        and positive_count >= minimum_class_samples
        and negative_count >= minimum_class_samples
        and len(np.unique(labels)) == 2
    )
    if not eligible:
        return {
            "fold_id": fold_id,
            "horizon": int(horizon),
            "eligible": False,
            "calibration_samples": int(len(labels)),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "reason": "insufficient_calibration_sample",
        }

    validation_model = _fit_binary_classifier_relaxed(x_train, y_train, config)
    raw_calibration = validation_model.predict_proba(x_calibration)[:, 1]
    calibrator = _fit_platt_calibrator(raw_calibration, y_calibration)
    probabilities = np.clip(calibrator.transform(raw_calibration), 0.0, 1.0)
    selected = select_threshold(labels, probabilities, metric=str(settings["selection_metric"]))
    auc = float(roc_auc_score(labels, probabilities))
    roc = roc_curve_payload(
        labels,
        probabilities,
        operating_threshold=float(selected["threshold"]),
        operating_point_role="roc_policy_threshold",
        threshold_origin="chronological_calibration_fold",
        validation_metric_name=str(settings["selection_metric"]),
        validation_metric_value=float(selected["selection_score"]),
        max_points=int(settings["max_curve_points"]),
    )
    return {
        "fold_id": fold_id,
        "horizon": int(horizon),
        "eligible": True,
        "threshold": float(selected["threshold"]),
        "selection_metric": str(settings["selection_metric"]),
        "selection_score": float(selected["selection_score"]),
        "calibration_auc": auc,
        "calibration_samples": int(len(labels)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "calibration_roc": roc,
    }
