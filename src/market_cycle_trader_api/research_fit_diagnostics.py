from __future__ import annotations

from statistics import median
from typing import Any, Iterable


FIT_DIAGNOSTICS_VERSION = "9.0.0"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _auc(metrics: dict[str, Any] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    return _number(metrics.get("auc"))


def assess_binary_fit(
    training_metrics: dict[str, Any] | None,
    validation_metrics: dict[str, Any] | None,
    oos_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify fit quality without changing model parameters or operational decisions.

    AUC is used because it does not depend on the threshold selected on validation data.
    Low skill even on the fit partition is treated as *possible* underfitting rather than
    proof: weak features/target signal is an explicit alternative explanation.
    """

    train_auc = _auc(training_metrics)
    validation_auc = _auc(validation_metrics)
    oos_auc = _auc(oos_metrics)
    train_to_validation_gap = None if train_auc is None or validation_auc is None else train_auc - validation_auc
    train_to_oos_gap = None if train_auc is None or oos_auc is None else train_auc - oos_auc
    validation_to_oos_gap = None if validation_auc is None or oos_auc is None else validation_auc - oos_auc

    status = "INCONCLUSIVE"
    reason = "insufficient_metrics"
    alternative_explanation = None

    if train_auc is not None and validation_auc is not None and oos_auc is not None:
        if train_auc >= 0.70 and train_to_oos_gap >= 0.15 and oos_auc < 0.60:
            status = "OVERFITTING_RISK"
            reason = "training_skill_does_not_generalize"
        elif train_auc <= 0.58 and validation_auc <= 0.58 and oos_auc <= 0.58:
            status = "POSSIBLE_UNDERFITTING"
            reason = "limited_skill_already_on_training_partition"
            alternative_explanation = "weak_feature_or_target_signal"
        elif train_auc >= 0.60 and oos_auc >= 0.56 and abs(train_to_oos_gap) <= 0.12:
            status = "HEALTHY_FIT"
            reason = "training_skill_is_preserved_out_of_sample"
        elif train_auc >= 0.62 and validation_auc >= 0.56 and oos_auc < 0.54:
            status = "UNSTABLE_GENERALIZATION"
            reason = "validation_skill_deteriorates_out_of_sample"
        elif train_auc < 0.60 and oos_auc < 0.55:
            status = "POSSIBLE_UNDERFITTING"
            reason = "low_training_and_out_of_sample_skill"
            alternative_explanation = "weak_feature_or_target_signal"

    return {
        "diagnostics_version": FIT_DIAGNOSTICS_VERSION,
        "status": status,
        "reason": reason,
        "alternative_explanation": alternative_explanation,
        "training_auc": train_auc,
        "validation_auc": validation_auc,
        "oos_auc": oos_auc,
        "train_to_validation_gap": train_to_validation_gap,
        "train_to_oos_gap": train_to_oos_gap,
        "validation_to_oos_gap": validation_to_oos_gap,
        "thresholds": {
            "possible_underfitting_max_auc": 0.58,
            "overfitting_min_training_auc": 0.70,
            "overfitting_min_train_to_oos_gap": 0.15,
            "healthy_min_training_auc": 0.60,
            "healthy_min_oos_auc": 0.56,
            "healthy_max_abs_train_to_oos_gap": 0.12,
        },
        "diagnostic_only": True,
        "changes_model_parameters": False,
        "changes_strategy_decisions": False,
    }


def aggregate_fit_assessments(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in items if isinstance(item, dict)]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "INCONCLUSIVE")
        counts[status] = counts.get(status, 0) + 1

    def _median(key: str) -> float | None:
        values = [_number(row.get(key)) for row in rows]
        clean = [value for value in values if value is not None]
        return float(median(clean)) if clean else None

    conclusive = len(rows) - counts.get("INCONCLUSIVE", 0)
    status = "INCONCLUSIVE"
    reason = "insufficient_conclusive_folds"
    if rows and counts.get("OVERFITTING_RISK", 0) >= max(1, conclusive // 2 + conclusive % 2):
        status = "OVERFITTING_RISK"
        reason = "overfitting_risk_repeats_across_folds"
    elif rows and counts.get("POSSIBLE_UNDERFITTING", 0) >= max(1, conclusive // 2 + conclusive % 2):
        status = "POSSIBLE_UNDERFITTING"
        reason = "low_training_skill_repeats_across_folds"
    elif rows and counts.get("HEALTHY_FIT", 0) >= max(1, conclusive // 2 + conclusive % 2):
        status = "HEALTHY_FIT"
        reason = "training_skill_generalizes_across_folds"
    elif rows and counts.get("UNSTABLE_GENERALIZATION", 0) >= max(1, conclusive // 2 + conclusive % 2):
        status = "UNSTABLE_GENERALIZATION"
        reason = "generalization_deterioration_repeats_across_folds"

    return {
        "diagnostics_version": FIT_DIAGNOSTICS_VERSION,
        "status": status,
        "reason": reason,
        "folds": len(rows),
        "status_counts": counts,
        "median_training_auc": _median("training_auc"),
        "median_validation_auc": _median("validation_auc"),
        "median_oos_auc": _median("oos_auc"),
        "median_train_to_oos_gap": _median("train_to_oos_gap"),
        "interpretation_guardrail": "Possible underfitting is not proof of insufficient model capacity; weak feature or target signal remains an alternative explanation until a controlled capacity probe is run.",
        "diagnostic_only": True,
        "requires_capacity_probe_before_parameter_change": True,
    }
