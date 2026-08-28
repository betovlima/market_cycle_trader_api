from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable


FIT_DIAGNOSTICS_VERSION = "9.1.0"


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


def _sample_evidence(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"rows": None, "positive_count": None, "negative_count": None, "positive_rate": None}
    rows_number = _number(metrics.get("rows"))
    rate = _number(metrics.get("positive_rate"))
    rows = int(rows_number) if rows_number is not None and rows_number >= 0 else None
    positive_count = None
    negative_count = None
    if rows is not None and rate is not None:
        positive_count = max(0, min(rows, int(round(rows * rate))))
        negative_count = rows - positive_count
    return {
        "rows": rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": rate,
    }


def _auc_interval(auc: float | None, positive_count: int | None, negative_count: int | None) -> dict[str, float | None]:
    """Approximate 95% AUC interval using the Hanley-McNeil variance estimate."""
    if auc is None or positive_count is None or negative_count is None or positive_count < 2 or negative_count < 2:
        return {"low": None, "high": None, "width": None}
    q1 = auc / (2.0 - auc) if auc < 2.0 else auc
    q2 = (2.0 * auc * auc) / (1.0 + auc) if auc > -1.0 else auc
    variance = (
        auc * (1.0 - auc)
        + (positive_count - 1) * (q1 - auc * auc)
        + (negative_count - 1) * (q2 - auc * auc)
    ) / float(positive_count * negative_count)
    variance = max(0.0, variance)
    se = math.sqrt(variance)
    low = max(0.0, auc - 1.96 * se)
    high = min(1.0, auc + 1.96 * se)
    return {"low": float(low), "high": float(high), "width": float(high - low)}


def _partition_reliability(evidence: dict[str, Any], level: str) -> tuple[str, list[str]]:
    rows = evidence.get("rows")
    positives = evidence.get("positive_count")
    negatives = evidence.get("negative_count")
    reasons: list[str] = []
    if rows is None or positives is None or negatives is None:
        return "LOW", ["missing_sample_counts"]

    if level == "monthly":
        if rows < 12:
            reasons.append("too_few_months")
        if positives < 3:
            reasons.append("too_few_positive_months")
        if negatives < 3:
            reasons.append("too_few_negative_months")
        if reasons:
            return "LOW", reasons
        if rows < 24 or positives < 5 or negatives < 5:
            return "MODERATE", ["limited_monthly_sample"]
        return "HIGH", []

    if rows < 80:
        reasons.append("too_few_sessions")
    if positives < 15:
        reasons.append("too_few_positive_sessions")
    if negatives < 15:
        reasons.append("too_few_negative_sessions")
    if reasons:
        return "LOW", reasons
    if rows < 250 or positives < 30 or negatives < 30:
        return "MODERATE", ["limited_session_sample"]
    return "HIGH", []


def _diagnostic_reliability(
    training: dict[str, Any], validation: dict[str, Any], oos: dict[str, Any], level: str
) -> dict[str, Any]:
    partitions = {"training": training, "validation": validation, "oos": oos}
    rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    partition_status: dict[str, str] = {}
    reasons: list[str] = []
    for name, evidence in partitions.items():
        status, current_reasons = _partition_reliability(evidence, level)
        partition_status[name] = status
        reasons.extend(f"{name}:{reason}" for reason in current_reasons)
    overall = min(partition_status.values(), key=lambda value: rank[value]) if partition_status else "LOW"
    return {"level": overall, "partition_levels": partition_status, "reasons": reasons}


def assess_binary_fit(
    training_metrics: dict[str, Any] | None,
    validation_metrics: dict[str, Any] | None,
    oos_metrics: dict[str, Any] | None,
    *,
    evaluation_level: str = "session",
) -> dict[str, Any]:
    """Classify fit quality and quantify how trustworthy that diagnosis is.

    The fit classification remains diagnostic-only. v9.1 adds evidence sufficiency so a
    large train-to-OOS gap from a tiny validation slice is not presented with the same
    confidence as the same gap measured on hundreds of observations.
    """

    train_auc = _auc(training_metrics)
    validation_auc = _auc(validation_metrics)
    oos_auc = _auc(oos_metrics)
    train_to_validation_gap = None if train_auc is None or validation_auc is None else train_auc - validation_auc
    train_to_oos_gap = None if train_auc is None or oos_auc is None else train_auc - oos_auc
    validation_to_oos_gap = None if validation_auc is None or oos_auc is None else validation_auc - oos_auc

    training_evidence = _sample_evidence(training_metrics)
    validation_evidence = _sample_evidence(validation_metrics)
    oos_evidence = _sample_evidence(oos_metrics)
    reliability = _diagnostic_reliability(training_evidence, validation_evidence, oos_evidence, evaluation_level)

    train_interval = _auc_interval(train_auc, training_evidence.get("positive_count"), training_evidence.get("negative_count"))
    validation_interval = _auc_interval(validation_auc, validation_evidence.get("positive_count"), validation_evidence.get("negative_count"))
    oos_interval = _auc_interval(oos_auc, oos_evidence.get("positive_count"), oos_evidence.get("negative_count"))

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

    if reliability["level"] == "LOW" and status != "INCONCLUSIVE":
        reason = f"{reason}_with_low_sample_confidence"

    return {
        "diagnostics_version": FIT_DIAGNOSTICS_VERSION,
        "evaluation_level": evaluation_level,
        "status": status,
        "reason": reason,
        "alternative_explanation": alternative_explanation,
        "reliability": reliability,
        "sample_evidence": {
            "training": training_evidence,
            "validation": validation_evidence,
            "oos": oos_evidence,
        },
        "auc_uncertainty_95": {
            "training": train_interval,
            "validation": validation_interval,
            "oos": oos_interval,
        },
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
            "monthly_min_rows_for_moderate_reliability": 12,
            "monthly_min_class_count": 3,
            "session_min_rows_for_moderate_reliability": 80,
            "session_min_class_count": 15,
        },
        "diagnostic_only": True,
        "changes_model_parameters": False,
        "changes_strategy_decisions": False,
    }


def aggregate_fit_assessments(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in items if isinstance(item, dict)]
    counts: dict[str, int] = {}
    reliability_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "INCONCLUSIVE")
        counts[status] = counts.get(status, 0) + 1
        reliability = str(((row.get("reliability") or {}).get("level")) or "LOW")
        reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1

    def _median(key: str) -> float | None:
        values = [_number(row.get(key)) for row in rows]
        clean = [value for value in values if value is not None]
        return float(median(clean)) if clean else None

    reliable_rows = [row for row in rows if str(((row.get("reliability") or {}).get("level")) or "LOW") != "LOW"]
    reliable_counts: dict[str, int] = {}
    for row in reliable_rows:
        status = str(row.get("status") or "INCONCLUSIVE")
        reliable_counts[status] = reliable_counts.get(status, 0) + 1

    conclusive = len(reliable_rows) - reliable_counts.get("INCONCLUSIVE", 0)
    status = "INCONCLUSIVE"
    reason = "insufficient_reliable_folds"
    minimum_conclusive_folds = 2
    if conclusive >= minimum_conclusive_folds and reliable_counts.get("OVERFITTING_RISK", 0) >= max(2, conclusive // 2 + conclusive % 2):
        status = "OVERFITTING_RISK"
        reason = "overfitting_risk_repeats_across_reliable_folds"
    elif conclusive >= minimum_conclusive_folds and reliable_counts.get("POSSIBLE_UNDERFITTING", 0) >= max(2, conclusive // 2 + conclusive % 2):
        status = "POSSIBLE_UNDERFITTING"
        reason = "low_training_skill_repeats_across_reliable_folds"
    elif conclusive >= minimum_conclusive_folds and reliable_counts.get("HEALTHY_FIT", 0) >= max(2, conclusive // 2 + conclusive % 2):
        status = "HEALTHY_FIT"
        reason = "training_skill_generalizes_across_reliable_folds"
    elif conclusive >= minimum_conclusive_folds and reliable_counts.get("UNSTABLE_GENERALIZATION", 0) >= max(2, conclusive // 2 + conclusive % 2):
        status = "UNSTABLE_GENERALIZATION"
        reason = "generalization_deterioration_repeats_across_reliable_folds"

    if not rows or not reliable_rows:
        reliability_level = "LOW"
    elif len(reliable_rows) < len(rows):
        reliability_level = "MODERATE"
    elif reliability_counts.get("MODERATE", 0):
        reliability_level = "MODERATE"
    else:
        reliability_level = "HIGH"

    return {
        "diagnostics_version": FIT_DIAGNOSTICS_VERSION,
        "status": status,
        "reason": reason,
        "reliability_level": reliability_level,
        "folds": len(rows),
        "reliable_folds": len(reliable_rows),
        "status_counts": counts,
        "reliable_status_counts": reliable_counts,
        "reliability_counts": reliability_counts,
        "median_training_auc": _median("training_auc"),
        "median_validation_auc": _median("validation_auc"),
        "median_oos_auc": _median("oos_auc"),
        "median_train_to_oos_gap": _median("train_to_oos_gap"),
        "interpretation_guardrail": "A fit warning is only strong evidence when the validation and OOS samples contain enough observations from both classes. Low-sample folds remain visible but do not drive the aggregate diagnosis.",
        "diagnostic_only": True,
        "requires_evidence_reliability_before_parameter_change": True,
    }
