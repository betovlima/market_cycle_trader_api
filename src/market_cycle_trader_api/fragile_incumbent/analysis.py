from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score

from .config import (
    ANALYSIS_VERSION,
    FEATURES,
    FOCUS_MONTHS,
    INNER_VALIDATION_SHARE,
    MIN_TRAIN_MONTHS,
    RANDOM_STATE,
    SCHEMA_VERSION,
    SEVERE_MONTH_THRESHOLD,
)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[Any]) -> float | None:
    clean = [_number(value) for value in values]
    valid = [value for value in clean if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    truth = np.asarray(y_true, dtype=int)
    prediction = clipped >= float(threshold)
    return {
        "rows": int(len(truth)),
        "positive_rate": float(np.mean(truth)) if len(truth) else None,
        "auc": _safe_auc(truth, clipped),
        "brier": float(brier_score_loss(truth, clipped)) if len(truth) else None,
        "log_loss": float(log_loss(truth, clipped, labels=[0, 1])) if len(truth) else None,
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)) if len(truth) else None,
        "precision": float(precision_score(truth, prediction, zero_division=0)) if len(truth) else None,
        "recall": float(recall_score(truth, prediction, zero_division=0)) if len(truth) else None,
        "average_probability": float(np.mean(clipped)) if len(clipped) else None,
    }


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(y_true, dtype=int)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return 0.5, 0.5
    ranked: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.15, 0.75, 61):
        score = float(balanced_accuracy_score(truth, probabilities >= float(threshold)))
        ranked.append((score, -abs(float(threshold) - 0.5), float(threshold)))
    score, _, threshold = max(ranked)
    return threshold, score


def _prepare_matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.reindex(columns=list(FEATURES)).apply(pd.to_numeric, errors="coerce")
    active_medians = medians if medians is not None else matrix.median(numeric_only=True)
    return matrix.fillna(active_medians).fillna(0.0), active_medians


def _month_outcome(value: float) -> str:
    if value <= SEVERE_MONTH_THRESHOLD:
        return "severe_negative"
    if value < 0.0:
        return "negative"
    return "positive"


def _fit_lightgbm(fit: pd.DataFrame, validation: pd.DataFrame):
    from lightgbm import LGBMClassifier, early_stopping

    x_fit, medians = _prepare_matrix(fit)
    x_validation, _ = _prepare_matrix(validation, medians)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    callbacks = []
    if len(validation) and validation["target_negative_month"].nunique() > 1:
        callbacks = [early_stopping(40, verbose=False)]
    model.fit(
        x_fit,
        fit["target_negative_month"].astype(int),
        sample_weight=fit["month_weight"].astype(float),
        eval_set=[(x_validation, validation["target_negative_month"].astype(int))] if len(validation) else None,
        callbacks=callbacks,
    )
    return model, medians


def _inner_month_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(str(value) for value in train["month"].dropna().unique())
    if len(months) < 2:
        return train.copy(), train.iloc[0:0].copy()
    validation_months = max(2, int(round(len(months) * INNER_VALIDATION_SHARE)))
    if len(months) >= MIN_TRAIN_MONTHS:
        validation_months = max(3, validation_months)
    validation_months = min(validation_months, len(months) - 1)
    fit_months = set(months[:-validation_months])
    validation_set = set(months[-validation_months:])
    return train[train["month"].isin(fit_months)].copy(), train[train["month"].isin(validation_set)].copy()


def _percentile(value: float | None, reference: list[float]) -> float | None:
    if value is None:
        return None
    clean = sorted(float(item) for item in reference if item is not None and math.isfinite(float(item)))
    if not clean:
        return None
    return float(sum(1 for item in clean if item <= value) / len(clean))


def _cohort_summary(monthly: list[dict[str, Any]], outcome: str | set[str]) -> dict[str, Any]:
    accepted = {outcome} if isinstance(outcome, str) else set(outcome)
    rows = [row for row in monthly if row.get("outcome") in accepted]
    return {
        "months": len(rows),
        "average_return": _mean([row.get("official_return") for row in rows]),
        "average_fragility_probability": _mean([row.get("oos_fragility_probability") for row in rows]),
        "average_weak_leader_share": _mean([row.get("weak_leader_share") for row in rows]),
        "average_hold_share": _mean([row.get("hold_share") for row in rows]),
        "average_rotation_share": _mean([row.get("rotation_share") for row in rows]),
        "features": {
            feature: _mean([((row.get("features") or {}).get(feature)) for row in rows])
            for feature in FEATURES
        },
    }


def _readiness(folds: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [_number((row.get("monthly_metrics") or {}).get("auc")) for row in folds]
    aucs = [value for value in aucs if value is not None]
    if not aucs:
        status = "insufficient_data"
    elif min(aucs) >= 0.65 and sum(aucs) / len(aucs) >= 0.70:
        status = "consistent_research_signal"
    elif min(aucs) >= 0.55 and sum(aucs) / len(aucs) >= 0.65:
        status = "promising_but_not_consistent"
    else:
        status = "not_stable_oos"
    return {
        "status": status,
        "policy_ready": False,
        "requires_counterfactual_replay": True,
        "reason": "Fragile Incumbent / Weak Leader remains diagnostic until HOLD, ROTATE and defensive-exit counterfactuals are validated walk-forward without damaging positive months.",
    }


def build_analysis(
    leadership: dict[str, Any],
    official_monthly_returns: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
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
    winner_by_day = {str(row.get("timestamp") or "")[:10]: dict(row) for row in winner_rows if isinstance(row, dict)}
    if not session_source or not official_map:
        raise ValueError("Fragile Incumbent Research requires Leadership Regime sessions and official Strategy monthly returns.")

    rows: list[dict[str, Any]] = []
    for item in session_source:
        month = str(item.get("month") or str(item.get("timestamp") or "")[:7])
        official_return = official_map.get(month)
        if official_return is None:
            continue
        features = item.get("features") if isinstance(item.get("features"), dict) else {}
        winner = winner_by_day.get(str(item.get("timestamp") or "")[:10], {})
        row = {
            "timestamp": item.get("timestamp"),
            "month": month,
            "fold_id": int(_number(item.get("fold_id")) or 0),
            "state": str(item.get("state") or ""),
            "selected_asset": item.get("selected_asset"),
            "best_asset": item.get("best_asset"),
            "official_return": official_return,
            "outcome": _month_outcome(official_return),
            "target_negative_month": int(official_return < 0.0),
            "decision_is_rotation": bool(winner.get("decision_is_rotation")),
            "decision_reason": winner.get("decision_reason"),
            "realized_forward_return_1": _number(item.get("realized_forward_return_1")),
            "realized_forward_return_5": _number(item.get("realized_forward_return_5")),
            "realized_forward_return_10": _number(item.get("realized_forward_return_10")),
            "position_drawdown_from_peak": _number(features.get("position_drawdown_from_peak")),
            "incumbent_risk_health": _number(features.get("incumbent_risk_health")),
            "position_return_since_entry": _number(features.get("position_return_since_entry")),
            "score_change_from_entry": _number(features.get("score_change_from_entry")),
            "best_vs_second_gap": _number(features.get("best_vs_second_gap")),
            "best_vs_current_gap": _number(winner.get("best_vs_current_gap")),
            "all_horizon_risk_safety": _number(features.get("all_horizon_risk_safety")),
            "best_score_zscore": _number(features.get("best_score_zscore")),
            "short_profit_consensus": _number(features.get("short_profit_consensus")),
            "long_profit_confirmation": _number(features.get("long_profit_confirmation")),
            "horizon_agreement": _number(features.get("horizon_agreement")),
            "current_asset_rank": _number(item.get("current_asset_rank")),
            "recent_rotations_10": _number(item.get("recent_rotations_10")),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Fragile Incumbent Research has no sessions bound to official monthly returns.")

    counts = frame.groupby("month").size().to_dict()
    frame["month_weight"] = frame["month"].map(lambda value: 1.0 / max(1, int(counts.get(value, 1))))
    month_fold = frame.groupby("month")["fold_id"].agg(lambda values: int(values.mode().iloc[-1]) if len(values.mode()) else int(values.iloc[-1])).to_dict()
    frame["research_fold_id"] = frame["month"].map(month_fold).astype(int)
    folds = sorted(int(value) for value in frame["research_fold_id"].unique() if int(value) > 0)
    fold_reports: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []

    for test_fold in folds[1:]:
        test = frame[frame["research_fold_id"] == test_fold].copy()
        test_months = set(str(value) for value in test["month"].dropna().unique())
        train = frame[(frame["research_fold_id"] < test_fold) & (~frame["month"].isin(test_months))].copy()
        if train["month"].nunique() < 6 or test.empty or train["target_negative_month"].nunique() < 2:
            continue
        fit, validation = _inner_month_split(train)
        if fit.empty or validation.empty or fit["target_negative_month"].nunique() < 2:
            continue
        model, medians = _fit_lightgbm(fit, validation)
        x_validation, _ = _prepare_matrix(validation, medians)
        validation_probability = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)
        threshold, threshold_score = _select_threshold(validation["target_negative_month"].to_numpy(dtype=int), validation_probability)
        x_test, _ = _prepare_matrix(test, medians)
        probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        contributions = np.asarray(model.predict(x_test, pred_contrib=True), dtype=float)
        output = test.copy()
        output["oos_fragility_probability"] = probability
        output["fragility_signal"] = probability >= threshold
        for index, feature in enumerate(FEATURES):
            output[f"contribution__{feature}"] = contributions[:, index]
        oos_frames.append(output)

        monthly_test = output.groupby("month", as_index=False).agg(
            target_negative_month=("target_negative_month", "first"),
            oos_fragility_probability=("oos_fragility_probability", "mean"),
            fragility_signal_share=("fragility_signal", "mean"),
        )
        fold_reports.append({
            "fold_id": int(test_fold),
            "train_months": int(train["month"].nunique()),
            "validation_months": int(validation["month"].nunique()),
            "test_months": int(test["month"].nunique()),
            "negative_test_months": int(monthly_test["target_negative_month"].sum()),
            "threshold": float(threshold),
            "threshold_validation_balanced_accuracy": float(threshold_score),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or getattr(model, "n_estimators_", 0) or 0),
            "session_metrics": _metrics(test["target_negative_month"].to_numpy(dtype=int), probability, threshold),
            "monthly_metrics": _metrics(
                monthly_test["target_negative_month"].to_numpy(dtype=int),
                monthly_test["oos_fragility_probability"].to_numpy(dtype=float),
                threshold,
            ),
        })

    oos = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    oos_by_month: dict[str, dict[str, Any]] = {}
    if not oos.empty:
        for month, group in oos.groupby("month"):
            drivers = []
            for feature in FEATURES:
                drivers.append({"feature": feature, "contribution": _mean(group[f"contribution__{feature}"].tolist())})
            drivers.sort(key=lambda row: abs(float(row.get("contribution") or 0.0)), reverse=True)
            oos_by_month[str(month)] = {
                "oos_fragility_probability": _mean(group["oos_fragility_probability"].tolist()),
                "fragility_signal_share": _mean(group["fragility_signal"].astype(float).tolist()),
                "top_drivers": drivers[:6],
            }

    monthly: list[dict[str, Any]] = []
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_month[row["month"]].append(row)
    for month in sorted(by_month):
        month_rows = by_month[month]
        rotations = sum(1 for row in month_rows if row.get("decision_is_rotation"))
        holds = len(month_rows) - rotations
        weak = sum(1 for row in month_rows if row.get("state") == "weak_relative_leader")
        monthly.append({
            "month": month,
            "fold_id": int(month_fold.get(month, month_rows[0].get("fold_id") or 0)),
            "sessions": len(month_rows),
            "official_return": float(official_map[month]),
            "outcome": _month_outcome(float(official_map[month])),
            "weak_leader_share": float(weak / len(month_rows)),
            "hold_sessions": int(holds),
            "rotation_sessions": int(rotations),
            "hold_share": float(holds / len(month_rows)),
            "rotation_share": float(rotations / len(month_rows)),
            "oos_fragility_probability": (oos_by_month.get(month) or {}).get("oos_fragility_probability"),
            "fragility_signal_share": (oos_by_month.get(month) or {}).get("fragility_signal_share"),
            "top_drivers": (oos_by_month.get(month) or {}).get("top_drivers") or [],
            "features": {feature: _mean([row.get(feature) for row in month_rows]) for feature in FEATURES},
        })

    oos_sessions: list[dict[str, Any]] = []
    if not oos.empty:
        for _, item in oos.iterrows():
            oos_sessions.append({
                "timestamp": item.get("timestamp"),
                "month": item.get("month"),
                "fold_id": int(item.get("research_fold_id") or item.get("fold_id") or 0),
                "official_return": _number(item.get("official_return")),
                "target_negative_month": int(item.get("target_negative_month") or 0),
                "oos_fragility_probability": _number(item.get("oos_fragility_probability")),
                "fragility_signal": bool(item.get("fragility_signal")),
                "decision_is_rotation": bool(item.get("decision_is_rotation")),
                "decision_reason": item.get("decision_reason"),
                "selected_asset": item.get("selected_asset"),
                "best_asset": item.get("best_asset"),
                "state": item.get("state"),
                "realized_forward_return_1": _number(item.get("realized_forward_return_1")),
                "realized_forward_return_5": _number(item.get("realized_forward_return_5")),
                "realized_forward_return_10": _number(item.get("realized_forward_return_10")),
                "features": {feature: _number(item.get(feature)) for feature in FEATURES},
                "contributions": {feature: _number(item.get(f"contribution__{feature}")) for feature in FEATURES},
            })

    feature_importance: list[dict[str, Any]] = []
    if not oos.empty:
        negative = oos[oos["target_negative_month"] == 1]
        positive = oos[oos["target_negative_month"] == 0]
        for feature in FEATURES:
            column = f"contribution__{feature}"
            feature_importance.append({
                "feature": feature,
                "mean_abs_contribution": _mean(np.abs(oos[column].to_numpy(dtype=float)).tolist()),
                "mean_contribution_negative_months": _mean(negative[column].tolist()),
                "mean_contribution_positive_months": _mean(positive[column].tolist()),
            })
        feature_importance.sort(key=lambda row: float(row.get("mean_abs_contribution") or 0.0), reverse=True)

    behavior_attribution: dict[str, Any] = {}
    if not oos.empty:
        flagged = oos[oos["fragility_signal"]]
        def behavior_rows(source: pd.DataFrame, rotate: bool) -> dict[str, Any]:
            subset = source[source["decision_is_rotation"] == rotate]
            return {
                "sessions": int(len(subset)),
                "share": float(len(subset) / len(source)) if len(source) else 0.0,
                "average_forward_return_1": _mean(subset["realized_forward_return_1"].tolist()),
                "average_forward_return_5": _mean(subset["realized_forward_return_5"].tolist()),
                "average_forward_return_10": _mean(subset["realized_forward_return_10"].tolist()),
            }
        behavior_attribution = {
            "flagged_sessions": int(len(flagged)),
            "flagged_share": float(len(flagged) / len(oos)) if len(oos) else 0.0,
            "hold": behavior_rows(flagged, False),
            "rotate": behavior_rows(flagged, True),
        }

    positive_months = [row for row in monthly if row.get("outcome") == "positive"]
    focus_details: list[dict[str, Any]] = []
    for focus_month in FOCUS_MONTHS:
        focus_row = next((row for row in monthly if row.get("month") == focus_month), None)
        if focus_row is None:
            continue
        detail = dict(focus_row)
        detail["is_oos_scored"] = focus_row.get("oos_fragility_probability") is not None
        detail["feature_percentiles_vs_positive_months"] = {
            feature: _percentile(
                _number((focus_row.get("features") or {}).get(feature)),
                [
                    _number((row.get("features") or {}).get(feature))
                    for row in positive_months
                    if _number((row.get("features") or {}).get(feature)) is not None
                ],
            )
            for feature in FEATURES
        }
        focus_details.append(detail)

    aggregate_month_metrics = None
    if not oos.empty:
        aggregated = oos.groupby("month", as_index=False).agg(
            target_negative_month=("target_negative_month", "first"),
            oos_fragility_probability=("oos_fragility_probability", "mean"),
        )
        aggregate_month_metrics = _metrics(
            aggregated["target_negative_month"].to_numpy(dtype=int),
            aggregated["oos_fragility_probability"].to_numpy(dtype=float),
            0.5,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "method": {
            "name": "fragile_incumbent_weak_leader_monthly_outcome_learning",
            "model": "lightgbm",
            "target": "official_strategy_monthly_return_below_zero",
            "severe_month_threshold": SEVERE_MONTH_THRESHOLD,
            "monthly_return_source": "strategy_reference_analytics.monthly_returns.simulation_return",
            "features": list(FEATURES),
            "primary_evaluation_unit": "month",
            "training_rows_are_session_level": True,
            "month_equal_weighting": True,
            "causal_features_only": True,
            "realized_forward_returns_role": "post_hoc_behavior_validation_only",
            "policy_effect": "none",
            "purpose": "test whether incumbent deterioration plus weak leadership is a stable OOS failure family before HOLD, ROTATE or defensive-exit intervention",
        },
        "summary": {
            "months": len(monthly),
            "positive_months": sum(1 for row in monthly if row.get("outcome") == "positive"),
            "negative_months": sum(1 for row in monthly if row.get("outcome") in {"negative", "severe_negative"}),
            "severe_negative_months": sum(1 for row in monthly if row.get("outcome") == "severe_negative"),
            "oos_folds": [int(row.get("fold_id") or 0) for row in fold_reports],
            "aggregate_oos_month_metrics": aggregate_month_metrics,
        },
        "cohorts": {
            "positive": _cohort_summary(monthly, "positive"),
            "negative": _cohort_summary(monthly, {"negative", "severe_negative"}),
            "severe_negative": _cohort_summary(monthly, "severe_negative"),
        },
        "folds": fold_reports,
        "feature_importance": feature_importance,
        "behavior_attribution": behavior_attribution,
        "focus_months": focus_details,
        "monthly": monthly,
        "oos_sessions": oos_sessions,
    }
    result["readiness"] = _readiness(fold_reports)
    return result
