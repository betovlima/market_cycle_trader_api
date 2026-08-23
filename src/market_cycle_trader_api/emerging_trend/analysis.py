from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score

from .config import (
    ANALYSIS_VERSION,
    FEATURES,
    FOCUS_MONTHS,
    FORWARD_RETURN_HORIZON,
    INNER_VALIDATION_SHARE,
    LEADER_PERSISTENCE_HORIZON,
    MIN_FORWARD_RETURN,
    MIN_LEADER_PERSISTENCE_SHARE,
    MIN_TRAIN_ROWS,
    RANDOM_STATE,
    SCHEMA_VERSION,
)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[Any]) -> float | None:
    valid = [value for value in (_number(v) for v in values) if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _prepare_matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.reindex(columns=list(FEATURES)).apply(pd.to_numeric, errors="coerce")
    active_medians = medians if medians is not None else matrix.median(numeric_only=True)
    return matrix.fillna(active_medians).fillna(0.0), active_medians


def _inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values("timestamp").copy()
    if len(ordered) < 10:
        return ordered, ordered.iloc[0:0].copy()
    validation_rows = max(40, int(round(len(ordered) * INNER_VALIDATION_SHARE)))
    validation_rows = min(validation_rows, max(1, len(ordered) - 1))
    return ordered.iloc[:-validation_rows].copy(), ordered.iloc[-validation_rows:].copy()


def _fit_model(fit: pd.DataFrame, validation: pd.DataFrame):
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
    if len(validation) and validation["persistent_emerging_leader"].nunique() > 1:
        callbacks = [early_stopping(40, verbose=False)]
    model.fit(
        x_fit,
        fit["persistent_emerging_leader"].astype(int),
        eval_set=[(x_validation, validation["persistent_emerging_leader"].astype(int))] if len(validation) else None,
        callbacks=callbacks,
    )
    return model, medians


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    ranked: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.15, 0.75, 61):
        score = float(balanced_accuracy_score(y_true, probabilities >= threshold))
        ranked.append((score, -abs(float(threshold) - 0.5), float(threshold)))
    score, _, threshold = max(ranked)
    return threshold, score


def _compound(values: list[Any]) -> float | None:
    clean = [_number(value) for value in values]
    if not clean or any(value is None for value in clean):
        return None
    result = 1.0
    for value in clean:
        result *= 1.0 + float(value)
    return float(result - 1.0)


def _readiness(folds: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [_number(row.get("auc")) for row in folds]
    aucs = [value for value in aucs if value is not None]
    if not aucs:
        status = "insufficient_data"
    elif min(aucs) >= 0.65 and sum(aucs) / len(aucs) >= 0.70:
        status = "consistent_research_signal"
    elif min(aucs) >= 0.55 and sum(aucs) / len(aucs) >= 0.60:
        status = "promising_but_not_consistent"
    else:
        status = "not_stable_oos"
    return {
        "status": status,
        "policy_ready": False,
        "requires_counterfactual_replay": True,
        "reason": "Emerging Trend / Delayed Confirmation remains diagnostic until a persistence intervention is replayed walk-forward and shown not to damage false starts or positive months.",
    }


def build_analysis(
    winner_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    *,
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    winner = pd.DataFrame([dict(row) for row in winner_rows if isinstance(row, dict)])
    assets = pd.DataFrame([dict(row) for row in observation_rows if isinstance(row, dict)])
    if winner.empty or assets.empty:
        raise ValueError("Emerging Trend Research requires winner reference daily rows and multi-horizon daily asset observations.")

    winner["timestamp"] = pd.to_datetime(winner["timestamp"], utc=True, errors="coerce")
    assets["timestamp"] = pd.to_datetime(assets["timestamp"], utc=True, errors="coerce")
    winner = winner[winner["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    assets = assets[assets["timestamp"].notna()].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    winner["month"] = winner["timestamp"].dt.strftime("%Y-%m")
    winner = winner[(winner["month"] >= period_start) & (winner["month"] <= period_end)].copy()
    if winner.empty:
        raise ValueError("Emerging Trend Research has no winner reference rows in the requested period.")

    returns = assets[["timestamp", "symbol", "open_to_open_return"]].copy()
    returns["open_to_open_return"] = pd.to_numeric(returns["open_to_open_return"], errors="coerce")
    forward_values: list[float | None] = [None] * len(returns)
    for _, indices in returns.groupby("symbol", sort=False).groups.items():
        idxs = list(indices)
        series = returns.loc[idxs, "open_to_open_return"].tolist()
        for offset, idx in enumerate(idxs):
            window = series[offset:offset + FORWARD_RETURN_HORIZON]
            forward_values[idx] = _compound(window) if len(window) == FORWARD_RETURN_HORIZON else None
    returns["best_forward_return_10"] = forward_values

    asset_features = assets.drop(columns=[column for column in ["open_to_open_return"] if column in assets.columns]).copy()
    frame = winner.merge(asset_features, left_on=["timestamp", "best_asset"], right_on=["timestamp", "symbol"], how="left", suffixes=("", "_best"))
    frame = frame.merge(returns[["timestamp", "symbol", "best_forward_return_10"]], on=["timestamp", "symbol"], how="left")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    leader_persistence: list[float | None] = []
    for index, row in frame.iterrows():
        future = frame.iloc[index:index + LEADER_PERSISTENCE_HORIZON]
        if len(future) < LEADER_PERSISTENCE_HORIZON:
            leader_persistence.append(None)
        else:
            leader_persistence.append(float((future["best_asset"] == row["best_asset"]).mean()))
    frame["future_leader_persistence_share"] = leader_persistence

    equity = pd.to_numeric(frame["strategy_equity"], errors="coerce")
    frame["strategy_forward_return_10"] = equity.shift(-FORWARD_RETURN_HORIZON) / equity - 1.0
    frame["persistent_emerging_leader"] = (
        (pd.to_numeric(frame["future_leader_persistence_share"], errors="coerce") >= MIN_LEADER_PERSISTENCE_SHARE)
        & (pd.to_numeric(frame["best_forward_return_10"], errors="coerce") >= MIN_FORWARD_RETURN)
    ).astype(int)
    frame["strategy_aligned_with_leader"] = frame["selected_asset"].astype(str) == frame["best_asset"].astype(str)
    frame["missed_edge_10"] = pd.to_numeric(frame["best_forward_return_10"], errors="coerce") - pd.to_numeric(frame["strategy_forward_return_10"], errors="coerce")
    frame["delayed_confirmation"] = (frame["persistent_emerging_leader"] == 1) & (~frame["strategy_aligned_with_leader"])

    usable = frame[
        frame["future_leader_persistence_share"].notna()
        & frame["best_forward_return_10"].notna()
        & frame["strategy_forward_return_10"].notna()
    ].copy()
    if usable.empty:
        raise ValueError("Emerging Trend Research has no sessions with complete forward validation data.")

    folds = sorted(int(value) for value in pd.to_numeric(usable.get("decision_fold_id"), errors="coerce").dropna().unique() if int(value) > 0)
    fold_reports: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []
    for test_fold in folds[1:]:
        train = usable[pd.to_numeric(usable["decision_fold_id"], errors="coerce") < test_fold].copy()
        test = usable[pd.to_numeric(usable["decision_fold_id"], errors="coerce") == test_fold].copy()
        if len(train) < MIN_TRAIN_ROWS or test.empty or train["persistent_emerging_leader"].nunique() < 2 or test["persistent_emerging_leader"].nunique() < 2:
            continue
        fit, validation = _inner_split(train)
        if fit.empty or validation.empty or fit["persistent_emerging_leader"].nunique() < 2:
            continue
        model, medians = _fit_model(fit, validation)
        x_validation, _ = _prepare_matrix(validation, medians)
        validation_probability = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)
        threshold, validation_balanced_accuracy = _select_threshold(validation["persistent_emerging_leader"].to_numpy(dtype=int), validation_probability)
        x_test, _ = _prepare_matrix(test, medians)
        probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        contributions = np.asarray(model.predict(x_test, pred_contrib=True), dtype=float)
        output = test.copy()
        output["oos_emerging_trend_probability"] = probability
        output["emerging_trend_signal"] = probability >= threshold
        for index, feature in enumerate(FEATURES):
            output[f"contribution__{feature}"] = contributions[:, index]
        oos_frames.append(output)
        truth = test["persistent_emerging_leader"].to_numpy(dtype=int)
        prediction = probability >= threshold
        fold_reports.append({
            "fold_id": int(test_fold),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "positive_test_rows": int(truth.sum()),
            "positive_rate": float(truth.mean()),
            "auc": _safe_auc(truth, probability),
            "threshold": float(threshold),
            "validation_balanced_accuracy": float(validation_balanced_accuracy),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or getattr(model, "n_estimators_", 0) or 0),
        })

    oos = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    oos_lookup: dict[str, dict[str, Any]] = {}
    if not oos.empty:
        for _, row in oos.iterrows():
            key = str(row.get("timestamp"))
            drivers = []
            for feature in FEATURES:
                drivers.append({"feature": feature, "contribution": _number(row.get(f"contribution__{feature}"))})
            drivers.sort(key=lambda item: abs(float(item.get("contribution") or 0.0)), reverse=True)
            oos_lookup[key] = {
                "probability": _number(row.get("oos_emerging_trend_probability")),
                "signal": bool(row.get("emerging_trend_signal")),
                "top_drivers": drivers[:6],
            }

    session_rows: list[dict[str, Any]] = []
    for _, row in usable.iterrows():
        key = str(row.get("timestamp"))
        oos_detail = oos_lookup.get(key) or {}
        session_rows.append({
            "timestamp": key,
            "month": row.get("month"),
            "fold_id": int(_number(row.get("decision_fold_id")) or 0),
            "best_asset": row.get("best_asset"),
            "selected_asset": row.get("selected_asset"),
            "decision_reason": row.get("decision_reason"),
            "persistent_emerging_leader": bool(row.get("persistent_emerging_leader")),
            "strategy_aligned_with_leader": bool(row.get("strategy_aligned_with_leader")),
            "delayed_confirmation": bool(row.get("delayed_confirmation")),
            "future_leader_persistence_share": _number(row.get("future_leader_persistence_share")),
            "best_forward_return_10": _number(row.get("best_forward_return_10")),
            "strategy_forward_return_10": _number(row.get("strategy_forward_return_10")),
            "missed_edge_10": _number(row.get("missed_edge_10")),
            "oos_emerging_trend_probability": oos_detail.get("probability"),
            "emerging_trend_signal": oos_detail.get("signal"),
            "top_drivers": oos_detail.get("top_drivers") or [],
            "features": {feature: _number(row.get(feature)) for feature in FEATURES},
        })

    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        by_month[str(row.get("month") or "")].append(row)
    monthly: list[dict[str, Any]] = []
    for month in sorted(key for key in by_month if key):
        rows = by_month[month]
        positives = [row for row in rows if row.get("persistent_emerging_leader")]
        delayed = [row for row in positives if row.get("delayed_confirmation")]
        monthly.append({
            "month": month,
            "sessions": len(rows),
            "emerging_leader_sessions": len(positives),
            "emerging_leader_share": float(len(positives) / len(rows)) if rows else 0.0,
            "delayed_confirmation_sessions": len(delayed),
            "delayed_confirmation_share": float(len(delayed) / len(positives)) if positives else 0.0,
            "average_best_forward_return_10": _mean([row.get("best_forward_return_10") for row in positives]),
            "average_strategy_forward_return_10": _mean([row.get("strategy_forward_return_10") for row in positives]),
            "average_missed_edge_10": _mean([row.get("missed_edge_10") for row in positives]),
            "average_oos_probability": _mean([row.get("oos_emerging_trend_probability") for row in rows]),
            "top_assets": [
                {"asset": asset, "sessions": count}
                for asset, count in pd.Series([row.get("best_asset") for row in positives if row.get("best_asset")]).value_counts().head(5).items()
            ],
        })

    feature_importance: list[dict[str, Any]] = []
    if not oos.empty:
        for feature in FEATURES:
            column = f"contribution__{feature}"
            feature_importance.append({
                "feature": feature,
                "mean_abs_contribution": _mean(np.abs(pd.to_numeric(oos[column], errors="coerce").dropna().to_numpy(dtype=float)).tolist()),
            })
        feature_importance.sort(key=lambda row: float(row.get("mean_abs_contribution") or 0.0), reverse=True)

    focus = [row for row in monthly if row.get("month") in FOCUS_MONTHS]
    total_positive = sum(int(row.get("emerging_leader_sessions") or 0) for row in monthly)
    total_delayed = sum(int(row.get("delayed_confirmation_sessions") or 0) for row in monthly)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "method": {
            "name": "emerging_trend_delayed_confirmation",
            "model": "lightgbm",
            "target": "best asset remains leader in >=60% of next 5 sessions and compounds >=3% over next 10 sessions",
            "features": list(FEATURES),
            "future_values_role": "label_and_post_hoc_validation_only",
            "policy_effect": "none",
            "purpose": "identify early signatures of leaders that later persist and quantify sessions where Strategy confirmation was delayed",
        },
        "summary": {
            "sessions": len(session_rows),
            "emerging_leader_sessions": total_positive,
            "emerging_leader_share": float(total_positive / len(session_rows)) if session_rows else 0.0,
            "delayed_confirmation_sessions": total_delayed,
            "delayed_confirmation_share": float(total_delayed / total_positive) if total_positive else 0.0,
            "average_missed_edge_10": _mean([row.get("missed_edge_10") for row in session_rows if row.get("persistent_emerging_leader")]),
            "oos_folds": [int(row.get("fold_id") or 0) for row in fold_reports],
        },
        "folds": fold_reports,
        "feature_importance": feature_importance,
        "focus_months": focus,
        "monthly": monthly,
        "sessions": session_rows,
    }
    result["readiness"] = _readiness(fold_reports)
    return result
