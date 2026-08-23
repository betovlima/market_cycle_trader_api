from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ANALYSIS_VERSION,
    INNER_VALIDATION_SHARE,
    MIN_TRAIN_DATES,
    OPPORTUNITY_FEATURES,
    PROBABILITY_BINS,
    RANDOM_STATE,
    SCHEMA_VERSION,
    TARGET_DESCRIPTION,
    TARGET_NAME,
    TRANSITION_BASE_FEATURES,
    TRANSITION_CONTEXT_FEATURES,
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1.0 else probabilities <= right)
        if not np.any(mask):
            continue
        error += (float(np.sum(mask)) / total) * abs(float(np.mean(probabilities[mask])) - float(np.mean(y_true[mask])))
    return float(error)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "auc": _safe_auc(y_true, clipped),
        "brier": float(brier_score_loss(y_true, clipped)) if len(y_true) else None,
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])) if len(y_true) else None,
        "calibration_error": _calibration_error(y_true, clipped),
        "average_probability": float(np.mean(clipped)) if len(clipped) else None,
    }


def _fit_model(kind: str, x: pd.DataFrame, y: pd.Series):
    if kind == "logistic_regression":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        model.fit(x, y)
        return model
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(x, y)
        return model
    raise ValueError(f"Unknown model kind: {kind}")


def _predict(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _balanced_accuracy(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    prediction = probabilities >= float(threshold)
    positive = y_true == 1
    negative = y_true == 0
    tpr = float(np.mean(prediction[positive])) if np.any(positive) else 0.0
    tnr = float(np.mean(~prediction[negative])) if np.any(negative) else 0.0
    return 0.5 * (tpr + tnr)


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.15, 0.75, 61)
    ranked = [
        (_balanced_accuracy(y_true, probabilities, float(value)), -abs(float(value) - 0.5), float(value))
        for value in candidates
    ]
    score, _, threshold = max(ranked)
    return float(threshold), float(score)


def _prepare_matrix(frame: pd.DataFrame, feature_names: Iterable[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    names = list(feature_names)
    numeric = frame.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
    active_medians = medians if medians is not None else numeric.median(numeric_only=True)
    return numeric.fillna(active_medians).fillna(0.0), active_medians


def _inner_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["timestamp"], utc=True).dropna().dt.normalize().unique())
    if len(dates) < MIN_TRAIN_DATES:
        cut = max(1, int(round(len(dates) * (1.0 - INNER_VALIDATION_SHARE))))
    else:
        validation_dates = max(30, int(round(len(dates) * INNER_VALIDATION_SHARE)))
        cut = max(1, len(dates) - validation_dates)
    boundary = dates[min(cut, len(dates) - 1)]
    stamps = pd.to_datetime(frame["timestamp"], utc=True).dt.normalize()
    fit = frame[stamps < boundary].copy()
    validation = frame[stamps >= boundary].copy()
    if fit.empty or validation.empty:
        split = max(1, int(len(frame) * 0.8))
        fit = frame.iloc[:split].copy()
        validation = frame.iloc[split:].copy()
    return fit, validation


def _model_selection(train: pd.DataFrame, features: tuple[str, ...], target: str) -> dict[str, Any]:
    fit, validation = _inner_split(train)
    x_fit, medians = _prepare_matrix(fit, features)
    x_validation, _ = _prepare_matrix(validation, features, medians)
    y_fit = fit[target].astype(int)
    y_validation = validation[target].astype(int).to_numpy()
    candidates: list[dict[str, Any]] = []
    for kind in ("logistic_regression", "lightgbm"):
        try:
            model = _fit_model(kind, x_fit, y_fit)
            probabilities = _predict(model, x_validation)
            metrics = _metrics(y_validation, probabilities)
            threshold, balanced = _select_threshold(y_validation, probabilities)
            candidates.append({
                "model": kind,
                "validation": metrics,
                "threshold": threshold,
                "threshold_balanced_accuracy": balanced,
            })
        except Exception as exc:
            candidates.append({"model": kind, "status": "unavailable", "message": str(exc)})
    available = [row for row in candidates if row.get("validation")]
    if not available:
        raise RuntimeError("No Decision Science model could be fitted.")
    available.sort(key=lambda row: (
        float((row.get("validation") or {}).get("brier") or 1.0),
        -float((row.get("validation") or {}).get("auc") or 0.0),
    ))
    return {"selected": available[0]["model"], "candidates": candidates, "threshold": available[0]["threshold"]}


def _probability_bins(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in zip(PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if not np.any(mask):
            continue
        rows.append({
            "lower": float(left),
            "upper": min(1.0, float(right)),
            "rows": int(np.sum(mask)),
            "average_probability": float(np.mean(probabilities[mask])),
            "realized_positive_rate": float(np.mean(y_true[mask])),
        })
    return rows


def _top_logistic_coefficients(model: Any, feature_names: tuple[str, ...], limit: int = 12) -> list[dict[str, Any]]:
    try:
        estimator = model.named_steps["model"]
        values = estimator.coef_[0]
    except Exception:
        return []
    rows = [{"feature": name, "coefficient": float(value)} for name, value in zip(feature_names, values)]
    return sorted(rows, key=lambda row: abs(row["coefficient"]), reverse=True)[:limit]


def _oos_opportunity(observations: pd.DataFrame, winner: pd.DataFrame) -> dict[str, Any]:
    feature_names = OPPORTUNITY_FEATURES
    data = observations.copy()
    data[TARGET_NAME] = pd.to_numeric(data[TARGET_NAME], errors="coerce")
    data["fold_id"] = pd.to_numeric(data["fold_id"], errors="coerce")
    data = data.dropna(subset=[TARGET_NAME, "fold_id", "timestamp", "symbol"]).copy()
    data["fold_id"] = data["fold_id"].astype(int)
    folds = sorted(int(value) for value in data["fold_id"].unique())
    fold_reports: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    last_logistic_coefficients: list[dict[str, Any]] = []

    winner_map = winner.copy()
    winner_map["timestamp"] = pd.to_datetime(winner_map["timestamp"], utc=True)
    winner_map = winner_map.set_index("timestamp").to_dict("index")

    for fold_id in folds[1:]:
        train = data[data["fold_id"] < fold_id].copy()
        test = data[data["fold_id"] == fold_id].copy()
        if train.empty or test.empty:
            continue
        selection = _model_selection(train, feature_names, TARGET_NAME)
        x_train, medians = _prepare_matrix(train, feature_names)
        x_test, _ = _prepare_matrix(test, feature_names, medians)
        y_train = train[TARGET_NAME].astype(int)
        y_test = test[TARGET_NAME].astype(int).to_numpy()
        model_metrics: dict[str, Any] = {}
        fitted: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        for kind in ("logistic_regression", "lightgbm"):
            try:
                model = _fit_model(kind, x_train, y_train)
                probability = _predict(model, x_test)
                fitted[kind] = model
                predictions[kind] = probability
                model_metrics[kind] = _metrics(y_test, probability)
            except Exception as exc:
                model_metrics[kind] = {"status": "unavailable", "message": str(exc)}
        selected_kind = str(selection["selected"])
        selected_probability = predictions[selected_kind]
        threshold = float(selection["threshold"])
        scored = test[["timestamp", "symbol", TARGET_NAME]].copy()
        scored["probability"] = selected_probability
        scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
        for timestamp, group in scored.groupby("timestamp", sort=True):
            best_index = group["probability"].idxmax()
            best = group.loc[best_index]
            meta = winner_map.get(timestamp, {})
            selected_asset = str(meta.get("selected_asset") or meta.get("strategy_research_control_asset") or meta.get("final_action_asset") or "")
            actual = group[group["symbol"].astype(str) == selected_asset]
            selected_target = None if actual.empty else _number(actual.iloc[0][TARGET_NAME])
            max_probability = float(best["probability"])
            shadow_rows.append({
                "timestamp": timestamp.isoformat(),
                "month": timestamp.strftime("%Y-%m"),
                "fold_id": int(fold_id),
                "model": selected_kind,
                "threshold": threshold,
                "best_probability_asset": str(best["symbol"]),
                "best_probability": max_probability,
                "shadow_decision": "INVEST" if max_probability >= threshold else "CASH",
                "strategy_selected_asset": selected_asset or None,
                "strategy_selected_significant_growth": selected_target,
            })
        if selected_kind == "logistic_regression" and selected_kind in fitted:
            last_logistic_coefficients = _top_logistic_coefficients(fitted[selected_kind], feature_names)
        fold_reports.append({
            "fold_id": int(fold_id),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "selection": selection,
            "models": model_metrics,
            "selected_model": selected_kind,
            "selected_model_probability_bins": _probability_bins(y_test, selected_probability),
        })

    cash = [row for row in shadow_rows if row["shadow_decision"] == "CASH"]
    invest = [row for row in shadow_rows if row["shadow_decision"] == "INVEST"]
    cash_with_label = [row for row in cash if row.get("strategy_selected_significant_growth") is not None]
    missed = [row for row in cash_with_label if float(row["strategy_selected_significant_growth"]) >= 0.5]
    avoided = [row for row in cash_with_label if float(row["strategy_selected_significant_growth"]) < 0.5]
    monthly: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shadow_rows:
        grouped[row["month"]].append(row)
    for month in sorted(grouped):
        rows = grouped[month]
        cash_rows = [row for row in rows if row["shadow_decision"] == "CASH"]
        invest_rows = [row for row in rows if row["shadow_decision"] == "INVEST"]
        labeled_cash_rows = [row for row in cash_rows if row.get("strategy_selected_significant_growth") is not None]
        missed_rows = [row for row in labeled_cash_rows if float(row["strategy_selected_significant_growth"]) >= 0.5]
        avoided_rows = [row for row in labeled_cash_rows if float(row["strategy_selected_significant_growth"]) < 0.5]
        thresholds = sorted({float(row["threshold"]) for row in rows if _number(row.get("threshold")) is not None})
        fold_ids = sorted({int(row["fold_id"]) for row in rows if _number(row.get("fold_id")) is not None})
        models = sorted({str(row["model"]) for row in rows if row.get("model")})
        average_threshold = float(np.mean([float(row["threshold"]) for row in rows])) if rows else None
        average_best_probability = float(np.mean([row["best_probability"] for row in rows])) if rows else None
        monthly.append({
            "month": month,
            "sessions": len(rows),
            "cash_sessions": len(cash_rows),
            "invest_sessions": len(invest_rows),
            "cash_share": len(cash_rows) / len(rows) if rows else 0.0,
            "average_best_probability": average_best_probability,
            "average_threshold": average_threshold,
            "average_probability_margin": (average_best_probability - average_threshold) if average_best_probability is not None and average_threshold is not None else None,
            "cash_average_best_probability": float(np.mean([row["best_probability"] for row in cash_rows])) if cash_rows else None,
            "invest_average_best_probability": float(np.mean([row["best_probability"] for row in invest_rows])) if invest_rows else None,
            "fold_ids": fold_ids,
            "models": models,
            "thresholds": thresholds,
            "labeled_cash_sessions": len(labeled_cash_rows),
            "missed_opportunity_sessions": len(missed_rows),
            "avoided_non_opportunity_sessions": len(avoided_rows),
            "missed_opportunity_rate": len(missed_rows) / len(labeled_cash_rows) if labeled_cash_rows else None,
            "avoided_non_opportunity_rate": len(avoided_rows) / len(labeled_cash_rows) if labeled_cash_rows else None,
        })
    return {
        "target": {
            "name": TARGET_NAME,
            "description": TARGET_DESCRIPTION,
            "future_label_used_for_training_only": True,
            "future_label_used_for_live_features": False,
        },
        "features": list(feature_names),
        "walk_forward": {
            "warmup_fold": folds[0] if folds else None,
            "scored_folds": [row["fold_id"] for row in fold_reports],
            "folds": fold_reports,
        },
        "shadow_cash": {
            "status": "research_only",
            "sessions": len(shadow_rows),
            "cash_sessions": len(cash),
            "cash_share": len(cash) / len(shadow_rows) if shadow_rows else 0.0,
            "invest_sessions": len(invest),
            "avoided_non_opportunity_rate": len(avoided) / len(cash_with_label) if cash_with_label else None,
            "missed_opportunity_rate": len(missed) / len(cash_with_label) if cash_with_label else None,
            "monthly": monthly,
            "recent_sessions": shadow_rows[-60:],
        },
        "logistic_interpretability": last_logistic_coefficients,
    }


def _transition_frame(observations: pd.DataFrame, winner: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    by_key = observations.copy()
    by_key["timestamp"] = pd.to_datetime(by_key["timestamp"], utc=True)
    lookup = by_key.set_index(["timestamp", "symbol"])
    rows: list[dict[str, Any]] = []
    diff_features = tuple(f"delta_{name}" for name in TRANSITION_BASE_FEATURES)
    feature_names = diff_features + TRANSITION_CONTEXT_FEATURES
    for item in winner.to_dict("records"):
        timestamp = pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        current = str(item.get("current_asset") or item.get("strategy_research_control_asset") or "")
        top1 = str(item.get("top_1_asset") or item.get("best_asset") or "")
        top2 = str(item.get("top_2_asset") or item.get("second_asset") or "")
        challenger = top1 if top1 and top1 != current else top2
        if not current or current == "CASH" or not challenger or challenger == current:
            continue
        try:
            current_row = lookup.loc[(timestamp, current)]
            challenger_row = lookup.loc[(timestamp, challenger)]
        except KeyError:
            continue
        if isinstance(current_row, pd.DataFrame):
            current_row = current_row.iloc[0]
        if isinstance(challenger_row, pd.DataFrame):
            challenger_row = challenger_row.iloc[0]
        current_target = _number(current_row.get(TARGET_NAME))
        challenger_target = _number(challenger_row.get(TARGET_NAME))
        if current_target is None or challenger_target is None:
            continue
        row: dict[str, Any] = {
            "timestamp": timestamp,
            "fold_id": item.get("walk_forward_fold") or item.get("decision_fold_id") or current_row.get("fold_id"),
            "current_asset": current,
            "challenger_asset": challenger,
            "transition_success": int(challenger_target >= 0.5 and current_target < 0.5),
        }
        for name in TRANSITION_BASE_FEATURES:
            left = _number(current_row.get(name))
            right = _number(challenger_row.get(name))
            row[f"delta_{name}"] = None if left is None or right is None else right - left
        for name in TRANSITION_CONTEXT_FEATURES:
            row[name] = _number(item.get(name))
        rows.append(row)
    return pd.DataFrame(rows), feature_names


def _oos_transition(observations: pd.DataFrame, winner: pd.DataFrame) -> dict[str, Any]:
    data, feature_names = _transition_frame(observations, winner)
    if data.empty:
        return {"status": "unavailable", "message": "No eligible incumbent/challenger transition rows were found."}
    data["fold_id"] = pd.to_numeric(data["fold_id"], errors="coerce")
    data = data.dropna(subset=["fold_id", "transition_success"]).copy()
    data["fold_id"] = data["fold_id"].astype(int)
    folds = sorted(int(value) for value in data["fold_id"].unique())
    reports: list[dict[str, Any]] = []
    for fold_id in folds[1:]:
        train = data[data["fold_id"] < fold_id].copy()
        test = data[data["fold_id"] == fold_id].copy()
        if train.empty or test.empty or train["transition_success"].nunique() < 2:
            continue
        selection = _model_selection(train, feature_names, "transition_success")
        x_train, medians = _prepare_matrix(train, feature_names)
        x_test, _ = _prepare_matrix(test, feature_names, medians)
        y_train = train["transition_success"].astype(int)
        y_test = test["transition_success"].astype(int).to_numpy()
        model_metrics: dict[str, Any] = {}
        for kind in ("logistic_regression", "lightgbm"):
            try:
                model = _fit_model(kind, x_train, y_train)
                model_metrics[kind] = _metrics(y_test, _predict(model, x_test))
            except Exception as exc:
                model_metrics[kind] = {"status": "unavailable", "message": str(exc)}
        reports.append({
            "fold_id": int(fold_id),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "positive_rate": float(test["transition_success"].mean()),
            "selection": selection,
            "models": model_metrics,
        })
    return {
        "status": "completed",
        "target": "challenger achieves significant-growth event while incumbent does not",
        "features": list(feature_names),
        "rows": int(len(data)),
        "walk_forward_folds": reports,
    }


def build_analysis(observation_rows: list[dict[str, Any]], winner_rows: list[dict[str, Any]], *, run: dict[str, Any]) -> dict[str, Any]:
    observations = pd.DataFrame(observation_rows)
    winner = pd.DataFrame(winner_rows)
    required = {"timestamp", "symbol", "fold_id", TARGET_NAME}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Decision Science observations are missing fields: {', '.join(missing)}")
    if winner.empty or "timestamp" not in winner.columns:
        raise ValueError("Decision Science requires winner_reference_daily artifacts.")
    observations["timestamp"] = pd.to_datetime(observations["timestamp"], utc=True, errors="coerce")
    winner["timestamp"] = pd.to_datetime(winner["timestamp"], utc=True, errors="coerce")
    observations = observations.dropna(subset=["timestamp"])
    winner = winner.dropna(subset=["timestamp"])
    opportunity = _oos_opportunity(observations, winner)
    transition = _oos_transition(observations, winner)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run.get("id") or ""),
        "strategy_profile_id": run.get("strategy_profile_id"),
        "strategy_profile_name": run.get("strategy_profile_name"),
        "strategy_profile_revision": run.get("strategy_profile_revision"),
        "strategy_configuration_hash": run.get("strategy_configuration_hash"),
        "processing_id": run.get("research_processing_id"),
        "market_data_snapshot_id": run.get("market_data_snapshot_id"),
        "analysis_end_date": run.get("analysis_end_date"),
        "method": {
            "name": "walk_forward_decision_science_shadow_lab",
            "models": ["logistic_regression", "lightgbm"],
            "selection_uses_current_test_fold": False,
            "future_labels_role": "training_and_post_hoc_validation_only",
            "decision_effect": "none_shadow_only",
        },
        "absolute_opportunity": opportunity,
        "leader_transition": transition,
    }
