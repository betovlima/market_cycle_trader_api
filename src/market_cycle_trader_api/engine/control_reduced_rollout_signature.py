"""v10.8.50 reduced rollout signature confirmation.

This is a diagnostic confirmation study, not a trading policy. It consumes the
frozen v10.8.49 causal decision dataset and evaluates a small predeclared
feature set with fixed logistic models and chronological transfer tests.

Confirmation criterion (declared before execution):
- same reduced multivariate model;
- balanced accuracy > 0.50 on fold 2 and fold 3;
- ROC AUC > 0.50 on fold 2 and fold 3.

Univariate models are explanatory only and cannot confirm the signature.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 20260930

REDUCED_FEATURES = (
    "incumbent__return_20",
    "incumbent__return_60",
    "incumbent__ema_distance_20",
    "incumbent__ema_distance_50",
    "incumbent__rsi_14",
    "incumbent__channel_position_50",
    "incumbent_capacity_equity_ratio",
    "state_shares",
    "candidate__channel_position_50",
)

MULTIVARIATE_MODELS = {
    "logistic_reduced_c1": LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "logistic_reduced_c025": LogisticRegression(
        C=0.25,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
}


def _pipeline(model: LogisticRegression, features: list[str]) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def _score(
    *,
    name: str,
    feature_set: str,
    features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: LogisticRegression,
    test_fold: int,
    train_folds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X_train = train[features]
    y_train = train["rotate_better"].astype(int)
    X_test = test[features]
    y_test = test["rotate_better"].astype(int)
    pipe = _pipeline(model, features)
    pipe.fit(X_train, y_train)
    probability = pipe.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, probability)) if y_test.nunique() == 2 else None
    result = {
        "model": name,
        "feature_set": feature_set,
        "test_fold": int(test_fold),
        "train_folds": ",".join(map(str, train_folds)),
        "feature_count": len(features),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y_test, probability)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "predicted_positive_rate": float(prediction.mean()),
        "mean_probability": float(np.mean(probability)),
    }
    coefficients = []
    fitted = pipe.named_steps["model"]
    for feature, coefficient in zip(features, fitted.coef_[0]):
        coefficients.append({
            "model": name,
            "feature_set": feature_set,
            "test_fold": int(test_fold),
            "train_folds": ",".join(map(str, train_folds)),
            "feature": feature,
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
        })
    return result, coefficients


def evaluate_reduced_signature(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "source_fold_id", "rotate_better",
        *REDUCED_FEATURES,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"v10.8.50 dataset missing columns: {missing}")
    if len(dataset) != 321:
        raise ValueError("v10.8.50 requires the exact 321-event v10.8.49 dataset.")

    results: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []

    for test_fold, train_folds in ((2, (1,)), (3, (1, 2))):
        train = dataset.loc[dataset["source_fold_id"].isin(train_folds)].copy()
        test = dataset.loc[dataset["source_fold_id"] == test_fold].copy()
        if train.empty or test.empty:
            raise ValueError("Expected chronological folds 1, 2 and 3.")

        for name, model in MULTIVARIATE_MODELS.items():
            row, coefs = _score(
                name=name,
                feature_set="reduced_9",
                features=list(REDUCED_FEATURES),
                train=train,
                test=test,
                model=model,
                test_fold=test_fold,
                train_folds=train_folds,
            )
            results.append(row)
            coefficients.extend(coefs)

        for feature in REDUCED_FEATURES:
            name = f"univariate__{feature}"
            model = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
            row, coefs = _score(
                name=name,
                feature_set="univariate",
                features=[feature],
                train=train,
                test=test,
                model=model,
                test_fold=test_fold,
                train_folds=train_folds,
            )
            results.append(row)
            coefficients.extend(coefs)

    frame = pd.DataFrame(results)
    coef_frame = pd.DataFrame(coefficients)

    confirmed = []
    for name in MULTIVARIATE_MODELS:
        group = frame.loc[frame["model"] == name]
        if set(group["test_fold"].tolist()) != {2, 3}:
            continue
        ok = bool(
            (group["balanced_accuracy"] > 0.50).all()
            and group["roc_auc"].notna().all()
            and (group["roc_auc"] > 0.50).all()
        )
        if ok:
            confirmed.append(name)

    univariate_consistent = []
    for feature in REDUCED_FEATURES:
        name = f"univariate__{feature}"
        group = frame.loc[frame["model"] == name]
        if len(group) == 2 and (
            (group["balanced_accuracy"] > 0.50).all()
            and group["roc_auc"].notna().all()
            and (group["roc_auc"] > 0.50).all()
        ):
            univariate_consistent.append(feature)

    summary = {
        "rows": int(len(dataset)),
        "reduced_features": list(REDUCED_FEATURES),
        "multivariate_models": list(MULTIVARIATE_MODELS.keys()),
        "confirmation_rule": (
            "Same reduced multivariate model must have balanced_accuracy > 0.50 "
            "and roc_auc > 0.50 on both chronological tests: fold2<-fold1 and "
            "fold3<-fold1+fold2."
        ),
        "confirmed_multivariate_models": confirmed,
        "reduced_signature_confirmed": bool(confirmed),
        "univariate_features_meeting_same_metrics_both_folds": univariate_consistent,
        "univariate_models_are_confirmation_eligible": False,
        "threshold": 0.50,
        "hyperparameter_search": False,
    }
    return frame, coef_frame, summary


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature), group in coefficients.groupby(["model", "feature"]):
        if set(group["test_fold"].tolist()) != {2, 3}:
            continue
        fold2 = float(group.loc[group["test_fold"] == 2, "coefficient"].iloc[0])
        fold3 = float(group.loc[group["test_fold"] == 3, "coefficient"].iloc[0])
        rows.append({
            "model": model,
            "feature": feature,
            "fold2_coefficient": fold2,
            "fold3_coefficient": fold3,
            "same_sign": bool((fold2 > 0 and fold3 > 0) or (fold2 < 0 and fold3 < 0)),
            "min_abs_coefficient": float(min(abs(fold2), abs(fold3))),
            "max_abs_coefficient": float(max(abs(fold2), abs(fold3))),
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "same_sign", "min_abs_coefficient"],
        ascending=[True, False, False],
    )


def write_reduced_signature_artifacts(
    *,
    output: Path,
    results: pd.DataFrame,
    coefficients: pd.DataFrame,
    stability: pd.DataFrame,
    summary: dict[str, Any],
) -> list[str]:
    output.mkdir(parents=True, exist_ok=False)
    results.to_csv(output / "reduced_model_results.csv", index=False, float_format="%.17g")
    coefficients.to_csv(
        output / "reduced_logistic_coefficients.csv",
        index=False,
        float_format="%.17g",
    )
    stability.to_csv(
        output / "coefficient_stability.csv",
        index=False,
        float_format="%.17g",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return [
        "summary.json",
        "reduced_model_results.csv",
        "reduced_logistic_coefficients.csv",
        "coefficient_stability.csv",
    ]
