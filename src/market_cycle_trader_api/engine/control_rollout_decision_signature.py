"""v10.8.49 diagnostic study of the 321 v10.8.48 rollout decisions.

This module does not create a trading policy.  It builds a causal, decision-time
feature table for the already generated paired rollout labels and evaluates
small fixed models with chronological fold transfer:

  fold 2 <- train fold 1
  fold 3 <- train folds 1+2

No hyperparameter search, threshold tuning, order path or Winner mutation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from .control_deep_learning_tcn import FEATURES
from .operational_control_contract import prepare_operational_control_panel

RANDOM_SEED = 20260930
LIQUIDITY_LOOKBACK = 20
PARTICIPATION_RATE = 0.10

MODEL_SPECS = {
    "logistic_regression": LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=8,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_SEED,
    ),
    "lightgbm_small": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=10,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=RANDOM_SEED,
        verbosity=-1,
    ),
}


def _safe_ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _loc(frame: pd.DataFrame, date: pd.Timestamp) -> int:
    idx = frame.index.get_indexer([pd.Timestamp(date)])
    return int(idx[0]) if len(idx) == 1 and int(idx[0]) >= 0 else -1


def _prior_liquidity(frame: pd.DataFrame, loc: int) -> dict[str, float]:
    start = max(0, loc - LIQUIDITY_LOOKBACK)
    prior = pd.to_numeric(frame["volume"].iloc[start:loc], errors="coerce")
    prior = prior[np.isfinite(prior) & (prior >= 0)]
    median = float(prior.median()) if len(prior) else float("nan")
    estimated_capacity = (
        math.floor(median * PARTICIPATION_RATE)
        if math.isfinite(median) else float("nan")
    )
    return {
        "prior_median_volume": median,
        "estimated_capacity_shares": float(estimated_capacity),
        "prior_volume_observations": float(len(prior)),
    }


def _state_lookup(curve: pd.DataFrame, date: pd.Timestamp) -> dict[str, Any]:
    rows = curve.loc[curve.index == pd.Timestamp(date)]
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {
        "state_cash": float(row.get("cash", np.nan)),
        "state_cash_weight": float(row.get("cash_weight", np.nan)),
        "state_shares": float(row.get("shares", np.nan)),
        "state_holding_days": float(row.get("holding_days", np.nan)),
        "state_selected_asset": str(row.get("selected_asset", "")),
        "state_equity_mark": float(row.get("strategy_equity", np.nan)),
    }


def build_rollout_decision_dataset(
    *,
    bars: dict[str, pd.DataFrame],
    completed_session: str,
    labels: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    frames, _, _audit = prepare_operational_control_panel(
        bars,
        completed_session=completed_session,
        config=config,
    )
    curve = baseline_curve.copy()
    curve.index = pd.DatetimeIndex(pd.to_datetime(curve.index, utc=True))
    rows: list[dict[str, Any]] = []

    for raw in labels.to_dict("records"):
        date = pd.Timestamp(raw["decision_date"])
        if date.tzinfo is None:
            date = date.tz_localize("UTC")
        else:
            date = date.tz_convert("UTC")
        incumbent = str(raw["incumbent_asset"]).upper()
        candidate = str(raw["control_target_asset"]).upper()
        if incumbent not in frames or candidate not in frames:
            raise ValueError("Rollout label references an asset outside the frozen panel.")
        iframe = frames[incumbent]
        cframe = frames[candidate]
        iloc, cloc = _loc(iframe, date), _loc(cframe, date)
        if iloc < 0 or cloc < 0:
            raise ValueError("Rollout decision date missing from frozen panel.")

        record: dict[str, Any] = {
            "decision_date": date,
            "source_fold_id": int(raw["source_fold_id"]),
            "incumbent_asset": incumbent,
            "candidate_asset": candidate,
            "initial_equity": float(raw["initial_equity"]),
            "rotate_ending_equity": float(raw["rotate_ending_equity"]),
            "hold_ending_equity": float(raw["hold_ending_equity"]),
            "delta_capital_usd": float(raw["delta_capital_usd"]),
            "delta_capital_fraction": float(raw["delta_capital_fraction"]),
            "rotate_better": int(bool(raw["rotate_better"])),
        }
        record.update(_state_lookup(curve, date))

        for feature in FEATURES:
            iv = float(iframe[feature].iloc[iloc])
            cv = float(cframe[feature].iloc[cloc])
            record[f"incumbent__{feature}"] = iv
            record[f"candidate__{feature}"] = cv
            record[f"relative__{feature}"] = cv - iv

        iprice = float(iframe["close"].iloc[iloc])
        cprice = float(cframe["close"].iloc[cloc])
        iliq = _prior_liquidity(iframe, iloc)
        cliq = _prior_liquidity(cframe, cloc)
        record.update({
            "incumbent_close": iprice,
            "candidate_close": cprice,
            "price_ratio_candidate_incumbent": _safe_ratio(cprice, iprice),
            "incumbent_prior_median_volume": iliq["prior_median_volume"],
            "candidate_prior_median_volume": cliq["prior_median_volume"],
            "liquidity_volume_ratio": _safe_ratio(
                cliq["prior_median_volume"], iliq["prior_median_volume"]
            ),
            "incumbent_estimated_capacity_shares": iliq["estimated_capacity_shares"],
            "candidate_estimated_capacity_shares": cliq["estimated_capacity_shares"],
            "incumbent_capacity_notional": (
                iliq["estimated_capacity_shares"] * iprice
            ),
            "candidate_capacity_notional": (
                cliq["estimated_capacity_shares"] * cprice
            ),
            "incumbent_capacity_equity_ratio": _safe_ratio(
                iliq["estimated_capacity_shares"] * iprice,
                float(raw["initial_equity"]),
            ),
            "candidate_capacity_equity_ratio": _safe_ratio(
                cliq["estimated_capacity_shares"] * cprice,
                float(raw["initial_equity"]),
            ),
        })
        rows.append(record)

    dataset = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    if len(dataset) != len(labels):
        raise AssertionError("Decision signature dataset lost rollout labels.")
    return dataset


def _numeric_features(dataset: pd.DataFrame) -> list[str]:
    excluded = {
        "source_fold_id", "initial_equity", "rotate_ending_equity",
        "hold_ending_equity", "delta_capital_usd", "delta_capital_fraction",
        "rotate_better",
    }
    return [
        column for column in dataset.columns
        if column not in excluded
        and column not in {"decision_date", "incumbent_asset", "candidate_asset",
                           "state_selected_asset"}
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def feature_diagnostics(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _numeric_features(dataset)
    correlation_rows = []
    class_rows = []
    quantile_rows = []
    target = dataset["delta_capital_fraction"].astype(float)

    for feature in features:
        series = pd.to_numeric(dataset[feature], errors="coerce")
        valid = series.notna() & target.notna()
        x, y = series[valid], target[valid]
        pearson = spearman = float("nan")
        if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1:
            pearson = float(pearsonr(x, y).statistic)
            spearman = float(spearmanr(x, y).statistic)
        correlation_rows.append({
            "feature": feature,
            "n": int(valid.sum()),
            "pearson_vs_delta_capital_fraction": pearson,
            "spearman_vs_delta_capital_fraction": spearman,
        })

        positive = series[dataset["rotate_better"] == 1].dropna()
        negative = series[dataset["rotate_better"] == 0].dropna()
        pooled = float(series.std(ddof=1))
        class_rows.append({
            "feature": feature,
            "rotate_mean": float(positive.mean()) if len(positive) else None,
            "hold_mean": float(negative.mean()) if len(negative) else None,
            "mean_difference_rotate_minus_hold": (
                float(positive.mean() - negative.mean())
                if len(positive) and len(negative) else None
            ),
            "standardized_mean_difference": (
                float((positive.mean() - negative.mean()) / pooled)
                if len(positive) and len(negative)
                and math.isfinite(pooled) and pooled > 1e-12 else None
            ),
        })

        if series.notna().sum() >= 20 and series.nunique(dropna=True) >= 4:
            try:
                bins = pd.qcut(series, 4, duplicates="drop")
                grouped = dataset.assign(_bin=bins).dropna(subset=["_bin"]).groupby(
                    "_bin", observed=True
                )
                for interval, group in grouped:
                    quantile_rows.append({
                        "feature": feature,
                        "quantile": str(interval),
                        "n": int(len(group)),
                        "rotate_better_rate": float(group["rotate_better"].mean()),
                        "mean_delta_capital_fraction": float(
                            group["delta_capital_fraction"].mean()
                        ),
                    })
            except ValueError:
                pass

    correlations = pd.DataFrame(correlation_rows).sort_values(
        "spearman_vs_delta_capital_fraction",
        key=lambda s: s.abs(),
        ascending=False,
    )
    classes = pd.DataFrame(class_rows).sort_values(
        "standardized_mean_difference",
        key=lambda s: s.abs(),
        ascending=False,
    )
    quantiles = pd.DataFrame(quantile_rows)
    return correlations, classes, quantiles


def fold_feature_stability(dataset: pd.DataFrame) -> pd.DataFrame:
    features = _numeric_features(dataset)
    rows = []
    for feature in features:
        values = []
        for fold in sorted(dataset["source_fold_id"].unique()):
            subset = dataset.loc[dataset["source_fold_id"] == fold]
            x = pd.to_numeric(subset[feature], errors="coerce")
            y = subset["delta_capital_fraction"].astype(float)
            valid = x.notna() & y.notna()
            corr = float("nan")
            if valid.sum() >= 5 and x[valid].nunique() > 1 and y[valid].nunique() > 1:
                corr = float(spearmanr(x[valid], y[valid]).statistic)
            values.append(corr)
            rows.append({
                "feature": feature,
                "fold_id": int(fold),
                "n": int(valid.sum()),
                "spearman_vs_delta_capital_fraction": corr,
            })
        finite = [v for v in values if math.isfinite(v)]
        sign_consistent = (
            len(finite) >= 2
            and (all(v > 0 for v in finite) or all(v < 0 for v in finite))
        )
        for row in rows[-len(values):]:
            row["sign_consistent_across_available_folds"] = sign_consistent
    return pd.DataFrame(rows)


def _pipeline(model: Any, features: list[str]) -> Pipeline:
    if isinstance(model, LogisticRegression):
        prep = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        prep = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
    return Pipeline([
        ("prep", ColumnTransformer([("numeric", prep, features)], remainder="drop")),
        ("model", model),
    ])


def temporal_model_evaluation(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _numeric_features(dataset)
    results = []
    importance_rows = []

    for test_fold, train_folds in ((2, (1,)), (3, (1, 2))):
        train = dataset.loc[dataset["source_fold_id"].isin(train_folds)].copy()
        test = dataset.loc[dataset["source_fold_id"] == test_fold].copy()
        if train.empty or test.empty:
            continue
        X_train, y_train = train[features], train["rotate_better"].astype(int)
        X_test, y_test = test[features], test["rotate_better"].astype(int)

        for name, spec in MODEL_SPECS.items():
            pipe = _pipeline(spec, features)
            pipe.fit(X_train, y_train)
            probability = pipe.predict_proba(X_test)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            auc = (
                float(roc_auc_score(y_test, probability))
                if y_test.nunique() == 2 else None
            )
            results.append({
                "model": name,
                "test_fold": test_fold,
                "train_folds": ",".join(map(str, train_folds)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_positive_rate": float(y_test.mean()),
                "accuracy": float(accuracy_score(y_test, prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, prediction)
                ),
                "roc_auc": auc,
                "brier": float(brier_score_loss(y_test, probability)),
                "precision": float(
                    precision_score(y_test, prediction, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, prediction, zero_division=0)
                ),
            })

            if test_fold == 3:
                fitted = pipe.named_steps["model"]
                if hasattr(fitted, "feature_importances_"):
                    importance = np.asarray(fitted.feature_importances_, dtype=float)
                elif hasattr(fitted, "coef_"):
                    importance = np.abs(np.asarray(fitted.coef_[0], dtype=float))
                else:
                    importance = np.zeros(len(features), dtype=float)
                for feature, value in zip(features, importance):
                    importance_rows.append({
                        "model": name,
                        "training_scope": "folds_1_2_only",
                        "feature": feature,
                        "importance": float(value),
                    })

    return pd.DataFrame(results), pd.DataFrame(importance_rows).sort_values(
        ["model", "importance"], ascending=[True, False]
    )


def summarize_signature(
    dataset: pd.DataFrame,
    correlations: pd.DataFrame,
    stability: pd.DataFrame,
    model_results: pd.DataFrame,
) -> dict[str, Any]:
    stable = []
    if not stability.empty:
        pivot = stability.groupby("feature").agg(
            folds=("fold_id", "count"),
            sign_consistent=("sign_consistent_across_available_folds", "max"),
            mean_abs_spearman=("spearman_vs_delta_capital_fraction",
                               lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
        ).reset_index()
        stable = pivot.loc[
            (pivot["folds"] >= 3)
            & pivot["sign_consistent"]
            & (pivot["mean_abs_spearman"] >= 0.10)
        ].sort_values("mean_abs_spearman", ascending=False)["feature"].tolist()

    predictive = []
    if not model_results.empty:
        for model, group in model_results.groupby("model"):
            fold2 = group.loc[group["test_fold"] == 2, "balanced_accuracy"]
            fold3 = group.loc[group["test_fold"] == 3, "balanced_accuracy"]
            if (
                not fold2.empty and not fold3.empty
                and float(fold2.iloc[0]) > 0.50
                and float(fold3.iloc[0]) > 0.50
            ):
                predictive.append(str(model))

    return {
        "rows": int(len(dataset)),
        "rotate_better_count": int(dataset["rotate_better"].sum()),
        "hold_better_or_equal_count": int(
            (dataset["rotate_better"] == 0).sum()
        ),
        "feature_count": len(_numeric_features(dataset)),
        "stable_descriptive_features": stable,
        "models_above_random_balanced_accuracy_both_test_folds": predictive,
        "predictive_signal_detected": bool(predictive),
        "interpretation_rule": (
            "Predictive evidence requires the same fixed model to exceed "
            "balanced accuracy 0.50 on both chronological transfer tests. "
            "Descriptive correlations alone are not sufficient."
        ),
    }


def write_signature_artifacts(
    *,
    output: Path,
    dataset: pd.DataFrame,
    correlations: pd.DataFrame,
    class_comparison: pd.DataFrame,
    quantiles: pd.DataFrame,
    stability: pd.DataFrame,
    model_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    summary: dict[str, Any],
) -> list[str]:
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "rollout_decision_dataset.csv": dataset,
        "feature_correlations.csv": correlations,
        "feature_class_comparison.csv": class_comparison,
        "feature_quantiles.csv": quantiles,
        "fold_feature_stability.csv": stability,
        "model_results.csv": model_results,
        "feature_importance.csv": feature_importance,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False, float_format="%.17g")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return ["summary.json", *artifacts.keys()]
