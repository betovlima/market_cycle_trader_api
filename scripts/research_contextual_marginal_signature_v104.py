from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.4"
EXPERIMENT_NAME = "contextual_marginal_signature_regularized_structure_test"

# Keep the exact feature space from v1.0.3 except for the one algebraically
# redundant feature documented below. The experiment consumes only the frozen
# counterfactual dataset; it performs no market-data fetch and no replay.
MODEL_FEATURES = [
    "relative__return_5",
    "relative__return_20",
    "relative__return_60",
    "relative__vol_20",
    "relative__vol_60",
    "relative__ema_distance_20",
    "relative__trend_efficiency_20",
    "corr_20_to_universe",
    "corr_60_to_universe",
    "universe_dispersion_return_20",
    "universe_dispersion_vol_20",
    "candidate_rank_return_20",
    "candidate_rank_return_60",
    "market_return_20",
    "market_vol_20",
]

REMOVED_EXACT_REDUNDANCY = {
    "feature": "relative__momentum_acceleration_5_20",
    "identity": "relative__momentum_acceleration_5_20 = relative__return_5 - 0.25 * relative__return_20",
}

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OLS, Ridge and Lasso on the frozen Contextual Marginal Signature "
            "counterfactual panel without running any new market replay."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to contextual_signature_dataset.csv from v1.0.3")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--output-dir", required=True)
    return parser


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"decision_date", "candidate", "evaluation_status", "delta_log_capital", *MODEL_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Dataset is missing required columns: " + ", ".join(missing))
    completed = frame.loc[frame["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    if completed.empty:
        raise RuntimeError("Dataset contains no completed counterfactual rows.")
    completed["decision_ts"] = pd.to_datetime(completed["decision_date"], utc=True, errors="raise")
    completed = completed.sort_values(["decision_ts", "candidate"]).reset_index(drop=True)
    return completed


def _prepare_split(
    dataset: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], pd.Series, pd.Series]:
    boundary = _utc_timestamp(validation_start)
    train = dataset.loc[dataset["decision_ts"] < boundary].copy()
    test = dataset.loc[dataset["decision_ts"] >= boundary].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Temporal split is empty: train={len(train)} test={len(test)}")

    train_x = train.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    test_x = test.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0, ddof=0)
    active = [feature for feature in MODEL_FEATURES if np.isfinite(stds[feature]) and float(stds[feature]) > 1e-12]
    if not active:
        raise RuntimeError("No active features remain after train-only standardization.")

    train_z = (train_x[active] - means[active]) / stds[active]
    test_z = (test_x[active] - means[active]) / stds[active]
    y_train = pd.to_numeric(train["delta_log_capital"], errors="raise")
    y_test = pd.to_numeric(test["delta_log_capital"], errors="raise")
    return train, test, train_z, test_z, y_train, y_test, active, means[active], stds[active]


def _date_forward_folds(train: pd.DataFrame, min_train_dates: int = 4) -> list[tuple[pd.Index, pd.Index]]:
    dates = list(pd.Series(train["decision_date"].astype(str).unique()).sort_values())
    if len(dates) <= min_train_dates:
        return []
    folds: list[tuple[pd.Index, pd.Index]] = []
    for split_position in range(min_train_dates, len(dates)):
        fit_dates = set(dates[:split_position])
        validation_date = dates[split_position]
        fit_idx = train.index[train["decision_date"].astype(str).isin(fit_dates)]
        valid_idx = train.index[train["decision_date"].astype(str) == validation_date]
        if len(fit_idx) and len(valid_idx):
            folds.append((fit_idx, valid_idx))
    return folds


def _fit_predict_ols(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    design_train = np.column_stack([np.ones(len(x_train)), x_train])
    design_test = np.column_stack([np.ones(len(x_test)), x_test])
    beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    return design_test @ beta, beta, None


def _fit_regularized(kind: str, alpha: float, x_train: np.ndarray, y_train: np.ndarray):
    if kind == "ridge":
        model = Ridge(alpha=float(alpha), fit_intercept=True)
    elif kind == "lasso":
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=100000, tol=1e-8, selection="cyclic")
    else:
        raise ValueError(kind)
    model.fit(x_train, y_train)
    return model


def _inner_cv_select_alpha(
    *,
    kind: str,
    alphas: list[float],
    train: pd.DataFrame,
    train_z: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[float, pd.DataFrame]:
    folds = _date_forward_folds(train)
    if not folds:
        raise RuntimeError("Not enough distinct training dates for chronological inner validation.")

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        fold_mae: list[float] = []
        fold_spearman: list[float] = []
        for fit_idx, valid_idx in folds:
            x_fit = train_z.loc[fit_idx].to_numpy(dtype=float)
            y_fit = y_train.loc[fit_idx].to_numpy(dtype=float)
            x_valid = train_z.loc[valid_idx].to_numpy(dtype=float)
            y_valid = y_train.loc[valid_idx].to_numpy(dtype=float)
            model = _fit_regularized(kind, alpha, x_fit, y_fit)
            prediction = np.asarray(model.predict(x_valid), dtype=float)
            fold_mae.append(float(np.mean(np.abs(prediction - y_valid))))
            if len(prediction) >= 3:
                value = pd.Series(prediction).corr(pd.Series(y_valid), method="spearman")
                if pd.notna(value):
                    fold_spearman.append(float(value))
        rows.append({
            "model": kind,
            "alpha": float(alpha),
            "folds": int(len(folds)),
            "mean_mae": float(np.mean(fold_mae)) if fold_mae else None,
            "mean_within_date_spearman": float(np.mean(fold_spearman)) if fold_spearman else None,
        })

    frame = pd.DataFrame(rows)
    # Selection is intentionally based on MAE only. The final OOS ranking metrics remain untouched.
    valid = frame.dropna(subset=["mean_mae"]).sort_values(["mean_mae", "alpha"], ascending=[True, True])
    if valid.empty:
        raise RuntimeError(f"Chronological alpha selection failed for {kind}.")
    return float(valid.iloc[0]["alpha"]), frame


def _prediction_report(
    *,
    model_name: str,
    prediction: np.ndarray,
    test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    predictions = test[["decision_date", "candidate", "ending_capital_delta_rate", "delta_log_capital"]].copy()
    predictions["model"] = model_name
    predictions["predicted_delta_log_capital"] = np.asarray(prediction, dtype=float)
    predictions["predicted_positive"] = predictions["predicted_delta_log_capital"] > 0
    predictions["actual_positive"] = y_test.to_numpy(dtype=float) > 0

    per_date: list[dict[str, Any]] = []
    for decision_date, group in predictions.groupby("decision_date", sort=True):
        chosen = group.sort_values(["predicted_delta_log_capital", "candidate"], ascending=[False, True]).iloc[0]
        oracle = group.sort_values(["delta_log_capital", "candidate"], ascending=[False, True]).iloc[0]
        spearman = group["predicted_delta_log_capital"].corr(group["delta_log_capital"], method="spearman")
        per_date.append({
            "model": model_name,
            "decision_date": decision_date,
            "chosen_candidate": chosen["candidate"],
            "chosen_actual_delta_log": float(chosen["delta_log_capital"]),
            "chosen_actual_delta_rate": float(chosen["ending_capital_delta_rate"]),
            "mean_candidate_delta_log": float(group["delta_log_capital"].mean()),
            "oracle_candidate": oracle["candidate"],
            "oracle_delta_log": float(oracle["delta_log_capital"]),
            "rank_spearman": None if pd.isna(spearman) else float(spearman),
        })
    per_date_frame = pd.DataFrame(per_date)

    pred_series = pd.Series(np.asarray(prediction, dtype=float), index=test.index)
    actual_series = pd.Series(y_test.to_numpy(dtype=float), index=test.index)
    majority_positive = bool(np.mean(y_train.to_numpy(dtype=float) > 0) >= 0.5)
    pearson = pred_series.corr(actual_series, method="pearson")
    spearman = pred_series.corr(actual_series, method="spearman")
    summary = {
        "model": model_name,
        "test_rows": int(len(test)),
        "test_dates": int(test["decision_date"].nunique()),
        "mae_delta_log_capital": float(np.mean(np.abs(pred_series.to_numpy() - actual_series.to_numpy()))),
        "pearson_prediction_vs_actual": None if pd.isna(pearson) else float(pearson),
        "spearman_prediction_vs_actual": None if pd.isna(spearman) else float(spearman),
        "sign_accuracy": float(np.mean((pred_series.to_numpy() > 0) == (actual_series.to_numpy() > 0))),
        "majority_sign_baseline_accuracy": float(np.mean((actual_series.to_numpy() > 0) == majority_positive)),
        "mean_cross_sectional_rank_spearman": float(per_date_frame["rank_spearman"].dropna().mean()) if not per_date_frame.empty else None,
        "top1_positive_rate": float((per_date_frame["chosen_actual_delta_log"] > 0).mean()) if not per_date_frame.empty else None,
        "top1_negative_rate": float((per_date_frame["chosen_actual_delta_log"] < 0).mean()) if not per_date_frame.empty else None,
        "mean_top1_actual_delta_log": float(per_date_frame["chosen_actual_delta_log"].mean()) if not per_date_frame.empty else None,
        "mean_candidate_delta_log": float(per_date_frame["mean_candidate_delta_log"].mean()) if not per_date_frame.empty else None,
        "mean_oracle_delta_log": float(per_date_frame["oracle_delta_log"].mean()) if not per_date_frame.empty else None,
    }
    return predictions, summary, per_date_frame


def _zero_regime_report(dataset: pd.DataFrame, tolerance: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset.copy()
    y = pd.to_numeric(frame["delta_log_capital"], errors="raise")
    frame["effect_regime"] = np.where(np.abs(y) <= tolerance, "zero", np.where(y > 0, "positive", "negative"))
    counts = (
        frame.groupby("effect_regime", as_index=False)
        .agg(rows=("candidate", "size"), dates=("decision_date", "nunique"), mean_delta_log=("delta_log_capital", "mean"))
    )
    feature_rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        for regime, subset_idx in frame.groupby("effect_regime").groups.items():
            subset = values.loc[subset_idx].dropna()
            feature_rows.append({
                "feature": feature,
                "effect_regime": regime,
                "n": int(len(subset)),
                "mean": float(subset.mean()) if len(subset) else None,
                "median": float(subset.median()) if len(subset) else None,
                "std": float(subset.std(ddof=0)) if len(subset) else None,
            })
    return counts, pd.DataFrame(feature_rows)


def _coefficient_frame(model_name: str, active: list[str], coefficients: np.ndarray, intercept: float) -> pd.DataFrame:
    rows = [{"model": model_name, "feature": "intercept", "coefficient": float(intercept), "selected": True}]
    for feature, value in zip(active, coefficients):
        rows.append({
            "model": model_name,
            "feature": feature,
            "coefficient": float(value),
            "selected": bool(abs(float(value)) > 1e-12),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = _parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    validation_start = _utc_timestamp(args.validation_start)
    train, test, train_z, test_z, y_train, y_test, active, means, stds = _prepare_split(dataset, validation_start)

    x_train = train_z.to_numpy(dtype=float)
    x_test = test_z.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float)

    ols_prediction, ols_beta, _ = _fit_predict_ols(x_train, y_train_np, x_test)
    ridge_alpha, ridge_cv = _inner_cv_select_alpha(
        kind="ridge", alphas=RIDGE_ALPHAS, train=train, train_z=train_z, y_train=y_train
    )
    lasso_alpha, lasso_cv = _inner_cv_select_alpha(
        kind="lasso", alphas=LASSO_ALPHAS, train=train, train_z=train_z, y_train=y_train
    )

    ridge_model = _fit_regularized("ridge", ridge_alpha, x_train, y_train_np)
    lasso_model = _fit_regularized("lasso", lasso_alpha, x_train, y_train_np)
    ridge_prediction = ridge_model.predict(x_test)
    lasso_prediction = lasso_model.predict(x_test)

    prediction_frames: list[pd.DataFrame] = []
    per_date_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for name, prediction in [
        ("ols", ols_prediction),
        ("ridge", ridge_prediction),
        ("lasso", lasso_prediction),
    ]:
        pred_frame, summary, per_date = _prediction_report(
            model_name=name, prediction=np.asarray(prediction, dtype=float), test=test, y_test=y_test, y_train=y_train
        )
        prediction_frames.append(pred_frame)
        per_date_frames.append(per_date)
        summaries.append(summary)

    coefficient_frames = [
        _coefficient_frame("ols", active, np.asarray(ols_beta[1:], dtype=float), float(ols_beta[0])),
        _coefficient_frame("ridge", active, np.asarray(ridge_model.coef_, dtype=float), float(ridge_model.intercept_)),
        _coefficient_frame("lasso", active, np.asarray(lasso_model.coef_, dtype=float), float(lasso_model.intercept_)),
    ]

    zero_counts, zero_features = _zero_regime_report(dataset)
    _write_frame(output_dir / "regularized_structure_predictions.csv", pd.concat(prediction_frames, ignore_index=True))
    _write_frame(output_dir / "regularized_structure_per_date.csv", pd.concat(per_date_frames, ignore_index=True))
    _write_frame(output_dir / "regularized_structure_coefficients.csv", pd.concat(coefficient_frames, ignore_index=True))
    _write_frame(output_dir / "regularized_structure_inner_cv.csv", pd.concat([ridge_cv, lasso_cv], ignore_index=True))
    _write_frame(output_dir / "zero_regime_counts.csv", zero_counts)
    _write_frame(output_dir / "zero_regime_feature_summary.csv", zero_features)

    matrix = np.column_stack([np.ones(len(x_train)), x_train])
    matrix_rank = int(np.linalg.matrix_rank(matrix))
    condition_number = float(np.linalg.cond(matrix))
    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "source_dataset": str(dataset_path),
        "dataset_rows": int(len(dataset)),
        "dataset_dates": int(dataset["decision_date"].nunique()),
        "validation_start": validation_start.date().isoformat(),
        "train_rows": int(len(train)),
        "train_dates": int(train["decision_date"].nunique()),
        "test_rows": int(len(test)),
        "test_dates": int(test["decision_date"].nunique()),
        "active_features": active,
        "removed_exact_redundancy": REMOVED_EXACT_REDUNDANCY,
        "train_design_rank": matrix_rank,
        "train_design_columns": int(matrix.shape[1]),
        "train_design_condition_number": condition_number,
        "ridge_selected_alpha": ridge_alpha,
        "lasso_selected_alpha": lasso_alpha,
        "lasso_selected_features": [
            feature for feature, value in zip(active, np.asarray(lasso_model.coef_, dtype=float)) if abs(float(value)) > 1e-12
        ],
        "models": summaries,
        "decision_rule": (
            "Regularized evidence is considered interesting only if OOS ranking remains positive and top-1 actual "
            "delta improves over the candidate mean; sign accuracy is descriptive because the target is zero-inflated."
        ),
        "research_policy": (
            "No replay, no market-data fetch, no tuning on 2024-2025. Ridge/Lasso alpha is selected only by "
            "chronological, whole-decision-date validation inside the pre-2024 training period."
        ),
    }
    _write_json(output_dir / "regularized_structure_summary.json", summary)

    print(json.dumps({
        "status": "completed",
        "script_version": SCRIPT_VERSION,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "lasso_selected_features": summary["lasso_selected_features"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
