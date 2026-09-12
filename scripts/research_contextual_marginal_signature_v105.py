from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.5"
EXPERIMENT_NAME = "contextual_marginal_signature_shallow_conditional_structure_test"

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

TREE_DEPTHS = [1, 2, 3]
MIN_SAMPLES_LEAF = 12
RANK_TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a deliberately shallow nonlinear rule improves contextual "
            "marginal-capital ranking on the frozen v1.0.3 counterfactual panel."
        )
    )
    parser.add_argument("--dataset", required=True)
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
        raise RuntimeError("Dataset contains no completed rows.")
    completed["decision_ts"] = pd.to_datetime(completed["decision_date"], utc=True, errors="raise")
    return completed.sort_values(["decision_ts", "candidate"]).reset_index(drop=True)


def _temporal_split(dataset: pd.DataFrame, boundary: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset.loc[dataset["decision_ts"] < boundary].copy()
    test = dataset.loc[dataset["decision_ts"] >= boundary].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Temporal split is empty: train={len(train)} test={len(test)}")
    return train, test


def _date_forward_folds(train: pd.DataFrame, min_train_dates: int = 4) -> list[tuple[list[str], str]]:
    dates = sorted(train["decision_date"].astype(str).unique().tolist())
    if len(dates) <= min_train_dates:
        return []
    return [(dates[:position], dates[position]) for position in range(min_train_dates, len(dates))]


def _fit_tree(depth: int, x: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(
        max_depth=int(depth),
        min_samples_leaf=int(MIN_SAMPLES_LEAF),
        criterion="squared_error",
        random_state=17,
    )
    model.fit(x, y)
    return model


def _ranking_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = frame[["decision_date", "candidate", "ending_capital_delta_rate", "delta_log_capital"]].copy()
    scored["model"] = model_name
    scored["predicted_delta_log_capital"] = np.asarray(prediction, dtype=float)

    per_date: list[dict[str, Any]] = []
    for decision_date, group in scored.groupby("decision_date", sort=True):
        prediction_range = float(group["predicted_delta_log_capital"].max() - group["predicted_delta_log_capital"].min())
        actual_range = float(group["delta_log_capital"].max() - group["delta_log_capital"].min())
        eligible = prediction_range > RANK_TOLERANCE and actual_range > RANK_TOLERANCE
        row: dict[str, Any] = {
            "model": model_name,
            "decision_date": decision_date,
            "ranking_eligible": bool(eligible),
            "prediction_range": prediction_range,
            "actual_range": actual_range,
            "chosen_candidate": None,
            "chosen_actual_delta_log": None,
            "chosen_actual_delta_rate": None,
            "mean_candidate_delta_log": float(group["delta_log_capital"].mean()),
            "oracle_candidate": None,
            "oracle_delta_log": None,
            "rank_spearman": None,
        }
        if eligible:
            chosen = group.sort_values(
                ["predicted_delta_log_capital", "candidate"], ascending=[False, True]
            ).iloc[0]
            oracle = group.sort_values(["delta_log_capital", "candidate"], ascending=[False, True]).iloc[0]
            rank_value = group["predicted_delta_log_capital"].corr(group["delta_log_capital"], method="spearman")
            row.update(
                {
                    "chosen_candidate": str(chosen["candidate"]),
                    "chosen_actual_delta_log": float(chosen["delta_log_capital"]),
                    "chosen_actual_delta_rate": float(chosen["ending_capital_delta_rate"]),
                    "oracle_candidate": str(oracle["candidate"]),
                    "oracle_delta_log": float(oracle["delta_log_capital"]),
                    "rank_spearman": None if pd.isna(rank_value) else float(rank_value),
                }
            )
        per_date.append(row)

    per_date_frame = pd.DataFrame(per_date)
    eligible = per_date_frame.loc[per_date_frame["ranking_eligible"]].copy()
    y = pd.to_numeric(scored["delta_log_capital"], errors="raise").to_numpy(dtype=float)
    p = scored["predicted_delta_log_capital"].to_numpy(dtype=float)
    pearson = pd.Series(p).corr(pd.Series(y), method="pearson")
    spearman = pd.Series(p).corr(pd.Series(y), method="spearman")
    summary = {
        "model": model_name,
        "rows": int(len(scored)),
        "dates": int(scored["decision_date"].nunique()),
        "ranking_eligible_dates": int(len(eligible)),
        "ranking_date_coverage": float(len(eligible) / max(1, scored["decision_date"].nunique())),
        "mae_delta_log_capital": float(np.mean(np.abs(p - y))),
        "pearson_prediction_vs_actual": None if pd.isna(pearson) else float(pearson),
        "spearman_prediction_vs_actual": None if pd.isna(spearman) else float(spearman),
        "mean_cross_sectional_rank_spearman": (
            float(eligible["rank_spearman"].dropna().mean()) if not eligible.empty else None
        ),
        "top1_positive_rate": (
            float((eligible["chosen_actual_delta_log"] > 0).mean()) if not eligible.empty else None
        ),
        "top1_negative_rate": (
            float((eligible["chosen_actual_delta_log"] < 0).mean()) if not eligible.empty else None
        ),
        "mean_top1_actual_delta_log": (
            float(eligible["chosen_actual_delta_log"].mean()) if not eligible.empty else None
        ),
        "mean_candidate_delta_log": float(per_date_frame["mean_candidate_delta_log"].mean()),
        "mean_oracle_delta_log": (
            float(eligible["oracle_delta_log"].mean()) if not eligible.empty else None
        ),
    }
    return per_date_frame, summary


def _inner_cv(train: pd.DataFrame) -> pd.DataFrame:
    folds = _date_forward_folds(train)
    if not folds:
        raise RuntimeError("Not enough training dates for chronological inner validation.")
    rows: list[dict[str, Any]] = []
    for depth in TREE_DEPTHS:
        fold_rank: list[float] = []
        fold_mae: list[float] = []
        eligible_folds = 0
        for fit_dates, valid_date in folds:
            fit = train.loc[train["decision_date"].astype(str).isin(fit_dates)].copy()
            valid = train.loc[train["decision_date"].astype(str) == valid_date].copy()
            x_fit = fit[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
            x_valid = valid[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
            medians = x_fit.median(axis=0)
            x_fit = x_fit.fillna(medians)
            x_valid = x_valid.fillna(medians)
            y_fit = pd.to_numeric(fit["delta_log_capital"], errors="raise")
            y_valid = pd.to_numeric(valid["delta_log_capital"], errors="raise")
            model = _fit_tree(depth, x_fit, y_fit)
            prediction = np.asarray(model.predict(x_valid), dtype=float)
            fold_mae.append(float(np.mean(np.abs(prediction - y_valid.to_numpy(dtype=float)))))
            prediction_range = float(np.ptp(prediction))
            actual_range = float(np.ptp(y_valid.to_numpy(dtype=float)))
            if prediction_range > RANK_TOLERANCE and actual_range > RANK_TOLERANCE:
                rank_value = pd.Series(prediction).corr(
                    pd.Series(y_valid.to_numpy(dtype=float)), method="spearman"
                )
                if pd.notna(rank_value):
                    fold_rank.append(float(rank_value))
                    eligible_folds += 1
        rows.append(
            {
                "max_depth": int(depth),
                "min_samples_leaf": int(MIN_SAMPLES_LEAF),
                "folds": int(len(folds)),
                "rank_valid_folds": int(eligible_folds),
                "mean_mae": float(np.mean(fold_mae)) if fold_mae else None,
                "mean_within_date_spearman": float(np.mean(fold_rank)) if fold_rank else None,
            }
        )
    return pd.DataFrame(rows)


def _select_depth(cv: pd.DataFrame) -> int:
    ranked = cv.dropna(subset=["mean_within_date_spearman"]).copy()
    if ranked.empty:
        raise RuntimeError("No shallow-tree depth produced rank-valid chronological folds.")
    ranked = ranked.sort_values(
        ["mean_within_date_spearman", "rank_valid_folds", "mean_mae", "max_depth"],
        ascending=[False, False, True, True],
    )
    return int(ranked.iloc[0]["max_depth"])


def _tree_structure(model: DecisionTreeRegressor, feature_names: list[str]) -> list[dict[str, Any]]:
    tree = model.tree_
    rows: list[dict[str, Any]] = []
    stack = [(0, 0, "root")]
    while stack:
        node, depth, path = stack.pop()
        is_leaf = int(tree.children_left[node]) == int(tree.children_right[node])
        feature_index = int(tree.feature[node])
        rows.append(
            {
                "node": int(node),
                "depth": int(depth),
                "path": path,
                "kind": "leaf" if is_leaf else "split",
                "feature": None if is_leaf else feature_names[feature_index],
                "threshold": None if is_leaf else float(tree.threshold[node]),
                "samples": int(tree.n_node_samples[node]),
                "value": float(tree.value[node][0][0]),
            }
        )
        if not is_leaf:
            stack.append((int(tree.children_right[node]), depth + 1, path + " -> right"))
            stack.append((int(tree.children_left[node]), depth + 1, path + " -> left"))
    return rows


def main() -> int:
    args = _parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    boundary = _utc_timestamp(args.validation_start)
    train, test = _temporal_split(dataset, boundary)

    cv = _inner_cv(train)
    selected_depth = _select_depth(cv)

    train_x = train[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    test_x = test[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    y_train = pd.to_numeric(train["delta_log_capital"], errors="raise")

    model = _fit_tree(selected_depth, train_x, y_train)
    prediction = np.asarray(model.predict(test_x), dtype=float)
    per_date, test_summary = _ranking_metrics(
        test, prediction, model_name=f"tree_depth_{selected_depth}"
    )

    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])

    structure = pd.DataFrame(_tree_structure(model, MODEL_FEATURES))
    rules_text = export_text(model, feature_names=MODEL_FEATURES, decimals=6)

    _write_frame(output_dir / "shallow_tree_inner_cv.csv", cv)
    _write_frame(output_dir / "shallow_tree_per_date.csv", per_date)
    _write_frame(output_dir / "shallow_tree_feature_importance.csv", feature_importance)
    _write_frame(output_dir / "shallow_tree_structure.csv", structure)
    (output_dir / "shallow_tree_rules.txt").write_text(rules_text, encoding="utf-8")

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "source_dataset": str(dataset_path),
        "dataset_rows": int(len(dataset)),
        "dataset_dates": int(dataset["decision_date"].nunique()),
        "validation_start": boundary.date().isoformat(),
        "train_rows": int(len(train)),
        "train_dates": int(train["decision_date"].nunique()),
        "test_rows": int(len(test)),
        "test_dates": int(test["decision_date"].nunique()),
        "features": MODEL_FEATURES,
        "candidate_depths": TREE_DEPTHS,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "selected_depth": selected_depth,
        "selection_metric": "mean_within_date_spearman on chronological whole-date folds before 2024",
        "test_interpretation": (
            "exploratory/post-hoc only: 2024-2025 was already inspected in v1.0.3/v1.0.4 and is no longer a virgin holdout"
        ),
        "test_metrics": test_summary,
        "top_features": feature_importance.head(8).to_dict(orient="records"),
        "decision_rule": (
            "Advance only if shallow nonlinear structure improves chronological pre-2024 ranking materially. "
            "Any 2024-2025 improvement is descriptive and must later be confirmed on fresh dates/candidates/universe."
        ),
    }
    _write_json(output_dir / "shallow_tree_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
