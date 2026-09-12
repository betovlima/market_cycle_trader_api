from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.10"
EXPERIMENT_NAME = "contextual_marginal_signature_reproducibility_audit"
TOLERANCE = 1e-12

KEY_COLUMNS = ["universe_name", "decision_date", "candidate"]
FEATURE_COLUMNS = [
    "relative__return_5", "relative__return_20", "relative__return_60",
    "relative__vol_20", "relative__vol_60", "relative__ema_distance_20",
    "relative__trend_efficiency_20", "relative__momentum_acceleration_5_20",
    "corr_20_to_universe", "corr_60_to_universe",
    "universe_dispersion_return_20", "universe_dispersion_vol_20",
    "candidate_rank_return_20", "candidate_rank_return_60",
    "market_return_20", "market_vol_20",
]
OUTPUT_COLUMNS = [
    "baseline_ending_capital", "candidate_ending_capital",
    "ending_capital_delta_rate", "delta_log_capital",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more Contextual Marginal Signature datasets row-by-row to detect "
            "input-feature drift, baseline drift, challenger drift and marginal-effect drift."
        )
    )
    parser.add_argument(
        "--dataset", action="append", required=True,
        help="Repeat as LABEL=path/to/multi_universe_contextual_dataset.csv; first dataset is reference.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise RuntimeError("--dataset must use LABEL=path syntax")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).resolve()
    if not label:
        raise RuntimeError("Dataset label cannot be blank.")
    if not path.exists():
        raise RuntimeError(f"Dataset does not exist: {path}")
    return label, path


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"evaluation_status", *KEY_COLUMNS, *FEATURE_COLUMNS, *OUTPUT_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Dataset is missing required columns: " + ", ".join(missing))
    completed = frame.loc[frame["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    if completed.empty:
        raise RuntimeError(f"Dataset contains no completed rows: {path}")
    if completed.duplicated(KEY_COLUMNS).any():
        raise RuntimeError(f"Dataset has duplicate experiment keys: {path}")
    for column in [*FEATURE_COLUMNS, *OUTPUT_COLUMNS]:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
    return completed.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _stable_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    normalized = frame.loc[:, [*KEY_COLUMNS, *columns]].copy()
    normalized = normalized.sort_values(KEY_COLUMNS).reset_index(drop=True)
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(15)
    payload = normalized.to_csv(index=False, na_rep="NaN", float_format="%.15g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sign(series: pd.Series, tolerance: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(values.abs() <= tolerance, "zero", np.where(values > 0, "positive", "negative")),
        index=series.index,
    )


def compare(reference: pd.DataFrame, challenger: pd.DataFrame, tolerance: float) -> tuple[dict[str, Any], pd.DataFrame]:
    left = reference.set_index(KEY_COLUMNS).sort_index()
    right = challenger.set_index(KEY_COLUMNS).sort_index()
    left_keys, right_keys = set(left.index), set(right.index)
    common = sorted(left_keys.intersection(right_keys))
    if not common:
        raise RuntimeError("Datasets have no common experiment keys.")

    l = left.loc[common]
    r = right.loc[common]
    differences: list[dict[str, Any]] = []
    for key in common:
        row: dict[str, Any] = dict(zip(KEY_COLUMNS, key))
        changed = False
        for column in [*FEATURE_COLUMNS, *OUTPUT_COLUMNS]:
            lv, rv = l.at[key, column], r.at[key, column]
            delta = float(rv - lv) if pd.notna(lv) and pd.notna(rv) else np.nan
            row[f"diff__{column}"] = delta
            if pd.isna(lv) != pd.isna(rv) or (pd.notna(delta) and abs(delta) > tolerance):
                changed = True
        if changed:
            differences.append(row)
    diff_frame = pd.DataFrame(differences)

    feature_diff = (r[FEATURE_COLUMNS] - l[FEATURE_COLUMNS]).abs()
    output_diff = (r[OUTPUT_COLUMNS] - l[OUTPUT_COLUMNS]).abs()
    reference_sign = _sign(l["delta_log_capital"], tolerance)
    challenger_sign = _sign(r["delta_log_capital"], tolerance)

    summary = {
        "reference_rows": int(len(reference)),
        "challenger_rows": int(len(challenger)),
        "common_rows": int(len(common)),
        "missing_from_challenger": int(len(left_keys.difference(right_keys))),
        "extra_in_challenger": int(len(right_keys.difference(left_keys))),
        "feature_hash_reference": _stable_hash(reference, FEATURE_COLUMNS),
        "feature_hash_challenger": _stable_hash(challenger, FEATURE_COLUMNS),
        "output_hash_reference": _stable_hash(reference, OUTPUT_COLUMNS),
        "output_hash_challenger": _stable_hash(challenger, OUTPUT_COLUMNS),
        "feature_rows_changed": int((feature_diff.max(axis=1) > tolerance).sum()),
        "max_abs_feature_diff": float(np.nanmax(feature_diff.to_numpy(dtype=float))) if feature_diff.size else 0.0,
        "baseline_rows_changed": int((output_diff["baseline_ending_capital"] > tolerance).sum()),
        "max_abs_baseline_diff": float(output_diff["baseline_ending_capital"].max()),
        "candidate_capital_rows_changed": int((output_diff["candidate_ending_capital"] > tolerance).sum()),
        "max_abs_candidate_capital_diff": float(output_diff["candidate_ending_capital"].max()),
        "delta_log_rows_changed": int((output_diff["delta_log_capital"] > tolerance).sum()),
        "max_abs_delta_log_diff": float(output_diff["delta_log_capital"].max()),
        "effect_sign_mismatches": int((reference_sign != challenger_sign).sum()),
        "exactly_reproducible": bool(
            len(left_keys) == len(right_keys) == len(common)
            and not (feature_diff.max(axis=1) > tolerance).any()
            and not (output_diff.max(axis=1) > tolerance).any()
        ),
    }
    return summary, diff_frame


def main() -> int:
    args = _parser().parse_args()
    items = [_parse_dataset(value) for value in args.dataset]
    labels = [label for label, _ in items]
    if len(items) < 2:
        raise RuntimeError("At least two --dataset arguments are required.")
    if len(set(labels)) != len(labels):
        raise RuntimeError("Dataset labels must be unique.")

    datasets = {label: _load(path) for label, path in items}
    reference_label = labels[0]
    reference = datasets[reference_label]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerance = max(0.0, float(args.tolerance))

    comparisons: list[dict[str, Any]] = []
    all_differences: list[pd.DataFrame] = []
    for challenger_label in labels[1:]:
        summary, diff_frame = compare(reference, datasets[challenger_label], tolerance)
        summary.update({"reference": reference_label, "challenger": challenger_label})
        comparisons.append(summary)
        if not diff_frame.empty:
            diff_frame.insert(0, "challenger", challenger_label)
            all_differences.append(diff_frame)

    comparison_frame = pd.DataFrame(comparisons)
    difference_frame = pd.concat(all_differences, ignore_index=True) if all_differences else pd.DataFrame()
    _write_frame(output_dir / "reproducibility_comparison.csv", comparison_frame)
    _write_frame(output_dir / "reproducibility_changed_rows.csv", difference_frame)

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "reference": reference_label,
        "tolerance": tolerance,
        "datasets": [
            {"label": label, "path": str(path), "rows": int(len(datasets[label]))}
            for label, path in items
        ],
        "comparisons": comparisons,
        "decision_rule": (
            "Do not interpret cross-campaign mechanism differences until repeated executions with the same intended inputs "
            "have matching feature and output hashes. Matching features with divergent baselines/outputs points to replay/configuration "
            "nondeterminism; divergent features points first to market-data/input drift."
        ),
    }
    _write_json(output_dir / "reproducibility_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
