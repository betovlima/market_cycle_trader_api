from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.12"
TOLERANCE = 1e-12
KEY_COLUMNS = ["decision_date", "universe_name", "candidate"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose native marginal contribution into frozen-switch-margin contribution plus recalibration effect."
    )
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_aggregate(root: Path) -> pd.DataFrame:
    path = root / "trace_aggregate_dataset.csv"
    if not path.exists():
        raise RuntimeError(f"Missing aggregate dataset: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS + ["baseline_ending_capital", "candidate_ending_capital", "delta_log_capital"] if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Aggregate dataset {path} is missing columns: {missing}")
    return frame


def _decompose(native: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    left = native.copy().rename(
        columns={
            "baseline_ending_capital": "native_baseline_ending_capital",
            "candidate_ending_capital": "native_candidate_ending_capital",
            "delta_log_capital": "native_marginal_log",
        }
    )
    right = frozen.copy().rename(
        columns={
            "baseline_ending_capital": "frozen_baseline_ending_capital",
            "candidate_ending_capital": "frozen_candidate_ending_capital",
            "delta_log_capital": "frozen_marginal_log",
        }
    )
    keep_left = KEY_COLUMNS + ["native_baseline_ending_capital", "native_candidate_ending_capital", "native_marginal_log"]
    keep_right = KEY_COLUMNS + ["frozen_baseline_ending_capital", "frozen_candidate_ending_capital", "frozen_marginal_log"]
    merged = left[keep_left].merge(right[keep_right], on=KEY_COLUMNS, how="outer", validate="one_to_one", indicator=True)
    if not bool((merged["_merge"] == "both").all()):
        missing = merged.loc[merged["_merge"] != "both", KEY_COLUMNS + ["_merge"]]
        raise RuntimeError("Native/frozen datasets do not contain the same cases:\n" + missing.to_string(index=False))
    merged = merged.drop(columns=["_merge"])

    merged["baseline_log_difference"] = np.log(
        merged["frozen_baseline_ending_capital"] / merged["native_baseline_ending_capital"]
    )
    if float(merged["baseline_log_difference"].abs().max()) > TOLERANCE:
        raise RuntimeError(
            "Baseline capital changed between native and frozen campaigns; this invalidates the one-factor decomposition. "
            f"max_abs_log_difference={merged['baseline_log_difference'].abs().max()}"
        )

    merged["switch_margin_recalibration_component"] = (
        merged["native_marginal_log"] - merged["frozen_marginal_log"]
    )
    merged["candidate_capital_recalibration_log"] = np.log(
        merged["native_candidate_ending_capital"] / merged["frozen_candidate_ending_capital"]
    )
    merged["reconstruction_error"] = (
        merged["frozen_marginal_log"]
        + merged["switch_margin_recalibration_component"]
        - merged["native_marginal_log"]
    )
    merged["native_sign"] = np.sign(merged["native_marginal_log"])
    merged["frozen_sign"] = np.sign(merged["frozen_marginal_log"])
    merged["sign_changed_after_freeze"] = merged["native_sign"] != merged["frozen_sign"]
    return merged


def _component_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for candidate, group in rows.groupby("candidate", sort=True):
        native_abs = float(group["native_marginal_log"].abs().mean())
        frozen_abs = float(group["frozen_marginal_log"].abs().mean())
        recal_abs = float(group["switch_margin_recalibration_component"].abs().mean())
        output.append(
            {
                "candidate": candidate,
                "rows": int(len(group)),
                "native_mean_abs_marginal_log": native_abs,
                "frozen_mean_abs_marginal_log": frozen_abs,
                "switch_margin_recalibration_mean_abs_log": recal_abs,
                "frozen_share_of_native_abs": None if native_abs <= TOLERANCE else frozen_abs / native_abs,
                "recalibration_share_of_native_abs": None if native_abs <= TOLERANCE else recal_abs / native_abs,
                "sign_changed_after_freeze_rows": int(group["sign_changed_after_freeze"].sum()),
                "nonzero_native_rows": int((group["native_marginal_log"].abs() > TOLERANCE).sum()),
                "nonzero_frozen_rows": int((group["frozen_marginal_log"].abs() > TOLERANCE).sum()),
            }
        )
    return pd.DataFrame(output)


def main() -> int:
    args = _parser().parse_args()
    native_dir = Path(args.native_dir).resolve()
    frozen_dir = Path(args.frozen_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    native = _load_aggregate(native_dir)
    frozen = _load_aggregate(frozen_dir)
    rows = _decompose(native, frozen)
    summary_table = _component_summary(rows)

    rows.to_csv(output_dir / "frozen_switch_margin_decomposition.csv", index=False)
    summary_table.to_csv(output_dir / "frozen_switch_margin_component_summary.csv", index=False)

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "rows": int(len(rows)),
        "max_baseline_log_difference": float(rows["baseline_log_difference"].abs().max()) if not rows.empty else 0.0,
        "max_reconstruction_error": float(rows["reconstruction_error"].abs().max()) if not rows.empty else 0.0,
        "sign_changed_after_freeze_rows": int(rows["sign_changed_after_freeze"].sum()),
        "decision_rule": (
            "If APD/VNCE native marginal effects collapse under the baseline switch margin while XSD retains its direct effect, "
            "the previous marginal signal was dominated by candidate-induced switch-margin recalibration rather than candidate trading."
        ),
        "candidate_summary": summary_table.to_dict(orient="records"),
    }
    _write_json(output_dir / "frozen_switch_margin_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
