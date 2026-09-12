from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.9"
EXPERIMENT_NAME = "contextual_marginal_signature_adm_adi_interaction_decomposition"
SIGN_TOLERANCE = 1e-12

UNIVERSES = {
    "base": "Original25",
    "minus_adm": "Original24_MinusADM",
    "minus_adi": "Original24_MinusADI",
    "minus_both": "Original23_MinusADM_ADI",
    "drop_last5": "Original20_DropLast5",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the Original25 -> DropLast5 marginal-capital change into ADM main effect, "
            "ADI main effect, ADM×ADI interaction and the residual collective effect of HD/ADC/ADEA."
        )
    )
    parser.add_argument("--dataset", required=True, help="multi_universe_contextual_dataset.csv from the focused v1.0.9 replay")
    parser.add_argument("--output-dir", required=True)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _effect_sign(value: float) -> str:
    if abs(float(value)) <= SIGN_TOLERANCE:
        return "zero"
    return "positive" if float(value) > 0 else "negative"


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"universe_name", "decision_date", "candidate", "evaluation_status", "delta_log_capital"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Dataset is missing required columns: " + ", ".join(missing))
    frame = frame.loc[frame["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    if frame.empty:
        raise RuntimeError("Dataset contains no completed rows.")
    frame["delta_log_capital"] = pd.to_numeric(frame["delta_log_capital"], errors="raise")
    expected = set(UNIVERSES.values())
    available = set(frame["universe_name"].astype(str))
    missing_universes = sorted(expected.difference(available))
    if missing_universes:
        raise RuntimeError("Focused dataset is missing universes: " + ", ".join(missing_universes))
    return frame


def build_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["decision_date", "candidate"],
        columns="universe_name",
        values="delta_log_capital",
        aggfunc="first",
    )
    required = list(UNIVERSES.values())
    pivot = pivot.dropna(subset=required).copy()
    if pivot.empty:
        raise RuntimeError("No candidate/date row has all focused universes completed.")

    base = pivot[UNIVERSES["base"]]
    minus_adm = pivot[UNIVERSES["minus_adm"]]
    minus_adi = pivot[UNIVERSES["minus_adi"]]
    minus_both = pivot[UNIVERSES["minus_both"]]
    drop_last5 = pivot[UNIVERSES["drop_last5"]]

    result = pivot.reset_index()[["decision_date", "candidate"]].copy()
    result["base_delta_log"] = base.to_numpy(dtype=float)
    result["minus_adm_delta_log"] = minus_adm.to_numpy(dtype=float)
    result["minus_adi_delta_log"] = minus_adi.to_numpy(dtype=float)
    result["minus_both_delta_log"] = minus_both.to_numpy(dtype=float)
    result["drop_last5_delta_log"] = drop_last5.to_numpy(dtype=float)

    result["adm_main"] = (minus_adm - base).to_numpy(dtype=float)
    result["adi_main"] = (minus_adi - base).to_numpy(dtype=float)
    result["adm_adi_interaction"] = (minus_both - minus_adm - minus_adi + base).to_numpy(dtype=float)
    result["residual_hd_adc_adea"] = (drop_last5 - minus_both).to_numpy(dtype=float)
    result["total_drop_last5"] = (drop_last5 - base).to_numpy(dtype=float)
    result["reconstructed_total"] = (
        result["adm_main"]
        + result["adi_main"]
        + result["adm_adi_interaction"]
        + result["residual_hd_adc_adea"]
    )
    result["reconstruction_error"] = result["reconstructed_total"] - result["total_drop_last5"]

    for label in ["base", "minus_adm", "minus_adi", "minus_both", "drop_last5"]:
        col = f"{label}_delta_log" if label != "drop_last5" else "drop_last5_delta_log"
        result[f"{label}_sign"] = result[col].map(_effect_sign)

    result["adm_sign_changed"] = result["base_sign"] != result["minus_adm_sign"]
    result["adi_sign_changed"] = result["base_sign"] != result["minus_adi_sign"]
    result["both_sign_changed"] = result["base_sign"] != result["minus_both_sign"]
    result["drop_last5_sign_changed"] = result["base_sign"] != result["drop_last5_sign"]
    result["adm_strict_flip"] = result.apply(
        lambda row: {row["base_sign"], row["minus_adm_sign"]} == {"positive", "negative"}, axis=1
    )
    result["drop_last5_strict_flip"] = result.apply(
        lambda row: {row["base_sign"], row["drop_last5_sign"]} == {"positive", "negative"}, axis=1
    )
    return result


def component_summary(decomposition: pd.DataFrame) -> pd.DataFrame:
    components = ["adm_main", "adi_main", "adm_adi_interaction", "residual_hd_adc_adea", "total_drop_last5"]
    rows: list[dict[str, Any]] = []
    total_abs_mean = float(decomposition["total_drop_last5"].abs().mean())
    for component in components:
        values = pd.to_numeric(decomposition[component], errors="coerce")
        rows.append({
            "component": component,
            "rows": int(values.notna().sum()),
            "mean": float(values.mean()),
            "mean_abs": float(values.abs().mean()),
            "median": float(values.median()),
            "negative_rows": int((values < -SIGN_TOLERANCE).sum()),
            "positive_rows": int((values > SIGN_TOLERANCE).sum()),
            "zero_rows": int((values.abs() <= SIGN_TOLERANCE).sum()),
            "mean_abs_share_of_total": (
                float(values.abs().mean() / total_abs_mean) if total_abs_mean > SIGN_TOLERANCE else None
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = _parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load(dataset_path)
    decomposition = build_decomposition(frame)
    summary_frame = component_summary(decomposition)

    _write_frame(output_dir / "adm_adi_decomposition_rows.csv", decomposition)
    _write_frame(output_dir / "adm_adi_component_summary.csv", summary_frame)

    max_reconstruction_error = float(decomposition["reconstruction_error"].abs().max())
    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "source_dataset": str(dataset_path),
        "rows": int(len(decomposition)),
        "dates": int(decomposition["decision_date"].nunique()),
        "candidates": int(decomposition["candidate"].nunique()),
        "max_reconstruction_error": max_reconstruction_error,
        "adm_sign_changes": int(decomposition["adm_sign_changed"].sum()),
        "adi_sign_changes": int(decomposition["adi_sign_changed"].sum()),
        "both_sign_changes": int(decomposition["both_sign_changed"].sum()),
        "drop_last5_sign_changes": int(decomposition["drop_last5_sign_changed"].sum()),
        "adm_strict_flips": int(decomposition["adm_strict_flip"].sum()),
        "drop_last5_strict_flips": int(decomposition["drop_last5_strict_flip"].sum()),
        "adm_matches_drop_last5_sign_change_mask": bool(
            (decomposition["adm_sign_changed"] == decomposition["drop_last5_sign_changed"]).all()
        ),
        "component_summary": summary_frame.to_dict(orient="records"),
        "decision_rule": (
            "ADM is a localized sign-state enabler if removing ADM reproduces the DropLast5 sign-change mask. "
            "ADI is a magnitude modulator if its main effect is material but causes few/no sign changes. "
            "A material ADM×ADI interaction means the two incumbents are not additive. "
            "A material residual_hd_adc_adea means HD/ADC/ADEA still matter conditionally after ADM and ADI are removed."
        ),
    }
    _write_json(output_dir / "adm_adi_decomposition_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
