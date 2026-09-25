"""CLMT Fold-1 root-cause audit: feature versus target versus inference precision.

Read-only; no full backtest, Alpaca fetch, MongoDB write, or TCC modification.
Every bar is validated against archived Strategy #11 hashes before fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from market_cycle_trader_api.core.environment import load_project_environment

# mongo_repository binds MONGO_URL at import time.
load_project_environment()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tcc_v106_fold1_calibration_parity import (
    _fold_context,
    _load_archive_control,
    _load_bars,
)
from scripts.audit_tcc_v106_fold1_data_parity import (
    TCC_FROZEN_MANIFEST_SHA,
    _assert_reference_files,
)
from scripts.audit_tcc_v106_fold1_model_crossover import (
    _read_reference_report,
    _read_unique_zip_member,
)
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES,
    _model_utilities,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    CONFIG as TCC_CONFIG,
    build_control_config,
)
from market_cycle_trader_api.tcc_v106_reference.research_challengers import (
    _lightgbm_fit_models,
)

SYMBOL = "CLMT"
TARGET = "forward_risk_adjusted_utility"
NAME = "tcc_v106_clmt_fold1_training_input_isolation"
ATOL = 1e-12
RTOL = 1e-12


def _numeric_precision_by_column(
    left: pd.DataFrame, right: pd.DataFrame, columns: list[str],
) -> list[dict]:
    if not left.index.equals(right.index):
        raise ValueError("CLMT row-date and order mismatch.")
    results = []
    for column in columns:
        a = left[column].to_numpy(dtype=np.float64)
        b = right[column].to_numpy(dtype=np.float64)
        identical = (a == b) | (np.isnan(a) & np.isnan(b))
        close = np.isclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
        different = np.flatnonzero(~identical)
        material = np.flatnonzero(~close)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.abs(a[finite] - b[finite])
        sample = int(different[0]) if different.size else None
        results.append({
            "column": column,
            "training_rows": int(len(left)),
            "exact_different_values": int(different.size),
            "material_different_values": int(material.size),
            "max_abs_delta_finite": float(delta.max()) if delta.size else 0.0,
            "first_exact_difference_date": (
                left.index[sample].isoformat() if sample is not None else None
            ),
            "first_tcc_float_hex": (
                float(a[sample]).hex() if sample is not None else None
            ),
            "first_mct_float_hex": (
                float(b[sample]).hex() if sample is not None else None
            ),
            "stage": "target" if column == TARGET else "feature",
        })
    return results


def _first_different_tree(a, b) -> dict | None:
    left = a.booster_.dump_model()["tree_info"]
    right = b.booster_.dump_model()["tree_info"]
    for i in range(min(len(left), len(right))):
        lt, rt = left[i]["tree_structure"], right[i]["tree_structure"]
        if lt != rt:
            return {
                "tree_index": i,
                "tcc_tree_leaves": left[i].get("num_leaves"),
                "mct_tree_leaves": right[i].get("num_leaves"),
                "tcc_root_feature": lt.get("split_feature"),
                "mct_root_feature": rt.get("split_feature"),
                "tcc_root_threshold": lt.get("threshold"),
                "mct_root_threshold": rt.get("threshold"),
            }
    if len(left) != len(right):
        return {"tree_count_tcc": len(left), "tree_count_mct": len(right)}
    return None


def _fit_single_clmt(frame: pd.DataFrame, train: pd.DatetimeIndex, config, phase: str):
    models = _lightgbm_fit_models(
        {SYMBOL: frame}, [SYMBOL], train, config,
        phase=phase, device_type="cpu",
    )
    if list(models) != [SYMBOL]:
        raise ValueError(f"{phase}: expected exactly one CLMT model.")
    return models[SYMBOL]


def _predict_single(model, frame, dates, config) -> np.ndarray:
    return np.asarray([
        _model_utilities(
            {SYMBOL: model}, {SYMBOL: frame}, [SYMBOL],
            pd.Timestamp(date), config,
        )[1]
        for date in dates
    ], dtype=np.float64)


def _verify_reference_predictions(
    calibration_zip: Path, dates: pd.DatetimeIndex,
    tcc: np.ndarray, mct: np.ndarray,
) -> float:
    records = pd.read_csv(io.BytesIO(_read_unique_zip_member(
        calibration_zip, "calibration_prediction_differences.csv"
    )))
    clmt = records.loc[records["symbol"] == SYMBOL].copy()
    clmt["date"] = pd.to_datetime(clmt["date"], utc=True)
    clmt = clmt.set_index("date").reindex(dates)
    if len(clmt) != len(dates) or clmt[["tcc_score", "mct_score"]].isna().any().any():
        raise ValueError("Prior calibration audit lacks complete CLMT prediction rows.")
    delta_tcc = np.abs(tcc - clmt["tcc_score"].to_numpy(dtype=np.float64))
    delta_mct = np.abs(mct - clmt["mct_score"].to_numpy(dtype=np.float64))
    biggest = max(float(delta_tcc.max()), float(delta_mct.max()))
    if biggest > 1e-10:
        raise ValueError(
            f"CLMT baseline prediction replay differs from prior audit by {biggest}."
        )
    return biggest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcc-data", type=Path, required=True)
    parser.add_argument("--strategy11-zip", type=Path, required=True)
    parser.add_argument("--strategy12-zip", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output") / "tcc_v106_clmt_fold1_input_isolation",
    )
    args = parser.parse_args()

    _assert_reference_files(args.tcc_data)
    archived, expected, _ = _load_archive_control(
        args.strategy11_zip, args.strategy12_zip
    )
    prior, _, material = _read_reference_report(args.calibration_zip)
    if (
        prior.get("market_data_signature_sha256")
        != archived.get("market_data_signature_sha256")
        or SYMBOL not in material
        or prior.get("tcc_snapshot_sha256") != TCC_FROZEN_MANIFEST_SHA
    ):
        raise ValueError("CLMT was not validated in the matching prior audit.")

    config = build_control_config(TCC_CONFIG)
    print("[1/3] Verify TCC frozen and original MCT archived data hashes", flush=True)
    tcc_bars, mct_bars = _load_bars(args.tcc_data, expected, config)
    tcc_frames, tcc_symbols, tcc_train, tcc_dates, *_ = _fold_context(
        tcc_bars, config
    )
    mct_frames, mct_symbols, mct_train, mct_dates, *_ = _fold_context(
        mct_bars, config
    )
    if (
        tcc_symbols != mct_symbols
        or not tcc_train.equals(mct_train)
        or not tcc_dates.equals(mct_dates)
    ):
        raise ValueError("Fold-1 asset/date ordering differs.")
    left = tcc_frames[SYMBOL]
    right = mct_frames[SYMBOL]
    columns = [*ROTATION_FEATURES, TARGET]
    a = left.loc[tcc_train].dropna(subset=columns)
    b = right.loc[mct_train].dropna(subset=columns)
    numeric = _numeric_precision_by_column(a, b, columns)
    if any(row["material_different_values"] for row in numeric):
        raise ValueError("CLMT training inputs have material source differences.")
    if len(a) != 700 or len(b) != 700:
        raise ValueError("Expected 700 valid CLMT Fold-1 training rows.")

    # Hybrid variants keep the exact row timestamps and do not change raw bars.
    # Only the indicated training columns differ; inference is separately
    # tested on the original frozen and MCT calibration frames.
    tcc_features_mct_target = left.copy()
    tcc_features_mct_target.loc[:, TARGET] = right[TARGET].to_numpy()
    mct_features_tcc_target = right.copy()
    mct_features_tcc_target.loc[:, TARGET] = left[TARGET].to_numpy()
    tcc_features_mct_features = left.copy()
    tcc_features_mct_features.loc[:, ROTATION_FEATURES] = (
        right[ROTATION_FEATURES].to_numpy(dtype=np.float64)
    )

    variants = {
        "tcc_original": left,
        "mct_original": right,
        "tcc_features_mct_target": tcc_features_mct_target,
        "mct_features_tcc_target": mct_features_tcc_target,
        "tcc_target_mct_features": tcc_features_mct_features,
    }
    print("[2/3] Fit five CLMT calibration models (two originals, three hybrids)", flush=True)
    models = {
        name: _fit_single_clmt(frame, tcc_train, config, f"clmt_isolation_{name}")
        for name, frame in variants.items()
    }
    print("[3/3] Cross-infer each model on the same two calibration panels", flush=True)
    predictions = {
        name + "_on_" + source: _predict_single(model, frame, tcc_dates, config)
        for name, model in models.items()
        for source, frame in (("tcc", left), ("mct", right))
    }
    replay_error = _verify_reference_predictions(
        args.calibration_zip, tcc_dates,
        predictions["tcc_original_on_tcc"],
        predictions["mct_original_on_mct"],
    )

    base = predictions["tcc_original_on_tcc"]
    fingerprints = []
    for name, model in models.items():
        serialized = model.booster_.model_to_string()
        relative = predictions[name + "_on_tcc"]
        delta = np.abs(relative - base)
        fingerprints.append({
            "variant": name,
            "model_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
            "model_matches_tcc_exact": bool(serialized == models[
                "tcc_original"
            ].booster_.model_to_string()),
            "first_tree_difference_vs_tcc": json.dumps(
                _first_different_tree(models["tcc_original"], model),
                sort_keys=True,
            ),
            "max_abs_prediction_delta_vs_tcc": float(delta.max()),
            "material_prediction_rows_vs_tcc": int(
                (~np.isclose(
                    relative, base, atol=ATOL, rtol=RTOL,
                    equal_nan=True,
                )).sum()
            ),
        })

    source_only = np.abs(
        predictions["tcc_original_on_tcc"] -
        predictions["tcc_original_on_mct"]
    )
    model_only = np.abs(
        predictions["tcc_original_on_tcc"] -
        predictions["mct_original_on_tcc"]
    )
    full = np.abs(
        predictions["tcc_original_on_tcc"] -
        predictions["mct_original_on_mct"]
    )
    import lightgbm
    report = {
        "schema_version": 1,
        "experiment": NAME,
        "status": "COMPLETED",
        "symbol": SYMBOL,
        "tcc_snapshot_sha256": TCC_FROZEN_MANIFEST_SHA,
        "mct_archived_signature": archived["market_data_signature_sha256"],
        "train_rows": int(len(a)),
        "calibration_rows": int(len(tcc_dates)),
        "training_precision": {
            "different_feature_columns": int(sum(
                x["exact_different_values"] > 0
                for x in numeric if x["stage"] == "feature"
            )),
            "different_target_values": next(
                x["exact_different_values"] for x in numeric
                if x["stage"] == "target"
            ),
            "material_different_values": int(sum(
                x["material_different_values"] for x in numeric
            )),
        },
        "prediction_deltas": {
            "same_model_different_inference_features_max_abs":
                float(source_only.max()),
            "different_model_same_inference_features_max_abs":
                float(model_only.max()),
            "different_model_and_features_max_abs": float(full.max()),
        },
        "prior_clmt_prediction_replay_max_abs_error": replay_error,
        "models": fingerprints,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm.__version__,
            "mct_model_threads_override": os.getenv(
                "MCT_MODEL_THREADS_OVERRIDE"
            ),
        },
        "scope": (
            "Fold-1 CLMT training/inference isolation only, same runtime, "
            "historically archived MCT data hashes. No inference that an "
            "asset has structural data failure based on float precision."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(numeric).to_csv(
        args.output / "clmt_training_column_precision.csv", index=False
    )
    pd.DataFrame(fingerprints).to_csv(
        args.output / "clmt_model_fingerprints.csv", index=False
    )
    pd.DataFrame({
        "date": [date.isoformat() for date in tcc_dates],
        **predictions,
    }).to_csv(args.output / "clmt_prediction_cross.csv", index=False)
    zip_path = args.output / "tcc_v106_clmt_fold1_input_isolation.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as output:
        for name in (
            "report.json",
            "clmt_training_column_precision.csv",
            "clmt_model_fingerprints.csv",
            "clmt_prediction_cross.csv",
        ):
            output.write(args.output / name, arcname=name)
    print(json.dumps({
        "report_zip": str(zip_path),
        "training_precision": report["training_precision"],
        "prediction_deltas": report["prediction_deltas"],
        "models": fingerprints,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
