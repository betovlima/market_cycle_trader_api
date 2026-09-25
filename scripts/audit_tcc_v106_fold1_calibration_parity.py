"""Fold-1 calibration parity experiment using the vendored TCC v1.0.6 engine.

Fits ONLY Fold-1 calibration regressors, once on each source dataset.
No full backtest, Alpaca request, MongoDB write, strategy mutation, or TCC edit.
The exact historical MCT snapshot is guarded by the archived #11 hashes.
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

# mongo_repository caches MONGO_URL/MONGO_DATABASE at import time.
load_project_environment()

# Direct invocation (python scripts/xxx.py) puts scripts/, not the repository
# root, on sys.path. Make the sibling audit import work both there and in CI.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tcc_v106_fold1_data_parity import (
    CUTOFF,
    TCC_FROZEN_MANIFEST_SHA,
    _assert_reference_files,
    _frozen_actions,
    _frozen_frame,
    _load_zip_manifest,
    _split_safe_float_frame,
)
from market_cycle_trader_api.engine.market_data import (
    _history_frame_sha256,
    _read_frame,
    inclusive_end_exclusive_boundary,
    validate_and_clean_bars,
)
from market_cycle_trader_api.engine.research_market_data import (
    ALPACA_CORPORATE_ACTIONS_COLLECTION,
    RAW_TOTAL_CAUSAL_PROTOCOL,
    split_normalize,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    create_client,
    get_database,
)
from market_cycle_trader_api.services.reproducibility import market_data_manifest
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES,
    _precompute_model_utilities,
    _simple_policy_growth,
    _utility_policy,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    CONFIG as TCC_CONFIG,
    build_control_config,
)
from market_cycle_trader_api.tcc_v106_reference.research_challengers import (
    _build_execution_context,
    _lightgbm_fit_models,
    _lightgbm_settings,
)

TARGET = "forward_risk_adjusted_utility"
EXPERIMENT_ID = "tcc_v106_fold1_calibration_parity"
MCT11_CALIBRATION_CSV = (
    "PORTFOLIO_lightgbm_utility/"
    "PORTFOLIO_lightgbm_utility_switch_margin_calibration_candidates.csv"
)


def _sha_float_matrix(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    stamps = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    digest.update(stamps.asi8.astype("<i8", copy=False).tobytes())
    digest.update(json.dumps(columns).encode("utf-8"))
    digest.update(frame[columns].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def _status_pair(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {
            "status": "SHAPE_MISMATCH", "shape_tcc": list(a.shape),
            "shape_mct": list(b.shape),
        }
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    near = np.isclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)
    bad = np.flatnonzero(~near.ravel())
    unequal = np.flatnonzero(~same.ravel())
    return {
        "status": (
            "DIFFERENT" if bad.size else "PRECISION_ONLY" if unequal.size else "MATCH"
        ),
        "exact_different_values": int(len(unequal)),
        "substantial_different_values": int(len(bad)),
        "first_substantial_flat_index": int(bad[0]) if len(bad) else None,
    }


def _load_archive_control(zip11: Path, zip12: Path) -> tuple[dict, dict, list[dict]]:
    left = _load_zip_manifest(zip11)
    right = _load_zip_manifest(zip12)
    if (left.get("strategy_profile_name"), right.get("strategy_profile_name")) != (
        "Strategy #11", "Strategy #12"
    ):
        raise ValueError("The supplied ZIP order is not Strategy #11, #12.")
    data_hash = left.get("market_data_signature_sha256")
    if not data_hash or data_hash != right.get("market_data_signature_sha256"):
        raise ValueError("Archived Strategy #11 and #12 data signatures differ.")
    expected = left.get("market_data_signatures") or {}
    if len(expected) != 55:
        raise ValueError(f"Expected 55 archived market signatures, got {len(expected)}.")
    with ZipFile(zip11) as source:
        archived = pd.read_csv(io.BytesIO(source.read(MCT11_CALIBRATION_CSV)))
    rows = archived.loc[archived["fold_id"] == 1]
    if len(rows) != 4:
        raise ValueError("Strategy #11 must have four Fold-1 candidate scores.")
    return left, expected, rows.to_dict(orient="records")


def _load_bars(
    frozen_path: Path,
    expected: dict,
    config: object,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    tcc_bars: dict[str, pd.DataFrame] = {}
    mct_bars: dict[str, pd.DataFrame] = {}
    start = pd.Timestamp("2016-01-01", tz="UTC")
    end = inclusive_end_exclusive_boundary(CUTOFF)
    client = create_client()
    try:
        db = get_database(client)
        bars_collection = db[ALPACA_MARKET_BARS_COLLECTION]
        actions_collection = db[ALPACA_CORPORATE_ACTIONS_COLLECTION]
        symbols = [
            symbol for symbol in TCC_CONFIG.assets if symbol != "DOC"
        ]
        for i, symbol in enumerate(symbols, 1):
            print(f"[data {i:02d}/55] {symbol}", flush=True)
            if symbol not in expected:
                raise ValueError(f"Archived Strategy #11 lacks {symbol}.")
            raw = _read_frame(
                bars_collection,
                {
                    "symbol": symbol, "interval": "1Day", "feed": "sip",
                    "adjustment": "raw",
                },
                start, end,
            )
            actual = _history_frame_sha256(raw)
            if actual != expected[symbol].get("raw_sha256"):
                raise ValueError(
                    f"MCT historical RAW cache changed since #11: {symbol}; "
                    "do not use it as the archived snapshot."
                )
            ca = actions_collection.find_one(
                {"symbol": symbol, "protocol": RAW_TOTAL_CAUSAL_PROTOCOL},
                {"_id": 0},
            )
            if ca is None or str(ca.get("query_end") or "") < CUTOFF:
                raise ValueError(f"Corporate Actions unavailable for {symbol}.")
            mct_actions = [
                dict(x) for x in (ca.get("actions") or [])
                if isinstance(x, dict)
            ]
            tcc_normalized, _ = split_normalize(
                _split_safe_float_frame(_frozen_frame(frozen_path, symbol)),
                _frozen_actions(frozen_path, symbol),
            )
            mct_normalized, _ = split_normalize(
                _split_safe_float_frame(raw), mct_actions,
            )
            tcc_normalized = validate_and_clean_bars(tcc_normalized, config)
            mct_normalized = validate_and_clean_bars(mct_normalized, config)
            normalized_sha = market_data_manifest({
                symbol: mct_normalized
            })[1][symbol]["sha256"]
            if normalized_sha != expected[symbol].get("sha256"):
                raise ValueError(
                    f"MCT normalized snapshot changed since #11: {symbol}."
                )
            tcc_bars[symbol] = tcc_normalized
            mct_bars[symbol] = mct_normalized
    finally:
        client.close()
    return tcc_bars, mct_bars


def _fold_context(bars: dict[str, pd.DataFrame], config: object):
    frames, common, symbols, folds, *_ = _build_execution_context(
        bars, config
    )
    fold = folds[0]
    if int(fold["fold_id"]) != 1:
        raise ValueError("First chronological walk-forward fold is not Fold 1.")
    train = common[: int(fold["train_end_index"])]
    calibration = common[
        int(fold["calibration_start_index"]):
        int(fold["calibration_end_index"])
    ]
    return frames, symbols, train, calibration, fold


def _fit_and_score(
    name: str,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train: pd.DatetimeIndex,
    calibration: pd.DatetimeIndex,
    config: object,
) -> tuple[dict, list[dict], np.ndarray, list[dict]]:
    matrices: dict[str, dict] = {}
    print(f"[{name}] auditing {len(symbols)} Fold-1 training matrices", flush=True)
    for symbol in symbols:
        frame = frames[symbol].loc[train].dropna(
            subset=[TARGET, *ROTATION_FEATURES]
        )
        columns = [*ROTATION_FEATURES, TARGET]
        matrices[symbol] = {
            "training_rows": int(len(frame)),
            "first_training_date": (
                frame.index.min().isoformat() if len(frame) else None
            ),
            "last_training_date": (
                frame.index.max().isoformat() if len(frame) else None
            ),
            "training_matrix_sha256": _sha_float_matrix(frame, columns),
            "frame": frame,
        }

    print(f"[{name}] fitting Fold-1 calibration LightGBM only", flush=True)
    models = _lightgbm_fit_models(
        frames, symbols, train, config,
        phase=f"parity_{name}_fold1_calibration",
        device_type="cpu",
    )
    scores, _ = _precompute_model_utilities(
        models, frames, symbols, calibration, config
    )
    matrix = np.stack(
        [scores[pd.Timestamp(day)] for day in calibration],
        axis=0,
    )
    records: list[dict] = []
    for candidate in tuple(float(x) for x in config.rotation_switch_margin_candidates):
        policy = _utility_policy(models, frames, symbols, config, candidate)
        score = _simple_policy_growth(
            policy, frames, symbols, calibration, config
        )
        records.append({
            "dataset": name,
            "fold_id": 1,
            "candidate_margin": candidate,
            "risk_adjusted_score": float(score),
        })
    selected = max(records, key=lambda r: r["risk_adjusted_score"])
    summary = {
        "dataset": name,
        "fitted_models": len(models),
        "selected_candidate_margin": selected["candidate_margin"],
        "selected_candidate_score": selected["risk_adjusted_score"],
        "effective_switch_margin": max(
            float(config.rotation_switch_margin),
            selected["candidate_margin"],
        ),
        "calibration_prediction_sha256": hashlib.sha256(
            matrix.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "calibration_dates": len(calibration),
        "training_sessions": len(train),
        "models_sha256": {
            symbol: hashlib.sha256(
                model.booster_.model_to_string().encode("utf-8")
            ).hexdigest()
            for symbol, model in models.items()
        },
    }
    public_matrices = [
        {k: v for k, v in val.items() if k != "frame"}
        | {"dataset": name, "symbol": symbol}
        for symbol, val in matrices.items()
    ]
    return summary, records, matrix, public_matrices


def _compare_training_matrices(
    frames_a: dict[str, pd.DataFrame],
    frames_b: dict[str, pd.DataFrame],
    symbols: list[str],
    train: pd.DatetimeIndex,
) -> list[dict]:
    result = []
    columns = [*ROTATION_FEATURES, TARGET]
    for symbol in symbols:
        a = frames_a[symbol].loc[train].dropna(subset=columns)
        b = frames_b[symbol].loc[train].dropna(subset=columns)
        dates_equal = a.index.equals(b.index)
        if dates_equal:
            status = _status_pair(
                a[columns].to_numpy(dtype=np.float64),
                b[columns].to_numpy(dtype=np.float64),
            )
        else:
            status = {"status": "DATE_OR_ROW_ORDER_MISMATCH"}
        result.append({
            "symbol": symbol,
            "same_row_dates_and_order": bool(dates_equal),
            "tcc_rows": int(len(a)), "mct_rows": int(len(b)),
            **status,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcc-data", type=Path, required=True)
    parser.add_argument("--strategy11-zip", type=Path, required=True)
    parser.add_argument("--strategy12-zip", type=Path, required=True)
    parser.add_argument(
        "--input-audit-zip", type=Path, required=True,
        help="Completed 55/55 data parity report from v10.8.28.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output") / "tcc_v106_fold1_calibration_parity",
    )
    args = parser.parse_args()

    _assert_reference_files(args.tcc_data)
    with ZipFile(args.input_audit_zip) as source:
        report_name = next(
            (x for x in source.namelist() if x.endswith("/report.json")),
            None,
        )
        if report_name is None:
            raise ValueError("Input audit ZIP has no report.json.")
        audited = json.loads(source.read(report_name))
    if (
        audited.get("audit_status") != "COMPLETED"
        or audited.get("fully_audited_assets") != 55
        or audited.get("mct_stale_assets")
        or audited.get("errors")
        or audited.get("tcc_snapshot_sha256") != TCC_FROZEN_MANIFEST_SHA
    ):
        raise ValueError("Input audit is incomplete or not the expected snapshot.")

    archived, expected, candidate_archive = _load_archive_control(
        args.strategy11_zip, args.strategy12_zip
    )
    if audited.get("mct_archived_data_signature_sha256") != archived.get(
        "market_data_signature_sha256"
    ):
        raise ValueError("Input audit and archived MCT snapshot do not match.")

    config = build_control_config(TCC_CONFIG)
    tcc_bars, mct_bars = _load_bars(
        args.tcc_data, expected, config
    )
    tcc_frames, tcc_symbols, tcc_train, tcc_calib, tcc_fold = _fold_context(
        tcc_bars, config
    )
    mct_frames, mct_symbols, mct_train, mct_calib, mct_fold = _fold_context(
        mct_bars, config
    )
    if (
        tcc_symbols != mct_symbols
        or not tcc_train.equals(mct_train)
        or not tcc_calib.equals(mct_calib)
    ):
        raise ValueError(
            "Fold-1 assets/calendar/train/calibration rows differ; "
            "do not compare model scores on unlike inputs."
        )

    training_comparison = _compare_training_matrices(
        tcc_frames, mct_frames, tcc_symbols, tcc_train
    )
    tcc_summary, tcc_scores, tcc_matrix, tcc_matrices = _fit_and_score(
        "tcc_frozen_main_snapshot", tcc_frames, tcc_symbols, tcc_train,
        tcc_calib, config,
    )
    mct_summary, mct_scores, mct_matrix, mct_matrices = _fit_and_score(
        "mct_archived_snapshot", mct_frames, mct_symbols, mct_train,
        mct_calib, config,
    )

    predict_comparison = _status_pair(tcc_matrix, mct_matrix)
    prediction_rows = []
    for row, date in enumerate(tcc_calib):
        for col, symbol in enumerate(["CASH", *tcc_symbols]):
            a = float(tcc_matrix[row, col])
            b = float(mct_matrix[row, col])
            same = a == b or (np.isnan(a) and np.isnan(b))
            if not same:
                prediction_rows.append({
                    "date": date.isoformat(), "symbol": symbol,
                    "tcc_score": a, "mct_score": b,
                    "abs_delta": abs(a - b),
                    "within_tolerance": bool(np.isclose(
                        a, b, atol=1e-12, rtol=1e-12, equal_nan=True
                    )),
                })

    selected_archived = [
        row for row in candidate_archive
        if str(row["selected"]).strip().lower() == "true"
    ]
    archived_by_candidate = {
        float(row["candidate_margin"]): float(row["risk_adjusted_score"])
        for row in candidate_archive
    }
    for row in mct_scores:
        row["archived_strategy11_score"] = archived_by_candidate[
            row["candidate_margin"]
        ]
        row["delta_vs_archived_strategy11"] = (
            row["risk_adjusted_score"] - row["archived_strategy11_score"]
        )

    import lightgbm
    import threadpoolctl

    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "status": "COMPLETED",
        "tcc_source_data_commit": audited["tcc_source_commit"],
        "tcc_snapshot_sha256": TCC_FROZEN_MANIFEST_SHA,
        "source_data_note": (
            "These frozen CSVs are from a later TCC main commit, not shipped "
            "inside the historical TCC engine v1.0.6 tag. This experiment "
            "does not assert exact historical original-run snapshot identity."
        ),
        "tcc_engine": "vendored_verbatim_v1.0.6",
        "mct_strategy11_job_id": archived.get("job_id"),
        "mct_strategy12_job_id": audited.get("mct_job12_id"),
        "market_data_signature_sha256": archived.get(
            "market_data_signature_sha256"
        ),
        "fold1_train_sessions": int(len(tcc_train)),
        "fold1_calibration_sessions": int(len(tcc_calib)),
        "fold1_train_start": tcc_train.min().isoformat(),
        "fold1_train_end": tcc_train.max().isoformat(),
        "fold1_calibration_start": tcc_calib.min().isoformat(),
        "fold1_calibration_end": tcc_calib.max().isoformat(),
        "fold1_tcc_dates_equal_mct": True,
        "training_matrix_stage_counts": {
            status: sum(
                row["status"] == status for row in training_comparison
            )
            for status in sorted(
                {row["status"] for row in training_comparison}
            )
        },
        "calibration_predictions": predict_comparison,
        "tcc_frozen_main_data_result": tcc_summary,
        "mct_archived_data_result": mct_summary,
        "archive_strategy11_selected_candidate_margin": (
            float(selected_archived[0]["candidate_margin"])
            if len(selected_archived) == 1 else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
            "threadpoolctl": threadpoolctl.__version__,
            "mct_model_threads_override": os.getenv(
                "MCT_MODEL_THREADS_OVERRIDE"
            ),
            "model_settings": _lightgbm_settings(config),
            "deterministic_execution": config.deterministic_execution,
            "numeric_thread_limit": config.numeric_thread_limit,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(training_comparison).to_csv(
        args.output / "training_matrix_comparison.csv", index=False
    )
    pd.DataFrame(tcc_matrices + mct_matrices).to_csv(
        args.output / "training_matrix_manifest.csv", index=False
    )
    pd.DataFrame(tcc_scores + mct_scores).to_csv(
        args.output / "calibration_candidate_scores.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "calibration_prediction_differences.csv", index=False
    )
    outzip = args.output / "tcc_v106_fold1_calibration_parity.zip"
    with ZipFile(outzip, "w", ZIP_DEFLATED) as package:
        for name in (
            "report.json", "training_matrix_comparison.csv",
            "training_matrix_manifest.csv", "calibration_candidate_scores.csv",
            "calibration_prediction_differences.csv",
        ):
            package.write(args.output / name, arcname=name)

    print(json.dumps({
        "report_zip": str(outzip),
        "training_matrix_stage_counts": report["training_matrix_stage_counts"],
        "calibration_predictions": report["calibration_predictions"],
        "selected_tcc": tcc_summary["selected_candidate_margin"],
        "selected_mct": mct_summary["selected_candidate_margin"],
        "selected_archived_mct11": report[
            "archive_strategy11_selected_candidate_margin"
        ],
        "mct_archive_score_deltas": {
            str(row["candidate_margin"]): row["delta_vs_archived_strategy11"]
            for row in mct_scores
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
