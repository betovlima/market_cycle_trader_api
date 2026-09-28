"""Causal diagnostic: exchange only differing Fold-1 prediction columns.

Re-train the same 55 frozen TCC v1.0.6 LightGBM models on each verified
dataset, then exchange calibration predictions without re-training or changing
source prices. This is NOT a new full backtest or a production policy change.
"""
from __future__ import annotations

import argparse
import csv
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

# MCT's Mongo constants are read at import time.
load_project_environment()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tcc_v106_fold1_calibration_parity import (
    _fold_context,
    _load_archive_control,
    _load_bars,
    _status_pair,
)
from scripts.audit_tcc_v106_fold1_data_parity import (
    TCC_FROZEN_MANIFEST_SHA,
    _assert_reference_files,
)
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    _model_utilities,
    _simple_policy_growth,
    _utility_policy,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    CONFIG as TCC_CONFIG,
    build_control_config,
)
from market_cycle_trader_api.tcc_v106_reference.research_challengers import (
    _lightgbm_fit_models,
)

EPS = 1e-10
AUDIT_NAME = "tcc_v106_fold1_four_model_crossover"


def _read_unique_zip_member(path: Path, basename: str) -> bytes:
    with ZipFile(path) as source:
        matches = [
            name for name in source.namelist()
            if name == basename or name.endswith("/" + basename)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {basename} in {path}; got {len(matches)}."
            )
        return source.read(matches[0])


def _read_reference_report(path: Path) -> tuple[dict, dict[float, dict], set[str]]:
    report = json.loads(_read_unique_zip_member(path, "report.json"))
    if (
        report.get("status") != "COMPLETED"
        or report.get("experiment") != "tcc_v106_fold1_calibration_parity"
        or report.get("tcc_snapshot_sha256") != TCC_FROZEN_MANIFEST_SHA
        or report.get("fold1_tcc_dates_equal_mct") is not True
    ):
        raise ValueError("The supplied calibration audit is not the verified Fold-1 result.")
    rows = list(csv.DictReader(io.StringIO(
        _read_unique_zip_member(path, "calibration_candidate_scores.csv").decode("utf-8-sig")
    )))
    tcc_rows = {
        float(item["candidate_margin"]): item
        for item in rows if item["dataset"] == "tcc_frozen_main_snapshot"
    }
    mct_rows = {
        float(item["candidate_margin"]): item
        for item in rows if item["dataset"] == "mct_archived_snapshot"
    }
    if set(tcc_rows) != {0.0, 0.0025, 0.005, 0.01} or set(mct_rows) != set(tcc_rows):
        raise ValueError("Missing candidate scores in the completed calibration audit.")
    diffs = list(csv.DictReader(io.StringIO(
        _read_unique_zip_member(path, "calibration_prediction_differences.csv").decode("utf-8-sig")
    )))
    symbols = {
        item["symbol"] for item in diffs
        if str(item["within_tolerance"]).strip().lower() == "false"
    }
    return report, {margin: {"tcc": float(tcc_rows[margin]["risk_adjusted_score"]),
                               "mct": float(mct_rows[margin]["risk_adjusted_score"])}
                    for margin in sorted(tcc_rows)}, symbols


def _prediction_matrix(models, frames, symbols, dates, config) -> np.ndarray:
    """Exact per-date inference used by the unbatched calibration policy."""
    return np.stack([
        _model_utilities(models, frames, symbols, pd.Timestamp(day), config)
        for day in dates
    ])


def _cache_from_matrix(matrix: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    return {
        pd.Timestamp(day): matrix[i].copy()
        for i, day in enumerate(dates)
    }


def _scenario_scores(
    *,
    name: str,
    data_source: str,
    prediction_source: str,
    swaps: list[str],
    matrix: np.ndarray,
    frames,
    models,
    symbols: list[str],
    dates: pd.DatetimeIndex,
    config,
) -> tuple[dict, list[dict], dict]:
    cache = _cache_from_matrix(matrix, dates)
    traces = {}
    scores = []
    for margin in tuple(float(m) for m in config.rotation_switch_margin_candidates):
        policy = _utility_policy(
            models, frames, symbols, config, margin, utility_cache=cache
        )
        transitions = []

        def recording_policy(now, position, holding):
            action, score = policy(now, position, holding)
            transitions.append({
                "date": pd.Timestamp(now).isoformat(),
                "position_before": int(position),
                "action": int(action),
                "holding_before": int(holding),
            })
            return action, score

        score = _simple_policy_growth(
            recording_policy, frames, symbols, dates, config
        )
        traces[margin] = transitions
        scores.append({
            "scenario": name,
            "base_data_source": data_source,
            "prediction_source": prediction_source,
            "swapped_symbols": ",".join(swaps),
            "candidate_margin": margin,
            "calibration_score": float(score),
        })
    selected = max(scores, key=lambda item: item["calibration_score"])
    summary = {
        "scenario": name,
        "data_source": data_source,
        "prediction_source": prediction_source,
        "swapped_symbols": swaps,
        "selected_margin": selected["candidate_margin"],
        "effective_margin": max(
            float(config.rotation_switch_margin),
            selected["candidate_margin"],
        ),
        "selected_score": selected["calibration_score"],
    }
    return summary, scores, traces


def _swap_columns(base: np.ndarray, donor: np.ndarray, symbols: list[str],
                  swap: list[str]) -> np.ndarray:
    result = np.array(base, dtype=np.float64, copy=True)
    for symbol in swap:
        column = symbols.index(symbol) + 1  # column zero denotes CASH
        result[:, column] = donor[:, column]
    return result


def _first_trace_difference(left: list[dict], right: list[dict]) -> dict | None:
    for a, b in zip(left, right):
        if a != b:
            return {
                "date": a["date"],
                "base_position": a["position_before"],
                "base_action": a["action"],
                "hybrid_position": b["position_before"],
                "hybrid_action": b["action"],
            }
    return None if len(left) == len(right) else {"reason": "length_mismatch"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcc-data", type=Path, required=True)
    parser.add_argument("--strategy11-zip", type=Path, required=True)
    parser.add_argument("--strategy12-zip", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output") / "tcc_v106_fold1_four_model_crossover",
    )
    args = parser.parse_args()

    _assert_reference_files(args.tcc_data)
    archived, expected, historic_candidates = _load_archive_control(
        args.strategy11_zip, args.strategy12_zip
    )
    prior, reference_scores, prior_diff_symbols = _read_reference_report(
        args.calibration_zip
    )
    if prior.get("market_data_signature_sha256") != archived.get(
        "market_data_signature_sha256"
    ):
        raise ValueError("Archived MCT signature differs from calibration audit.")
    config = build_control_config(TCC_CONFIG)
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
        raise ValueError("Fold 1 calendars or symbol order differ.")
    symbols = tcc_symbols

    print("[1/4] Fit frozen TCC source models, Fold 1 calibration only", flush=True)
    tcc_models = _lightgbm_fit_models(
        tcc_frames, symbols, tcc_train, config,
        phase="crossover_tcc_fold1", device_type="cpu",
    )
    print("[2/4] Fit archived MCT source models, Fold 1 calibration only", flush=True)
    mct_models = _lightgbm_fit_models(
        mct_frames, symbols, mct_train, config,
        phase="crossover_mct_fold1", device_type="cpu",
    )
    if len(tcc_models) != 55 or len(mct_models) != 55:
        raise ValueError("Expected 55 fitted models in both Fold 1 variants.")

    print("[3/4] Build row-wise calibration predictions", flush=True)
    tcc_matrix = _prediction_matrix(
        tcc_models, tcc_frames, symbols, tcc_dates, config
    )
    mct_matrix = _prediction_matrix(
        mct_models, mct_frames, symbols, mct_dates, config
    )
    material = sorted({
        symbol for i, symbol in enumerate(symbols, 1)
        if not np.isclose(
            tcc_matrix[:, i], mct_matrix[:, i],
            atol=1e-12, rtol=1e-12, equal_nan=True,
        ).all()
    })
    if set(material) != prior_diff_symbols:
        raise ValueError(
            f"Prediction differences no longer reproduce the prior audit: "
            f"now={material}, archived={sorted(prior_diff_symbols)}. "
            "Check runtime and numeric conditions before interpreting a crossover."
        )

    all_summaries = []
    all_scores = []
    first_differences = []
    baseline_traces = {}

    print(f"[4/4] Score baseline and hybrid policies for {material}", flush=True)
    scenarios = [
        ("tcc_original", "tcc", "tcc", [], tcc_matrix, tcc_frames, tcc_models),
        ("mct_original", "mct", "mct", [], mct_matrix, mct_frames, mct_models),
        (
            "tcc_prices_mct_predictions_all", "tcc", "mct", material,
            _swap_columns(tcc_matrix, mct_matrix, symbols, material),
            tcc_frames, tcc_models,
        ),
        (
            "mct_prices_tcc_predictions_all", "mct", "tcc", material,
            _swap_columns(mct_matrix, tcc_matrix, symbols, material),
            mct_frames, mct_models,
        ),
    ]
    for symbol in material:
        scenarios.append((
            f"tcc_prices_swap_{symbol}", "tcc", "mct", [symbol],
            _swap_columns(tcc_matrix, mct_matrix, symbols, [symbol]),
            tcc_frames, tcc_models,
        ))
        scenarios.append((
            f"mct_prices_swap_{symbol}", "mct", "tcc", [symbol],
            _swap_columns(mct_matrix, tcc_matrix, symbols, [symbol]),
            mct_frames, mct_models,
        ))

    for name, data_origin, pred_origin, swaps, matrix, frames, models in scenarios:
        summary, scores, traces = _scenario_scores(
            name=name, data_source=data_origin,
            prediction_source=pred_origin, swaps=swaps, matrix=matrix,
            frames=frames, models=models, symbols=symbols,
            dates=tcc_dates, config=config,
        )
        all_summaries.append(summary)
        all_scores.extend(scores)
        if name in {"tcc_original", "mct_original"}:
            baseline_traces[data_origin] = traces
            key = data_origin
            for row in scores:
                target = reference_scores[row["candidate_margin"]][key]
                if abs(row["calibration_score"] - target) > EPS:
                    raise ValueError(
                        f"{name}: row-wise replay does not reproduce the "
                        f"previous audit for margin {row['candidate_margin']}: "
                        f"{row['calibration_score']} vs {target}."
                    )
                row["prior_audit_score"] = target
        else:
            base_traces = baseline_traces[data_origin]
            for row in scores:
                margin = row["candidate_margin"]
                first = _first_trace_difference(base_traces[margin], traces[margin])
                first_differences.append({
                    "scenario": name,
                    "candidate_margin": margin,
                    "first_transition_difference": (
                        json.dumps(first, sort_keys=True) if first is not None else ""
                    ),
                    "calibration_score": row["calibration_score"],
                    "base_score": reference_scores[margin][data_origin],
                    "score_delta": (
                        row["calibration_score"]
                        - reference_scores[margin][data_origin]
                    ),
                })
    archived_scores = {
        float(row["candidate_margin"]): float(row["risk_adjusted_score"])
        for row in historic_candidates
    }
    for row in all_scores:
        if row["scenario"] == "mct_original":
            archived_score = archived_scores[row["candidate_margin"]]
            if abs(row["calibration_score"] - archived_score) > EPS:
                raise ValueError("MCT baseline no longer reproduces Strategy #11 archive.")
            row["strategy11_archived_score"] = archived_score

    import lightgbm
    report = {
        "schema_version": 1,
        "experiment": AUDIT_NAME,
        "status": "COMPLETED",
        "source_data_note": prior["source_data_note"],
        "tcc_snapshot_sha256": TCC_FROZEN_MANIFEST_SHA,
        "mct_archived_data_signature_sha256": archived[
            "market_data_signature_sha256"
        ],
        "fold1_training_sessions": len(tcc_train),
        "fold1_calibration_sessions": len(tcc_dates),
        "symbols_with_material_prediction_difference": material,
        "prediction_comparison": _status_pair(tcc_matrix, mct_matrix),
        "scenarios": all_summaries,
        "baseline_replays_verified": True,
        "runtime": {
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "model_threads_override": os.getenv("MCT_MODEL_THREADS_OVERRIDE"),
        },
        "interpretation_limit": (
            "Counterfactual exchange of prediction vectors, not a "
            "production portfolio replay. Source data has sub-1e-12 "
            "differences and the TCC main CSVs postdate the historic "
            "TCC v1.0.6 tag."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "report.json": json.dumps(
            report, indent=2, ensure_ascii=False
        ) + "\n",
    }
    (args.output / "report.json").write_text(
        files["report.json"], encoding="utf-8"
    )
    pd.DataFrame(all_scores).to_csv(
        args.output / "crossover_candidate_scores.csv", index=False
    )
    pd.DataFrame(first_differences).to_csv(
        args.output / "first_transition_differences.csv", index=False
    )
    output_zip = args.output / "tcc_v106_fold1_four_model_crossover.zip"
    with ZipFile(output_zip, "w", ZIP_DEFLATED) as output:
        for basename in (
            "report.json", "crossover_candidate_scores.csv",
            "first_transition_differences.csv",
        ):
            output.write(args.output / basename, arcname=basename)
    print(json.dumps({
        "output_zip": str(output_zip),
        "materially_different_prediction_symbols": material,
        "scenarios": [
            {"scenario": row["scenario"],
             "selected_margin": row["selected_margin"],
             "selected_score": row["selected_score"]}
            for row in all_summaries
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
