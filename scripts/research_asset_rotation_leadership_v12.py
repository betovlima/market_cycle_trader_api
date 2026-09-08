from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_asset_rotation_leadership as research  # noqa: E402

SCRIPT_VERSION = "asset-rotation-leadership-v1.2"
_LOG_LOCK = threading.Lock()


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _LOG_LOCK:
        print(f"[{stamp}] {message}", flush=True)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _install_progress_logging() -> None:
    # Replace only the research script logger. This also removes the pandas
    # Timestamp.utcnow deprecation warning without touching research math.
    research._log = _log

    original_analyse_asset = research._analyse_asset
    original_calibrate_and_fit = research._calibrate_and_fit

    def logged_calibrate_and_fit(
        symbol: str,
        frame: pd.DataFrame,
        common_dates: pd.DatetimeIndex,
        fold: dict[str, Any],
        config: Any,
    ):
        fold_id = int(fold["fold_id"])
        _log(
            f"{symbol}: fold {fold_id} - calibration + final LightGBM training started."
        )
        result = original_calibrate_and_fit(
            symbol,
            frame,
            common_dates,
            fold,
            config,
        )
        calibrated = float(result[2])
        effective = float(result[3])
        _log(
            f"{symbol}: fold {fold_id} - model ready "
            f"(calibrated_margin={calibrated:.6f}, effective_margin={effective:.6f}); "
            "running OOS timing simulation and leadership predictions."
        )
        return result

    def logged_analyse_asset(*args: Any, **kwargs: Any):
        symbol = str(args[0] if args else kwargs.get("symbol") or "unknown").upper()
        _log(
            f"{symbol}: START - intrinsic timing diagnostic + OOS rotation-opportunity predictions."
        )
        result = original_analyse_asset(*args, **kwargs)
        aggregate = result[0]
        elapsed = float(aggregate.get("elapsed_seconds") or 0.0)
        _log(
            f"{symbol}: ALL FOLDS COMPLETE in {elapsed:.1f}s - "
            "waiting for cross-sectional rotation-opportunity ranking."
        )
        return result

    research._calibrate_and_fit = logged_calibrate_and_fit
    research._analyse_asset = logged_analyse_asset


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--output-dir", default=None)
    args, _ = parser.parse_known_args()
    return args


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    snapshot = pd.Timestamp(args.snapshot_end).date().isoformat()
    return (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{args.strategy_sequence}_{snapshot}"
    ).resolve()


def _postprocess_rotation_only_selection(output_dir: Path) -> None:
    qualification_path = output_dir / "asset_rotation_qualification.csv"
    frozen_path = output_dir / "rotation_leadership_snapshot_frozen.json"
    manifest_path = output_dir / "experiment_manifest.json"

    if not qualification_path.exists() or not frozen_path.exists():
        raise RuntimeError(
            "Rotation-leadership qualification artifacts are missing after phase 1."
        )

    combined = pd.read_csv(qualification_path)
    if combined.empty or "leadership_qualified" not in combined.columns:
        raise RuntimeError("Rotation leadership qualification table is invalid or empty.")

    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    leadership_ok = _bool_series(combined["leadership_qualified"])

    # Scientific rule agreed for v1.2:
    # - same-asset Timing vs Buy & Hold remains a diagnostic;
    # - final Backtest participation is decided only by useful OOS rotation opportunity.
    combined["qualified"] = leadership_ok
    combined["qualification_path"] = np.where(
        leadership_ok,
        "rotation_opportunity",
        "rejected_rotation_opportunity",
    )
    combined = combined.sort_values(
        [
            "qualified",
            "leadership_event_realized_utility_median",
            "leadership_event_positive_utility_rate",
            "leadership_event_forward_net_log_return_mean",
            "symbol",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    combined["research_rank"] = np.arange(1, len(combined) + 1)
    research._write_csv(qualification_path, combined)

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    original_assets = [str(x).upper() for x in frozen.get("original_assets") or []]
    original_set = set(original_assets)
    external = [
        str(x).upper() for x in frozen.get("complete_external_candidates") or []
    ]

    qualified_assets = combined.loc[combined["qualified"], "symbol"].astype(str).tolist()
    qualified_set = set(qualified_assets)
    retained_existing = [x for x in original_assets if x in qualified_set]
    removed_existing = [x for x in original_assets if x not in qualified_set]
    added_candidates = [x for x in qualified_assets if x not in original_set]
    rejected_candidates = [x for x in external if x not in qualified_set]

    ranking_records = combined.to_dict(orient="records")
    ranking_sha256 = research._sha256_json(ranking_records)
    qualified_assets_sha256 = research._sha256_json(qualified_assets)

    frozen.pop("decision_snapshot_sha256", None)
    frozen.update(
        {
            "schema_version": 5,
            "script_version": SCRIPT_VERSION,
            "selection_rule": "rotation_leadership_qualified only; same-asset timing vs Buy & Hold is diagnostic only",
            "benchmark_definition": (
                "Buy & Hold versus causal point-in-time intelligent rotation across available assets; "
                "the study does not require prediction of exact tops or bottoms."
            ),
            "intrinsic_timing_used_for_selection": False,
            "same_asset_buy_hold_used_for_selection": False,
            "rotation_opportunity_used_for_selection": True,
            "qualified_assets": qualified_assets,
            "retained_existing_assets": retained_existing,
            "removed_existing_assets": removed_existing,
            "added_candidate_assets": added_candidates,
            "rejected_candidate_assets": rejected_candidates,
            "ranking_sha256": ranking_sha256,
            "qualified_assets_sha256": qualified_assets_sha256,
        }
    )
    frozen["decision_snapshot_sha256"] = research._sha256_json(frozen)
    research._write_json(frozen_path, frozen)

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_counts = dict(manifest.get("result_counts") or {})
    result_counts.update(
        {
            "qualified_for_final_backtest": len(qualified_assets),
            "retained_existing": len(retained_existing),
            "removed_existing": len(removed_existing),
            "added_candidates": len(added_candidates),
            "rejected_candidates": len(rejected_candidates),
        }
    )
    manifest.update(
        {
            "schema_version": 5,
            "script_version": SCRIPT_VERSION,
            "selection_rule": "rotation_leadership_qualified only",
            "intrinsic_timing_used_for_selection": False,
            "rotation_opportunity_used_for_selection": True,
            "result_counts": result_counts,
            "ranking_sha256": ranking_sha256,
            "qualified_assets_sha256": qualified_assets_sha256,
            "decision_snapshot_sha256": frozen["decision_snapshot_sha256"],
        }
    )
    research._write_json(manifest_path, manifest)

    _log(
        "FINAL ROTATION-OPPORTUNITY DECISION - Timing vs Buy & Hold above is diagnostic only."
    )
    total_folds = int(manifest.get("walk_forward_fold_count") or 0)
    for row in combined.sort_values("symbol").to_dict(orient="records"):
        include = bool(row.get("qualified"))
        decision = "INCLUDE IN BACKTEST" if include else "EXCLUDE FROM BACKTEST"
        if include:
            reason = "useful_rotation_opportunity"
        else:
            reason = str(row.get("leadership_fail_reasons") or "rotation_opportunity_not_qualified")
            if not reason or reason.lower() == "nan":
                reason = "rotation_opportunity_not_qualified"

        event_count = int(float(row.get("leadership_event_count") or 0))
        event_folds = int(float(row.get("leadership_event_fold_count") or 0))
        positive_rate = float(row.get("leadership_event_positive_utility_rate") or 0.0)
        median_utility = float(row.get("leadership_event_realized_utility_median") or 0.0)
        mean_forward = float(row.get("leadership_event_forward_net_log_return_mean") or 0.0)
        _log(
            f"FINAL {row['symbol']}: {decision} | events={event_count} | "
            f"leadership_folds={event_folds}/{total_folds or '?'} | "
            f"positive_events={positive_rate:.1%} | median_utility={median_utility:.2%} | "
            f"mean_forward_return={mean_forward:.2%} | reason={reason}."
        )

    _log(
        f"Frozen final universe: {len(qualified_assets)} assets = "
        f"{len(retained_existing)} retained existing + {len(added_candidates)} new."
    )
    _log(f"Frozen snapshot updated for rotation-only selection: {frozen_path}")


def main() -> int:
    args = _arguments()
    _install_progress_logging()
    research.SCRIPT_VERSION = SCRIPT_VERSION
    result = int(research.main())
    if result != 0:
        return result
    _postprocess_rotation_only_selection(_output_dir(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
