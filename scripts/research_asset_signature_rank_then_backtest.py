from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import research_asset_signature_leave_one_out as calibration
from market_cycle_trader_api.core.environment import load_project_environment
from research_asset_signature_pipeline_backtest import (
    JOBS_COLLECTION,
    compare_results,
    exact_baseline_job,
    experiment_job,
)
from research_asset_signature_pipeline_ranking import (
    RANKING_METRICS,
    RANKING_POLICY_VERSION,
    analyse_candidate,
    build_model_payloads,
    canonical_hash,
    history_diagnostics,
    load_frames,
    rank_candidates,
    return_correlations,
)

ASSET_DISCOVERY_RESEARCH_COLLECTION = "asset_discovery_research"


def write_json(path: Path, value: Any) -> None:
    calibration._write_json(path, value)


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    rows = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value
    calibration._write_csv(path, rows)


def candidate_source(db: Any, baseline_assets: list[str], explicit: str | None) -> tuple[list[str], dict[str, Any]]:
    baseline = set(baseline_assets)
    if explicit:
        raw = [part.strip().upper() for part in explicit.split(",") if part.strip()]
        source = {"type": "explicit_cli", "run_id": None}
    else:
        doc = db[ASSET_DISCOVERY_RESEARCH_COLLECTION].find_one({"_id": "current"}) or {}
        if not isinstance(doc.get("results"), list) or not doc.get("results"):
            doc = db[ASSET_DISCOVERY_RESEARCH_COLLECTION].find_one(
                {"results.0": {"$exists": True}}, sort=[("updated_at", -1)]
            ) or {}
        raw = [
            str(item.get("symbol") or "").strip().upper()
            for item in list(doc.get("results") or [])
            if isinstance(item, dict) and str(item.get("symbol") or "").strip()
        ]
        source = {
            "type": "asset_discovery_research",
            "run_id": doc.get("run_id"),
            "status": doc.get("status"),
            "phase": doc.get("phase"),
            "api_version": doc.get("api_version"),
            "source_result_count": len(doc.get("results") or []),
        }
    unique = list(dict.fromkeys(raw))
    external = [symbol for symbol in unique if symbol not in baseline]
    if not external:
        raise RuntimeError("No new candidate symbols were found outside the selected Strategy.")
    source["candidate_symbols"] = external
    source["candidate_symbols_sha256"] = canonical_hash(external)
    source["excluded_existing_strategy_symbols"] = [symbol for symbol in unique if symbol in baseline]
    return external, source


def freeze_ranking(
    output_dir: Path,
    ranked: pd.DataFrame,
    policy: dict[str, Any],
    strategy: dict[str, Any],
    source: dict[str, Any],
    snapshot_end: pd.Timestamp,
) -> dict[str, Any]:
    allowed = [column for column in ranked.columns if not column.startswith("oracle_")]
    records = ranked[allowed].replace({np.nan: None}).to_dict(orient="records")
    selected = [str(row["symbol"]) for row in records if row.get("selected_for_backtest")]
    frozen = {
        "schema_version": 1,
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "snapshot_end": snapshot_end.date().isoformat(),
        "candidate_source": source,
        "ranking_policy": policy,
        "ranking": records,
        "selected_symbols": selected,
        "backtest_used_for_ranking": False,
        "oracle_metrics_used_for_ranking": False,
    }
    frozen["ranking_sha256"] = canonical_hash(records)
    frozen["selected_symbols_sha256"] = canonical_hash(selected)
    frozen["decision_snapshot_sha256"] = canonical_hash(frozen)
    write_json(output_dir / "ranking_snapshot_frozen.json", frozen)
    return frozen


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Use the existing Strategy signature to rank new assets without Backtest, freeze the decision, "
            "then run one combined validation Backtest against the exact certified baseline."
        )
    )
    result.add_argument("--strategy-sequence", type=int, default=10)
    result.add_argument("--strategy-id", default=None)
    result.add_argument("--snapshot-end", default="2026-09-04")
    result.add_argument("--mongo-uri", default=None)
    result.add_argument("--database", default=None)
    result.add_argument("--env-file", default=None)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--random-state", type=int, default=42)
    result.add_argument("--candidate-symbols", default=None)
    result.add_argument(
        "--selection-count",
        type=int,
        default=0,
        help="0 selects Pareto front layer 1; positive N selects the first N ranked candidates.",
    )
    result.add_argument("--baseline-job-id", default=None)
    result.add_argument("--ranking-only", action="store_true")
    result.add_argument("--output-dir", default=None)
    result.add_argument("--allow-remote-mongo", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    load_project_environment(args.env_file)
    mongo_uri = str(
        args.mongo_uri or os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    calibration._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))
    os.environ["MONGO_URL"] = mongo_uri
    os.environ["MONGO_URI"] = mongo_uri
    os.environ["MONGO_DATABASE"] = database_name

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, retryWrites=False)
    client.admin.command("ping")
    db = client[database_name]
    strategy = calibration._strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = calibration._configuration(strategy)
    baseline_assets = list(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in list(configuration.get("assets") or [])
        if str(symbol).strip()
    ))
    start = calibration._normalize_date(configuration.get("start_date") or "2016-01-01")
    snapshot_end = calibration._normalize_date(args.snapshot_end)
    identity = calibration._market_identity(configuration)
    source_symbols, source = candidate_source(db, baseline_assets, args.candidate_symbols)
    baseline_job = None if args.ranking_only else exact_baseline_job(db, strategy, snapshot_end, args.baseline_job_id)

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "research_output" / f"asset_signature_strategy_{args.strategy_sequence}_{snapshot_end.date().isoformat()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration._log(
        f"Start rank-then-backtest: Strategy #{args.strategy_sequence}, baseline={len(baseline_assets)}, "
        f"candidate_source={len(source_symbols)}, snapshot={snapshot_end.date().isoformat()}."
    )

    calibration._log("Step 1/7 - Loading baseline + candidate histories once from local MongoDB.")
    expected = calibration._expected_sessions(start, snapshot_end)
    requested = list(dict.fromkeys([*baseline_assets, *source_symbols]))
    frames = load_frames(db[calibration.ALPACA_MARKET_BARS_COLLECTION], requested, identity, start, snapshot_end)
    baseline_history = history_diagnostics(frames, baseline_assets, expected)
    candidate_history = history_diagnostics(frames, source_symbols, expected)
    write_csv(output_dir / "baseline_history_integrity.csv", baseline_history)
    write_csv(output_dir / "candidate_history_integrity.csv", candidate_history)
    if not baseline_history["history_complete"].all():
        raise RuntimeError("Selected Strategy does not have complete Full Strategy History in local MongoDB.")
    valid_candidates = candidate_history.loc[candidate_history["history_complete"], "symbol"].tolist()
    if not valid_candidates:
        raise RuntimeError("No candidate has complete Full Strategy History.")
    frames = {symbol: frames[symbol] for symbol in [*baseline_assets, *valid_candidates]}
    all_features, all_training = calibration._build_panels(frames)
    baseline_features = all_features[all_features["symbol"].isin(set(baseline_assets))].copy()
    baseline_training = all_training[all_training["symbol"].isin(set(baseline_assets))].copy()
    unique_dates = [pd.Timestamp(value) for value in sorted(pd.unique(baseline_training["date"]))]
    folds = calibration._fold_specs(unique_dates)
    latest_date = pd.Timestamp(max(pd.unique(baseline_features["date"])))
    full_corr, corr_60 = return_correlations(frames)
    client.close()
    calibration._log("MongoDB connection closed. Ranking is now entirely in RAM/CPU.")

    calibration._log("Step 2/7 - Loading or computing the 56-asset leave-one-out calibration.")
    calibration_path = output_dir / "asset_signatures.csv"
    calibration_folds_path = output_dir / "asset_signature_folds.csv"
    if calibration_path.exists() and len(pd.read_csv(calibration_path)) == len(baseline_assets):
        calibration_frame = pd.read_csv(calibration_path)
        calibration._log("Existing complete calibration reused.")
    else:
        results: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(args.workers, len(baseline_assets))) as executor:
            futures = {
                executor.submit(
                    calibration._analyse_asset,
                    symbol,
                    baseline_assets,
                    baseline_features,
                    baseline_training,
                    folds,
                    latest_date,
                    full_corr.loc[baseline_assets, baseline_assets],
                    corr_60.loc[baseline_assets, baseline_assets],
                    set(),
                    args.random_state,
                ): symbol
                for symbol in baseline_assets
            }
            for future in as_completed(futures):
                row, rows = future.result()
                results.append(row)
                fold_rows.extend(rows)
                calibration._log(f"Calibration {len(results)}/{len(baseline_assets)} - {row['symbol']}.")
        calibration_frame = pd.DataFrame(results).sort_values("symbol").reset_index(drop=True)
        write_csv(calibration_path, calibration_frame)
        write_csv(calibration_folds_path, fold_rows)
    write_json(output_dir / "signature_distribution.json", {
        "schema_version": 2,
        "metric_distributions": calibration._distribution_summary(calibration_frame),
        "ranking_metrics": [{"metric": metric, "direction": direction} for metric, direction in RANKING_METRICS],
        "backtest_used": False,
    })

    calibration._log("Step 3/7 - Fitting baseline-only fold models once and evaluating all external candidates.")
    payloads, latest = build_model_payloads(
        calibration,
        baseline_training,
        all_features,
        all_training,
        baseline_assets,
        valid_candidates,
        folds,
        latest_date,
        args.random_state,
    )
    candidate_rows: list[dict[str, Any]] = []
    candidate_fold_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(valid_candidates))) as executor:
        futures = {
            executor.submit(analyse_candidate, symbol, baseline_assets, payloads, latest, full_corr, corr_60): symbol
            for symbol in valid_candidates
        }
        for future in as_completed(futures):
            row, rows = future.result()
            candidate_rows.append(row)
            candidate_fold_rows.extend(rows)
            calibration._log(f"Candidate {len(candidate_rows)}/{len(valid_candidates)} - {row['symbol']}.")
    candidate_frame = pd.DataFrame(candidate_rows).sort_values("symbol").reset_index(drop=True)

    calibration._log("Step 4/7 - Building the final mathematical ranking without Backtest inputs.")
    ranked, policy = rank_candidates(candidate_frame, calibration_frame, int(args.selection_count))
    oracle_columns = [column for column in ranked.columns if column.startswith("oracle_")]
    write_csv(output_dir / "candidate_ranking.csv", ranked[[column for column in ranked.columns if column not in oracle_columns]])
    write_csv(output_dir / "candidate_ranking_folds.csv", candidate_fold_rows)
    write_csv(
        output_dir / "candidate_oracle_validation.csv",
        ranked[["symbol", "candidate_rank", "selected_for_backtest", *oracle_columns]],
    )
    write_json(output_dir / "ranking_policy.json", policy)

    calibration._log("Step 5/7 - Freezing ranking and selected assets before Backtest initialization.")
    frozen = freeze_ranking(output_dir, ranked, policy, strategy, source, snapshot_end)
    selected = list(frozen["selected_symbols"])
    write_json(output_dir / "experiment_manifest.json", {
        "schema_version": 2,
        "strategy_id": strategy.get("_id"),
        "strategy_revision": strategy.get("revision"),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "snapshot_end": snapshot_end.date().isoformat(),
        "baseline_assets": baseline_assets,
        "candidate_source": source,
        "valid_candidates": valid_candidates,
        "selected_assets": selected,
        "ranking_policy_version": RANKING_POLICY_VERSION,
        "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
        "backtest_used_for_ranking": False,
        "oracle_metrics_used_for_ranking": False,
        "ranking_data_source": "local_mongodb_only",
    })
    if args.ranking_only or not selected:
        write_json(output_dir / "experiment_summary.json", {
            "status": "ranking_frozen",
            "selected_assets": selected,
            "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
            "backtest_run": False,
        })
        calibration._log("Ranking frozen. Backtest skipped by configuration or empty selection.")
        return 0

    calibration._log("Step 6/7 - Running ONE combined full Backtest with the frozen selected assets.")
    from market_cycle_trader_api.core.runtime import close_mongo, database, initialize_mongo
    from market_cycle_trader_api.services.jobs import run_job
    from market_cycle_trader_api.services.results import build_results

    initialize_mongo(role="research")
    runtime_db = database()
    baseline_results = build_results(str(baseline_job["id"]))
    job, request = experiment_job(baseline_job, strategy, baseline_assets, selected, snapshot_end, frozen)
    write_json(output_dir / "baseline_backtest_job.json", {key: value for key, value in baseline_job.items() if key != "_id"})
    write_json(output_dir / "baseline_backtest_results.json", baseline_results)
    write_json(output_dir / "candidate_backtest_request.json", request)
    runtime_db[JOBS_COLLECTION].insert_one(deepcopy(job))
    started = time.perf_counter()
    try:
        run_job(job["id"])
        completed = runtime_db[JOBS_COLLECTION].find_one({"id": job["id"]}) or {}
        write_json(output_dir / "candidate_backtest_job.json", {key: value for key, value in completed.items() if key != "_id"})
        if completed.get("status") != "completed":
            write_json(output_dir / "experiment_summary.json", {
                "status": "backtest_failed",
                "selected_assets": selected,
                "backtest_job_id": job["id"],
                "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
            })
            raise RuntimeError(f"Final validation Backtest failed: {job['id']}")
        candidate_results = build_results(job["id"])
        comparison = compare_results(baseline_results, candidate_results)
        comparison.update({
            "baseline_job_id": baseline_job["id"],
            "candidate_job_id": job["id"],
            "selected_assets": selected,
            "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
        })
        write_json(output_dir / "candidate_backtest_results.json", candidate_results)
        write_json(output_dir / "backtest_comparison.json", comparison)
        calibration._log("Step 7/7 - Capturing final experiment summary.")
        write_json(output_dir / "experiment_summary.json", {
            "schema_version": 2,
            "status": "completed",
            "strategy_id": strategy.get("_id"),
            "strategy_revision": strategy.get("revision"),
            "snapshot_end": snapshot_end.date().isoformat(),
            "baseline_asset_count": len(baseline_assets),
            "candidate_count": len(valid_candidates),
            "selected_asset_count": len(selected),
            "selected_assets": selected,
            "ranking_policy_version": RANKING_POLICY_VERSION,
            "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
            "ranking_sha256": frozen["ranking_sha256"],
            "selected_symbols_sha256": frozen["selected_symbols_sha256"],
            "backtest_used_for_ranking": False,
            "oracle_metrics_used_for_ranking": False,
            "baseline_backtest_job_id": baseline_job["id"],
            "candidate_backtest_job_id": job["id"],
            "backtest_elapsed_seconds": time.perf_counter() - started,
            "backtest_comparison": comparison,
        })
    finally:
        close_mongo()
    calibration._log(f"Completed. Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
