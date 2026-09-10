from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_signature_leave_one_out as common  # noqa: E402
from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.engine.market_data import validate_and_clean_bars  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402
from market_cycle_trader_api.services.asset_discovery_ranker import train_ranker  # noqa: E402
from market_cycle_trader_api.services.exact_marginal_capital_search import (  # noqa: E402
    annotate_preselector_ranks,
    build_preselector_recall,
    economic_outcome,
    summarize_exact_search,
)

SCRIPT_VERSION = "exact-marginal-capital-search-v1.0.0"
EXPERIMENT_NAME = "exact_marginal_capital_search"
DEFAULT_WORKERS = 1


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256_strings(values: list[str]) -> str:
    normalized = "|".join(str(value).strip().upper() for value in values)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_fresh_output(path: Path) -> None:
    research_root = (PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(research_root)
    except ValueError as exc:
        raise RuntimeError(
            "--fresh-run may delete only output under PROJECT_ROOT/research_output. "
            f"Resolved path: {resolved}"
        ) from exc
    if not relative.parts or not relative.parts[0].startswith("exact_marginal_capital_search_"):
        raise RuntimeError(f"Refusing to delete unexpected research output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the original economic judge directly: rank external assets for diagnostics, "
            "but evaluate every history-complete candidate with the exact Strategy engine and persist "
            "positive, negative and failed outcomes."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _candidate_symbols(
    collection: Any,
    identity: dict[str, str],
    baseline_assets: list[str],
    explicit: list[str] | None,
) -> list[str]:
    baseline = set(baseline_assets)
    if explicit is not None:
        values = [str(value).strip().upper() for value in explicit if str(value).strip()]
        values = list(dict.fromkeys(values))
        return [symbol for symbol in values if symbol not in baseline]
    cached = [
        str(value).strip().upper()
        for value in collection.distinct("symbol", dict(identity))
        if str(value).strip()
    ]
    return sorted(set(cached).difference(baseline))


def _load_candidate_frame(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: BacktestRequest,
    required_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = common._load_frames_once(
        collection,
        [symbol],
        identity,
        history_start,
        snapshot_end,
    )
    frame = validate_and_clean_bars(frames[symbol], config)
    coverage = discovery._history_coverage_against_baseline(
        symbol,
        frame,
        config,
        required_sessions,
    )
    return frame, coverage


def _candidate_evaluation(
    *,
    db: Any,
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: BacktestRequest,
    strategy_id: str,
    baseline_assets: list[str],
    baseline_frames: dict[str, pd.DataFrame],
    baseline_metrics: dict[str, Any],
    baseline_sessions: pd.DatetimeIndex,
    required_sessions: pd.DatetimeIndex,
    ranker_bundle: Any,
    baseline_returns: pd.DataFrame,
) -> dict[str, Any]:
    overall_started = time.perf_counter()
    row: dict[str, Any] = {
        "symbol": symbol,
        "evaluation_status": "running",
        "preselector_raw_score": None,
        "preselector_error": None,
        "history_window_complete": False,
        "history_load_seconds": None,
        "preselector_seconds": None,
        "exact_replay_seconds": None,
        "total_seconds": None,
    }

    load_started = time.perf_counter()
    try:
        frame, coverage = _load_candidate_frame(
            collection,
            symbol,
            identity,
            history_start,
            snapshot_end,
            config,
            required_sessions,
        )
        row.update(coverage)
        row["history_window_complete"] = True
    except RuntimeError as exc:
        reason = str(exc).strip().lower()
        row["history_load_seconds"] = float(time.perf_counter() - load_started)
        if reason in {
            "insufficient_history",
            "discontinuous_history",
            "ticker_identity_discontinuity",
        } or "no in-window bars were loaded" in reason.lower():
            row.update(
                {
                    "evaluation_status": "history_rejected",
                    "rejection_reason": reason[:500],
                    "total_seconds": float(time.perf_counter() - overall_started),
                }
            )
            return row
        row.update(
            {
                "evaluation_status": "failed",
                "error": str(exc)[:700],
                "total_seconds": float(time.perf_counter() - overall_started),
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "history_load_seconds": float(time.perf_counter() - load_started),
                "evaluation_status": "failed",
                "error": str(exc)[:700],
                "total_seconds": float(time.perf_counter() - overall_started),
            }
        )
        return row
    row["history_load_seconds"] = float(time.perf_counter() - load_started)

    score_started = time.perf_counter()
    try:
        score = discovery._score_candidate(ranker_bundle, symbol, frame, baseline_returns)
        row.update(
            {
                "preselector_raw_score": score.get("raw_score"),
                "feature_at": score.get("feature_at"),
                "latest_close": score.get("latest_close"),
                "median_dollar_volume": score.get("median_dollar_volume"),
                "return_20": score.get("return_20"),
                "return_60": score.get("return_60"),
                "volatility_20": score.get("volatility_20"),
                "drawdown_60": score.get("drawdown_60"),
                "trend_efficiency_20": score.get("trend_efficiency_20"),
                "max_baseline_correlation_60": score.get("max_baseline_correlation_60"),
            }
        )
    except Exception as exc:
        # The preselector is not the judge. An unrankable candidate still receives
        # the exact economic evaluation when its Strategy history is complete.
        row["preselector_error"] = str(exc)[:700]
    row["preselector_seconds"] = float(time.perf_counter() - score_started)

    exact_started = time.perf_counter()
    try:
        candidate_assets = list(dict.fromkeys([*baseline_assets, symbol]))
        request = discovery._marginal_execution_request(
            db,
            config,
            {"id": strategy_id},
            config,
            snapshot_end.date().isoformat(),
            assets=candidate_assets,
            reference_assets=baseline_assets,
            candidate_assets=[symbol],
            analysis_start_date=history_start.date().isoformat(),
            analysis_end_date=snapshot_end.date().isoformat(),
        )
        candidate_frames = dict(baseline_frames)
        candidate_frames[symbol] = frame
        metrics, sessions = discovery._run_rotation_replay(candidate_frames, request)
        context = discovery._research_context_compatibility(baseline_sessions, sessions)
        row.update(context)
        if not bool(context.get("research_context_compatible")):
            row.update(
                {
                    "evaluation_status": "context_rejected",
                    "rejection_reason": "research_context_incomplete",
                }
            )
        else:
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            candidate_capital = discovery._finite_number(metrics.get("ending_capital"))
            row.update(
                {
                    "evaluation_status": "completed",
                    "baseline_ending_capital": baseline_capital,
                    "candidate_ending_capital": candidate_capital,
                    "ending_capital_delta": discovery._delta(candidate_capital, baseline_capital),
                    "ending_capital_delta_rate": discovery._capital_delta_rate(candidate_capital, baseline_capital),
                    "candidate_cagr": metrics.get("cagr"),
                    "candidate_sharpe": metrics.get("sharpe"),
                    "candidate_maximum_drawdown": metrics.get("maximum_drawdown"),
                    "candidate_worst_fold_return": metrics.get("worst_fold_return"),
                    "candidate_switches": metrics.get("switches"),
                    "candidate_cash_days": metrics.get("cash_days"),
                    "candidate_market_exposure": metrics.get("market_exposure"),
                    "candidate_negative_months": metrics.get("negative_months"),
                    "candidate_severe_negative_months": metrics.get("severe_negative_months"),
                    "cagr_delta": discovery._delta(metrics.get("cagr"), baseline_metrics.get("cagr")),
                    "sharpe_delta": discovery._delta(metrics.get("sharpe"), baseline_metrics.get("sharpe")),
                    "maximum_drawdown_delta": discovery._delta(
                        metrics.get("maximum_drawdown"), baseline_metrics.get("maximum_drawdown")
                    ),
                    "worst_fold_return_delta": discovery._delta(
                        metrics.get("worst_fold_return"), baseline_metrics.get("worst_fold_return")
                    ),
                    "switches_delta": discovery._delta(metrics.get("switches"), baseline_metrics.get("switches")),
                    "cash_days_delta": discovery._delta(metrics.get("cash_days"), baseline_metrics.get("cash_days")),
                }
            )
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700]})
    row["exact_replay_seconds"] = float(time.perf_counter() - exact_started)
    row["total_seconds"] = float(time.perf_counter() - overall_started)
    row["economic_outcome"] = economic_outcome(row)
    return row


def _manifest_contract(
    *,
    strategy: dict[str, Any],
    strategy_id: str,
    strategy_sequence: int,
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    baseline_assets: list[str],
    candidates: list[str],
    workers: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "api_version": API_VERSION,
        "strategy_id": strategy_id,
        "strategy_sequence": strategy_sequence,
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": str(strategy.get("configuration_hash") or ""),
        "history_start": history_start.date().isoformat(),
        "snapshot_end": snapshot_end.date().isoformat(),
        "baseline_assets": baseline_assets,
        "baseline_asset_count": len(baseline_assets),
        "candidate_count": len(candidates),
        "candidate_symbols_sha256": _sha256_strings(candidates),
        "workers": workers,
        "candidate_source": "local_mongodb_cached_symbols",
        "market_data_source": "local_mongodb_only",
        "alpaca_network_used": False,
        "mongo_writes": False,
        "preselector_role": "diagnostic_queue_order_only",
        "preselector_filters_exact_evaluation": False,
        "exact_judge": "run_rotation_models_via_asset_discovery_run_rotation_replay",
        "baseline_exact_replay_count": 1,
        "candidate_exact_retrain_and_replay": True,
        "all_candidate_outcomes_persisted": True,
        "acceptance_boundary": "ending_capital_delta_rate > 0",
        "note": (
            "This is a benchmark of the expensive economic judge, not an outer-walk-forward "
            "generalization claim. Every history-complete candidate is evaluated regardless of rank."
        ),
    }


def _verify_resume_contract(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = (
        "script_version",
        "strategy_id",
        "strategy_sequence",
        "strategy_revision",
        "strategy_configuration_hash",
        "history_start",
        "snapshot_end",
        "candidate_symbols_sha256",
    )
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "Existing Exact Marginal Capital Search output belongs to another experiment "
            f"({', '.join(mismatches)} differ). Use --fresh-run."
        )


def main() -> int:
    args = _parser().parse_args()
    load_project_environment(args.env_file)
    if args.mongo_uri:
        os.environ["MONGO_URI"] = str(args.mongo_uri)
        os.environ["MONGO_URL"] = str(args.mongo_uri)
    if args.database:
        os.environ["MONGO_DATABASE"] = str(args.database)

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))
    workers = max(1, int(args.workers or DEFAULT_WORKERS))

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 4),
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
        configuration = common._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        baseline_assets = list(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in list(configuration.get("assets") or [])
                if str(symbol).strip()
            )
        )
        if len(baseline_assets) < 3:
            raise RuntimeError("The selected Strategy must contain at least three baseline assets.")

        configured_start = common._normalize_date(configuration.get("start_date"))
        history_start = common._normalize_date(args.history_start)
        if configured_start != history_start:
            raise RuntimeError(
                "--history-start must match the Strategy start date for the exact benchmark: "
                f"strategy={configured_start.date().isoformat()}, requested={history_start.date().isoformat()}."
            )
        snapshot_end = common._normalize_date(args.snapshot_end)
        identity = common._market_identity(configuration)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
        candidates = _candidate_symbols(collection, identity, baseline_assets, args.candidate_symbols)
        if not candidates:
            raise RuntimeError("No external cached candidate symbols are available for this benchmark.")

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT
            / "research_output"
            / f"exact_marginal_capital_search_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = _manifest_contract(
            strategy=strategy,
            strategy_id=strategy_id,
            strategy_sequence=strategy_sequence,
            history_start=history_start,
            snapshot_end=snapshot_end,
            baseline_assets=baseline_assets,
            candidates=candidates,
            workers=workers,
        )
        manifest_path = output_dir / "exact_search_manifest.json"
        if manifest_path.exists() and not args.fresh_run:
            _verify_resume_contract(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
        _write_json(manifest_path, manifest)

        _log(
            f"Exact Marginal Capital Search v1: Strategy #{strategy_sequence}, "
            f"baseline={len(baseline_assets)}, candidates={len(candidates)}, workers={workers}."
        )
        _log("MongoDB read-only. No Alpaca request. Preselector never filters the exact judge.")

        locked_config = BacktestRequest.model_validate(configuration).model_copy(
            update={"end_date": snapshot_end.date().isoformat()}
        )
        expected_sessions = common._expected_sessions(history_start, snapshot_end)

        _log("Phase 1/4 - Loading and validating immutable baseline history...")
        baseline_raw = common._load_frames_once(
            collection,
            baseline_assets,
            identity,
            history_start,
            snapshot_end,
        )
        history_rows = common._validate_complete_history(
            baseline_raw,
            baseline_assets,
            expected_sessions,
        )
        _write_frame(output_dir / "baseline_history_integrity.csv", pd.DataFrame(history_rows))
        baseline_frames = {
            symbol: validate_and_clean_bars(frame, locked_config)
            for symbol, frame in baseline_raw.items()
        }
        required_sessions = discovery._baseline_required_sessions(
            baseline_frames,
            locked_config,
            snapshot_end,
        )

        _log("Phase 2/4 - Training the existing preselector for queue-quality diagnostics...")
        seed = discovery._research_sample_seed(
            str(strategy.get("configuration_hash") or ""),
            snapshot_end.date().isoformat(),
        )
        ranker_bundle = train_ranker(baseline_frames, random_state=seed)
        baseline_returns = discovery._baseline_recent_returns(baseline_frames)

        _log("Phase 3/4 - Running the exact baseline Strategy once...")
        baseline_request = discovery._marginal_execution_request(
            db,
            locked_config,
            {"id": strategy_id},
            locked_config,
            snapshot_end.date().isoformat(),
            assets=baseline_assets,
            reference_assets=baseline_assets,
            candidate_assets=[],
            analysis_start_date=history_start.date().isoformat(),
            analysis_end_date=snapshot_end.date().isoformat(),
        )
        baseline_started = time.perf_counter()
        baseline_metrics, baseline_sessions = discovery._run_rotation_replay(
            baseline_frames,
            baseline_request,
        )
        baseline_elapsed = float(time.perf_counter() - baseline_started)
        _write_json(
            output_dir / "exact_baseline.json",
            {
                "metrics": baseline_metrics,
                "elapsed_seconds": baseline_elapsed,
                "decision_session_count": int(len(baseline_sessions)),
            },
        )
        _log(
            f"Exact baseline completed in {baseline_elapsed:.1f}s; "
            f"ending capital={baseline_metrics.get('ending_capital')}."
        )

        result_path = output_dir / "exact_candidate_evaluations.csv"
        existing_rows: list[dict[str, Any]] = []
        completed_symbols: set[str] = set()
        if result_path.exists() and not args.fresh_run:
            existing = pd.read_csv(result_path)
            existing_rows = existing.to_dict(orient="records")
            completed_symbols = {
                str(row.get("symbol") or "").strip().upper()
                for row in existing_rows
                if str(row.get("evaluation_status") or "").strip().lower()
                in {"completed", "history_rejected", "context_rejected", "failed"}
            }
            if completed_symbols:
                _log(f"Resume: {len(completed_symbols)} candidate outcomes already persisted.")

        pending = [symbol for symbol in candidates if symbol not in completed_symbols]
        _log(
            f"Phase 4/4 - Exact candidate retrain + Strategy replay: pending={len(pending)}. "
            "Every complete-history candidate will be evaluated."
        )
        rows = list(existing_rows)
        if pending:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {
                    executor.submit(
                        _candidate_evaluation,
                        db=db,
                        collection=collection,
                        symbol=symbol,
                        identity=identity,
                        history_start=history_start,
                        snapshot_end=snapshot_end,
                        config=locked_config,
                        strategy_id=strategy_id,
                        baseline_assets=baseline_assets,
                        baseline_frames=baseline_frames,
                        baseline_metrics=baseline_metrics,
                        baseline_sessions=baseline_sessions,
                        required_sessions=required_sessions,
                        ranker_bundle=ranker_bundle,
                        baseline_returns=baseline_returns,
                    ): symbol
                    for symbol in pending
                }
                processed = len(completed_symbols)
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "symbol": symbol,
                            "evaluation_status": "failed",
                            "error": str(exc)[:700],
                            "economic_outcome": "failed",
                        }
                    rows = [
                        item
                        for item in rows
                        if str(item.get("symbol") or "").strip().upper() != symbol
                    ]
                    rows.append(row)
                    processed += 1

                    ranked_rows = annotate_preselector_ranks(rows)
                    ranked_rows.sort(
                        key=lambda item: (
                            int(item.get("preselector_rank") or 10**9),
                            str(item.get("symbol") or ""),
                        )
                    )
                    _write_frame(result_path, pd.DataFrame(ranked_rows))
                    delta = row.get("ending_capital_delta_rate")
                    delta_text = "n/a" if delta is None or pd.isna(delta) else f"{float(delta):+.4%}"
                    _log(
                        f"{processed}/{len(candidates)} {symbol}: "
                        f"status={row.get('evaluation_status')}, ΔCapital={delta_text}, "
                        f"exact={float(row.get('exact_replay_seconds') or 0.0):.1f}s."
                    )

        rows = annotate_preselector_ranks(rows)
        for row in rows:
            row["economic_outcome"] = economic_outcome(row)
        rows.sort(
            key=lambda item: (
                int(item.get("preselector_rank") or 10**9),
                str(item.get("symbol") or ""),
            )
        )
        result_frame = pd.DataFrame(rows)
        _write_frame(result_path, result_frame)

        recall_rows = build_preselector_recall(rows)
        _write_frame(output_dir / "preselector_recall.csv", pd.DataFrame(recall_rows))
        summary = summarize_exact_search(rows)
        summary.update(
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "api_version": API_VERSION,
                "strategy_id": strategy_id,
                "strategy_sequence": strategy_sequence,
                "baseline": baseline_metrics,
                "baseline_replay_seconds": baseline_elapsed,
                "preselector_diagnostics": ranker_bundle.diagnostics,
                "preselector_recall": recall_rows,
                "conclusion_rule": (
                    "Use this run to quantify exact-judge cost and preselector enrichment. "
                    "Do not claim generalization until an outer walk-forward is run."
                ),
            }
        )
        _write_json(output_dir / "exact_search_summary.json", summary)
        _log(f"Completed. Outputs: {output_dir}")
        _log("Main files: exact_candidate_evaluations.csv, preselector_recall.csv, exact_search_summary.json.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
