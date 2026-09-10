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
import research_exact_marginal_capital_search_v102 as exact_v102  # noqa: E402
from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.engine.market_data import validate_and_clean_bars  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "sequential-exact-marginal-search-v1.0.0"
EXPERIMENT_NAME = "sequential_exact_marginal_search"
DEFAULT_WORKERS = 1
ORIGINAL_25 = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY",
    "AVGO", "NFLX", "CRM", "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V", "HD",
    "ADC", "ADEA", "ADI", "ADM",
]


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def _sha256(values: list[str]) -> str:
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def _safe_fresh(path: Path) -> None:
    root = (PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing --fresh-run outside {root}: {resolved}") from exc
    if not relative.parts or not relative.parts[0].startswith("sequential_exact_marginal_search_"):
        raise RuntimeError(f"Refusing to delete unexpected research output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the mechanism that made brute-force Asset Discovery useful: evaluate exact "
            "marginal capital against the CURRENT universe, add only the best positive candidate, then "
            "re-evaluate the remaining candidates against the changed universe."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--seed-assets", nargs="*", default=None)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def normalize_symbols(values: list[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value).strip().upper() for value in list(values or []) if str(value).strip()))


def choose_best_positive(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("evaluation_status") or "").lower() != "completed":
            continue
        try:
            delta = float(row.get("ending_capital_delta_rate"))
        except (TypeError, ValueError):
            continue
        if pd.notna(delta) and delta > 0.0:
            eligible.append(row)
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["ending_capital_delta_rate"]), str(row.get("symbol") or "")))


def _load_local_frame(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: BacktestRequest,
    required_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = common._load_frames_once(collection, [symbol], identity, history_start, snapshot_end)
    frame = validate_and_clean_bars(frames[symbol], config)
    coverage = discovery._history_coverage_against_baseline(symbol, frame, config, required_sessions)
    frame.attrs["sequential_history_source"] = "local_mongodb_cache"
    return frame, coverage


def _candidate_frame(
    *,
    db: Any,
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: BacktestRequest,
    required_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        return _load_local_frame(
            collection, symbol, identity, history_start, snapshot_end, config, required_sessions
        )
    except (RuntimeError, ValueError):
        frame, coverage = exact_v102._transient_full_history(
            db, symbol, history_start, snapshot_end, config, required_sessions
        )
        frame.attrs["sequential_history_source"] = "alpaca_transient_full_history"
        return frame, coverage


def _exact_request(
    db: Any,
    config: BacktestRequest,
    strategy_id: str,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
):
    return discovery._marginal_execution_request(
        db,
        config,
        {"id": strategy_id},
        config,
        snapshot_end.date().isoformat(),
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        analysis_start_date=history_start.date().isoformat(),
        analysis_end_date=snapshot_end.date().isoformat(),
    )


def _run_baseline(
    *,
    db: Any,
    strategy_id: str,
    config: BacktestRequest,
    assets: list[str],
    frames: dict[str, pd.DataFrame],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DatetimeIndex, float]:
    request = _exact_request(
        db, config, strategy_id, assets, assets, [], history_start, snapshot_end
    )
    started = time.perf_counter()
    metrics, sessions = discovery._run_rotation_replay(frames, request)
    return metrics, sessions, float(time.perf_counter() - started)


def _evaluate_candidate(
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
    candidate_cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    identity_integrity: dict[str, dict[str, Any]],
    round_number: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "round": round_number,
        "symbol": symbol,
        "baseline_asset_count": len(baseline_assets),
        "baseline_assets_sha256": _sha256(baseline_assets),
        "evaluation_status": "running",
    }
    integrity = dict(identity_integrity.get(symbol) or {"status": "unknown", "checked": False})
    row.update(
        {
            "identity_integrity_status": integrity.get("status"),
            "identity_integrity_checked": bool(integrity.get("checked")),
            "identity_integrity_reason": integrity.get("reason"),
            "identity_integrity_break_count": int(integrity.get("comparability_break_count") or 0),
        }
    )
    if str(integrity.get("status") or "").lower() == "rejected":
        row.update(
            {
                "evaluation_status": "context_rejected",
                "rejection_reason": "economic_identity_discontinuity",
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        return row

    try:
        if symbol not in candidate_cache:
            candidate_cache[symbol] = _candidate_frame(
                db=db,
                collection=collection,
                symbol=symbol,
                identity=identity,
                history_start=history_start,
                snapshot_end=snapshot_end,
                config=config,
                required_sessions=required_sessions,
            )
        frame, coverage = candidate_cache[symbol]
        row.update(coverage)
        row["history_source"] = frame.attrs.get("sequential_history_source")
    except RuntimeError as exc:
        reason = str(exc).strip().lower()
        row.update(
            {
                "evaluation_status": "history_rejected" if reason in {
                    "insufficient_history", "discontinuous_history", "ticker_identity_discontinuity"
                } else "failed",
                "rejection_reason": reason if reason in {
                    "insufficient_history", "discontinuous_history", "ticker_identity_discontinuity"
                } else None,
                "error": None if reason in {
                    "insufficient_history", "discontinuous_history", "ticker_identity_discontinuity"
                } else str(exc)[:700],
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        return row
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700], "total_seconds": float(time.perf_counter() - started)})
        return row

    exact_started = time.perf_counter()
    try:
        expanded_assets = [*baseline_assets, symbol]
        expanded_frames = dict(baseline_frames)
        expanded_frames[symbol] = frame
        request = _exact_request(
            db, config, strategy_id, expanded_assets, baseline_assets, [symbol], history_start, snapshot_end
        )
        metrics, sessions = discovery._run_rotation_replay(expanded_frames, request)
        context = discovery._research_context_compatibility(baseline_sessions, sessions)
        row.update(context)
        if not bool(context.get("research_context_compatible")):
            row.update({"evaluation_status": "context_rejected", "rejection_reason": "research_context_incomplete"})
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
                }
            )
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700]})
    row["exact_replay_seconds"] = float(time.perf_counter() - exact_started)
    row["total_seconds"] = float(time.perf_counter() - started)
    return row


def main() -> int:
    args = _parser().parse_args()
    load_project_environment(args.env_file)
    if args.mongo_uri:
        os.environ["MONGO_URI"] = str(args.mongo_uri)
        os.environ["MONGO_URL"] = str(args.mongo_uri)
    if args.database:
        os.environ["MONGO_DATABASE"] = str(args.database)

    mongo_uri = str(args.mongo_uri or os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "mongodb://localhost:27017").strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))
    workers = max(1, int(args.workers or 1))
    max_rounds = max(1, int(args.max_rounds or 1))

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3_000, connectTimeoutMS=3_000, maxPoolSize=max(8, workers + 4), retryWrites=False)
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored_configuration = common._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        strategy_assets = normalize_symbols(list(stored_configuration.get("assets") or []))

        seed_assets = normalize_symbols(args.seed_assets) if args.seed_assets else list(ORIGINAL_25)
        if not seed_assets:
            raise RuntimeError("At least one seed asset is required.")
        missing_seed = sorted(set(seed_assets).difference(strategy_assets))
        if missing_seed:
            raise RuntimeError(f"Seed assets are not present in Strategy #{strategy_sequence}: {missing_seed}")

        candidates = normalize_symbols(args.candidate_symbols)
        if not candidates:
            candidates = [symbol for symbol in strategy_assets if symbol not in set(seed_assets)]
        candidates = [symbol for symbol in candidates if symbol not in set(seed_assets)]
        if not candidates:
            raise RuntimeError("No candidate symbols remain after excluding the seed universe.")

        history_start = common._normalize_date(args.history_start)
        configured_start = common._normalize_date(stored_configuration.get("start_date"))
        if history_start != configured_start:
            raise RuntimeError(
                f"--history-start must match Strategy start date: strategy={configured_start.date()}, requested={history_start.date()}"
            )
        snapshot_end = common._normalize_date(args.snapshot_end)
        config = BacktestRequest.model_validate(stored_configuration).model_copy(
            update={"assets": list(seed_assets), "end_date": snapshot_end.date().isoformat()}
        )
        identity = common._market_identity(stored_configuration)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT / "research_output" / f"sequential_exact_marginal_search_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "experiment": EXPERIMENT_NAME,
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "strategy_revision": int(strategy.get("revision") or 0),
            "strategy_configuration_hash": str(strategy.get("configuration_hash") or ""),
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "seed_assets": seed_assets,
            "seed_asset_count": len(seed_assets),
            "candidate_symbols": candidates,
            "candidate_count": len(candidates),
            "max_rounds": max_rounds,
            "workers": workers,
            "selection_rule": "choose maximum exact ending_capital_delta_rate strictly above zero; then re-evaluate remaining candidates",
            "preselector_used": False,
            "candidate_history_persistence": "none",
            "purpose": "test state-dependent/path-dependent marginal capital contribution; not a generalization claim",
        }
        _write_json(output_dir / "sequential_exact_manifest.json", manifest)

        expected_sessions = common._expected_sessions(history_start, snapshot_end)
        current_assets = list(seed_assets)
        baseline_raw = common._load_frames_once(collection, current_assets, identity, history_start, snapshot_end)
        common._validate_complete_history(baseline_raw, current_assets, expected_sessions)
        current_frames = {symbol: validate_and_clean_bars(frame, config) for symbol, frame in baseline_raw.items()}
        required_sessions = discovery._baseline_required_sessions(current_frames, config, snapshot_end)

        _log(f"Sequential Exact Marginal Search v1: seed={len(current_assets)}, candidates={len(candidates)}, max_rounds={max_rounds}.")
        _log("No ML preselector. Every round uses the exact Strategy capital judge and zero as the only economic boundary.")

        metadata = discovery._discover_asset_metadata(db)
        identity_integrity = discovery._identity_integrity_for_symbols(
            db,
            candidates,
            start_date=history_start.date().isoformat(),
            end_date=snapshot_end.date().isoformat(),
            asset_metadata=metadata,
        )

        all_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        remaining = list(candidates)
        candidate_cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

        for round_number in range(1, max_rounds + 1):
            if not remaining:
                break
            round_config = config.model_copy(update={"assets": list(current_assets)})
            _log(f"Round {round_number}: exact baseline with {len(current_assets)} assets; remaining={len(remaining)}.")
            baseline_metrics, baseline_sessions, baseline_seconds = _run_baseline(
                db=db,
                strategy_id=strategy_id,
                config=round_config,
                assets=current_assets,
                frames=current_frames,
                history_start=history_start,
                snapshot_end=snapshot_end,
            )
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            _log(f"Round {round_number}: baseline={baseline_capital}, elapsed={baseline_seconds:.1f}s.")

            round_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(remaining))) as executor:
                futures = {
                    executor.submit(
                        _evaluate_candidate,
                        db=db,
                        collection=collection,
                        symbol=symbol,
                        identity=identity,
                        history_start=history_start,
                        snapshot_end=snapshot_end,
                        config=round_config,
                        strategy_id=strategy_id,
                        baseline_assets=current_assets,
                        baseline_frames=current_frames,
                        baseline_metrics=baseline_metrics,
                        baseline_sessions=baseline_sessions,
                        required_sessions=required_sessions,
                        candidate_cache=candidate_cache,
                        identity_integrity=identity_integrity,
                        round_number=round_number,
                    ): symbol
                    for symbol in remaining
                }
                for future in as_completed(futures):
                    row = future.result()
                    round_rows.append(row)
                    all_rows.append(row)
                    delta = row.get("ending_capital_delta_rate")
                    delta_text = "n/a" if delta is None or pd.isna(delta) else f"{float(delta):+.4%}"
                    _log(f"Round {round_number} {row.get('symbol')}: {row.get('evaluation_status')} ΔCapital={delta_text}")
                    _write_frame(output_dir / "sequential_round_evaluations.csv", all_rows)

            chosen = choose_best_positive(round_rows)
            if chosen is None:
                path_rows.append(
                    {
                        "round": round_number,
                        "baseline_asset_count": len(current_assets),
                        "baseline_ending_capital": baseline_capital,
                        "chosen_symbol": None,
                        "chosen_delta_rate": None,
                        "stop_reason": "no_positive_exact_marginal_candidate",
                    }
                )
                _write_frame(output_dir / "sequential_path.csv", path_rows)
                _log(f"Round {round_number}: no candidate has exact ΔCapital > 0. Search stops.")
                break

            chosen_symbol = str(chosen["symbol"])
            chosen_frame, _ = candidate_cache[chosen_symbol]
            path_rows.append(
                {
                    "round": round_number,
                    "baseline_asset_count": len(current_assets),
                    "baseline_ending_capital": baseline_capital,
                    "chosen_symbol": chosen_symbol,
                    "chosen_candidate_ending_capital": chosen.get("candidate_ending_capital"),
                    "chosen_delta_rate": chosen.get("ending_capital_delta_rate"),
                    "stop_reason": None,
                }
            )
            current_assets.append(chosen_symbol)
            current_frames[chosen_symbol] = chosen_frame
            remaining.remove(chosen_symbol)
            _write_frame(output_dir / "sequential_path.csv", path_rows)
            _log(f"Round {round_number}: added {chosen_symbol}; the next round will re-evaluate every remaining candidate against the changed universe.")

        _write_frame(output_dir / "sequential_round_evaluations.csv", all_rows)
        _write_frame(output_dir / "sequential_path.csv", path_rows)
        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "strategy_sequence": strategy_sequence,
            "initial_seed_asset_count": len(seed_assets),
            "initial_candidate_count": len(candidates),
            "rounds_executed": len({int(row["round"]) for row in all_rows}) if all_rows else 0,
            "selected_assets": [row["chosen_symbol"] for row in path_rows if row.get("chosen_symbol")],
            "final_assets": current_assets,
            "final_asset_count": len(current_assets),
            "remaining_candidates": remaining,
            "path": path_rows,
            "conclusion_rule": (
                "A sign/rank change for the same candidate across nested universes is direct evidence that marginal contribution is state-dependent. "
                "This experiment reconstructs greedy exact selection mechanics; it does not establish out-of-sample generalization."
            ),
        }
        _write_json(output_dir / "sequential_exact_summary.json", summary)
        _log(f"Completed. Outputs: {output_dir}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
