from __future__ import annotations

import argparse
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
import research_exact_universe_pruning_v1 as pruning  # noqa: E402
import research_sequential_exact_marginal_search as sequential  # noqa: E402
import research_sequential_exact_marginal_search_v110 as accelerator  # noqa: E402
from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.engine.market_data import validate_and_clean_bars  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "exact-bidirectional-universe-search-v1.0.0"
EXPERIMENT_NAME = "exact_bidirectional_universe_search"
DEFAULT_WORKERS = 4


def _log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
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


def _safe_fresh(path: Path) -> None:
    root = (PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing --fresh-run outside {root}: {resolved}") from exc
    if not relative.parts or not relative.parts[0].startswith("exact_bidirectional_universe_search_"):
        raise RuntimeError(f"Refusing to delete unexpected research output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search the current universe in both directions with the exact Strategy capital judge. "
            "Every round evaluates every possible single addition from the fixed pool and every possible "
            "single removal from the current universe, then applies only the strongest exact positive move. "
            "A round with any unresolved evaluation is never allowed to choose a winner."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--seed-assets", nargs="+", required=True)
    parser.add_argument("--pool-symbols", nargs="*", default=None)
    parser.add_argument("--max-rounds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def normalize_move_row(row: dict[str, Any], move_type: str, symbol: str) -> dict[str, Any]:
    normalized = dict(row)
    if move_type == "add":
        delta_rate = normalized.get("ending_capital_delta_rate")
        resulting_capital = normalized.get("candidate_ending_capital")
    elif move_type == "remove":
        delta_rate = normalized.get("removal_capital_delta_rate")
        resulting_capital = normalized.get("reduced_ending_capital")
    else:
        raise ValueError(f"Unsupported move_type: {move_type}")
    normalized.update(
        {
            "move_type": move_type,
            "move_symbol": str(symbol).strip().upper(),
            "move_key": f"{move_type}:{str(symbol).strip().upper()}",
            "move_delta_rate": delta_rate,
            "resulting_ending_capital": resulting_capital,
        }
    )
    return normalized


def choose_best_positive_move(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("evaluation_status") or "").lower() != "completed":
            continue
        try:
            delta = float(row.get("move_delta_rate"))
        except (TypeError, ValueError):
            continue
        if pd.notna(delta) and delta > 0.0:
            eligible.append(row)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["move_delta_rate"]),
            str(row.get("move_key") or ""),
        ),
    )


def round_completeness_issues(
    rows: list[dict[str, Any]],
    expected_move_keys: set[str],
) -> list[str]:
    issues: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = str(row.get("move_key") or "")
        if key in seen:
            duplicates.append(key)
        seen[key] = row

    missing = sorted(expected_move_keys.difference(seen))
    unexpected = sorted(set(seen).difference(expected_move_keys))
    non_completed = sorted(
        f"{key}={seen[key].get('evaluation_status')}"
        for key in expected_move_keys.intersection(seen)
        if str(seen[key].get("evaluation_status") or "").lower() != "completed"
    )

    if missing:
        issues.append("missing:" + ",".join(missing))
    if unexpected:
        issues.append("unexpected:" + ",".join(unexpected))
    if duplicates:
        issues.append("duplicates:" + ",".join(sorted(set(duplicates))))
    if non_completed:
        issues.append("non_completed:" + ",".join(non_completed))
    return issues


def _failed_future_row(
    *,
    round_number: int,
    move_type: str,
    symbol: str,
    exc: Exception,
) -> dict[str, Any]:
    return normalize_move_row(
        {
            "round": round_number,
            "evaluation_status": "failed",
            "error": str(exc)[:700],
        },
        move_type,
        symbol,
    )


def _summary(
    *,
    strategy_sequence: int,
    initial_assets: list[str],
    pool_symbols: list[str],
    current_assets: list[str],
    path_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    last_baseline_metrics: dict[str, Any] | None,
    abort_reason: str | None,
) -> dict[str, Any]:
    completed_moves = [
        {
            "round": row.get("round"),
            "move_type": row.get("chosen_move_type"),
            "symbol": row.get("chosen_symbol"),
            "delta_rate": row.get("chosen_delta_rate"),
            "resulting_ending_capital": row.get("chosen_resulting_ending_capital"),
        }
        for row in path_rows
        if row.get("chosen_symbol")
    ]
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "api_version": API_VERSION,
        "strategy_sequence": strategy_sequence,
        "initial_assets": initial_assets,
        "initial_asset_count": len(initial_assets),
        "pool_symbols": pool_symbols,
        "pool_asset_count": len(pool_symbols),
        "final_assets": current_assets,
        "final_asset_count": len(current_assets),
        "rounds_evaluated": len({int(row["round"]) for row in all_rows}) if all_rows else 0,
        "accepted_moves": completed_moves,
        "path": path_rows,
        "last_baseline_metrics": last_baseline_metrics,
        "abort_reason": abort_reason,
        "conclusion_rule": (
            "A local 1-opt stop is valid only after a complete round in which every single addition from the fixed "
            "pool and every single removal from the current universe completed under the same exact capital judge "
            "and none produced move_delta_rate > 0. This remains historical/in-sample research."
        ),
    }


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
        raise RuntimeError("MONGO_DATABASE is required.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    workers = max(1, int(args.workers or 1))
    max_rounds = max(1, int(args.max_rounds or 1))
    seed_assets = sequential.normalize_symbols(args.seed_assets)
    if len(seed_assets) < 2:
        raise RuntimeError("Exact Bidirectional Universe Search requires at least two seed assets.")

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
        stored_configuration = common._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        strategy_assets = sequential.normalize_symbols(list(stored_configuration.get("assets") or []))

        pool_symbols = sequential.normalize_symbols(args.pool_symbols)
        if not pool_symbols:
            pool_symbols = list(strategy_assets)
        missing_pool = sorted(set(pool_symbols).difference(strategy_assets))
        if missing_pool:
            raise RuntimeError(
                f"Pool symbols are not present in Strategy #{strategy_sequence}: {missing_pool}"
            )
        missing_seed = sorted(set(seed_assets).difference(pool_symbols))
        if missing_seed:
            raise RuntimeError(f"Seed assets are not present in the fixed pool: {missing_seed}")
        if len(pool_symbols) < 2:
            raise RuntimeError("The fixed pool must contain at least two symbols.")

        history_start = common._normalize_date(args.history_start)
        configured_start = common._normalize_date(stored_configuration.get("start_date"))
        if history_start != configured_start:
            raise RuntimeError(
                f"--history-start must match Strategy start date: "
                f"strategy={configured_start.date()}, requested={history_start.date()}"
            )
        snapshot_end = common._normalize_date(args.snapshot_end)
        config = BacktestRequest.model_validate(stored_configuration).model_copy(
            update={"assets": list(seed_assets), "end_date": snapshot_end.date().isoformat()}
        )
        identity = common._market_identity(stored_configuration)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT
            / "research_output"
            / f"exact_bidirectional_universe_search_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # The accelerator writes its diagnostics from worker threads. Bind it explicitly to the
        # already-created run directory and reset process-local diagnostics before installing it.
        accelerator._OUTPUT_DIR = output_dir
        accelerator._INDEX_ROWS.clear()
        accelerator._REPLAY_COUNTER = 0
        accelerator.install_v110(parity_mode=False)

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
            "initial_assets": seed_assets,
            "initial_asset_count": len(seed_assets),
            "pool_symbols": pool_symbols,
            "pool_asset_count": len(pool_symbols),
            "max_rounds": max_rounds,
            "workers": workers,
            "selection_rule": (
                "on a complete round, choose the single add/remove move with maximum exact ending-capital "
                "delta rate strictly above zero; then re-evaluate every possible move"
            ),
            "complete_round_guard": (
                "every expected add/remove move must finish with evaluation_status=completed; otherwise no "
                "winner is selected and the run aborts after writing partial outputs"
            ),
            "objective": "exact ending capital only",
            "preselector_used": False,
            "purpose": (
                "test bidirectional state/path-dependent local universe optimization inside a fixed known pool; "
                "not an OOS generalization claim"
            ),
        }
        _write_json(output_dir / "bidirectional_manifest.json", manifest)

        expected_sessions = common._expected_sessions(history_start, snapshot_end)
        current_assets = list(seed_assets)
        baseline_raw = common._load_frames_once(
            collection,
            current_assets,
            identity,
            history_start,
            snapshot_end,
        )
        common._validate_complete_history(baseline_raw, current_assets, expected_sessions)
        current_frames = {
            symbol: validate_and_clean_bars(frame, config)
            for symbol, frame in baseline_raw.items()
        }
        required_sessions = discovery._baseline_required_sessions(
            current_frames,
            config,
            snapshot_end,
        )

        metadata = discovery._discover_asset_metadata(db)
        identity_integrity = discovery._identity_integrity_for_symbols(
            db,
            pool_symbols,
            start_date=history_start.date().isoformat(),
            end_date=snapshot_end.date().isoformat(),
            asset_metadata=metadata,
        )
        candidate_cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

        all_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        abort_reason: str | None = None
        last_baseline_metrics: dict[str, Any] | None = None

        _log(
            f"Exact Bidirectional Universe Search v1: seed={len(current_assets)}, "
            f"pool={len(pool_symbols)}, max_rounds={max_rounds}, workers={workers}."
        )
        _log(
            "Every round evaluates all single additions and removals. "
            "Any unresolved move makes the round incomplete and prevents selection."
        )

        for round_number in range(1, max_rounds + 1):
            round_config = config.model_copy(update={"assets": list(current_assets)})
            current_set = set(current_assets)
            outside = [symbol for symbol in pool_symbols if symbol not in current_set]
            expected_keys = {
                *(f"remove:{symbol}" for symbol in current_assets),
                *(f"add:{symbol}" for symbol in outside),
            }

            _log(
                f"Round {round_number}: exact baseline with {len(current_assets)} assets; "
                f"additions={len(outside)}, removals={len(current_assets)}, total_moves={len(expected_keys)}."
            )
            baseline_metrics, baseline_sessions, baseline_seconds = sequential._run_baseline(
                db=db,
                strategy_id=strategy_id,
                config=round_config,
                assets=current_assets,
                frames=current_frames,
                history_start=history_start,
                snapshot_end=snapshot_end,
            )
            last_baseline_metrics = dict(baseline_metrics)
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            _log(
                f"Round {round_number}: baseline={baseline_capital}, "
                f"elapsed={baseline_seconds:.1f}s."
            )

            round_rows: list[dict[str, Any]] = []
            max_workers = min(workers, max(1, len(expected_keys)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Any, tuple[str, str]] = {}

                for symbol in current_assets:
                    future = executor.submit(
                        pruning._evaluate_removal,
                        db=db,
                        strategy_id=strategy_id,
                        config=round_config,
                        current_assets=current_assets,
                        current_frames=current_frames,
                        baseline_metrics=baseline_metrics,
                        baseline_sessions=baseline_sessions,
                        removed_symbol=symbol,
                        history_start=history_start,
                        snapshot_end=snapshot_end,
                        round_number=round_number,
                    )
                    futures[future] = ("remove", symbol)

                for symbol in outside:
                    future = executor.submit(
                        sequential._evaluate_candidate,
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
                    )
                    futures[future] = ("add", symbol)

                for future in as_completed(futures):
                    move_type, symbol = futures[future]
                    try:
                        raw_row = future.result()
                        row = normalize_move_row(raw_row, move_type, symbol)
                    except Exception as exc:
                        row = _failed_future_row(
                            round_number=round_number,
                            move_type=move_type,
                            symbol=symbol,
                            exc=exc,
                        )
                    round_rows.append(row)
                    all_rows.append(row)
                    delta = row.get("move_delta_rate")
                    delta_text = (
                        "n/a"
                        if delta is None or pd.isna(delta)
                        else f"{float(delta):+.4%}"
                    )
                    _log(
                        f"Round {round_number} {move_type} {symbol}: "
                        f"{row.get('evaluation_status')} ΔCapital={delta_text}"
                    )
                    _write_frame(
                        output_dir / "bidirectional_round_evaluations.csv",
                        all_rows,
                    )

            issues = round_completeness_issues(round_rows, expected_keys)
            if issues:
                abort_reason = "BidirectionalRoundIncomplete: " + " | ".join(issues)
                path_rows.append(
                    {
                        "round": round_number,
                        "baseline_asset_count": len(current_assets),
                        "baseline_ending_capital": baseline_capital,
                        "chosen_move_type": None,
                        "chosen_symbol": None,
                        "chosen_delta_rate": None,
                        "chosen_resulting_ending_capital": None,
                        "stop_reason": "incomplete_round",
                        "detail": abort_reason,
                    }
                )
                _write_frame(output_dir / "bidirectional_path.csv", path_rows)
                _log(
                    "Round is incomplete. No move will be selected; partial outputs were saved."
                )
                break

            chosen = choose_best_positive_move(round_rows)
            if chosen is None:
                path_rows.append(
                    {
                        "round": round_number,
                        "baseline_asset_count": len(current_assets),
                        "baseline_ending_capital": baseline_capital,
                        "chosen_move_type": None,
                        "chosen_symbol": None,
                        "chosen_delta_rate": None,
                        "chosen_resulting_ending_capital": None,
                        "stop_reason": "no_positive_exact_single_asset_move",
                        "detail": None,
                    }
                )
                _write_frame(output_dir / "bidirectional_path.csv", path_rows)
                _log(
                    "Complete round: no single addition or removal has exact ΔCapital > 0. "
                    "Bidirectional search stops at a local 1-opt state."
                )
                break

            chosen_type = str(chosen["move_type"])
            chosen_symbol = str(chosen["move_symbol"])
            path_rows.append(
                {
                    "round": round_number,
                    "baseline_asset_count": len(current_assets),
                    "baseline_ending_capital": baseline_capital,
                    "chosen_move_type": chosen_type,
                    "chosen_symbol": chosen_symbol,
                    "chosen_delta_rate": chosen.get("move_delta_rate"),
                    "chosen_resulting_ending_capital": chosen.get(
                        "resulting_ending_capital"
                    ),
                    "stop_reason": None,
                    "detail": None,
                }
            )

            if chosen_type == "add":
                chosen_frame, _coverage = candidate_cache[chosen_symbol]
                current_assets.append(chosen_symbol)
                current_frames[chosen_symbol] = chosen_frame
                _log(
                    f"Round {round_number}: added {chosen_symbol}; next round recomputes all adds/removes."
                )
            elif chosen_type == "remove":
                current_assets.remove(chosen_symbol)
                current_frames.pop(chosen_symbol, None)
                _log(
                    f"Round {round_number}: removed {chosen_symbol}; next round recomputes all adds/removes."
                )
            else:
                raise RuntimeError(f"Unsupported chosen move type: {chosen_type}")

            _write_frame(output_dir / "bidirectional_path.csv", path_rows)

        _write_frame(output_dir / "bidirectional_round_evaluations.csv", all_rows)
        _write_frame(output_dir / "bidirectional_path.csv", path_rows)
        summary = _summary(
            strategy_sequence=strategy_sequence,
            initial_assets=seed_assets,
            pool_symbols=pool_symbols,
            current_assets=current_assets,
            path_rows=path_rows,
            all_rows=all_rows,
            last_baseline_metrics=last_baseline_metrics,
            abort_reason=abort_reason,
        )
        _write_json(output_dir / "bidirectional_summary.json", summary)

        if abort_reason:
            raise RuntimeError(abort_reason)

        _log(
            f"Completed. final_assets={len(current_assets)}; outputs={output_dir}"
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
