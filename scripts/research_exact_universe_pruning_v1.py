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
import research_sequential_exact_marginal_search as sequential  # noqa: E402
import research_sequential_exact_marginal_search_v110 as accelerator  # noqa: E402
from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "exact-universe-pruning-v1.0.0"
EXPERIMENT_NAME = "exact_universe_pruning"
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
    if not relative.parts or not relative.parts[0].startswith("exact_universe_pruning_"):
        raise RuntimeError(f"Refusing to delete unexpected research output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a greedy exact-selected universe by testing every single-asset removal with the same exact "
            "Strategy capital judge. If removing an asset increases ending capital, remove the best one and "
            "repeat until no single removal improves capital."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--seed-assets", nargs="+", required=True)
    parser.add_argument("--max-rounds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def choose_best_positive_removal(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("evaluation_status") or "").lower() != "completed":
            continue
        try:
            delta = float(row.get("removal_capital_delta_rate"))
        except (TypeError, ValueError):
            continue
        if pd.notna(delta) and delta > 0.0:
            eligible.append(row)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["removal_capital_delta_rate"]),
            str(row.get("removed_symbol") or ""),
        ),
    )


def _evaluate_removal(
    *,
    db: Any,
    strategy_id: str,
    config: BacktestRequest,
    current_assets: list[str],
    current_frames: dict[str, pd.DataFrame],
    baseline_metrics: dict[str, Any],
    baseline_sessions: pd.DatetimeIndex,
    removed_symbol: str,
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    round_number: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    reduced_assets = [symbol for symbol in current_assets if symbol != removed_symbol]
    row: dict[str, Any] = {
        "round": round_number,
        "removed_symbol": removed_symbol,
        "baseline_asset_count": len(current_assets),
        "reduced_asset_count": len(reduced_assets),
        "evaluation_status": "running",
    }
    if len(reduced_assets) < 2:
        row.update(
            {
                "evaluation_status": "rejected",
                "rejection_reason": "rotation_requires_at_least_two_assets",
                "total_seconds": float(time.perf_counter() - started),
            }
        )
        return row

    try:
        reduced_frames = {symbol: current_frames[symbol] for symbol in reduced_assets}
        request = sequential._exact_request(
            db,
            config,
            strategy_id,
            reduced_assets,
            reduced_assets,
            [],
            history_start,
            snapshot_end,
        )
        metrics, sessions = discovery._run_rotation_replay(reduced_frames, request)
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
            reduced_capital = discovery._finite_number(metrics.get("ending_capital"))
            row.update(
                {
                    "evaluation_status": "completed",
                    "baseline_ending_capital": baseline_capital,
                    "reduced_ending_capital": reduced_capital,
                    "removal_capital_delta": discovery._delta(reduced_capital, baseline_capital),
                    "removal_capital_delta_rate": discovery._capital_delta_rate(reduced_capital, baseline_capital),
                    "reduced_cagr": metrics.get("cagr"),
                    "reduced_sharpe": metrics.get("sharpe"),
                    "reduced_maximum_drawdown": metrics.get("maximum_drawdown"),
                    "reduced_worst_fold_return": metrics.get("worst_fold_return"),
                    "reduced_switches": metrics.get("switches"),
                    "reduced_cash_days": metrics.get("cash_days"),
                }
            )
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700]})
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
    seed_assets = sequential.normalize_symbols(args.seed_assets)
    if len(seed_assets) < 2:
        raise RuntimeError("Exact Universe Pruning requires at least two seed assets.")

    accelerator.install_v110(parity_mode=False)

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
        missing = sorted(set(seed_assets).difference(strategy_assets))
        if missing:
            raise RuntimeError(f"Seed assets are not present in Strategy #{strategy_sequence}: {missing}")

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
        expected_sessions = common._expected_sessions(history_start, snapshot_end)

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT / "research_output" / f"exact_universe_pruning_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        current_frames: dict[str, pd.DataFrame] = {}
        for symbol in seed_assets:
            frame, _coverage = sequential._candidate_frame(
                db=db,
                collection=collection,
                symbol=symbol,
                identity=identity,
                history_start=history_start,
                snapshot_end=snapshot_end,
                config=config,
                required_sessions=expected_sessions,
            )
            current_frames[symbol] = frame

        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "experiment": EXPERIMENT_NAME,
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "initial_assets": seed_assets,
            "initial_asset_count": len(seed_assets),
            "max_rounds": max_rounds,
            "workers": workers,
            "selection_rule": "remove the single asset with maximum exact positive ending-capital delta; repeat until no single removal improves capital",
            "purpose": "audit whether greedy forward additions remain necessary after later universe changes; not an OOS generalization claim",
        }
        _write_json(output_dir / "exact_pruning_manifest.json", manifest)

        all_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        current_assets = list(seed_assets)

        for round_number in range(1, max_rounds + 1):
            if len(current_assets) <= 2:
                break
            round_config = config.model_copy(update={"assets": list(current_assets)})
            _log(f"Pruning round {round_number}: exact baseline with {len(current_assets)} assets.")
            baseline_metrics, baseline_sessions, baseline_seconds = sequential._run_baseline(
                db=db,
                strategy_id=strategy_id,
                config=round_config,
                assets=current_assets,
                frames=current_frames,
                history_start=history_start,
                snapshot_end=snapshot_end,
            )
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            _log(f"Pruning round {round_number}: baseline={baseline_capital}, elapsed={baseline_seconds:.1f}s.")

            round_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(current_assets))) as executor:
                futures = {
                    executor.submit(
                        _evaluate_removal,
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
                    ): symbol
                    for symbol in current_assets
                }
                for future in as_completed(futures):
                    row = future.result()
                    round_rows.append(row)
                    all_rows.append(row)
                    delta = row.get("removal_capital_delta_rate")
                    delta_text = "n/a" if delta is None or pd.isna(delta) else f"{float(delta):+.4%}"
                    _log(
                        f"Pruning round {round_number} remove {row.get('removed_symbol')}: "
                        f"{row.get('evaluation_status')} ΔCapital={delta_text}"
                    )
                    _write_frame(output_dir / "pruning_round_evaluations.csv", all_rows)

            chosen = choose_best_positive_removal(round_rows)
            if chosen is None:
                path_rows.append(
                    {
                        "round": round_number,
                        "baseline_asset_count": len(current_assets),
                        "baseline_ending_capital": baseline_capital,
                        "removed_symbol": None,
                        "reduced_ending_capital": None,
                        "removal_delta_rate": None,
                        "stop_reason": "no_positive_exact_single_asset_removal",
                    }
                )
                _write_frame(output_dir / "pruning_path.csv", path_rows)
                _log("No single removal has exact ΔCapital > 0. Pruning stops.")
                break

            removed_symbol = str(chosen["removed_symbol"])
            path_rows.append(
                {
                    "round": round_number,
                    "baseline_asset_count": len(current_assets),
                    "baseline_ending_capital": baseline_capital,
                    "removed_symbol": removed_symbol,
                    "reduced_ending_capital": chosen.get("reduced_ending_capital"),
                    "removal_delta_rate": chosen.get("removal_capital_delta_rate"),
                    "stop_reason": None,
                }
            )
            current_assets.remove(removed_symbol)
            current_frames.pop(removed_symbol, None)
            _write_frame(output_dir / "pruning_path.csv", path_rows)
            _log(f"Removed {removed_symbol}; next round re-tests every surviving asset against the changed universe.")

        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "strategy_sequence": strategy_sequence,
            "initial_asset_count": len(seed_assets),
            "final_asset_count": len(current_assets),
            "removed_assets": [row["removed_symbol"] for row in path_rows if row.get("removed_symbol")],
            "final_assets": current_assets,
            "rounds_executed": len(path_rows),
            "path": path_rows,
            "conclusion_rule": (
                "If a previously selected asset can be removed with exact positive capital delta, greedy forward selection "
                "was path-dependent and that asset became redundant or harmful in the later universe. Stop only when "
                "no single removal improves capital."
            ),
        }
        _write_frame(output_dir / "pruning_round_evaluations.csv", all_rows)
        _write_frame(output_dir / "pruning_path.csv", path_rows)
        _write_json(output_dir / "exact_pruning_summary.json", summary)
        _log(f"Exact Universe Pruning finished with {len(current_assets)} assets.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
