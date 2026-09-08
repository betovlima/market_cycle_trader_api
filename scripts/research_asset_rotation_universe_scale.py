from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_signature_leave_one_out as common  # noqa: E402
import research_asset_rotation_leadership as leadership  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.capital_rotation import run_rotation_models  # noqa: E402
from market_cycle_trader_api.engine.compound_rotation_backtest import (  # noqa: E402
    apply_slippage,
    calculate_reference_fees,
)
from market_cycle_trader_api.services.asset_universe_scale_candidates import (  # noqa: E402
    CandidateUniverseLoader,
)

SCRIPT_VERSION = "asset-rotation-universe-scale-v1.1.0"
DEFAULT_SERIES = (56, 82, 250, 500)
DEFAULT_WORKERS = 4


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".mct_write.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".mct_write.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same Strategy LightGBM rotation engine across nested US-equity "
            "universes. Candidate history is RAM-only unless it already existed in MongoDB."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--analysis-end", required=True)
    parser.add_argument("--series", type=int, nargs="+", default=list(DEFAULT_SERIES))
    parser.add_argument("--seed-universe-file", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-candidate-scans", type=int, default=2500)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _default_seed_file(strategy_sequence: int, snapshot_end: str) -> Path:
    return (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{strategy_sequence}_{snapshot_end}"
        / "asset_history_integrity.csv"
    ).resolve()


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _summary(size: int, result: Any, elapsed: float) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    return {
        "universe_size": int(size),
        "strategy_ending_capital": _metric(metrics, "strategy_ending_capital", "ending_capital"),
        "buy_hold_ending_capital": _metric(metrics, "buy_hold_ending_capital", "benchmark_ending_capital"),
        "strategy_total_return": _metric(metrics, "strategy_total_return", "strategy_return", "total_return"),
        "buy_hold_total_return": _metric(metrics, "buy_hold_total_return", "benchmark_return"),
        "strategy_cagr": _metric(metrics, "strategy_cagr", "cagr"),
        "buy_hold_cagr": _metric(metrics, "buy_hold_cagr", "benchmark_cagr"),
        "strategy_sharpe": _metric(metrics, "strategy_sharpe", "sharpe"),
        "buy_hold_sharpe": _metric(metrics, "buy_hold_sharpe", "benchmark_sharpe"),
        "maximum_drawdown": _metric(metrics, "maximum_drawdown", "strategy_max_drawdown", "max_drawdown"),
        "buy_hold_maximum_drawdown": _metric(metrics, "buy_hold_maximum_drawdown", "benchmark_max_drawdown"),
        "capital_rotations": _metric(metrics, "capital_rotations", "rotation_count"),
        "cash_days": _metric(metrics, "cash_days"),
        "market_exposure": _metric(metrics, "market_exposure", "market_exposure_ratio"),
        "elapsed_seconds": float(elapsed),
        "backend": str(getattr(result, "backend", "")),
    }


def _progress_logger(size: int):
    last_bucket = {"value": -1}

    def callback(progress: float, message: str, _completed: int) -> None:
        bucket = int(float(progress) // 5) * 5
        if bucket != last_bucket["value"]:
            last_bucket["value"] = bucket
            _log(f"Universe {size}: {float(progress):.1f}% - {message}")

    return callback


def _output_dir(args: argparse.Namespace, strategy: dict[str, Any]) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
    return (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_universe_scale_strategy_{sequence}_{args.analysis_start}_to_{args.analysis_end}"
    ).resolve()


def _execution_config(
    configuration: dict[str, Any],
    assets: list[str],
    baseline_assets: list[str],
    snapshot_end: pd.Timestamp,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    family: str,
    model_settings: dict[str, Any],
) -> Any:
    baseline_set = set(baseline_assets)
    return leadership._execution_config(
        configuration,
        assets,
        baseline_assets,
        snapshot_end,
        family,
        model_settings,
    ).model_copy(
        update={
            "analysis_start_date": analysis_start.date().isoformat(),
            "analysis_end_date": analysis_end.date().isoformat(),
            "calendar_anchor_assets": list(baseline_assets),
            "research_reference_assets": list(baseline_assets),
            "research_candidate_assets": [symbol for symbol in assets if symbol not in baseline_set],
        }
    )


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)

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

    history_start = common._normalize_date(args.history_start)
    snapshot_end = common._normalize_date(args.snapshot_end)
    analysis_start = common._normalize_date(args.analysis_start)
    analysis_end = common._normalize_date(args.analysis_end)
    if not (history_start < analysis_start <= analysis_end <= snapshot_end):
        raise RuntimeError(
            "Expected --history-start < --analysis-start <= --analysis-end <= --snapshot-end."
        )

    series = sorted(set(int(value) for value in args.series))
    workers = max(1, min(16, int(args.workers)))

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 4),
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]

    try:
        _log("STEP 1/6 - Loading Strategy, dates and immutable LightGBM snapshot.")
        strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
        configuration = common._configuration(strategy)
        configured_start = common._normalize_date(configuration.get("start_date") or history_start)
        if configured_start != history_start:
            raise RuntimeError(
                "--history-start must match the Strategy research start: "
                f"strategy={configured_start.date().isoformat()}, requested={history_start.date().isoformat()}."
            )

        baseline_assets = list(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in list(configuration.get("assets") or [])
                if str(symbol).strip()
            )
        )
        if not series or min(series) != len(baseline_assets):
            raise RuntimeError(
                "The first universe size must equal the Strategy baseline asset count: "
                f"baseline={len(baseline_assets)}, series={series}."
            )

        family, model_settings, settings_hash = _immutable_model_snapshot(strategy)
        identity = common._market_identity(configuration)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
        expected = common._expected_sessions(history_start, snapshot_end)

        _log(
            f"Execution sequence: Strategy #{int(strategy.get('strategy_sequence') or args.strategy_sequence)} "
            f"({len(baseline_assets)} assets) -> nested universes {series} -> "
            "same LightGBM/settings -> Rotation vs equal-weight Buy & Hold."
        )
        _log(
            f"History={history_start.date().isoformat()} -> {snapshot_end.date().isoformat()}; "
            f"economic comparison={analysis_start.date().isoformat()} -> "
            f"{analysis_end.date().isoformat()}; settings_hash={settings_hash}."
        )

        _log(
            f"STEP 2/6 - Loading and validating the {len(baseline_assets)} baseline assets from MongoDB."
        )
        baseline_frames = common._load_frames_once(
            collection,
            baseline_assets,
            identity,
            history_start,
            snapshot_end,
        )
        baseline_history = common._validate_complete_history(
            baseline_frames,
            baseline_assets,
            expected,
        )
        history_rows: list[dict[str, Any]] = [
            {
                **row,
                "accepted": True,
                "data_source": "mongo_baseline_permanent",
                "source": "strategy",
            }
            for row in baseline_history
        ]

        required_external = max(series) - len(baseline_assets)
        candidate_frames: dict[str, pd.DataFrame] = {}
        accepted_external: list[str] = []
        candidate_scan_count = 0
        seed_path: Path | None = None

        _log(
            f"STEP 3/6 - Preparing external assets for largest universe={max(series)} "
            f"(required external={required_external})."
        )
        if required_external > 0:
            seed_path = (
                Path(args.seed_universe_file).resolve()
                if args.seed_universe_file
                else _default_seed_file(args.strategy_sequence, snapshot_end.date().isoformat())
            )
            loader = CandidateUniverseLoader(
                db=db,
                collection=collection,
                configuration=configuration,
                configuration_hash=str(strategy.get("configuration_hash") or ""),
                identity=identity,
                history_start=history_start,
                snapshot_end=snapshot_end,
                expected_sessions=expected,
                baseline_assets=baseline_assets,
                workers=workers,
                max_candidate_scans=int(args.max_candidate_scans),
                log=_log,
            )
            candidate_frames, accepted_external, candidate_rows, candidate_scan_count = loader.prepare(
                required_external,
                seed_file=seed_path,
            )
            history_rows.extend(candidate_rows)
        else:
            _log("No external assets are required. Alpaca universe discovery is skipped.")
    finally:
        client.close()

    _log(
        "STEP 4/6 - MongoDB closed. Candidate history remains RAM-only; "
        "this experiment performed no MongoDB writes."
    )

    output_dir = _output_dir(args, strategy)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_set = set(baseline_assets)
    all_frames = {**baseline_frames, **candidate_frames}

    inventory = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "source": "strategy" if symbol in baseline_set else "candidate",
                "rows": int(len(frame)),
                "frame_memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
                "persisted_by_experiment": False,
            }
            for symbol, frame in all_frames.items()
        ]
    )
    _write_csv(output_dir / "universe_candidate_history_integrity.csv", pd.DataFrame(history_rows))
    _write_csv(output_dir / "universe_data_inventory.csv", inventory)
    _write_json(
        output_dir / "universe_scale_manifest.json",
        {
            "schema_version": 2,
            "script_version": SCRIPT_VERSION,
            "experiment": "nested_eligible_universe_scale_vs_equal_weight_buy_hold",
            "strategy_id": str(strategy.get("_id") or ""),
            "strategy_sequence": int(strategy.get("strategy_sequence") or args.strategy_sequence),
            "strategy_revision": int(strategy.get("revision") or 0),
            "strategy_configuration_hash": strategy.get("configuration_hash"),
            "model_family": family,
            "model_settings_hash": settings_hash,
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "analysis_start": analysis_start.date().isoformat(),
            "analysis_end": analysis_end.date().isoformat(),
            "series": series,
            "baseline_assets": baseline_assets,
            "external_assets_order": accepted_external,
            "candidate_scan_count": candidate_scan_count,
            "seed_universe_file": str(seed_path) if seed_path and seed_path.exists() else None,
            "persistence": {
                "mongo_writes": False,
                "external_history": "RAM only unless already present before experiment",
                "rejected_candidates_persisted": False,
                "persistent_candidate_history_added": 0,
            },
            "raw_frame_memory_bytes": int(inventory["frame_memory_bytes"].sum()) if not inventory.empty else 0,
            "benchmark": "Equal-weight Buy & Hold across the same universe used by each series",
        },
    )

    _log("STEP 5/6 - Executing nested universes with identical model/settings/costs/dates.")
    summary_rows: list[dict[str, Any]] = []
    for size in series:
        external_count = size - len(baseline_assets)
        assets = [*baseline_assets, *accepted_external[:external_count]]
        if len(assets) != size:
            raise RuntimeError(f"Unable to construct nested universe={size}; built={len(assets)}.")

        _write_json(
            output_dir / f"universe_{size}_assets.json",
            {
                "universe_size": size,
                "baseline_asset_count": len(baseline_assets),
                "external_asset_count": external_count,
                "assets": assets,
                "assets_sha256": hashlib.sha256("\n".join(assets).encode("utf-8")).hexdigest(),
            },
        )

        result_path = output_dir / f"universe_{size}_result.json"
        if result_path.exists() and not args.no_resume:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            summary_rows.append(dict(previous.get("summary") or {}))
            _log(f"Universe {size}: existing completed result found; resume skips recomputation.")
            continue

        config = _execution_config(
            configuration,
            assets,
            baseline_assets,
            snapshot_end,
            analysis_start,
            analysis_end,
            family,
            model_settings,
        )
        bars = {symbol: all_frames[symbol] for symbol in assets}
        _log(
            f"=== SERIES {size}: Rotation vs equal-weight Buy & Hold; external={external_count} ==="
        )
        started = time.perf_counter()
        results = run_rotation_models(
            bars,
            config,
            calculate_reference_fees,
            apply_slippage,
            progress_callback=_progress_logger(size),
            technical_log_callback=lambda message, n=size: _log(f"Universe {n}: {message}"),
        )
        if not results:
            raise RuntimeError(f"Universe {size}: rotation engine returned no result.")

        result = results[0]
        elapsed = float(time.perf_counter() - started)
        row = _summary(size, result, elapsed)
        summary_rows.append(row)

        _write_csv(
            output_dir / f"universe_{size}_predictions.csv",
            result.predictions.reset_index() if not result.predictions.empty else pd.DataFrame(),
        )
        _write_csv(output_dir / f"universe_{size}_trades.csv", result.trades.copy())
        _write_json(
            result_path,
            {
                "schema_version": 2,
                "script_version": SCRIPT_VERSION,
                "universe_size": size,
                "assets": assets,
                "summary": row,
                "metrics": dict(result.metrics or {}),
                "elapsed_seconds": elapsed,
            },
        )
        _log(
            f"Universe {size} completed: capital=${float(row.get('strategy_ending_capital') or 0):,.2f}; "
            f"Buy&Hold=${float(row.get('buy_hold_ending_capital') or 0):,.2f}; "
            f"elapsed={elapsed / 60.0:.1f} min."
        )
        gc.collect()

    _log("STEP 6/6 - Writing compact cross-series comparison.")
    summary = pd.DataFrame(summary_rows).sort_values("universe_size")
    _write_csv(output_dir / "universe_scale_comparison.csv", summary)
    _write_json(
        output_dir / "universe_scale_summary.json",
        {
            "schema_version": 2,
            "script_version": SCRIPT_VERSION,
            "series": summary.to_dict(orient="records"),
            "mongo_writes": False,
            "persistent_candidate_history_added": 0,
        },
    )

    print("\n=== UNIVERSE SCALE SUMMARY ===", flush=True)
    for row in summary.to_dict(orient="records"):
        _log(
            f"{int(row['universe_size'])} assets: rotation=${float(row.get('strategy_ending_capital') or 0):,.2f}; "
            f"Buy&Hold=${float(row.get('buy_hold_ending_capital') or 0):,.2f}; "
            f"CAGR={float(row.get('strategy_cagr') or 0) * 100:.2f}%; "
            f"Sharpe={float(row.get('strategy_sharpe') or 0):.3f}; "
            f"MaxDD={float(row.get('maximum_drawdown') or 0) * 100:.2f}%."
        )
    _log(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
