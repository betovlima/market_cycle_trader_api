from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from market_cycle_trader_api.infrastructure.market_data.alpaca import download_stock_bars  # noqa: E402
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    get_alpaca_credentials,
)
from market_cycle_trader_api.services import asset_discovery_market as market  # noqa: E402
from market_cycle_trader_api.services.asset_discovery_behavior import behavior_risk_profile  # noqa: E402
from market_cycle_trader_api.services.asset_discovery_settings import (  # noqa: E402
    DEFAULT_SETTINGS,
    normalized_asset_discovery_settings,
)

SCRIPT_VERSION = "asset-rotation-universe-scale-v1.0"
DEFAULT_SERIES = (56, 82, 250, 500)
DEFAULT_WORKERS = 4
ASSET_DISCOVERY_SETTINGS_COLLECTION = "asset_discovery_settings"


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how the Strategy #10 intelligent-rotation engine behaves as its eligible "
            "US-equity universe expands. External histories are read from MongoDB when already "
            "complete or downloaded directly into RAM; this research script never writes market "
            "history or research results to MongoDB."
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


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _default_seed_file(strategy_sequence: int, snapshot_end: str) -> Path:
    return (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{strategy_sequence}_{snapshot_end}"
        / "asset_history_integrity.csv"
    ).resolve()


def _seed_candidates(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path, usecols=lambda name: name in {"symbol", "source", "history_complete"})
    required = {"symbol", "source", "history_complete"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Seed universe must contain symbol, source and history_complete. "
            "No qualification/ranking outcome is read."
        )
    rows = frame.loc[
        (frame["source"].astype(str).str.lower() == "candidate")
        & frame["history_complete"].map(_truthy),
        "symbol",
    ]
    return sorted({str(item).strip().upper() for item in rows if str(item).strip()})


def _stable_priority(configuration_hash: str, snapshot_end: str, symbol: str) -> str:
    material = f"{configuration_hash}|{snapshot_end}|{symbol.upper()}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _settings_read_only(db: Any) -> dict[str, Any]:
    document = db[ASSET_DISCOVERY_SETTINGS_COLLECTION].find_one({"_id": "default"}) or {}
    raw = document.get("settings") if isinstance(document.get("settings"), dict) else DEFAULT_SETTINGS
    return normalized_asset_discovery_settings(raw)


def _frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    columns = [name for name in ("open", "high", "low", "close", "volume", "vwap", "trade_count") if name in frame]
    return frame[columns]


def _cached_frame(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> pd.DataFrame:
    start = history_start.tz_localize("UTC").to_pydatetime()
    end = (snapshot_end + pd.Timedelta(days=1)).tz_localize("UTC").to_pydatetime()
    rows = list(
        collection.find(
            {
                "symbol": symbol,
                **identity,
                "timestamp": {"$gte": start, "$lt": end},
            },
            {
                "_id": 0,
                "timestamp": 1,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "vwap": 1,
                "trade_count": 1,
            },
        ).sort("timestamp", 1)
    )
    return _frame_from_rows(rows)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in result.columns for column in required):
        return pd.DataFrame()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    return result


def _history_diagnostics(
    symbol: str,
    frame: pd.DataFrame,
    expected: pd.DatetimeIndex,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "symbol": symbol,
            "history_complete": False,
            "observed_rows": 0,
            "expected_sessions": int(len(expected)),
            "missing_sessions": int(len(expected)),
            "actual_start": None,
            "actual_end": None,
        }
    observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)).unique()
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    return {
        "symbol": symbol,
        "history_complete": bool(len(missing) == 0),
        "observed_rows": int(len(frame)),
        "expected_sessions": int(len(expected)),
        "missing_sessions": int(len(missing)),
        "extra_sessions": int(len(extra)),
        "actual_start": observed.min().date().isoformat() if len(observed) else None,
        "actual_end": observed.max().date().isoformat() if len(observed) else None,
        "missing_sample": ",".join(item.date().isoformat() for item in missing[:5]),
    }


def _quality_diagnostics(frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    recent = frame.tail(min(63, len(frame)))
    close = pd.to_numeric(recent.get("close"), errors="coerce")
    volume = pd.to_numeric(recent.get("volume"), errors="coerce")
    latest_close = float(close.dropna().iloc[-1]) if not close.dropna().empty else float("nan")
    dollar_volume = (close * volume).dropna()
    median_dollar_volume = float(dollar_volume.median()) if not dollar_volume.empty else 0.0
    nonzero_volume_ratio = float((volume > 0).mean()) if len(volume) else 0.0
    behavior = behavior_risk_profile(frame, settings)
    checks = {
        "price_ready": bool(np.isfinite(latest_close) and latest_close >= float(settings["min_price"])),
        "liquidity_ready": bool(median_dollar_volume >= float(settings["min_median_dollar_volume"])),
        "volume_quality_ready": bool(nonzero_volume_ratio >= float(settings["min_nonzero_volume_ratio"])),
        "behavior_ready": bool(not behavior.get("sample_ready") or behavior.get("passed")),
    }
    return {
        "market_quality_passed": bool(all(checks.values())),
        "latest_close": latest_close,
        "median_dollar_volume_63d": median_dollar_volume,
        "nonzero_volume_ratio": nonzero_volume_ratio,
        "behavior_passed": behavior.get("passed"),
        "behavior_reason_codes": list(behavior.get("reason_codes") or []),
        "failed_quality_checks": [name for name, passed in checks.items() if not passed],
    }


def _download_frame(
    symbol: str,
    credentials: dict[str, str],
    configuration: dict[str, Any],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> pd.DataFrame:
    start = history_start.tz_localize("UTC").to_pydatetime()
    end = (snapshot_end + pd.Timedelta(days=1)).tz_localize("UTC").to_pydatetime()
    frame = download_stock_bars(
        api_key_id=credentials["api_key_id"],
        secret_key=credentials["secret_key"],
        symbol=symbol,
        timeframe=str(configuration.get("timeframe") or "1Day"),
        start=start,
        end=end,
        feed=str(configuration.get("alpaca_historical_feed") or "sip"),
        adjustment=str(configuration.get("alpaca_adjustment") or "all"),
    )
    return _clean_frame(frame)


def _candidate_frame(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    credentials: dict[str, str],
    configuration: dict[str, Any],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    expected: pd.DatetimeIndex,
    settings: dict[str, Any],
) -> tuple[str, pd.DataFrame | None, dict[str, Any]]:
    started = time.perf_counter()
    source = "mongo_read_only"
    try:
        cached = _clean_frame(_cached_frame(collection, symbol, identity, history_start, snapshot_end))
        history = _history_diagnostics(symbol, cached, expected)
        if history["history_complete"]:
            frame = cached
        else:
            source = "alpaca_ram_only"
            frame = _download_frame(symbol, credentials, configuration, history_start, snapshot_end)
            history = _history_diagnostics(symbol, frame, expected)
        quality = _quality_diagnostics(frame, settings) if history["history_complete"] else {
            "market_quality_passed": False,
            "failed_quality_checks": ["full_strategy_history"],
        }
        accepted = bool(history["history_complete"] and quality["market_quality_passed"])
        diagnostics = {
            **history,
            **quality,
            "accepted": accepted,
            "data_source": source,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        return symbol, frame if accepted else None, diagnostics
    except Exception as exc:
        return symbol, None, {
            "symbol": symbol,
            "history_complete": False,
            "market_quality_passed": False,
            "accepted": False,
            "data_source": source,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": float(time.perf_counter() - started),
        }


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


def _normalized_result(size: int, result: Any, elapsed: float) -> dict[str, Any]:
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
    if not series or any(value < 2 for value in series):
        raise RuntimeError("--series must contain universe sizes >= 2.")
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
    if len(baseline_assets) < 2:
        raise RuntimeError("Strategy has fewer than two baseline assets.")
    if min(series) != len(baseline_assets):
        raise RuntimeError(
            "The first universe size must equal the Strategy baseline asset count so the control is exact: "
            f"baseline={len(baseline_assets)}, series={series}."
        )

    family, model_settings, settings_hash = _immutable_model_snapshot(strategy)
    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
    expected = common._expected_sessions(history_start, snapshot_end)
    discovery_settings = _settings_read_only(db)
    credentials = get_alpaca_credentials(db)

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / (
            f"asset_rotation_universe_scale_strategy_{int(strategy.get('strategy_sequence') or args.strategy_sequence)}_"
            f"{analysis_start.date().isoformat()}_to_{analysis_end.date().isoformat()}"
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("ASSET ROTATION UNIVERSE SCALE EXPERIMENT")
    _log(
        f"Full Strategy History={history_start.date().isoformat()} -> {snapshot_end.date().isoformat()}, "
        f"economic comparison={analysis_start.date().isoformat()} -> {analysis_end.date().isoformat()}."
    )
    _log(
        f"Nested series={series}; baseline assets={len(baseline_assets)}; model={family}; "
        f"settings_hash={settings_hash}; workers={workers}."
    )
    _log(
        "Persistence policy: MongoDB READ ONLY. External candidate histories downloaded from Alpaca remain RAM-only."
    )

    baseline_frames = common._load_frames_once(
        collection,
        baseline_assets,
        identity,
        history_start,
        snapshot_end,
    )
    baseline_history = common._validate_complete_history(baseline_frames, baseline_assets, expected)
    history_rows: list[dict[str, Any]] = [
        {**row, "accepted": True, "data_source": "mongo_baseline_permanent", "source": "strategy"}
        for row in baseline_history
    ]

    max_target = max(series)
    required_external = max_target - len(baseline_assets)
    if required_external < 0:
        raise RuntimeError("Largest requested universe is smaller than the Strategy baseline.")

    seed_path = (
        Path(args.seed_universe_file).resolve()
        if args.seed_universe_file
        else _default_seed_file(args.strategy_sequence, snapshot_end.date().isoformat())
    )
    seed = [symbol for symbol in _seed_candidates(seed_path) if symbol not in set(baseline_assets)]
    if seed:
        _log(f"Seed universe: {len(seed)} prior full-history candidates will be revalidated without reading their rankings.")
    else:
        _log("Seed universe: none found; all external slots will come from deterministic Alpaca sampling.")

    discovered = market.discover_alpaca_symbols()
    baseline_set = set(baseline_assets)
    seed_set = set(seed)
    remaining = [symbol for symbol in discovered if symbol not in baseline_set and symbol not in seed_set]
    configuration_hash = str(strategy.get("configuration_hash") or "")
    remaining.sort(
        key=lambda symbol: _stable_priority(
            configuration_hash,
            snapshot_end.date().isoformat(),
            symbol,
        )
    )
    candidate_order = list(dict.fromkeys([*seed, *remaining]))
    scan_limit = min(len(candidate_order), max(required_external, int(args.max_candidate_scans)))

    candidate_frames: dict[str, pd.DataFrame] = {}
    accepted_external: list[str] = []
    scanned = 0
    batch_size = workers

    while len(accepted_external) < required_external and scanned < scan_limit:
        batch = candidate_order[scanned : min(scan_limit, scanned + batch_size)]
        if not batch:
            break
        scanned += len(batch)
        futures = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="universe-history") as executor:
            for symbol in batch:
                futures[executor.submit(
                    _candidate_frame,
                    collection,
                    symbol,
                    identity,
                    credentials,
                    configuration,
                    history_start,
                    snapshot_end,
                    expected,
                    discovery_settings,
                )] = symbol
            completed: dict[str, tuple[pd.DataFrame | None, dict[str, Any]]] = {}
            for future in as_completed(futures):
                symbol, frame, diagnostics = future.result()
                completed[symbol] = (frame, diagnostics)

        for symbol in batch:
            frame, diagnostics = completed[symbol]
            diagnostics = {**diagnostics, "source": "candidate"}
            history_rows.append(diagnostics)
            if frame is not None and diagnostics.get("accepted"):
                candidate_frames[symbol] = frame
                accepted_external.append(symbol)
                _log(
                    f"Candidate accepted {len(accepted_external)}/{required_external} - {symbol}: "
                    f"history={diagnostics.get('observed_rows')}, "
                    f"median$vol={float(diagnostics.get('median_dollar_volume_63d') or 0):,.0f}, "
                    f"source={diagnostics.get('data_source')}."
                )
            else:
                reasons = diagnostics.get("failed_quality_checks") or diagnostics.get("error") or "incomplete_history"
                _log(f"Candidate rejected - {symbol}: {reasons}.")
            if len(accepted_external) >= required_external:
                break

    if len(accepted_external) < required_external:
        _write_csv(output_dir / "universe_candidate_history_integrity.csv", pd.DataFrame(history_rows))
        raise RuntimeError(
            f"Could only qualify {len(accepted_external)} external assets after scanning {scanned}; "
            f"{required_external} are required for universe={max_target}. Increase --max-candidate-scans."
        )

    client.close()
    _log(
        f"Candidate preparation complete: scanned={scanned}, qualified={len(accepted_external)}. "
        "MongoDB connection closed; all remaining work is RAM/CPU."
    )

    all_frames = {**baseline_frames, **candidate_frames}
    memory_rows = []
    for symbol, frame in all_frames.items():
        memory_rows.append(
            {
                "symbol": symbol,
                "source": "strategy" if symbol in baseline_set else "candidate",
                "rows": int(len(frame)),
                "frame_memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
                "persisted_by_experiment": False,
            }
        )
    inventory = pd.DataFrame(memory_rows)
    _write_csv(output_dir / "universe_candidate_history_integrity.csv", pd.DataFrame(history_rows))
    _write_csv(output_dir / "universe_data_inventory.csv", inventory)

    manifest = {
        "schema_version": 1,
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
        "expected_xnys_sessions": int(len(expected)),
        "series": series,
        "baseline_asset_count": len(baseline_assets),
        "baseline_assets": baseline_assets,
        "external_assets_required": required_external,
        "external_assets_qualified": len(accepted_external),
        "external_assets_order": accepted_external,
        "seed_universe_file": str(seed_path) if seed_path.exists() else None,
        "seed_candidate_count": len(seed),
        "candidate_scan_count": scanned,
        "candidate_order_method": "seed_full_history_candidates_then_sha256_strategy_snapshot_symbol",
        "eligibility": {
            "active_tradable_us_equity": True,
            "supported_exchanges": sorted(market.SUPPORTED_EXCHANGES),
            "full_strategy_history_required": True,
            "market_quality_settings": discovery_settings,
            "economic_backtest_outcome_used_for_candidate_selection": False,
        },
        "persistence": {
            "mongo_writes": False,
            "external_history": "RAM only unless it already existed in MongoDB before this experiment",
            "rejected_candidates_persisted": False,
            "research_results_persisted_to_mongo": False,
        },
        "raw_frame_memory_bytes": int(inventory["frame_memory_bytes"].sum()),
        "benchmark": "Equal-weight Buy & Hold across the same universe used by each series",
        "scientific_question": (
            "With the same immutable Strategy model/settings and economic window, does giving the "
            "intelligent rotation engine a larger objectively eligible opportunity set improve or "
            "degrade portfolio performance?"
        ),
    }
    _write_json(output_dir / "universe_scale_manifest.json", manifest)

    summary_rows: list[dict[str, Any]] = []
    for size in series:
        external_count = size - len(baseline_assets)
        assets = [*baseline_assets, *accepted_external[:external_count]]
        if len(assets) != size:
            raise RuntimeError(f"Unable to construct nested universe={size}; built={len(assets)}.")
        universe_file = output_dir / f"universe_{size}_assets.json"
        _write_json(
            universe_file,
            {
                "universe_size": size,
                "baseline_asset_count": len(baseline_assets),
                "external_asset_count": external_count,
                "assets": assets,
                "assets_sha256": hashlib.sha256("\n".join(assets).encode("utf-8")).hexdigest(),
            },
        )

        result_file = output_dir / f"universe_{size}_result.json"
        if result_file.exists() and not args.no_resume:
            previous = json.loads(result_file.read_text(encoding="utf-8"))
            summary_rows.append(dict(previous.get("summary") or {}))
            _log(f"Universe {size}: completed result already exists; resume skips recomputation.")
            continue

        config = leadership._execution_config(
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
        bars = {symbol: all_frames[symbol] for symbol in assets}
        _log(
            f"=== SERIES {size}: intelligent rotation vs equal-weight Buy & Hold; "
            f"external={external_count} ==="
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
        summary = _normalized_result(size, result, elapsed)
        summary_rows.append(summary)

        predictions = result.predictions.reset_index() if not result.predictions.empty else pd.DataFrame()
        trades = result.trades.copy()
        _write_csv(output_dir / f"universe_{size}_predictions.csv", predictions)
        _write_csv(output_dir / f"universe_{size}_trades.csv", trades)
        _write_json(
            result_file,
            {
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "universe_size": size,
                "assets": assets,
                "summary": summary,
                "metrics": dict(result.metrics or {}),
                "elapsed_seconds": elapsed,
            },
        )
        _log(
            f"Universe {size} completed: capital=${float(summary.get('strategy_ending_capital') or 0):,.2f}; "
            f"Buy&Hold=${float(summary.get('buy_hold_ending_capital') or 0):,.2f}; "
            f"elapsed={elapsed / 60.0:.1f} min."
        )
        gc.collect()

    summary_frame = pd.DataFrame(summary_rows).sort_values("universe_size")
    _write_csv(output_dir / "universe_scale_comparison.csv", summary_frame)
    _write_json(
        output_dir / "universe_scale_summary.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "series": summary_frame.to_dict(orient="records"),
            "mongo_writes": False,
            "persistent_candidate_history_added": 0,
        },
    )

    print("\n=== UNIVERSE SCALE SUMMARY ===", flush=True)
    for row in summary_frame.to_dict(orient="records"):
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
