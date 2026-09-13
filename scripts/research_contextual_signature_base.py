from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.uri_parser import parse_uri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.11"
EXPERIMENT_NAME = "contextual_marginal_signature_path_attribution"
STRATEGY_PROFILES_COLLECTION = "strategy_profiles"
LOCAL_MONGO_HOSTS = {"localhost", "127.0.0.1", "::1", "mongo", "host.docker.internal"}
TOLERANCE = 1e-12

TRACE_DECISION_COLUMNS = [
    "selected_asset",
    "previous_asset",
    "trade_action",
    "trade_reason",
    "decision_score",
    "selected_score",
    "cash_weight",
    "market_exposure_weight",
    "assets_held",
    "strategy_equity",
]


def _log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Focused serial replay audit that captures complete RotationRunResult predictions/trades "
            "and attributes cross-universe marginal-capital changes to baseline versus challenger paths."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--horizon-sessions", type=int, default=40)
    parser.add_argument("--universe-spec", required=True)
    parser.add_argument("--cases-spec", required=True)
    parser.add_argument("--data-source", choices=("yahoo", "snapshot"), default="yahoo")
    parser.add_argument("--market-snapshot", default=None, help="Frozen CSV/CSV.GZ produced by a prior v1.0.11 run.")
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _normalize_date(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid date: {value}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _utc_index(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(values), utc=True)).sort_values()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_fresh(path: Path) -> None:
    root = (PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing --fresh-run outside {root}: {resolved}") from exc
    if not relative.parts or not relative.parts[0].startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected research output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_local_mongo(uri: str, allow_remote: bool) -> None:
    if allow_remote:
        return
    parsed = parse_uri(uri)
    hosts = {str(host).strip().lower() for host, _ in parsed.get("nodelist") or []}
    if hosts and hosts.issubset(LOCAL_MONGO_HOSTS):
        return
    raise RuntimeError(
        "This audit keeps Strategy/configuration access on local MongoDB. "
        f"Resolved hosts={sorted(hosts) or ['unknown']}. Use --allow-remote-mongo only intentionally."
    )


def _strategy_document(db: Any, sequence: int, strategy_id: str | None) -> dict[str, Any]:
    query = {"_id": strategy_id} if strategy_id else {"strategy_sequence": int(sequence)}
    document = db[STRATEGY_PROFILES_COLLECTION].find_one(query)
    if document is None:
        raise RuntimeError(f"Strategy not found: {strategy_id or f'Strategy #{sequence}'}")
    return document


def _configuration(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("configuration")
    return dict(value) if isinstance(value, dict) else dict(document)


def _normalize_symbols(values: list[str] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        symbol = str(value).strip().upper()
        if symbol and symbol not in output:
            output.append(symbol)
    return output


def _load_universe_spec(path: Path, selected_names: list[str]) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("universes") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Universe spec must contain universes=[...].")
    by_name = {
        str(item.get("name") or "").strip(): _normalize_symbols(item.get("assets") if isinstance(item.get("assets"), list) else [])
        for item in items if isinstance(item, dict)
    }
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise RuntimeError("Cases spec references missing universes: " + ", ".join(missing))
    return [{"name": name, "assets": by_name[name]} for name in selected_names]


def _load_cases(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    universes = [str(item).strip() for item in raw.get("universes") or [] if str(item).strip()]
    raw_cases = raw.get("cases")
    if len(universes) < 2 or not isinstance(raw_cases, list) or not raw_cases:
        raise RuntimeError("Cases spec requires at least two universes and a non-empty cases list.")
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        date = _normalize_date(item.get("decision_date")).date().isoformat()
        candidates = _normalize_symbols(item.get("candidates") if isinstance(item.get("candidates"), list) else [])
        if not candidates:
            raise RuntimeError(f"Cases spec has no candidates for {date}.")
        for candidate in candidates:
            key = (date, candidate)
            if key in seen:
                raise RuntimeError(f"Duplicate candidate/date case: {date}/{candidate}")
            seen.add(key)
        cases.append({"decision_date": date, "candidates": candidates})
    return universes, cases


def _expected_sessions(start_date: pd.Timestamp, snapshot_end: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(calendar.date_to_session(start_date, direction="next"))
    last = pd.Timestamp(calendar.date_to_session(snapshot_end, direction="previous"))
    return _utc_index(calendar.sessions_in_range(first, last)).normalize()


def _horizon_end(sessions: pd.DatetimeIndex, decision: pd.Timestamp, horizon_sessions: int) -> pd.Timestamp:
    ordered = _utc_index(sessions)
    decision_utc = _utc_timestamp(decision)
    position = int(ordered.searchsorted(decision_utc, side="left"))
    if position >= len(ordered) or pd.Timestamp(ordered[position]) != decision_utc:
        raise RuntimeError(f"Decision date is not an XNYS session: {decision_utc.date()}")
    end_position = position + max(1, int(horizon_sessions)) - 1
    if end_position >= len(ordered):
        raise RuntimeError(f"Horizon exceeds snapshot for {decision_utc.date()}")
    return pd.Timestamp(ordered[end_position])


def _load_yahoo_frames(symbols: list[str], start_date: pd.Timestamp, snapshot_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Yahoo source requires yfinance. Install with: pip install yfinance") from exc
    frames: dict[str, pd.DataFrame] = {}
    end_exclusive = (snapshot_end + pd.Timedelta(days=1)).date().isoformat()
    for symbol in symbols:
        _log(f"Yahoo: loading {symbol}...")
        data = yf.download(
            symbol,
            start=start_date.date().isoformat(),
            end=end_exclusive,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            raise RuntimeError(f"Yahoo returned no history for {symbol}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(item[0]) for item in data.columns]
        renamed = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise RuntimeError(f"Yahoo history for {symbol} is missing columns: {missing}")
        frame = renamed.loc[:, required].copy()
        frame.index = pd.to_datetime(frame.index, utc=True).normalize()
        frames[symbol] = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frames


def _snapshot_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in sorted(frames):
        frame = frames[symbol].copy().sort_index()
        frame = frame.loc[:, [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]]
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        frame.insert(0, "symbol", symbol)
        rows.append(frame)
    snapshot = pd.concat(rows, ignore_index=True)
    snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"], utc=True)
    return snapshot.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _snapshot_hash(snapshot: pd.DataFrame) -> str:
    normalized = snapshot.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(12)
    payload = normalized.to_csv(index=False, na_rep="NaN", float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frames_from_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    table = pd.read_csv(path)
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError("Frozen market snapshot is missing columns: " + ", ".join(missing))
    table["symbol"] = table["symbol"].astype(str).str.upper()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in table.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        frames[str(symbol)] = frame
    return frames


def _window_request(
    *, db: Any, config: BacktestRequest, strategy_id: str, assets: list[str], reference_assets: list[str],
    candidate_assets: list[str], decision_session: pd.Timestamp, horizon_end: pd.Timestamp,
):
    return discovery._marginal_execution_request(
        db,
        config,
        {"id": strategy_id},
        config,
        horizon_end.date().isoformat(),
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        analysis_start_date=decision_session.date().isoformat(),
        analysis_end_date=horizon_end.date().isoformat(),
    )


def _run_with_capture(frames: dict[str, pd.DataFrame], request: Any) -> tuple[dict[str, Any], pd.DatetimeIndex, list[Any]]:
    captured: list[Any] = []
    original = discovery.run_rotation_models

    def wrapper(*args: Any, **kwargs: Any):
        results = original(*args, **kwargs)
        captured.extend(list(results or []))
        return results

    discovery.run_rotation_models = wrapper
    try:
        metrics, sessions = discovery._run_rotation_replay(frames, request)
    finally:
        discovery.run_rotation_models = original
    if not captured:
        raise RuntimeError("Replay completed without captured RotationRunResult objects.")
    return metrics, sessions, captured


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _prediction_frame(result: Any) -> pd.DataFrame:
    frame = getattr(result, "predictions", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    if "timestamp" not in output.columns:
        output = output.reset_index()
        if "timestamp" not in output.columns:
            output = output.rename(columns={output.columns[0]: "timestamp"})
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    return output.sort_values("timestamp").reset_index(drop=True)


def _trade_frame(result: Any) -> pd.DataFrame:
    frame = getattr(result, "trades", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    if "timestamp" in output.columns:
        output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    return output.reset_index(drop=True)


def _backend_key(result: Any, index: int) -> str:
    backend = str(getattr(result, "backend", "") or "backend").strip()
    return f"{_safe_name(backend)}_{index:02d}"


def _save_capture(base: Path, results: list[Any]) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(results):
        key = _backend_key(result, index)
        path = base / key
        path.mkdir(parents=True, exist_ok=True)
        predictions = _prediction_frame(result)
        trades = _trade_frame(result)
        _write_frame(path / "predictions.csv", predictions)
        _write_frame(path / "trades.csv", trades)
        _write_json(path / "metrics.json", getattr(result, "metrics", {}) or {})
        (path / "summary.txt").write_text(str(getattr(result, "summary", "") or ""), encoding="utf-8")
        _write_json(path / "prediction_columns.json", list(predictions.columns))
        captured[key] = {
            "predictions": predictions,
            "trades": trades,
            "metrics": dict(getattr(result, "metrics", {}) or {}),
        }
    return captured


def _first_divergence(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = TOLERANCE) -> dict[str, Any]:
    if left.empty and right.empty:
        return {"first_divergence": None, "changed_columns": []}
    l = left.copy()
    r = right.copy()
    for frame in (l, r):
        if "timestamp" not in frame.columns:
            raise RuntimeError("Prediction trace has no timestamp column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    l = l.set_index("timestamp").sort_index()
    r = r.set_index("timestamp").sort_index()
    timestamps = l.index.union(r.index).sort_values()
    columns = [column for column in TRACE_DECISION_COLUMNS if column in l.columns or column in r.columns]
    for timestamp in timestamps:
        if timestamp not in l.index or timestamp not in r.index:
            return {"first_divergence": timestamp.isoformat(), "changed_columns": ["row_presence"]}
        changed: list[str] = []
        for column in columns:
            lv = l.at[timestamp, column] if column in l.columns else np.nan
            rv = r.at[timestamp, column] if column in r.columns else np.nan
            ln = pd.to_numeric(pd.Series([lv]), errors="coerce").iloc[0]
            rn = pd.to_numeric(pd.Series([rv]), errors="coerce").iloc[0]
            if pd.notna(ln) and pd.notna(rn):
                if abs(float(rn) - float(ln)) > tolerance:
                    changed.append(column)
            else:
                ls = "" if pd.isna(lv) else str(lv)
                rs = "" if pd.isna(rv) else str(rv)
                if ls != rs:
                    changed.append(column)
        if changed:
            return {"first_divergence": timestamp.isoformat(), "changed_columns": changed}
    return {"first_divergence": None, "changed_columns": []}


def _candidate_trace_stats(predictions: pd.DataFrame, trades: pd.DataFrame, candidate: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "candidate_selected_sessions": None,
        "candidate_trade_rows": None,
        "candidate_buy_rows": None,
        "candidate_sell_rows": None,
    }
    if not predictions.empty and "selected_asset" in predictions.columns:
        selected = predictions["selected_asset"].astype(str).str.upper()
        stats["candidate_selected_sessions"] = int((selected == candidate).sum())
    if not trades.empty:
        symbol_column = next((column for column in ["symbol", "asset", "selected_asset"] if column in trades.columns), None)
        if symbol_column:
            candidate_rows = trades.loc[trades[symbol_column].astype(str).str.upper() == candidate].copy()
            stats["candidate_trade_rows"] = int(len(candidate_rows))
            if "action" in candidate_rows.columns:
                actions = candidate_rows["action"].astype(str).str.upper()
                stats["candidate_buy_rows"] = int(actions.str.contains("BUY", regex=False).sum())
                stats["candidate_sell_rows"] = int(actions.str.contains("SELL", regex=False).sum())
    return stats


def _common_backend(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> str | None:
    common = sorted(set(left).intersection(right))
    return common[0] if common else None


def _pairwise_decomposition(aggregate: pd.DataFrame, universe_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (date, candidate), group in aggregate.groupby(["decision_date", "candidate"], sort=True):
        indexed = {str(row["universe_name"]): row for _, row in group.iterrows()}
        for left_name, right_name in combinations(universe_order, 2):
            if left_name not in indexed or right_name not in indexed:
                continue
            left = indexed[left_name]
            right = indexed[right_name]
            b_left, b_right = float(left["baseline_ending_capital"]), float(right["baseline_ending_capital"])
            c_left, c_right = float(left["candidate_ending_capital"]), float(right["candidate_ending_capital"])
            m_left, m_right = float(left["delta_log_capital"]), float(right["delta_log_capital"])
            delta_log_b = float(math.log(b_right / b_left))
            delta_log_c = float(math.log(c_right / c_left))
            delta_m = m_right - m_left
            rows.append({
                "decision_date": date,
                "candidate": candidate,
                "universe_left": left_name,
                "universe_right": right_name,
                "delta_log_baseline": delta_log_b,
                "delta_log_candidate_capital": delta_log_c,
                "delta_marginal": delta_m,
                "identity_error": delta_log_c - delta_log_b - delta_m,
            })
    return pd.DataFrame(rows)


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
    _assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    output_dir = Path(args.output_dir).resolve()
    if args.fresh_run:
        _safe_fresh(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_order, cases = _load_cases(Path(args.cases_spec).resolve())
    universes = _load_universe_spec(Path(args.universe_spec).resolve(), universe_order)
    all_candidates = sorted({candidate for case in cases for candidate in case["candidates"]})
    history_start = _normalize_date(args.history_start)
    snapshot_end = _normalize_date(args.snapshot_end)
    expected_sessions = _expected_sessions(history_start, snapshot_end)

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, retryWrites=False)
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = _strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored_configuration = _configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        configured_start = _normalize_date(stored_configuration.get("start_date"))
        if configured_start != history_start:
            raise RuntimeError(
                f"--history-start must match Strategy start date: strategy={configured_start.date()}, requested={history_start.date()}"
            )
        strategy_assets = set(_normalize_symbols(list(stored_configuration.get("assets") or [])))
        requested_symbols = _normalize_symbols(
            [symbol for universe in universes for symbol in universe["assets"]] + all_candidates
        )
        missing = sorted(set(requested_symbols).difference(strategy_assets))
        if missing:
            raise RuntimeError(f"Requested symbols are not present in Strategy #{strategy_sequence}: {missing}")

        if args.data_source == "snapshot":
            if not args.market_snapshot:
                raise RuntimeError("--data-source snapshot requires --market-snapshot.")
            raw_frames = _frames_from_snapshot(Path(args.market_snapshot).resolve())
            source_detail = f"frozen snapshot: {Path(args.market_snapshot).resolve()}"
        else:
            raw_frames = _load_yahoo_frames(requested_symbols, history_start, snapshot_end)
            source_detail = "Yahoo Finance via yfinance, auto_adjust=True, threads=False"

        absent = sorted(set(requested_symbols).difference(raw_frames))
        if absent:
            raise RuntimeError("Market source did not provide: " + ", ".join(absent))

        snapshot = _snapshot_table({symbol: raw_frames[symbol] for symbol in requested_symbols})
        market_hash = _snapshot_hash(snapshot)
        snapshot_path = output_dir / "market_snapshot.csv.gz"
        snapshot.to_csv(snapshot_path, index=False, compression="gzip")

        universe_configs = {
            universe["name"]: BacktestRequest.model_validate(stored_configuration).model_copy(
                update={"assets": list(universe["assets"]), "end_date": snapshot_end.date().isoformat()}
            )
            for universe in universes
        }

        config_snapshot = BacktestRequest.model_validate(stored_configuration).model_dump(mode="json")
        _write_json(output_dir / "strategy_configuration.json", config_snapshot)

        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "experiment": EXPERIMENT_NAME,
            "git_commit": _git_commit(),
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "strategy_configuration_hash": _canonical_hash(config_snapshot),
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "horizon_sessions": int(args.horizon_sessions),
            "data_source": args.data_source,
            "data_source_detail": source_detail,
            "market_snapshot_hash": market_hash,
            "market_snapshot_path": str(snapshot_path),
            "universes": universes,
            "cases": cases,
            "execution_policy": "strictly serial; no research ThreadPoolExecutor; native asset_discovery replay; no v1.0.6 accelerator",
            "trace_contract": (
                "Capture every underlying RotationRunResult prediction/trade table and preserve baseline/challenger paths. "
                "No predictive model is fitted in this audit."
            ),
        }
        _write_json(output_dir / "trace_manifest.json", manifest)

        baseline_cache: dict[tuple[str, str], dict[str, Any]] = {}
        candidate_trace_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        aggregate_rows: list[dict[str, Any]] = []
        within_divergence_rows: list[dict[str, Any]] = []

        for case in cases:
            decision = _utc_timestamp(case["decision_date"])
            horizon_end = _horizon_end(expected_sessions, decision, int(args.horizon_sessions))
            _log(f"Decision {decision.date()} -> {horizon_end.date()}")
            for universe in universes:
                name = universe["name"]
                config = universe_configs[name]
                seed_frames = {symbol: raw_frames[symbol] for symbol in universe["assets"]}
                baseline_key = (case["decision_date"], name)
                if baseline_key not in baseline_cache:
                    _log(f"  {name}: baseline trace")
                    baseline_request = _window_request(
                        db=db,
                        config=config,
                        strategy_id=strategy_id,
                        assets=list(universe["assets"]),
                        reference_assets=list(universe["assets"]),
                        candidate_assets=[],
                        decision_session=decision,
                        horizon_end=horizon_end,
                    )
                    baseline_metrics, baseline_sessions, baseline_results = _run_with_capture(seed_frames, baseline_request)
                    baseline_capture = _save_capture(
                        output_dir / "traces" / case["decision_date"] / name / "BASELINE",
                        baseline_results,
                    )
                    baseline_cache[baseline_key] = {
                        "metrics": baseline_metrics,
                        "sessions": baseline_sessions,
                        "capture": baseline_capture,
                    }

                baseline_info = baseline_cache[baseline_key]
                baseline_capital = discovery._finite_number(baseline_info["metrics"].get("ending_capital"))
                if baseline_capital is None or baseline_capital <= 0:
                    raise RuntimeError(f"Invalid baseline capital for {case['decision_date']}/{name}")

                for candidate in case["candidates"]:
                    _log(f"  {name}/{candidate}: challenger trace")
                    challenger_frames = dict(seed_frames)
                    challenger_frames[candidate] = raw_frames[candidate]
                    challenger_request = _window_request(
                        db=db,
                        config=config,
                        strategy_id=strategy_id,
                        assets=[*universe["assets"], candidate],
                        reference_assets=list(universe["assets"]),
                        candidate_assets=[candidate],
                        decision_session=decision,
                        horizon_end=horizon_end,
                    )
                    challenger_metrics, challenger_sessions, challenger_results = _run_with_capture(challenger_frames, challenger_request)
                    compatibility = discovery._research_context_compatibility(baseline_info["sessions"], challenger_sessions)
                    if not bool(compatibility.get("research_context_compatible")):
                        raise RuntimeError(f"Trace context mismatch for {case['decision_date']}/{name}/{candidate}")
                    candidate_capital = discovery._finite_number(challenger_metrics.get("ending_capital"))
                    if candidate_capital is None or candidate_capital <= 0:
                        raise RuntimeError(f"Invalid challenger capital for {case['decision_date']}/{name}/{candidate}")
                    delta_log = float(math.log(candidate_capital / baseline_capital))
                    capture = _save_capture(
                        output_dir / "traces" / case["decision_date"] / name / candidate,
                        challenger_results,
                    )
                    candidate_trace_cache[(case["decision_date"], candidate, name)] = capture

                    backend = _common_backend(baseline_info["capture"], capture)
                    divergence = {"first_divergence": None, "changed_columns": []}
                    stats: dict[str, Any] = {}
                    if backend:
                        base_predictions = baseline_info["capture"][backend]["predictions"]
                        candidate_predictions = capture[backend]["predictions"]
                        divergence = _first_divergence(base_predictions, candidate_predictions)
                        stats = _candidate_trace_stats(
                            candidate_predictions,
                            capture[backend]["trades"],
                            candidate,
                        )
                    within_divergence_rows.append({
                        "decision_date": case["decision_date"],
                        "universe_name": name,
                        "candidate": candidate,
                        "backend": backend,
                        "first_baseline_vs_candidate_divergence": divergence["first_divergence"],
                        "changed_columns": ",".join(divergence["changed_columns"]),
                        **stats,
                    })
                    aggregate_rows.append({
                        "decision_date": case["decision_date"],
                        "horizon_end": horizon_end.date().isoformat(),
                        "universe_name": name,
                        "candidate": candidate,
                        "baseline_ending_capital": float(baseline_capital),
                        "candidate_ending_capital": float(candidate_capital),
                        "ending_capital_delta_rate": float(candidate_capital / baseline_capital - 1.0),
                        "delta_log_capital": delta_log,
                        "baseline_backend_count": int(len(baseline_info["capture"])),
                        "candidate_backend_count": int(len(capture)),
                        "baseline_total_fees": baseline_info["metrics"].get("total_transaction_fees"),
                        "candidate_total_fees": challenger_metrics.get("total_transaction_fees"),
                        **compatibility,
                    })

        aggregate = pd.DataFrame(aggregate_rows)
        _write_frame(output_dir / "trace_aggregate_dataset.csv", aggregate)
        _write_frame(output_dir / "trace_within_universe_first_divergence.csv", pd.DataFrame(within_divergence_rows))

        pairwise = _pairwise_decomposition(aggregate, universe_order)
        cross_rows: list[dict[str, Any]] = []
        for _, row in pairwise.iterrows():
            date = str(row["decision_date"])
            candidate = str(row["candidate"])
            left_name = str(row["universe_left"])
            right_name = str(row["universe_right"])
            left_candidate = candidate_trace_cache.get((date, candidate, left_name), {})
            right_candidate = candidate_trace_cache.get((date, candidate, right_name), {})
            candidate_backend = _common_backend(left_candidate, right_candidate)
            candidate_div = {"first_divergence": None, "changed_columns": []}
            if candidate_backend:
                candidate_div = _first_divergence(
                    left_candidate[candidate_backend]["predictions"],
                    right_candidate[candidate_backend]["predictions"],
                )
            left_base = baseline_cache[(date, left_name)]["capture"]
            right_base = baseline_cache[(date, right_name)]["capture"]
            base_backend = _common_backend(left_base, right_base)
            baseline_div = {"first_divergence": None, "changed_columns": []}
            if base_backend:
                baseline_div = _first_divergence(
                    left_base[base_backend]["predictions"],
                    right_base[base_backend]["predictions"],
                )
            cross_rows.append({
                "decision_date": date,
                "candidate": candidate,
                "universe_left": left_name,
                "universe_right": right_name,
                "baseline_backend": base_backend,
                "candidate_backend": candidate_backend,
                "first_baseline_path_divergence": baseline_div["first_divergence"],
                "baseline_changed_columns": ",".join(baseline_div["changed_columns"]),
                "first_candidate_path_divergence": candidate_div["first_divergence"],
                "candidate_changed_columns": ",".join(candidate_div["changed_columns"]),
            })
        cross = pd.DataFrame(cross_rows)
        pairwise = pairwise.merge(
            cross,
            on=["decision_date", "candidate", "universe_left", "universe_right"],
            how="left",
        )
        _write_frame(output_dir / "trace_universe_pair_decomposition.csv", pairwise)

        max_identity_error = float(pairwise["identity_error"].abs().max()) if not pairwise.empty else 0.0
        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
            "status": "completed",
            "git_commit": manifest["git_commit"],
            "strategy_configuration_hash": manifest["strategy_configuration_hash"],
            "market_snapshot_hash": market_hash,
            "aggregate_rows": int(len(aggregate)),
            "pairwise_rows": int(len(pairwise)),
            "max_delta_identity_error": max_identity_error,
            "baseline_trace_count": int(len(baseline_cache)),
            "challenger_trace_count": int(len(candidate_trace_cache)),
            "decision_rule": (
                "Explain each marginal-effect change through path evidence before fitting another signature model. "
                "Inspect delta_log_baseline, delta_log_candidate_capital and delta_marginal together with the first baseline/candidate path divergence."
            ),
        }
        _write_json(output_dir / "trace_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
