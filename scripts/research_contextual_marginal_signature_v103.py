from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from market_cycle_trader_api.engine import capital_rotation as rotation  # noqa: E402
from market_cycle_trader_api.engine.market_data import validate_and_clean_bars  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.3"
EXPERIMENT_NAME = "contextual_marginal_signature"
DEFAULT_WORKERS = 4
STRATEGY_PROFILES_COLLECTION = "strategy_profiles"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
LOCAL_MONGO_HOSTS = {"localhost", "127.0.0.1", "::1", "mongo", "host.docker.internal"}

BASE_FEATURES = [
    "return_5", "return_20", "return_60", "vol_20", "vol_60",
    "ema_distance_20", "trend_efficiency_20", "momentum_acceleration_5_20",
]
MODEL_FEATURES = [
    "relative__return_5", "relative__return_20", "relative__return_60",
    "relative__vol_20", "relative__vol_60", "relative__ema_distance_20",
    "relative__trend_efficiency_20", "relative__momentum_acceleration_5_20",
    "corr_20_to_universe", "corr_60_to_universe",
    "universe_dispersion_return_20", "universe_dispersion_vol_20",
    "candidate_rank_return_20", "candidate_rank_return_60",
    "market_return_20", "market_vol_20",
]

_ORIGINAL_MODEL_UTILITIES = rotation._model_utilities
_ORIGINAL_REPLAY = discovery._run_rotation_replay
_ACCELERATOR_THREAD = threading.local()


def _log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small point-in-time counterfactual experiment for contextual marginal asset value."
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--seed-assets", nargs="+", required=True)
    parser.add_argument("--candidate-symbols", nargs="+", required=True)
    parser.add_argument("--decision-start", required=True)
    parser.add_argument("--decision-end", required=True)
    parser.add_argument("--decision-count", type=int, default=20)
    parser.add_argument("--horizon-sessions", type=int, default=40)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--market-proxy", default="SPY")
    parser.add_argument("--data-source", choices=("mongo", "yahoo"), default="mongo")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
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


def normalize_symbols(values: list[str] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        symbol = str(value).strip().upper()
        if symbol and symbol not in output:
            output.append(symbol)
    return output


def _assert_local_mongo(uri: str, allow_remote: bool) -> None:
    if allow_remote:
        return
    parsed = parse_uri(uri)
    hosts = {str(host).strip().lower() for host, _ in parsed.get("nodelist") or []}
    if hosts and hosts.issubset(LOCAL_MONGO_HOSTS):
        return
    raise RuntimeError(
        "This research keeps Strategy/configuration access on local MongoDB. "
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


def _market_identity(configuration: dict[str, Any]) -> dict[str, str]:
    return {
        "interval": str(configuration.get("timeframe") or "1Day"),
        "feed": str(configuration.get("alpaca_historical_feed") or "sip"),
        "adjustment": str(configuration.get("alpaca_adjustment") or "all"),
    }


def _expected_sessions(start_date: pd.Timestamp, snapshot_end: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(calendar.date_to_session(start_date, direction="next"))
    last = pd.Timestamp(calendar.date_to_session(snapshot_end, direction="previous"))
    return _utc_index(calendar.sessions_in_range(first, last)).normalize()


def _load_mongo_frames(
    collection: Any,
    assets: list[str],
    identity: dict[str, str],
    start_date: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    start = _utc_timestamp(start_date).to_pydatetime()
    end = _utc_timestamp(snapshot_end + pd.Timedelta(days=1)).to_pydatetime()
    query = {"symbol": {"$in": assets}, **identity, "timestamp": {"$gte": start, "$lt": end}}
    projection = {
        "_id": 0, "symbol": 1, "timestamp": 1, "open": 1, "high": 1,
        "low": 1, "close": 1, "volume": 1, "vwap": 1, "trade_count": 1,
    }
    rows = list(collection.find(query, projection).sort([("symbol", 1), ("timestamp", 1)]))
    if not rows:
        raise RuntimeError("Local MongoDB returned no market bars for the selected symbols.")
    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        frames[str(symbol)] = frame[~frame.index.duplicated(keep="last")]
    absent = [symbol for symbol in assets if symbol not in frames]
    if absent:
        raise RuntimeError("No in-window Mongo bars were loaded for: " + ", ".join(absent))
    return frames


def _load_yahoo_frames(
    assets: list[str],
    start_date: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Yahoo source requires yfinance. Install the research dependency with: pip install yfinance"
        ) from exc

    frames: dict[str, pd.DataFrame] = {}
    end_exclusive = (snapshot_end + pd.Timedelta(days=1)).date().isoformat()
    for symbol in assets:
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
        renamed = data.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise RuntimeError(f"Yahoo history for {symbol} is missing columns: {missing}")
        frame = renamed.loc[:, required].copy()
        frame.index = pd.to_datetime(frame.index, utc=True).normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frames[symbol] = frame
    return frames


def _validate_complete_history(
    frames: dict[str, pd.DataFrame],
    assets: list[str],
    expected: pd.DatetimeIndex,
) -> None:
    expected_utc = _utc_index(expected).normalize()
    failures: list[str] = []
    for symbol in assets:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            failures.append(f"{symbol}: no history")
            continue
        observed = _utc_index(frame.index).normalize().unique()
        missing = expected_utc.difference(observed)
        if len(missing):
            sample = ",".join(item.date().isoformat() for item in missing[:5])
            failures.append(f"{symbol}: missing={len(missing)} sample={sample}")
    if failures:
        raise RuntimeError("Complete-history validation failed:\n- " + "\n- ".join(failures))


def _horizon_end(
    sessions: pd.DatetimeIndex,
    decision_session: pd.Timestamp,
    horizon_sessions: int,
) -> pd.Timestamp:
    ordered = _utc_index(sessions)
    decision = _utc_timestamp(decision_session)
    position = int(ordered.searchsorted(decision, side="left"))
    if position >= len(ordered) or pd.Timestamp(ordered[position]) != decision:
        raise RuntimeError(f"Decision session is not in the expected calendar: {decision.date()}")
    end_position = position + max(1, int(horizon_sessions)) - 1
    if end_position >= len(ordered):
        raise RuntimeError(f"Horizon exceeds snapshot for decision session {decision.date()}")
    return pd.Timestamp(ordered[end_position])


def _window_is_executable(
    common_dates: pd.DatetimeIndex,
    decision_session: pd.Timestamp,
    horizon_end: pd.Timestamp,
    config: BacktestRequest,
) -> bool:
    ordered = _utc_index(common_dates)
    decision = _utc_timestamp(decision_session)
    horizon = _utc_timestamp(horizon_end)
    truncated = ordered[ordered <= horizon]
    if len(truncated) < 2:
        return False
    try:
        folds = rotation._build_walk_forward_folds(truncated, config)
    except ValueError:
        return False
    start = max(int(folds[0]["test_start_index"]), int(truncated.searchsorted(decision, side="left")))
    end = min(int(folds[-1]["test_end_index"]), int(truncated.searchsorted(horizon, side="right")))
    return start < end


def select_executable_decision_sessions(
    common_dates: pd.DatetimeIndex,
    calendar_sessions: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    count: int,
    horizon_sessions: int,
    config: BacktestRequest,
) -> list[pd.Timestamp]:
    ordered = _utc_index(common_dates)
    start_utc = _utc_timestamp(start)
    end_utc = _utc_timestamp(end)
    monthly: list[pd.Timestamp] = []
    seen: set[tuple[int, int]] = set()
    for session in ordered:
        ts = pd.Timestamp(session)
        if ts < start_utc or ts > end_utc:
            continue
        key = (ts.year, ts.month)
        if key in seen:
            continue
        try:
            horizon_end = _horizon_end(calendar_sessions, ts, horizon_sessions)
        except RuntimeError:
            continue
        if not _window_is_executable(ordered, ts, horizon_end, config):
            continue
        seen.add(key)
        monthly.append(ts)
    if not monthly:
        raise RuntimeError(
            "No executable point-in-time windows exist for the requested range under the locked walk-forward protocol."
        )
    count = max(1, int(count))
    if len(monthly) <= count:
        return monthly
    indexes = []
    for value in np.linspace(0, len(monthly) - 1, num=count):
        index = int(round(float(value)))
        if index not in indexes:
            indexes.append(index)
    for index in range(len(monthly)):
        if len(indexes) >= count:
            break
        if index not in indexes:
            indexes.append(index)
    return [monthly[index] for index in sorted(indexes[:count])]


def _asof_rotation_row(frame: pd.DataFrame, decision: pd.Timestamp) -> pd.Series:
    decision = _utc_timestamp(decision)
    index = _utc_index(frame.index)
    normalized = frame.copy()
    normalized.index = index
    eligible = normalized.loc[normalized.index <= decision]
    if eligible.empty:
        raise RuntimeError(f"No point-in-time feature row is available at {decision.date()}")
    return eligible.iloc[-1]


def _equal_weight_return_series(
    raw_frames: dict[str, pd.DataFrame], symbols: list[str], decision: pd.Timestamp
) -> pd.Series:
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = raw_frames[symbol].copy()
        frame.index = _utc_index(frame.index)
        close = pd.to_numeric(frame.loc[frame.index <= decision, "close"], errors="coerce")
        columns[symbol] = close.pct_change()
    return pd.DataFrame(columns).mean(axis=1, skipna=True)


def _candidate_correlation(
    raw_frames: dict[str, pd.DataFrame], seed_assets: list[str], candidate: str,
    decision: pd.Timestamp, window: int,
) -> float:
    market = _equal_weight_return_series(raw_frames, seed_assets, decision)
    frame = raw_frames[candidate].copy()
    frame.index = _utc_index(frame.index)
    candidate_return = pd.to_numeric(frame.loc[frame.index <= decision, "close"], errors="coerce").pct_change()
    aligned = pd.concat([candidate_return.rename("candidate"), market.rename("universe")], axis=1).dropna().tail(window)
    if len(aligned) < max(10, window // 2):
        return float("nan")
    return float(aligned["candidate"].corr(aligned["universe"]))


def build_feature_snapshot(
    *, candidate: str, seed_assets: list[str], rotation_frames: dict[str, pd.DataFrame],
    raw_frames: dict[str, pd.DataFrame], decision: pd.Timestamp, market_proxy: str,
) -> dict[str, float]:
    candidate_row = _asof_rotation_row(rotation_frames[candidate], decision)
    universe_rows = pd.DataFrame({
        symbol: _asof_rotation_row(rotation_frames[symbol], decision) for symbol in seed_assets
    }).T
    result: dict[str, float] = {}
    for feature in BASE_FEATURES:
        if feature not in rotation.ROTATION_FEATURES:
            raise RuntimeError(f"Feature is not backward-looking ROTATION_FEATURES input: {feature}")
        candidate_value = float(candidate_row.get(feature, float("nan")))
        universe_mean = float(pd.to_numeric(universe_rows[feature], errors="coerce").mean())
        result[f"relative__{feature}"] = candidate_value - universe_mean
    r20 = pd.concat([
        pd.to_numeric(universe_rows["return_20"], errors="coerce"),
        pd.Series({candidate: float(candidate_row.get("return_20", float("nan")))}, dtype=float),
    ])
    r60 = pd.concat([
        pd.to_numeric(universe_rows["return_60"], errors="coerce"),
        pd.Series({candidate: float(candidate_row.get("return_60", float("nan")))}, dtype=float),
    ])
    result["candidate_rank_return_20"] = float(r20.rank(pct=True).get(candidate, float("nan")))
    result["candidate_rank_return_60"] = float(r60.rank(pct=True).get(candidate, float("nan")))
    result["universe_dispersion_return_20"] = float(pd.to_numeric(universe_rows["return_20"], errors="coerce").std())
    result["universe_dispersion_vol_20"] = float(pd.to_numeric(universe_rows["vol_20"], errors="coerce").std())
    result["corr_20_to_universe"] = _candidate_correlation(raw_frames, seed_assets, candidate, decision, 20)
    result["corr_60_to_universe"] = _candidate_correlation(raw_frames, seed_assets, candidate, decision, 60)
    proxy = str(market_proxy).strip().upper()
    if proxy in rotation_frames:
        proxy_row = _asof_rotation_row(rotation_frames[proxy], decision)
        result["market_return_20"] = float(proxy_row.get("return_20", float("nan")))
        result["market_vol_20"] = float(proxy_row.get("vol_20", float("nan")))
    else:
        result["market_return_20"] = float(pd.to_numeric(universe_rows["return_20"], errors="coerce").mean())
        result["market_vol_20"] = float(pd.to_numeric(universe_rows["vol_20"], errors="coerce").mean())
    return result


def _accelerated_model_utilities(
    models: dict[str, Any], frames: dict[str, pd.DataFrame], symbols: list[str],
    timestamp: pd.Timestamp, config: Any,
) -> np.ndarray:
    state = getattr(_ACCELERATOR_THREAD, "state", None)
    if state is None:
        return _ORIGINAL_MODEL_UTILITIES(models, frames, symbols, timestamp, config)
    key = (id(frames), tuple((symbol, id(models.get(symbol))) for symbol in symbols))
    cached = state.get(key)
    if cached is None:
        cached = {}
        for symbol in symbols:
            model = models.get(symbol)
            frame = frames.get(symbol)
            if model is None or frame is None or frame.empty:
                cached[symbol] = pd.Series(dtype=float)
                continue
            features = frame.loc[:, rotation.ROTATION_FEATURES]
            valid = ~features.isna().any(axis=1)
            if not bool(valid.any()):
                cached[symbol] = pd.Series(dtype=float)
                continue
            batch = features.loc[valid]
            cached[symbol] = pd.Series(np.asarray(model.predict(batch), dtype=float), index=batch.index)
        state[key] = cached
    ts = pd.Timestamp(timestamp)
    values = [0.0]
    for symbol in symbols:
        value = cached[symbol].get(ts, np.nan)
        values.append(float(value) if pd.notna(value) and np.isfinite(float(value)) else float("-inf"))
    return np.asarray(values, dtype=np.float64)


def _accelerated_replay(*args: Any, **kwargs: Any):
    _ACCELERATOR_THREAD.state = {}
    try:
        return _ORIGINAL_REPLAY(*args, **kwargs)
    finally:
        _ACCELERATOR_THREAD.state = None


def install_accelerator() -> None:
    rotation._model_utilities = _accelerated_model_utilities
    discovery._run_rotation_replay = _accelerated_replay


def _window_request(
    *, db: Any, config: BacktestRequest, strategy_id: str, assets: list[str],
    reference_assets: list[str], candidate_assets: list[str],
    decision_session: pd.Timestamp, horizon_end: pd.Timestamp,
):
    return discovery._marginal_execution_request(
        db, config, {"id": strategy_id}, config, horizon_end.date().isoformat(),
        assets=assets, reference_assets=reference_assets, candidate_assets=candidate_assets,
        analysis_start_date=decision_session.date().isoformat(),
        analysis_end_date=horizon_end.date().isoformat(),
    )


def _evaluate_candidate_window(
    *, db: Any, config: BacktestRequest, strategy_id: str, seed_assets: list[str],
    seed_frames: dict[str, pd.DataFrame], baseline_metrics: dict[str, Any],
    baseline_sessions: pd.DatetimeIndex, candidate: str, candidate_frame: pd.DataFrame,
    decision_session: pd.Timestamp, horizon_end: pd.Timestamp,
    feature_snapshot: dict[str, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "decision_date": decision_session.date().isoformat(),
        "horizon_end": horizon_end.date().isoformat(),
        "candidate": candidate, "evaluation_status": "running", **feature_snapshot,
    }
    try:
        assets = [*seed_assets, candidate]
        frames = dict(seed_frames)
        frames[candidate] = candidate_frame
        request = _window_request(
            db=db, config=config, strategy_id=strategy_id, assets=assets,
            reference_assets=seed_assets, candidate_assets=[candidate],
            decision_session=decision_session, horizon_end=horizon_end,
        )
        metrics, sessions = discovery._run_rotation_replay(frames, request)
        context = discovery._research_context_compatibility(baseline_sessions, sessions)
        row.update(context)
        if not bool(context.get("research_context_compatible")):
            row.update({"evaluation_status": "context_rejected", "error": "research_context_incomplete"})
        else:
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            candidate_capital = discovery._finite_number(metrics.get("ending_capital"))
            if baseline_capital is None or candidate_capital is None or baseline_capital <= 0 or candidate_capital <= 0:
                raise RuntimeError("Exact window returned non-positive/invalid ending capital.")
            delta_rate = discovery._capital_delta_rate(candidate_capital, baseline_capital)
            row.update({
                "evaluation_status": "completed",
                "baseline_ending_capital": baseline_capital,
                "candidate_ending_capital": candidate_capital,
                "ending_capital_delta_rate": delta_rate,
                "delta_log_capital": float(math.log(candidate_capital / baseline_capital)),
                "candidate_sharpe": metrics.get("sharpe"),
                "candidate_maximum_drawdown": metrics.get("maximum_drawdown"),
            })
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700]})
    row["total_seconds"] = float(time.perf_counter() - started)
    return row


def feature_report(dataset: pd.DataFrame) -> pd.DataFrame:
    completed = dataset.loc[dataset["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    y = pd.to_numeric(completed["delta_log_capital"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        x = pd.to_numeric(completed[feature], errors="coerce")
        pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(pair) < 3:
            rows.append({"feature": feature, "n": len(pair), "pearson": None, "spearman": None})
            continue
        median = float(pair["x"].median())
        low = pair.loc[pair["x"] <= median, "y"]
        high = pair.loc[pair["x"] > median, "y"]
        rows.append({
            "feature": feature, "n": int(len(pair)),
            "pearson": float(pair["x"].corr(pair["y"], method="pearson")),
            "spearman": float(pair["x"].corr(pair["y"], method="spearman")),
            "positive_rate_low": float((low > 0).mean()) if len(low) else None,
            "positive_rate_high": float((high > 0).mean()) if len(high) else None,
            "mean_delta_log_low": float(low.mean()) if len(low) else None,
            "mean_delta_log_high": float(high.mean()) if len(high) else None,
        })
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False, na_position="last")


def temporal_ols_validation(
    dataset: pd.DataFrame, validation_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    completed = dataset.loc[dataset["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    completed["decision_ts"] = pd.to_datetime(completed["decision_date"], utc=True)
    boundary = _utc_timestamp(validation_start)
    train = completed.loc[completed["decision_ts"] < boundary].copy()
    test = completed.loc[completed["decision_ts"] >= boundary].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_temporal_split", "train_rows": int(len(train)), "test_rows": int(len(test))
        }
    train_x = train.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    test_x = test.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0)
    train_x, test_x = train_x.fillna(medians), test_x.fillna(medians)
    means, stds = train_x.mean(axis=0), train_x.std(axis=0, ddof=0)
    active = [f for f in MODEL_FEATURES if np.isfinite(stds[f]) and float(stds[f]) > 1e-12]
    x_train = ((train_x[active] - means[active]) / stds[active]).to_numpy(dtype=float)
    x_test = ((test_x[active] - means[active]) / stds[active]).to_numpy(dtype=float)
    y_train = pd.to_numeric(train["delta_log_capital"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test["delta_log_capital"], errors="coerce").to_numpy(dtype=float)
    design_train = np.column_stack([np.ones(len(x_train)), x_train])
    design_test = np.column_stack([np.ones(len(x_test)), x_test])
    beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    prediction = design_test @ beta
    predictions = test[["decision_date", "candidate", "ending_capital_delta_rate", "delta_log_capital"]].copy()
    predictions["predicted_delta_log_capital"] = prediction
    predictions["predicted_positive"] = prediction > 0
    predictions["actual_positive"] = y_test > 0
    coefficients = pd.DataFrame({"feature": ["intercept", *active], "standardized_ols_coefficient": beta})
    per_date = []
    for decision_date, group in predictions.groupby("decision_date", sort=True):
        chosen = group.sort_values("predicted_delta_log_capital", ascending=False).iloc[0]
        oracle = group.sort_values("delta_log_capital", ascending=False).iloc[0]
        per_date.append({
            "decision_date": decision_date,
            "chosen_actual_delta_log": float(chosen["delta_log_capital"]),
            "mean_candidate_delta_log": float(group["delta_log_capital"].mean()),
            "oracle_delta_log": float(oracle["delta_log_capital"]),
            "rank_spearman": float(group["predicted_delta_log_capital"].corr(group["delta_log_capital"], method="spearman")),
        })
    per_date_frame = pd.DataFrame(per_date)
    summary = {
        "status": "completed",
        "validation_start": boundary.date().isoformat(),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_dates": int(train["decision_date"].nunique()), "test_dates": int(test["decision_date"].nunique()),
        "mae_delta_log_capital": float(np.mean(np.abs(prediction - y_test))),
        "pearson_prediction_vs_actual": float(pd.Series(prediction).corr(pd.Series(y_test), method="pearson")),
        "spearman_prediction_vs_actual": float(pd.Series(prediction).corr(pd.Series(y_test), method="spearman")),
        "sign_accuracy": float(np.mean((prediction > 0) == (y_test > 0))),
        "majority_sign_baseline_accuracy": float(np.mean((y_test > 0) == bool(np.mean(y_train > 0) >= 0.5))),
        "mean_cross_sectional_rank_spearman": float(per_date_frame["rank_spearman"].mean()),
        "top1_positive_rate": float((per_date_frame["chosen_actual_delta_log"] > 0).mean()),
        "mean_top1_actual_delta_log": float(per_date_frame["chosen_actual_delta_log"].mean()),
        "mean_candidate_delta_log": float(per_date_frame["mean_candidate_delta_log"].mean()),
        "mean_oracle_delta_log": float(per_date_frame["oracle_delta_log"].mean()),
    }
    return predictions, coefficients, summary


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

    workers = max(1, int(args.workers or 1))
    seed_assets = normalize_symbols(args.seed_assets)
    candidates = normalize_symbols(args.candidate_symbols)
    overlap = sorted(set(seed_assets).intersection(candidates))
    if overlap:
        raise RuntimeError(f"Candidates must be outside the seed universe: {overlap}")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, retryWrites=False)
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = _strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored_configuration = _configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        strategy_assets = normalize_symbols(list(stored_configuration.get("assets") or []))
        requested_symbols = [*seed_assets, *candidates]
        missing = sorted(set(requested_symbols).difference(strategy_assets))
        if missing:
            raise RuntimeError(f"Requested symbols are not present in Strategy #{strategy_sequence}: {missing}")

        history_start = _normalize_date(args.history_start)
        configured_start = _normalize_date(stored_configuration.get("start_date"))
        if history_start != configured_start:
            raise RuntimeError(
                f"--history-start must match Strategy start date: strategy={configured_start.date()}, requested={history_start.date()}"
            )
        snapshot_end = _normalize_date(args.snapshot_end)
        decision_start = _normalize_date(args.decision_start)
        decision_end = _normalize_date(args.decision_end)
        validation_start = _normalize_date(args.validation_start)
        horizon_sessions = max(1, int(args.horizon_sessions))

        config = BacktestRequest.model_validate(stored_configuration).model_copy(
            update={"assets": list(seed_assets), "end_date": snapshot_end.date().isoformat()}
        )
        expected_sessions = _expected_sessions(history_start, snapshot_end)
        if args.data_source == "yahoo":
            raw_frames = _load_yahoo_frames(requested_symbols, history_start, snapshot_end)
            source_detail = "Yahoo Finance via yfinance, auto_adjust=True"
        else:
            collection = db[ALPACA_MARKET_BARS_COLLECTION]
            raw_frames = _load_mongo_frames(
                collection, requested_symbols, _market_identity(stored_configuration), history_start, snapshot_end
            )
            source_detail = "frozen local MongoDB market-bar snapshot; no live Alpaca request"
        _validate_complete_history(raw_frames, requested_symbols, expected_sessions)

        cleaned_frames = {symbol: validate_and_clean_bars(frame.copy(), config) for symbol, frame in raw_frames.items()}
        seed_frames = {symbol: cleaned_frames[symbol] for symbol in seed_assets}
        rotation_frames = {
            symbol: rotation.build_rotation_frame(frame.copy(), config) for symbol, frame in cleaned_frames.items()
        }
        _, common_dates = rotation.prepare_rotation_panel(seed_frames, config)
        decision_sessions = select_executable_decision_sessions(
            common_dates, expected_sessions, decision_start, decision_end,
            int(args.decision_count), horizon_sessions, config,
        )

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT / "research_output" / f"contextual_marginal_signature_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        install_accelerator()

        manifest = {
            "schema_version": 1, "script_version": SCRIPT_VERSION, "api_version": API_VERSION,
            "experiment": EXPERIMENT_NAME, "strategy_id": strategy_id, "strategy_sequence": strategy_sequence,
            "history_start": history_start.date().isoformat(), "snapshot_end": snapshot_end.date().isoformat(),
            "seed_assets": seed_assets, "candidate_symbols": candidates,
            "requested_decision_start": decision_start.date().isoformat(),
            "requested_decision_end": decision_end.date().isoformat(),
            "selected_decision_dates": [item.date().isoformat() for item in decision_sessions],
            "horizon_sessions": horizon_sessions, "validation_start": validation_start.date().isoformat(),
            "research_data_source": args.data_source, "research_data_source_detail": source_detail,
            "processor_policy": "standalone versioned processor; no import from earlier research processor files",
            "decision_window_policy": "sample only windows executable under the locked Strategy walk-forward protocol",
        }
        _write_json(output_dir / "contextual_signature_manifest.json", manifest)
        _log(
            f"Contextual Marginal Signature v1.0.3: source={args.data_source}, seed={len(seed_assets)}, "
            f"candidates={len(candidates)}, executable_dates={len(decision_sessions)}."
        )
        _log(f"Selected range: {decision_sessions[0].date()} -> {decision_sessions[-1].date()}.")

        all_rows: list[dict[str, Any]] = []
        abort_reason: str | None = None
        for index, decision_session in enumerate(decision_sessions, start=1):
            horizon_end = _horizon_end(expected_sessions, decision_session, horizon_sessions)
            _log(f"Decision {index}/{len(decision_sessions)}: {decision_session.date()} -> {horizon_end.date()}")
            baseline_request = _window_request(
                db=db, config=config, strategy_id=strategy_id, assets=seed_assets,
                reference_assets=seed_assets, candidate_assets=[], decision_session=decision_session,
                horizon_end=horizon_end,
            )
            baseline_metrics, baseline_sessions = discovery._run_rotation_replay(seed_frames, baseline_request)
            snapshots = {
                candidate: build_feature_snapshot(
                    candidate=candidate, seed_assets=seed_assets, rotation_frames=rotation_frames,
                    raw_frames=cleaned_frames, decision=decision_session, market_proxy=args.market_proxy,
                ) for candidate in candidates
            }
            date_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
                futures = {
                    executor.submit(
                        _evaluate_candidate_window, db=db, config=config, strategy_id=strategy_id,
                        seed_assets=seed_assets, seed_frames=seed_frames, baseline_metrics=baseline_metrics,
                        baseline_sessions=baseline_sessions, candidate=candidate,
                        candidate_frame=cleaned_frames[candidate], decision_session=decision_session,
                        horizon_end=horizon_end, feature_snapshot=snapshots[candidate],
                    ): candidate for candidate in candidates
                }
                for future in as_completed(futures):
                    row = future.result()
                    date_rows.append(row)
                    all_rows.append(row)
                    delta = row.get("ending_capital_delta_rate")
                    delta_text = "n/a" if delta is None or pd.isna(delta) else f"{float(delta):+.4%}"
                    _log(f"{row['decision_date']} {row['candidate']}: {row['evaluation_status']} ΔCapital={delta_text}")
                    _write_frame(output_dir / "contextual_signature_dataset.csv", pd.DataFrame(all_rows))
            incomplete = [row for row in date_rows if row.get("evaluation_status") != "completed"]
            if incomplete:
                abort_reason = "ContextualSignatureDateIncomplete: " + ", ".join(
                    f"{row.get('candidate')}={row.get('evaluation_status')}:{row.get('error') or ''}" for row in incomplete
                )
                _log(abort_reason)
                break

        dataset = pd.DataFrame(all_rows)
        _write_frame(output_dir / "contextual_signature_dataset.csv", dataset)
        report = feature_report(dataset) if not dataset.empty else pd.DataFrame()
        _write_frame(output_dir / "contextual_signature_feature_report.csv", report)
        if not dataset.empty:
            predictions, coefficients, validation = temporal_ols_validation(dataset, validation_start)
        else:
            predictions, coefficients, validation = pd.DataFrame(), pd.DataFrame(), {"status": "empty_dataset"}
        _write_frame(output_dir / "contextual_signature_validation_predictions.csv", predictions)
        _write_frame(output_dir / "contextual_signature_ols_coefficients.csv", coefficients)
        completed = dataset.loc[dataset["evaluation_status"] == "completed"] if not dataset.empty else pd.DataFrame()
        summary = {
            "schema_version": 1, "script_version": SCRIPT_VERSION,
            "status": "aborted" if abort_reason else "completed", "abort_reason": abort_reason,
            "research_data_source": args.data_source, "rows": int(len(dataset)),
            "completed_rows": int(len(completed)),
            "positive_label_rate": float((completed["delta_log_capital"] > 0).mean()) if not completed.empty else None,
            "top_absolute_spearman_features": report[["feature", "spearman"]].head(8).to_dict(orient="records") if not report.empty else [],
            "temporal_ols_validation": validation,
            "interpretation_rule": "Only future-held-out sign/rank stability counts as evidence; in-sample correlation alone does not.",
        }
        _write_json(output_dir / "contextual_signature_summary.json", summary)
        _log(f"Completed with status={summary['status']}. Outputs: {output_dir}")
        return 2 if abort_reason else 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
