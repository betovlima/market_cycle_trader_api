from __future__ import annotations

import os
import random
import re
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
import exchange_calendars as xcals
from pymongo.database import Database

from ..core.config import API_VERSION
from ..schemas.requests import BacktestExecutionRequest, BacktestRequest
from ..engine.market_data import (
    _download_alpaca_bars,
    _market_data_identity,
    _upsert_frame,
    complete_market_history,
    latest_safe_completed_xnys_session,
    load_market_bars,
    validate_and_clean_bars,
)
from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    ASSET_DISCOVERY_CATALOG_COLLECTION,
    ASSET_DISCOVERY_RESEARCH_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    bson_value,
    get_alpaca_credentials,
)
from .asset_discovery_ranker import FEATURE_COLUMNS, latest_feature_snapshot, market_quality, train_ranker
from ..engine.capital_rotation import run_rotation_models
from ..engine.compound_rotation_backtest import apply_slippage, calculate_reference_fees
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings
from .strategy_lab import (
    _configuration_hash,
    create_strategy,
    get_research_strategy_context,
    get_strategy,
    get_strategy_model_snapshot,
    get_trader_winner_context,
)

COLLECTION = ASSET_DISCOVERY_RESEARCH_COLLECTION
CATALOG_COLLECTION = ASSET_DISCOVERY_CATALOG_COLLECTION
CURRENT_ID = "current"
BATCH_SIZE = 8
CANDIDATE_HISTORY_DAYS = 900
SUPPORTED_EXCHANGES = frozenset({"AMEX", "ARCA", "BATS", "NASDAQ", "NYSE"})
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
ACTIVE_STATUSES = frozenset({"queued", "running", "stopping"})
TICKER_IDENTITY_GAP_SESSIONS = 20

_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None


class AssetDiscoveryConflict(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trading_base_url() -> str:
    return str(os.getenv("ALPACA_TRADING_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")


def _alpaca_headers(credentials: dict[str, str]) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": credentials["api_key_id"],
        "APCA-API-SECRET-KEY": credentials["secret_key"],
    }


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result.pop("_id", None)
    return result


def _campaign(db: Database) -> dict[str, Any] | None:
    return db[COLLECTION].find_one({"_id": CURRENT_ID})


def _worker_alive() -> bool:
    with _worker_lock:
        return bool(_worker_thread and _worker_thread.is_alive())


def _event(db: Database, run_id: str, message: str, *, phase: str | None = None, changes: dict[str, Any] | None = None) -> None:
    update: dict[str, Any] = {
        "$set": {"updated_at": utc_now()},
        "$push": {
            "events": {
                "$each": [{"at": utc_now(), "message": str(message)[:500]}],
                "$slice": -24,
            }
        },
    }
    if phase:
        update["$set"]["phase"] = phase
    if changes:
        update["$set"].update(bson_value(changes))
    db[COLLECTION].update_one({"_id": CURRENT_ID, "run_id": run_id}, update)


def _increment(db: Database, run_id: str, values: dict[str, int]) -> None:
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {"$inc": {key: int(value) for key, value in values.items()}, "$set": {"updated_at": utc_now()}},
    )


def _reject(db: Database, run_id: str, reason: str) -> None:
    safe_reason = str(reason or "unknown").strip().lower().replace(".", "_").replace("$", "_")[:80]
    _increment(db, run_id, {"rejected_count": 1, f"rejection_summary.{safe_reason}": 1})


def _discover_symbols(db: Database) -> list[str]:
    credentials = get_alpaca_credentials(db)
    response = requests.get(
        f"{_trading_base_url()}/v2/assets",
        params={"status": "active", "asset_class": "us_equity"},
        headers=_alpaca_headers(credentials),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Alpaca returned an unexpected US-equity universe payload.")
    result: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        exchange = str(item.get("exchange") or "").strip().upper()
        if not symbol or not SYMBOL_PATTERN.fullmatch(symbol):
            continue
        if exchange not in SUPPORTED_EXCHANGES or not bool(item.get("tradable")):
            continue
        result.append(symbol)
    return sorted(set(result))


def _baseline_frames(config: Any, end_session: str) -> dict[str, pd.DataFrame]:
    research_config = config.model_copy(
        update={
            "end_date": end_session,
            "mongo_cache_enabled": True,
            "market_data_history_backfill_enabled": False,
        }
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in config.assets:
        frame = load_market_bars(str(symbol), research_config)
        if frame is None or frame.empty:
            continue
        frames[str(symbol).upper()] = frame
    if len(frames) < 3:
        raise RuntimeError("Asset Discovery requires at least three baseline assets with usable historical data.")
    return frames




def _normalized_sessions(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DatetimeIndex([])
    values = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("UTC").tz_localize(None).normalize()
    return pd.DatetimeIndex(values.unique()).sort_values()


def _required_xnys_sessions(config: Any, end_session: str | pd.Timestamp) -> pd.DatetimeIndex:
    requested = pd.Timestamp(config.start_date)
    if requested.tzinfo is not None:
        requested = requested.tz_convert("UTC").tz_localize(None)
    requested = requested.normalize()
    end = pd.Timestamp(end_session)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    end = end.normalize()
    calendar = xcals.get_calendar("XNYS")
    first_session = pd.Timestamp(calendar.date_to_session(requested, direction="next"))
    last_session = pd.Timestamp(calendar.date_to_session(end, direction="previous"))
    sessions = pd.DatetimeIndex(calendar.sessions_in_range(first_session, last_session))
    if sessions.empty:
        raise RuntimeError("Asset Discovery could not derive the XNYS research session calendar.")
    if sessions.tz is not None:
        sessions = sessions.tz_convert("UTC").tz_localize(None)
    return sessions.normalize().unique().sort_values()


def _baseline_required_sessions(
    frames: dict[str, pd.DataFrame],
    config: Any,
    end_session: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    required = _required_xnys_sessions(config, end_session)
    for symbol, frame in frames.items():
        try:
            _history_coverage_against_baseline(symbol, frame, config, required)
        except RuntimeError as exc:
            reason = str(exc).strip().lower()
            if reason == "ticker_identity_discontinuity":
                raise RuntimeError(
                    f"Baseline asset {symbol} contains a long internal market-history gap consistent with a ticker identity change."
                ) from exc
            if reason == "discontinuous_history":
                raise RuntimeError(
                    f"Baseline asset {symbol} does not cover every XNYS session required by the research window."
                ) from exc
            if reason == "insufficient_history":
                raise RuntimeError(
                    f"Baseline asset {symbol} does not cover the full historical start required by the research window."
                ) from exc
            raise
    return required


def _history_gap_diagnostics(
    actual: pd.DatetimeIndex,
    required: pd.DatetimeIndex,
) -> dict[str, int]:
    required_values = list(pd.DatetimeIndex(required).tz_localize(None).normalize().unique().sort_values())
    actual_set = set(pd.DatetimeIndex(actual).tz_localize(None).normalize().unique())
    present_positions = [index for index, session in enumerate(required_values) if session in actual_set]
    missing_positions = [index for index, session in enumerate(required_values) if session not in actual_set]
    if not present_positions:
        return {
            "missing_count": len(required_values),
            "internal_missing_count": 0,
            "longest_internal_missing_run": 0,
        }

    first_present = min(present_positions)
    last_present = max(present_positions)
    internal = [index for index in missing_positions if first_present < index < last_present]
    longest = 0
    current = 0
    previous = None
    for index in internal:
        if previous is not None and index == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = index
    return {
        "missing_count": len(missing_positions),
        "internal_missing_count": len(internal),
        "longest_internal_missing_run": longest,
    }


def _history_coverage_against_baseline(
    symbol: str,
    frame: pd.DataFrame,
    config: Any,
    required_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("insufficient_history")

    requested = pd.Timestamp(config.start_date)
    requested = requested.tz_localize("UTC") if requested.tzinfo is None else requested.tz_convert("UTC")
    tolerance_days = int(getattr(config, "market_data_history_start_tolerance_days", 0) or 0)
    latest_allowed = requested.normalize() + pd.Timedelta(days=tolerance_days)
    actual = _normalized_sessions(frame)
    if actual.empty:
        raise RuntimeError("insufficient_history")
    first = pd.Timestamp(actual.min()).tz_localize("UTC")
    last = pd.Timestamp(actual.max()).tz_localize("UTC")
    if first.normalize() > latest_allowed:
        raise RuntimeError("insufficient_history")

    required = pd.DatetimeIndex(required_sessions).tz_localize(None).normalize().unique().sort_values()
    gap = _history_gap_diagnostics(actual, required)
    if gap["missing_count"]:
        if gap["longest_internal_missing_run"] >= TICKER_IDENTITY_GAP_SESSIONS:
            raise RuntimeError("ticker_identity_discontinuity")
        raise RuntimeError("discontinuous_history")

    return {
        "history_window_complete": True,
        "history_required_start": requested.date().isoformat(),
        "history_required_end": pd.Timestamp(required[-1]).date().isoformat(),
        "history_start_tolerance_days": tolerance_days,
        "history_actual_start": first.isoformat(),
        "history_actual_end": last.isoformat(),
        "history_expected_sessions": int(len(required)),
        "history_observed_required_sessions": int(len(required) - gap["missing_count"]),
        "history_missing_required_sessions": int(gap["missing_count"]),
        "history_internal_missing_sessions": int(gap["internal_missing_count"]),
        "history_longest_internal_gap_sessions": int(gap["longest_internal_missing_run"]),
    }


def _candidate_history_coverage(
    db: Database,
    symbol: str,
    config: Any,
    end_session: str | pd.Timestamp,
    required_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the complete candidate window transiently against the baseline calendar.

    The full frame is downloaded only for the manually bounded external sample and is
    not persisted during Discovery.  A candidate must cover every research session
    already required by the selected baseline Strategy.
    """
    candidate_config = config.model_copy(
        update={
            "end_date": pd.Timestamp(end_session).date().isoformat(),
            "mongo_cache_enabled": False,
            "market_data_history_backfill_enabled": False,
        }
    )
    frame = _download_alpaca_bars(
        symbol,
        candidate_config,
        candidate_config.start_date,
        candidate_config.end_date,
    )
    coverage = _history_coverage_against_baseline(symbol, frame, candidate_config, required_sessions)
    return frame, coverage


def _persist_selected_asset_history(
    db: Database,
    symbol: str,
    config: Any,
    end_date: str | None,
    required_sessions: pd.DatetimeIndex,
) -> int:
    """Persist complete market history only for a user-selected Discovery asset."""
    selected_config = config.model_copy(
        update={
            "end_date": end_date or config.end_date,
            "mongo_cache_enabled": True,
            "market_data_history_backfill_enabled": False,
        }
    )
    downloaded = _download_alpaca_bars(
        symbol,
        selected_config,
        selected_config.start_date,
        end_date or selected_config.end_date,
    )
    if downloaded is None or downloaded.empty:
        raise AssetDiscoveryConflict(
            f"{symbol} does not provide the complete historical window required by the Strategy."
        )
    try:
        _history_coverage_against_baseline(symbol, downloaded, selected_config, required_sessions)
        complete_market_history(symbol, downloaded, selected_config, provider="alpaca")
    except RuntimeError as exc:
        reason = str(exc).strip().lower()
        if reason == "ticker_identity_discontinuity":
            raise AssetDiscoveryConflict(
                f"{symbol} contains a long internal historical gap consistent with ticker reuse and cannot be used for a comparable replay."
            ) from exc
        if reason == "discontinuous_history":
            raise AssetDiscoveryConflict(
                f"{symbol} does not cover every historical session required by the source Strategy and cannot be used for a comparable replay."
            ) from exc
        raise AssetDiscoveryConflict(str(exc)) from exc
    collection = db[ALPACA_MARKET_BARS_COLLECTION]
    identity = _market_data_identity(symbol, selected_config)
    collection.delete_many(identity)
    _upsert_frame(collection, downloaded, identity, selected_config.mongo_write_batch_size)

    try:
        persisted = load_market_bars(symbol, selected_config)
        persisted = validate_and_clean_bars(persisted, selected_config)
        _history_coverage_against_baseline(symbol, persisted, selected_config, required_sessions)
    except Exception as exc:
        collection.delete_many(identity)
        reason = str(exc).strip().lower()
        if "ticker_identity_discontinuity" in reason:
            raise AssetDiscoveryConflict(
                f"{symbol} contains a long internal historical gap consistent with ticker reuse and cannot be used in a comparable Strategy."
            ) from exc
        raise AssetDiscoveryConflict(
            f"{symbol} failed the persisted market-history integrity readback and cannot be used in a comparable Strategy: {str(exc)}"
        ) from exc
    return int(len(downloaded))

def _candidate_frame(db: Database, symbol: str, config: Any, end_session: pd.Timestamp) -> pd.DataFrame:
    credentials = get_alpaca_credentials(db)
    session = pd.Timestamp(end_session)
    end = session.tz_localize("UTC") if session.tzinfo is None else session.tz_convert("UTC")
    end = end + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=CANDIDATE_HISTORY_DAYS)
    return download_stock_bars(
        api_key_id=credentials["api_key_id"],
        secret_key=credentials["secret_key"],
        symbol=symbol,
        timeframe="1Day",
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=config.alpaca_historical_feed,
        adjustment=config.alpaca_adjustment,
    )


def _baseline_recent_returns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series: list[pd.Series] = []
    for symbol, frame in frames.items():
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        returns = close.pct_change().tail(90).rename(symbol)
        series.append(returns)
    return pd.concat(series, axis=1) if series else pd.DataFrame()


def _redundancy(candidate: pd.DataFrame, baseline_returns: pd.DataFrame) -> float | None:
    if baseline_returns.empty or candidate is None or candidate.empty:
        return None
    candidate_returns = pd.to_numeric(candidate.get("close"), errors="coerce").pct_change().rename("candidate")
    joined = baseline_returns.join(candidate_returns, how="inner").tail(60)
    if len(joined) < 30:
        return None
    correlations = joined.drop(columns=["candidate"]).corrwith(joined["candidate"]).abs().dropna()
    return float(correlations.max()) if not correlations.empty else None


def _score_candidate(bundle: Any, symbol: str, frame: pd.DataFrame, baseline_returns: pd.DataFrame) -> dict[str, Any]:
    quality = market_quality(frame)
    feature_row, feature_at = latest_feature_snapshot(frame)
    vector = pd.DataFrame([[float(feature_row[column]) for column in FEATURE_COLUMNS]], columns=list(FEATURE_COLUMNS))
    raw_score = float(bundle.model.predict(vector)[0])
    redundancy = _redundancy(frame, baseline_returns)
    return {
        "symbol": symbol,
        "raw_score": raw_score,
        "feature_at": feature_at.to_pydatetime(),
        "latest_close": quality["latest_close"],
        "median_dollar_volume": quality["median_dollar_volume"],
        "return_20": float(feature_row["return_20"]),
        "return_60": float(feature_row["return_60"]),
        "volatility_20": float(feature_row["volatility_20"]),
        "drawdown_60": float(feature_row["drawdown_60"]),
        "trend_efficiency_20": float(feature_row["trend_efficiency_20"]),
        "max_baseline_correlation_60": redundancy,
    }


def _rank_results(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: float(item["raw_score"]), reverse=True)
    count = len(ordered)
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
        row["rank_score"] = 1.0 if count <= 1 else float(1.0 - ((index - 1) / (count - 1)))
    return ordered[: max(1, min(limit, count))]



def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None


def _median_metric(results: list[Any], key: str) -> float | None:
    values = [_finite_number((result.metrics or {}).get(key)) for result in results]
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _aggregate_rotation_replay(results: list[Any]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Marginal Capital Replay produced no rotation result.")
    fold_returns: list[float] = []
    for result in results:
        folds = (result.metrics or {}).get("walk_forward_folds") or []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            value = _finite_number(fold.get("strategy_return"))
            if value is not None:
                fold_returns.append(value)
    return {
        "ending_capital": _median_metric(results, "strategy_ending_capital"),
        "cagr": _median_metric(results, "strategy_cagr"),
        "sharpe": _median_metric(results, "strategy_sharpe"),
        "maximum_drawdown": _median_metric(results, "strategy_maximum_drawdown"),
        "market_exposure": _median_metric(results, "market_exposure"),
        "cash_days": _median_metric(results, "cash_days"),
        "switches": _median_metric(results, "capital_rotations"),
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "repetition_count": len(results),
    }


def _marginal_execution_request(
    db: Database,
    base_config: BacktestRequest,
    strategy: dict[str, Any],
    winner_config: BacktestRequest,
    end_session: str,
    *,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
) -> BacktestExecutionRequest:
    snapshot = get_strategy_model_snapshot(db, str(strategy.get("id") or ""))
    family = str(snapshot.get("family") or "xgboost_utility")
    settings = dict(snapshot.get("settings_snapshot") or {}) if isinstance(snapshot.get("settings_snapshot"), dict) else {}
    locked = base_config.model_copy(update={"assets": assets, "end_date": end_session})
    locked = apply_training_runtime_settings(db, locked)
    locked = apply_execution_profile(locked, family, settings)
    selected_set = set(assets)
    anchors = [symbol for symbol in winner_config.assets if symbol in selected_set]
    if len(anchors) < 2:
        anchors = list(assets)
    return BacktestExecutionRequest.model_validate({
        **locked.model_dump(mode="python"),
        "analysis_start_date": locked.start_date,
        "analysis_end_date": end_session,
        "calendar_anchor_assets": anchors,
        "research_reference_assets": reference_assets,
        "research_candidate_assets": candidate_assets,
        "research_model_family": family,
        "research_model_settings": settings,
        "research_market_data_mode": "database_only",
    })


def _run_rotation_replay(
    frames: dict[str, pd.DataFrame],
    request: BacktestExecutionRequest,
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    cleaned: dict[str, pd.DataFrame] = {}
    for symbol in request.assets:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            raise RuntimeError(f"Marginal Capital Replay is missing market data for {symbol}.")
        cleaned[symbol] = validate_and_clean_bars(frame.copy(), request)
    results = run_rotation_models(
        cleaned,
        request,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    return _aggregate_rotation_replay(results)


def _marginal_progress_callback(
    db: Database,
    run_id: str,
    *,
    run_position: int,
    total_runs: int,
    current_symbol: str,
    current_index: int,
    completed_count: int,
) -> Any:
    last_percent = -1.0
    last_stage = ""

    def emit(local_percent: float, stage: str, _completed: int) -> None:
        nonlocal last_percent, last_stage
        local = max(0.0, min(100.0, float(local_percent or 0.0)))
        global_percent = 100.0 * (float(run_position) + local / 100.0) / max(1, int(total_runs))
        rounded = round(global_percent, 1)
        safe_stage = str(stage or "").strip()[:240]
        if rounded < 100.0 and rounded - last_percent < 0.5 and safe_stage == last_stage:
            return
        last_percent = rounded
        last_stage = safe_stage
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id},
            {"$set": {
                "updated_at": utc_now(),
                "marginal_replay.progress_percent": rounded,
                "marginal_replay.current_symbol": current_symbol,
                "marginal_replay.current_index": int(current_index),
                "marginal_replay.completed_count": int(completed_count),
                "marginal_replay.current_stage": safe_stage,
            }},
        )

    return emit


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def _capital_delta_rate(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0.0):
        return None
    return float(candidate / baseline - 1.0)


def _run_marginal_capital_replay(
    db: Database,
    run_id: str,
    *,
    config: BacktestRequest,
    strategy: dict[str, Any],
    winner_config: BacktestRequest,
    end_session: str,
    baseline_frames: dict[str, pd.DataFrame],
    required_sessions: pd.DatetimeIndex,
    shortlist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_assets = [str(symbol).strip().upper() for symbol in config.assets]
    baseline_request = _marginal_execution_request(
        db, config, strategy, winner_config, end_session,
        assets=baseline_assets,
        reference_assets=baseline_assets,
        candidate_assets=[],
    )
    _event(
        db, run_id,
        "Running the baseline replay once before testing shortlisted assets.",
        phase="marginal_replay",
        changes={
            "marginal_replay": {
                "status": "running",
                "total_count": len(shortlist),
                "completed_count": 0,
                "current_symbol": "BASELINE",
                "current_index": 0,
                "current_stage": "Preparing baseline replay",
                "progress_percent": 0.0,
                "baseline": None,
                "results": [],
            },
            "results": shortlist,
            "shortlisted_count": len(shortlist),
        },
    )
    total_runs = len(shortlist) + 1
    baseline_metrics = _run_rotation_replay(
        baseline_frames,
        baseline_request,
        progress_callback=_marginal_progress_callback(
            db,
            run_id,
            run_position=0,
            total_runs=total_runs,
            current_symbol="BASELINE",
            current_index=0,
            completed_count=0,
        ),
    )
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {"$set": {
            "updated_at": utc_now(),
            "marginal_replay.progress_percent": round(100.0 / total_runs, 1),
            "marginal_replay.current_symbol": None,
            "marginal_replay.current_index": 0,
            "marginal_replay.current_stage": "Baseline replay completed",
            "marginal_replay.baseline": bson_value(baseline_metrics),
        }},
    )
    replay_rows: list[dict[str, Any]] = []
    updated_results = [dict(item) for item in shortlist]
    result_map = {str(item.get("symbol") or "").upper(): item for item in updated_results}

    for index, item in enumerate(shortlist, start=1):
        symbol = str(item.get("symbol") or "").strip().upper()
        _event(
            db, run_id,
            f"Marginal Capital Replay {index}/{len(shortlist)}: {symbol}.",
            phase="marginal_replay",
            changes={
                "marginal_replay.current_symbol": symbol,
                "marginal_replay.current_index": index,
                "marginal_replay.current_stage": "Preparing asset replay",
                "marginal_replay.completed_count": index - 1,
                "marginal_replay.progress_percent": round(100.0 * index / total_runs, 1),
            },
        )
        row: dict[str, Any] = {"symbol": symbol, "status": "completed"}
        try:
            candidate_frame, coverage = _candidate_history_coverage(
                db, symbol, config, pd.Timestamp(end_session), required_sessions
            )
            candidate_assets = list(dict.fromkeys([*baseline_assets, symbol]))
            candidate_request = _marginal_execution_request(
                db, config, strategy, winner_config, end_session,
                assets=candidate_assets,
                reference_assets=baseline_assets,
                candidate_assets=[symbol],
            )
            candidate_frames = dict(baseline_frames)
            candidate_frames[symbol] = candidate_frame
            candidate_metrics = _run_rotation_replay(
                candidate_frames,
                candidate_request,
                progress_callback=_marginal_progress_callback(
                    db,
                    run_id,
                    run_position=index,
                    total_runs=total_runs,
                    current_symbol=symbol,
                    current_index=index,
                    completed_count=index - 1,
                ),
            )
            row.update({
                "history_window_complete": bool(coverage.get("history_window_complete")),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "ending_capital_delta": _delta(candidate_metrics.get("ending_capital"), baseline_metrics.get("ending_capital")),
                "ending_capital_delta_rate": _capital_delta_rate(candidate_metrics.get("ending_capital"), baseline_metrics.get("ending_capital")),
                "cagr_delta": _delta(candidate_metrics.get("cagr"), baseline_metrics.get("cagr")),
                "sharpe_delta": _delta(candidate_metrics.get("sharpe"), baseline_metrics.get("sharpe")),
                "maximum_drawdown_delta": _delta(candidate_metrics.get("maximum_drawdown"), baseline_metrics.get("maximum_drawdown")),
                "worst_fold_return_delta": _delta(candidate_metrics.get("worst_fold_return"), baseline_metrics.get("worst_fold_return")),
                "switches_delta": _delta(candidate_metrics.get("switches"), baseline_metrics.get("switches")),
                "cash_days_delta": _delta(candidate_metrics.get("cash_days"), baseline_metrics.get("cash_days")),
                "market_exposure_delta": _delta(candidate_metrics.get("market_exposure"), baseline_metrics.get("market_exposure")),
            })
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)[:700]})
        replay_rows.append(row)
        target = result_map.get(symbol)
        if target is not None:
            target["marginal_replay"] = row
        _event(
            db, run_id,
            f"Marginal Capital Replay finished for {symbol}.",
            changes={
                "marginal_replay": {
                    "status": "running",
                    "total_count": len(shortlist),
                    "completed_count": index,
                    "current_symbol": None,
                    "current_index": index,
                    "current_stage": f"Replay completed for {symbol}",
                    "progress_percent": round(100.0 * (index + 1) / total_runs, 1),
                    "baseline": baseline_metrics,
                    "results": replay_rows,
                },
                "results": updated_results,
            },
        )
        if _should_stop_after_batch(db, run_id):
            break

    comparable = [
        row for row in replay_rows
        if row.get("status") == "completed" and row.get("ending_capital_delta_rate") is not None
    ]
    comparable.sort(key=lambda row: float(row.get("ending_capital_delta_rate") or 0.0), reverse=True)
    marginal_rank = {str(row.get("symbol") or ""): index for index, row in enumerate(comparable, start=1)}
    for item in updated_results:
        symbol = str(item.get("symbol") or "")
        replay = item.get("marginal_replay") if isinstance(item.get("marginal_replay"), dict) else None
        if replay is not None and symbol in marginal_rank:
            replay["marginal_rank"] = marginal_rank[symbol]
    for row in replay_rows:
        symbol = str(row.get("symbol") or "")
        if symbol in marginal_rank:
            row["marginal_rank"] = marginal_rank[symbol]

    replay_summary = {
        "status": "completed" if len(replay_rows) == len(shortlist) else "stopped",
        "total_count": len(shortlist),
        "completed_count": len(replay_rows),
        "current_symbol": None,
        "current_index": len(replay_rows),
        "current_stage": "Marginal Capital Replay completed" if len(replay_rows) == len(shortlist) else "Marginal Capital Replay stopped",
        "progress_percent": 100.0 if len(replay_rows) == len(shortlist) else round(100.0 * (len(replay_rows) + 1) / total_runs, 1),
        "baseline": baseline_metrics,
        "results": replay_rows,
    }
    return updated_results, replay_summary

def _catalog_metrics(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "raw_score",
        "latest_close",
        "median_dollar_volume",
        "return_20",
        "return_60",
        "volatility_20",
        "drawdown_60",
        "trend_efficiency_20",
        "max_baseline_correlation_60",
    )
    metrics = {key: item.get(key) for key in keys}
    marginal = item.get("marginal_replay") if isinstance(item.get("marginal_replay"), dict) else None
    if marginal is not None:
        candidate = marginal.get("candidate") if isinstance(marginal.get("candidate"), dict) else {}
        metrics["marginal_replay"] = {
            "status": marginal.get("status"),
            "marginal_rank": marginal.get("marginal_rank"),
            "ending_capital_delta": marginal.get("ending_capital_delta"),
            "ending_capital_delta_rate": marginal.get("ending_capital_delta_rate"),
            "cagr_delta": marginal.get("cagr_delta"),
            "sharpe_delta": marginal.get("sharpe_delta"),
            "maximum_drawdown_delta": marginal.get("maximum_drawdown_delta"),
            "worst_fold_return_delta": marginal.get("worst_fold_return_delta"),
            "candidate_ending_capital": candidate.get("ending_capital"),
        }
    return metrics


def _persist_shortlist_to_catalog(db: Database, document: dict[str, Any], results: list[dict[str, Any]]) -> None:
    run_id = str(document.get("run_id") or "").strip()
    if not run_id:
        return
    baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}
    winner = document.get("winner_source") if isinstance(document.get("winner_source"), dict) else {}
    now = utc_now()
    for item in results:
        if not isinstance(item, dict) or not bool(item.get("history_window_complete")):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        existing = db[CATALOG_COLLECTION].find_one({"_id": symbol}) or {}
        recent_run_ids = [str(value) for value in existing.get("recent_run_ids") or []]
        already_counted = run_id in recent_run_ids
        rank = int(item.get("rank") or 0) or None
        raw_score = item.get("raw_score")
        discovery = {
            "run_id": run_id,
            "seen_at": now,
            "rank": rank,
            "raw_score": raw_score,
            "snapshot_end": baseline.get("market_snapshot_end"),
        }
        set_fields = {
            "symbol": symbol,
            "status": str(existing.get("status") or "discovered"),
            "last_seen_at": now,
            "latest_run_id": run_id,
            "latest_rank": rank,
            "latest_evaluated_count": int(document.get("evaluated_count") or 0),
            "latest_model_score": raw_score,
            "latest_snapshot_end": baseline.get("market_snapshot_end"),
            "latest_metrics": _catalog_metrics(item),
            "history_window_complete": True,
            "history_required_start": item.get("history_required_start"),
            "history_start_tolerance_days": item.get("history_start_tolerance_days"),
            "history_actual_start": item.get("history_actual_start"),
            "history_actual_end": item.get("history_actual_end"),
            "history_expected_sessions": item.get("history_expected_sessions"),
            "history_missing_required_sessions": item.get("history_missing_required_sessions"),
            "latest_baseline_strategy_id": baseline.get("strategy_id"),
            "latest_baseline_strategy_sequence": baseline.get("strategy_sequence"),
            "latest_winner_strategy_id": winner.get("strategy_id"),
            "latest_winner_strategy_sequence": winner.get("strategy_sequence"),
            "updated_at": now,
        }
        update: dict[str, Any] = {
            "$set": bson_value(set_fields),
            "$setOnInsert": {"first_seen_at": now, "strategy_created_count": 0},
        }
        if not already_counted:
            update["$inc"] = {"times_discovered": 1}
            update["$push"] = {
                "recent_run_ids": {"$each": [run_id], "$slice": -12},
                "recent_discoveries": {"$each": [bson_value(discovery)], "$slice": -12},
            }
            if rank is not None:
                update["$min"] = {"best_rank": rank}
            if isinstance(raw_score, (int, float)):
                update["$max"] = {"best_model_score": float(raw_score)}
        db[CATALOG_COLLECTION].update_one({"_id": symbol}, update, upsert=True)


def get_discovery_catalog(db: Database) -> dict[str, Any]:
    document = _campaign(db) or {}
    if str(document.get("status") or "") in {"completed", "stopped"} and document.get("results"):
        _persist_shortlist_to_catalog(db, document, list(document.get("results") or []))
    assets = [_public(item) for item in db[CATALOG_COLLECTION].find({}).sort([("last_seen_at", -1), ("times_discovered", -1), ("best_rank", 1)])]
    clean_assets = [item for item in assets if item is not None]
    return {
        "api_version": API_VERSION,
        "count": len(clean_assets),
        "assets": clean_assets,
        "persistence_policy": {
            "market_bars": "not_stored_by_catalog",
            "rejected_assets": "not_stored",
            "recent_discoveries_per_asset": 12,
        },
    }


def _finish(db: Database, run_id: str, status: str, message: str, *, results: list[dict[str, Any]] | None = None) -> None:
    changes: dict[str, Any] = {
        "status": status,
        "phase": "completed" if status == "completed" else status,
        "completed_at": utc_now(),
        "current_symbol": None,
        "message": message,
    }
    if results is not None:
        changes["results"] = results
        changes["shortlisted_count"] = len(results)
    _event(db, run_id, message, changes=changes)
    if results and status in {"completed", "stopped"}:
        try:
            current = _campaign(db) or {}
            _persist_shortlist_to_catalog(db, current, results)
        except Exception as exc:
            _event(db, run_id, f"Discovery Catalog update failed: {str(exc)[:300]}")


def _should_stop_after_batch(db: Database, run_id: str) -> bool:
    document = _campaign(db) or {}
    return str(document.get("run_id") or "") != run_id or bool(document.get("cancel_requested"))


def _run_worker(db: Database, run_id: str) -> None:
    try:
        document = _campaign(db) or {}
        requested = int(document.get("research_size") or 24)
        seed = int(document.get("random_seed") or 0)
        _event(db, run_id, "Loading the Strategy Research baseline.", phase="baseline", changes={"status": "running", "started_at": utc_now()})
        config, strategy = get_research_strategy_context(db)
        winner_config, winner_strategy = get_trader_winner_context(db)
        safe_session = latest_safe_completed_xnys_session()
        end_session = pd.Timestamp(safe_session).date().isoformat()
        baseline_frames = _baseline_frames(config, end_session)
        baseline_returns = _baseline_recent_returns(baseline_frames)
        _event(
            db,
            run_id,
            "Training the Learning-to-Rank model on the selected Strategy Research universe.",
            phase="training_ranker",
            changes={
                "baseline": {
                    "strategy_id": strategy.get("id"),
                    "strategy_name": strategy.get("name"),
                    "strategy_sequence": strategy.get("strategy_sequence"),
                    "configuration_hash": strategy.get("configuration_hash"),
                    "asset_count": len(config.assets),
                    "assets": list(config.assets),
                    "market_snapshot_end": end_session,
                },
                "winner_source": {
                    "strategy_id": winner_strategy.get("id"),
                    "strategy_name": winner_strategy.get("name"),
                    "strategy_sequence": winner_strategy.get("strategy_sequence"),
                    "strategy_revision": winner_strategy.get("revision"),
                    "configuration_hash": winner_strategy.get("configuration_hash"),
                    "asset_count": len(winner_config.assets),
                    "assets": list(winner_config.assets),
                },
            },
        )
        required_sessions = _baseline_required_sessions(baseline_frames, config, safe_session)
        bundle = train_ranker(baseline_frames, random_state=seed)
        _event(db, run_id, "Learning-to-Rank training completed.", phase="scanning", changes={"model": bundle.diagnostics})
        if _should_stop_after_batch(db, run_id):
            _finish(db, run_id, "stopped", "Asset Discovery stopped after Learning-to-Rank training and before the first external batch.", results=[])
            return

        universe = _discover_symbols(db)
        baseline_symbols = {str(item).upper() for item in config.assets}
        external = [symbol for symbol in universe if symbol not in baseline_symbols]
        random.Random(seed).shuffle(external)
        scan_budget = min(len(external), requested)
        selected = external[:scan_budget]
        _event(
            db,
            run_id,
            f"Scanning exactly the manually bounded external sample of at most {scan_budget} symbols.",
            changes={"universe_size": len(universe), "external_universe_size": len(external), "scan_budget": scan_budget},
        )
        if _should_stop_after_batch(db, run_id):
            _finish(db, run_id, "stopped", "Asset Discovery stopped before the first external batch.", results=[])
            return

        evaluated: list[dict[str, Any]] = []
        for batch_index, batch_start in enumerate(range(0, len(selected), BATCH_SIZE), start=1):
            batch = selected[batch_start: batch_start + BATCH_SIZE]
            _event(db, run_id, f"Processing batch {batch_index}.", changes={"current_batch": batch_index})
            for symbol in batch:
                _event(db, run_id, "Evaluating the current external asset.", changes={"current_symbol": symbol})
                try:
                    frame, coverage = _candidate_history_coverage(
                        db, symbol, config, safe_session, required_sessions
                    )
                    result = _score_candidate(bundle, symbol, frame, baseline_returns)
                    result.update(coverage)
                    evaluated.append(result)
                    _increment(db, run_id, {"attempted_count": 1, "evaluated_count": 1})
                except RuntimeError as exc:
                    reason = str(exc).strip().lower()
                    if reason in {"insufficient_history", "discontinuous_history", "ticker_identity_discontinuity", "price_filter", "liquidity_filter", "volume_quality_filter"}:
                        _increment(db, run_id, {"attempted_count": 1})
                        _reject(db, run_id, reason)
                    else:
                        _increment(db, run_id, {"attempted_count": 1, "technical_failure_count": 1})
                except Exception:
                    _increment(db, run_id, {"attempted_count": 1, "technical_failure_count": 1})

            if _should_stop_after_batch(db, run_id):
                shortlist = _rank_results(evaluated)
                _finish(
                    db,
                    run_id,
                    "stopped",
                    f"Asset Discovery stopped after batch {batch_index}; completed batch results were preserved.",
                    results=shortlist,
                )
                return

        shortlist = _rank_results(evaluated)
        if shortlist:
            _event(
                db,
                run_id,
                f"Asset Discovery ranked {len(evaluated)} evaluable external assets; starting Marginal Capital Replay for all {len(shortlist)} shortlisted assets.",
                phase="marginal_replay",
                changes={"results": shortlist, "shortlisted_count": len(shortlist)},
            )
            shortlist, marginal_replay = _run_marginal_capital_replay(
                db,
                run_id,
                config=config,
                strategy=strategy,
                winner_config=winner_config,
                end_session=end_session,
                baseline_frames=baseline_frames,
                required_sessions=required_sessions,
                shortlist=shortlist,
            )
            if _should_stop_after_batch(db, run_id):
                _event(db, run_id, "Asset Discovery stopped after the current Marginal Capital Replay asset.", changes={"marginal_replay": marginal_replay})
                _finish(
                    db,
                    run_id,
                    "stopped",
                    f"Asset Discovery stopped during Marginal Capital Replay; {marginal_replay.get('completed_count', 0)} of {len(shortlist)} shortlisted assets were replayed.",
                    results=shortlist,
                )
                return
            _event(db, run_id, "Marginal Capital Replay completed for the full shortlist.", changes={"marginal_replay": marginal_replay})
        _finish(
            db,
            run_id,
            "completed",
            f"Asset Discovery ranked {len(evaluated)} evaluable external assets and completed Marginal Capital Replay for {len(shortlist)} shortlisted assets.",
            results=shortlist,
        )
    except Exception as exc:
        _finish(db, run_id, "failed", f"Asset Discovery failed: {str(exc)[:900]}")
    finally:
        global _worker_thread
        with _worker_lock:
            _worker_thread = None


def get_asset_discovery_status(db: Database) -> dict[str, Any]:
    document = _campaign(db)
    if document and str(document.get("status") or "") in ACTIVE_STATUSES and not _worker_alive():
        phase = str(document.get("phase") or "").strip().lower()
        interrupted_marginal = phase == "marginal_replay"
        changes: dict[str, Any] = {
            "status": "interrupted",
            "phase": "marginal_replay" if interrupted_marginal else "interrupted",
            "message": (
                "Marginal Capital Replay worker was interrupted. Run Marginal Capital Replay again to restart the replay."
                if interrupted_marginal
                else "The previous Asset Discovery worker is no longer active. Start a new manual campaign to continue research."
            ),
            "completed_at": utc_now(),
            "updated_at": utc_now(),
        }
        if interrupted_marginal:
            changes.update({
                "marginal_replay.status": "interrupted",
                "marginal_replay.current_symbol": None,
                "marginal_replay.current_stage": "Replay interrupted",
            })
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": document.get("run_id")},
            {"$set": changes},
        )
        document = _campaign(db)
    try:
        config, strategy = get_research_strategy_context(db)
        selected_baseline = {
            "strategy_id": strategy.get("id"),
            "strategy_name": strategy.get("name"),
            "strategy_sequence": strategy.get("strategy_sequence"),
            "configuration_hash": strategy.get("configuration_hash"),
            "asset_count": len(config.assets),
            "assets": list(config.assets),
        }
    except Exception:
        selected_baseline = None
    return {
        "api_version": API_VERSION,
        "mode": "manual",
        "batch_size": BATCH_SIZE,
        "baseline": selected_baseline,
        "research_size_options": [8, 16, 24, 32, 40, 48, 56, 64],
        "persistence_policy": {
            "external_market_bars": "memory_only",
            "technical_failures": "aggregate_only",
            "rejected_assets": "aggregate_only",
            "stored_shortlist_limit": 8,
            "marginal_replay": "shortlist_only_in_memory_market_data",
            "history": "latest_campaign_only",
            "discovery_catalog": "shortlist_assets_only",
        },
        "campaign": _public(document),
    }


def start_asset_discovery(db: Database, *, research_size: int) -> dict[str, Any]:
    global _worker_thread
    requested = int(research_size)
    if requested < 8 or requested > 64 or requested % 8 != 0:
        raise AssetDiscoveryConflict("Research size must be one of 8, 16, 24, 32, 40, 48, 56 or 64 assets.")

    with _worker_lock:
        current = _campaign(db) or {}
        if _worker_thread and _worker_thread.is_alive():
            raise AssetDiscoveryConflict("An Asset Discovery campaign is already running.")
        if str(current.get("status") or "") in ACTIVE_STATUSES:
            db[COLLECTION].update_one(
                {"_id": CURRENT_ID},
                {"$set": {"status": "interrupted", "phase": "interrupted", "completed_at": utc_now(), "updated_at": utc_now()}},
            )

        run_id = f"asset-discovery-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        seed = int.from_bytes(uuid4().bytes[:4], "big")
        document = {
            "_id": CURRENT_ID,
            "run_id": run_id,
            "schema_version": 1,
            "api_version": API_VERSION,
            "status": "queued",
            "phase": "queued",
            "mode": "manual",
            "research_size": requested,
            "batch_size": BATCH_SIZE,
            "random_seed": seed,
            "scan_budget": requested,
            "attempted_count": 0,
            "evaluated_count": 0,
            "rejected_count": 0,
            "technical_failure_count": 0,
            "shortlisted_count": 0,
            "rejection_summary": {},
            "results": [],
            "marginal_replay": {"status": "pending", "total_count": 0, "completed_count": 0, "current_symbol": None, "baseline": None, "results": []},
            "events": [{"at": utc_now(), "message": "Manual Asset Discovery campaign queued."}],
            "cancel_requested": False,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "current_symbol": None,
            "current_batch": 0,
            "message": "Manual Asset Discovery campaign queued.",
        }
        db[COLLECTION].replace_one({"_id": CURRENT_ID}, bson_value(document), upsert=True)
        _worker_thread = threading.Thread(target=_run_worker, args=(db, run_id), name="asset-discovery-ranker", daemon=True)
        _worker_thread.start()
    return get_asset_discovery_status(db)


def stop_asset_discovery(db: Database) -> dict[str, Any]:
    document = _campaign(db)
    if not document or str(document.get("status") or "") not in ACTIVE_STATUSES:
        return get_asset_discovery_status(db)
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": document.get("run_id")},
        {
            "$set": {
                "cancel_requested": True,
                "status": "stopping",
                "message": "Stop requested. The current batch will finish before the campaign stops.",
                "updated_at": utc_now(),
            },
            "$push": {"events": {"$each": [{"at": utc_now(), "message": "Stop requested; finishing the current batch."}], "$slice": -24}},
        },
    )
    return get_asset_discovery_status(db)





def _raw_strategy(db: Database, strategy_id: str | None) -> dict[str, Any] | None:
    normalized = str(strategy_id or "").strip()
    if not normalized:
        return None
    return db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": normalized})


def _marginal_campaign_context(
    db: Database,
    document: dict[str, Any],
) -> tuple[BacktestRequest, dict[str, Any], BacktestRequest, str, dict[str, pd.DataFrame], pd.DatetimeIndex]:
    baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}
    baseline_id = str(baseline.get("strategy_id") or "").strip()
    baseline_raw = _raw_strategy(db, baseline_id)
    if baseline_raw is None:
        raise AssetDiscoveryConflict("The Strategy used by this Asset Discovery campaign is no longer available.")
    config = BacktestRequest.model_validate(baseline_raw.get("configuration") or {})
    strategy = get_strategy(db, baseline_id)

    winner_source = document.get("winner_source") if isinstance(document.get("winner_source"), dict) else {}
    winner_raw = _raw_strategy(db, winner_source.get("strategy_id"))
    if winner_raw is not None:
        winner_config = BacktestRequest.model_validate(winner_raw.get("configuration") or {})
    else:
        winner_config, _winner = get_trader_winner_context(db)

    end_session = str(baseline.get("market_snapshot_end") or "").strip()
    if not end_session:
        end_session = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()
    baseline_frames = _baseline_frames(config, end_session)
    required_sessions = _baseline_required_sessions(baseline_frames, config, end_session)
    return config, strategy, winner_config, end_session, baseline_frames, required_sessions


def _run_existing_marginal_worker(db: Database, run_id: str) -> None:
    try:
        document = _campaign(db) or {}
        if str(document.get("run_id") or "") != run_id:
            return
        shortlist = [dict(item) for item in document.get("results") or [] if isinstance(item, dict)]
        if not shortlist:
            raise AssetDiscoveryConflict("The current Asset Discovery campaign has no shortlist to replay.")
        _event(
            db,
            run_id,
            "Preparing Marginal Capital Replay for the existing shortlist.",
            phase="marginal_replay",
            changes={"status": "running", "started_at": document.get("started_at") or utc_now(), "completed_at": None, "cancel_requested": False},
        )
        config, strategy, winner_config, end_session, baseline_frames, required_sessions = _marginal_campaign_context(db, document)
        updated_results, replay = _run_marginal_capital_replay(
            db,
            run_id,
            config=config,
            strategy=strategy,
            winner_config=winner_config,
            end_session=end_session,
            baseline_frames=baseline_frames,
            required_sessions=required_sessions,
            shortlist=shortlist,
        )
        if _should_stop_after_batch(db, run_id):
            _event(db, run_id, "Marginal Capital Replay stopped after the current asset.", changes={"marginal_replay": replay})
            _finish(
                db,
                run_id,
                "stopped",
                f"Marginal Capital Replay stopped after {replay.get('completed_count', 0)} of {len(shortlist)} assets.",
                results=updated_results,
            )
            return
        _event(db, run_id, "Marginal Capital Replay completed for the existing shortlist.", changes={"marginal_replay": replay})
        _finish(
            db,
            run_id,
            "completed",
            f"Marginal Capital Replay completed for all {len(shortlist)} shortlisted assets.",
            results=updated_results,
        )
    except Exception as exc:
        current = _campaign(db) or {}
        current_results = [dict(item) for item in current.get("results") or [] if isinstance(item, dict)]
        _event(
            db,
            run_id,
            f"Marginal Capital Replay failed: {str(exc)[:700]}",
            changes={"marginal_replay.status": "failed", "marginal_replay.error": str(exc)[:700]},
        )
        _finish(db, run_id, "failed", f"Marginal Capital Replay failed: {str(exc)[:700]}", results=current_results)
    finally:
        global _worker_thread
        with _worker_lock:
            _worker_thread = None


def start_marginal_capital_replay(db: Database) -> dict[str, Any]:
    global _worker_thread
    with _worker_lock:
        document = _campaign(db) or {}
        if _worker_thread and _worker_thread.is_alive():
            raise AssetDiscoveryConflict("An Asset Discovery operation is already running.")
        run_id = str(document.get("run_id") or "").strip()
        shortlist = [item for item in document.get("results") or [] if isinstance(item, dict)]
        if not run_id or not shortlist:
            raise AssetDiscoveryConflict("Complete an Asset Discovery search before running Marginal Capital Replay.")
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id},
            {"$set": {
                "status": "queued",
                "phase": "marginal_replay",
                "cancel_requested": False,
                "completed_at": None,
                "updated_at": utc_now(),
                "message": "Marginal Capital Replay queued for the existing shortlist.",
                "marginal_replay": {
                    "status": "queued",
                    "total_count": len(shortlist),
                    "completed_count": 0,
                    "current_symbol": None,
                    "current_index": 0,
                    "current_stage": "Queued",
                    "progress_percent": 0.0,
                    "baseline": None,
                    "results": [],
                },
            }},
        )
        _worker_thread = threading.Thread(
            target=_run_existing_marginal_worker,
            args=(db, run_id),
            name="asset-discovery-marginal-replay",
            daemon=True,
        )
        _worker_thread.start()
    return get_asset_discovery_status(db)


def _creation_source_profiles(db: Database, document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve a practical source and template without imposing product gates.

    Asset Discovery is a manual research tool. The campaign Winner snapshot is the
    preferred configuration source. If it is unavailable, use the current Winner;
    if that is unavailable, use the campaign Strategy Research baseline. A template
    profile is used only to reserve a normal catalog Strategy document; the resulting
    Strategy is always materialized as a standard research Strategy with the selected
    assets added to the chosen source configuration.
    """
    winner_source = document.get("winner_source") if isinstance(document.get("winner_source"), dict) else {}
    baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}

    campaign_winner = _raw_strategy(db, winner_source.get("strategy_id"))
    campaign_baseline = _raw_strategy(db, baseline.get("strategy_id"))

    current_winner: dict[str, Any] | None = None
    try:
        _current_config, current_winner_public = get_trader_winner_context(db)
        current_winner = _raw_strategy(db, current_winner_public.get("id"))
    except Exception:
        current_winner = None

    if campaign_winner is not None:
        source = campaign_winner
        source_kind = "campaign_winner"
    elif current_winner is not None:
        source = current_winner
        source_kind = "current_winner"
    elif campaign_baseline is not None:
        source = campaign_baseline
        source_kind = "campaign_baseline"
    else:
        _research_config, research_public = get_research_strategy_context(db)
        source = _raw_strategy(db, research_public.get("id"))
        if source is None:
            raise AssetDiscoveryConflict("No Strategy source is available for creating the research Strategy.")
        source_kind = "current_research_strategy"

    template = campaign_baseline or source
    return source, template, source_kind


def create_research_strategy_from_discovery(
    db: Database,
    *,
    run_id: str | None,
    symbols: list[str],
    actor_email: str | None,
) -> dict[str, Any]:
    current_document = _campaign(db) or {}
    normalized_run_id = str(run_id or "").strip()
    campaign_document = (
        current_document
        if normalized_run_id and str(current_document.get("run_id") or "") == normalized_run_id
        else {}
    )

    requested_symbols = list(dict.fromkeys(str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()))
    if not requested_symbols:
        raise AssetDiscoveryConflict("Select at least one discovered asset.")

    shortlist = {
        str(item.get("symbol") or "").strip().upper(): item
        for item in campaign_document.get("results") or []
        if isinstance(item, dict)
    }
    catalog_documents = {
        str(item.get("symbol") or item.get("_id") or "").strip().upper(): item
        for item in db[CATALOG_COLLECTION].find({"_id": {"$in": requested_symbols}})
        if isinstance(item, dict)
    }
    discovery_metadata: dict[str, dict[str, Any]] = {}
    for symbol in requested_symbols:
        if symbol in shortlist:
            discovery_metadata[symbol] = shortlist[symbol]
        elif symbol in catalog_documents:
            item = catalog_documents[symbol]
            metrics = item.get("latest_metrics") if isinstance(item.get("latest_metrics"), dict) else {}
            discovery_metadata[symbol] = {
                "symbol": symbol,
                "rank": item.get("latest_rank"),
                "raw_score": item.get("latest_model_score"),
                "history_window_complete": item.get("history_window_complete"),
                "history_required_start": item.get("history_required_start"),
                "history_actual_start": item.get("history_actual_start"),
            "history_actual_end": item.get("history_actual_end"),
            "history_expected_sessions": item.get("history_expected_sessions"),
            "history_missing_required_sessions": item.get("history_missing_required_sessions"),
                **metrics,
            }
        else:
            raise AssetDiscoveryConflict(f"{symbol} is not available in the current Discovery shortlist or Discovery Catalog.")

    source_document = campaign_document if campaign_document else {}
    source_raw, template_raw, source_resolution = _creation_source_profiles(db, source_document)
    source_id = str(source_raw.get("_id") or "")
    template_id = str(template_raw.get("_id") or "")
    source_config = BacktestRequest.model_validate(source_raw.get("configuration") or {})
    source_assets = [str(item).strip().upper() for item in source_config.assets]

    snapshot_end = None
    if campaign_document and isinstance(campaign_document.get("baseline"), dict):
        snapshot_end = (campaign_document.get("baseline") or {}).get("market_snapshot_end")
    if not snapshot_end:
        snapshot_end = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()

    source_baseline_frames = _baseline_frames(source_config, snapshot_end)
    required_sessions = _baseline_required_sessions(source_baseline_frames, source_config, snapshot_end)
    persisted_history_rows: dict[str, int] = {}
    valid_symbols: list[str] = []
    discarded_assets: list[dict[str, str]] = []
    source_asset_set = set(source_assets)
    for symbol in requested_symbols:
        if symbol in source_asset_set:
            valid_symbols.append(symbol)
            continue
        try:
            persisted_history_rows[symbol] = _persist_selected_asset_history(
                db, symbol, source_config, snapshot_end, required_sessions
            )
            valid_symbols.append(symbol)
        except AssetDiscoveryConflict as exc:
            discarded_assets.append({"symbol": symbol, "reason": str(exc)})
            db[CATALOG_COLLECTION].delete_one({"_id": symbol})

    if not valid_symbols:
        discarded = ", ".join(item["symbol"] for item in discarded_assets) or "the selected assets"
        raise AssetDiscoveryConflict(
            f"No selected Discovery asset has complete continuous history for the source Strategy. Discarded: {discarded}."
        )

    combined_assets = list(dict.fromkeys([*source_assets, *valid_symbols]))
    updated_configuration = source_config.model_copy(update={"assets": combined_assets})
    configuration_payload = updated_configuration.model_dump(mode="json")

    created = create_strategy(
        db,
        name="Asset Discovery Research Strategy",
        description="Research Strategy created from Asset Discovery selected assets.",
        clone_from_strategy_id=template_id,
        actor_email=actor_email,
    )
    created_id = str(created.get("id") or "")

    try:
        source_model_snapshot = get_strategy_model_snapshot(db, source_id)
    except Exception:
        source_model_snapshot = template_raw.get("research_model_snapshot") if isinstance(template_raw.get("research_model_snapshot"), dict) else None

    baseline = campaign_document.get("baseline") if isinstance(campaign_document.get("baseline"), dict) else {}
    winner = campaign_document.get("winner_source") if isinstance(campaign_document.get("winner_source"), dict) else {}
    origin = {
        "run_id": normalized_run_id or None,
        "source_resolution": source_resolution,
        "source_strategy_id": source_id,
        "source_strategy_revision": source_raw.get("revision"),
        "source_strategy_hash": source_raw.get("configuration_hash"),
        "source_strategy_kind": source_raw.get("strategy_kind"),
        "campaign_winner_strategy_id": winner.get("strategy_id"),
        "discovery_baseline_strategy_id": baseline.get("strategy_id"),
        "discovery_baseline_hash": baseline.get("configuration_hash"),
        "discovery_snapshot_end": baseline.get("market_snapshot_end") or snapshot_end,
        "selected_assets": valid_symbols,
        "discarded_assets": discarded_assets,
        "persisted_history_rows": persisted_history_rows,
        "ranked_assets": [
            {
                "symbol": symbol,
                "rank": discovery_metadata[symbol].get("rank"),
                "raw_score": discovery_metadata[symbol].get("raw_score"),
            }
            for symbol in valid_symbols
        ],
        "created_at": utc_now(),
    }

    set_payload: dict[str, Any] = {
        "configuration": bson_value(configuration_payload),
        "configuration_hash": _configuration_hash(configuration_payload),
        "source_strategy_id": source_id,
        "source_strategy_revision": int(source_raw.get("revision") or 1),
        "strategy_kind": "standard",
        "tuning_target": "model_strategy",
        "status": "draft",
        "locked": False,
        "research_reference_assets": list(source_assets),
        "discovery_origin": bson_value(origin),
        "last_change_note": "Created from Asset Discovery selected assets.",
        "updated_at": utc_now(),
        "updated_by": (actor_email or "").strip().lower() or None,
    }
    if isinstance(source_model_snapshot, dict):
        set_payload["research_model_snapshot"] = bson_value(source_model_snapshot)
        set_payload["research_model_revision"] = 1

    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": created_id},
        {
            "$set": set_payload,
            "$unset": {
                "source_temporal_run_id": "",
                "source_temporal_experiment": "",
                "temporal_strategy_variant": "",
                "source_stateful_replay_id": "",
                "source_stateful_processing_id": "",
                "stateful_candidate_key": "",
                "stateful_candidate_label": "",
                "temporal_policy_revision": "",
                "temporal_policy_snapshot": "",
                "temporal_validation_status": "",
                "temporal_validation_id": "",
                "temporal_validation_at": "",
                "temporal_validation_by": "",
                "temporal_trader_eligible": "",
                "temporal_trader_block_reason": "",
            },
        },
    )

    now = utc_now()
    for symbol in valid_symbols:
        db[CATALOG_COLLECTION].update_one(
            {"_id": symbol},
            {
                "$set": {
                    "status": "strategy_created",
                    "last_strategy_id": created_id,
                    "last_strategy_created_at": now,
                    "updated_at": now,
                },
                "$inc": {"strategy_created_count": 1},
            },
        )

    strategy = get_strategy(db, created_id)
    return {
        "strategy": strategy,
        "source_strategy": {
            "strategy_id": source_id,
            "strategy_sequence": source_raw.get("strategy_sequence"),
            "strategy_kind": source_raw.get("strategy_kind"),
            "configuration_hash": source_raw.get("configuration_hash"),
            "resolution": source_resolution,
        },
        "selected_assets": valid_symbols,
        "discarded_assets": discarded_assets,
        "persisted_history_rows": persisted_history_rows,
        "asset_count": len(combined_assets),
        "added_asset_count": len([symbol for symbol in valid_symbols if symbol not in source_asset_set]),
    }

def export_asset_discovery(db: Database, *, front_version: str | None = None) -> dict[str, Any]:
    document = _campaign(db)
    if not document:
        raise AssetDiscoveryConflict("There is no Asset Discovery campaign to export.")
    payload = _public(document) or {}
    return {
        "schema_version": 2,
        "package_kind": "market_cycle_trader_asset_discovery_research",
        "api_version": API_VERSION,
        "front_version": str(front_version or "") or None,
        "generated_at": utc_now(),
        "storage_policy": {
            "raw_external_market_data_persisted": False,
            "rejected_symbols_persisted": False,
            "technical_failure_symbols_persisted": False,
            "campaign_history_persisted": False,
            "discovery_catalog_persisted": True,
        },
        "campaign": payload,
    }
