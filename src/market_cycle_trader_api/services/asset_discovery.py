from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    refresh_market_data_to_live_cutoff,
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
from .asset_discovery_ranker import (
    FEATURE_COLUMNS,
    AssetDiscoveryRankerCancelled,
    latest_feature_snapshot,
    market_quality,
    train_ranker,
)
from ..engine.capital_rotation import run_rotation_models
from ..engine.compound_rotation_backtest import apply_slippage, calculate_reference_fees
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings
from .temporal_research_settings import temporal_research_settings_snapshot
from .strategy_lab import (
    _configuration_hash,
    create_strategy,
    get_research_strategy_context,
    get_strategy,
    get_strategy_model_snapshot,
    get_trader_winner_context,
    update_strategy,
    StrategyLabConflict,
)

COLLECTION = ASSET_DISCOVERY_RESEARCH_COLLECTION
CATALOG_COLLECTION = ASSET_DISCOVERY_CATALOG_COLLECTION
CURRENT_ID = "current"
DEFAULT_RESEARCH_SIZE = 64
CANDIDATE_HISTORY_DAYS = 900
SUPPORTED_EXCHANGES = frozenset({"AMEX", "ARCA", "BATS", "NASDAQ", "NYSE"})
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
ACTIVE_STATUSES = frozenset({"queued", "running", "stopping"})
TICKER_IDENTITY_GAP_SESSIONS = 20
MIN_PERSISTENT_MARGINAL_CAPITAL_DELTA_RATE = 0.0
DEFAULT_SEVERE_MONTH_THRESHOLD = -0.05
WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0
WORKER_HEARTBEAT_STALE_SECONDS = 20.0
CAUSAL_HOLDOUT_SESSIONS = 252
CAUSAL_VALIDATION_SESSIONS = 126
CAUSAL_CERTIFICATION_SESSIONS = 126
CAUSAL_MIN_TRAINING_SESSIONS = 720
CERTIFICATION_LEDGER_COLLECTION = "asset_discovery_certification_ledger"

logger = logging.getLogger(__name__)


def _positive_env_int(name: str, fallback: int) -> int:
    try:
        value = int(str(os.getenv(name) or fallback).strip())
    except (TypeError, ValueError):
        value = int(fallback)
    return max(1, value)


def _scan_worker_count() -> int:
    fallback = max(8, min(32, max(1, int(os.cpu_count() or 1)) * 4))
    return _positive_env_int("ASSET_DISCOVERY_SCAN_WORKERS", fallback)


def _replay_worker_count() -> int:
    fallback = max(2, min(4, max(1, int(os.cpu_count() or 1))))
    return _positive_env_int("ASSET_DISCOVERY_REPLAY_WORKERS", fallback)


def _scan_batch_size() -> int:
    return max(_scan_worker_count(), _scan_worker_count() * 4)


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


def _sanitize_completed_campaign_persistence(db: Database, document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return document
    if str(document.get("status") or "").strip().lower() in ACTIVE_STATUSES:
        return document

    stored_results = list(document.get("results") or []) if isinstance(document.get("results"), list) else []
    visible_results = [item for item in stored_results if _item_is_persistent_candidate(item)]

    marginal = dict(document.get("marginal_replay") or {}) if isinstance(document.get("marginal_replay"), dict) else {}
    stored_replay_rows = list(marginal.get("results") or []) if isinstance(marginal.get("results"), list) else []
    visible_replay_rows = [row for row in stored_replay_rows if _marginal_replay_is_persistent_candidate(row)]

    changed = len(visible_results) != len(stored_results) or len(visible_replay_rows) != len(stored_replay_rows)
    if not changed:
        return document

    marginal["results"] = visible_replay_rows
    marginal["persistent_candidate_count"] = len(visible_replay_rows)
    changes = {
        "results": bson_value(visible_results),
        "shortlisted_count": len(visible_results),
        "marginal_replay": bson_value(marginal),
        "updated_at": utc_now(),
    }
    db[COLLECTION].update_one({"_id": document.get("_id", CURRENT_ID)}, {"$set": changes})
    sanitized = dict(document)
    sanitized.update(changes)
    return sanitized


def purge_legacy_non_persistent_asset_discovery_records(db: Database) -> dict[str, int]:
    research_documents_scanned = 0
    research_results_removed = 0
    marginal_rows_removed = 0
    catalog_records_removed = 0

    for document in list(db[COLLECTION].find({})):
        if not isinstance(document, dict):
            continue
        research_documents_scanned += 1
        if str(document.get("status") or "").strip().lower() in ACTIVE_STATUSES:
            continue
        stored_results = list(document.get("results") or []) if isinstance(document.get("results"), list) else []
        marginal = dict(document.get("marginal_replay") or {}) if isinstance(document.get("marginal_replay"), dict) else {}
        stored_replay_rows = list(marginal.get("results") or []) if isinstance(marginal.get("results"), list) else []
        sanitized = _sanitize_completed_campaign_persistence(db, document) or document
        current_results = list(sanitized.get("results") or []) if isinstance(sanitized.get("results"), list) else []
        current_marginal = sanitized.get("marginal_replay") if isinstance(sanitized.get("marginal_replay"), dict) else {}
        current_replay_rows = list(current_marginal.get("results") or []) if isinstance(current_marginal.get("results"), list) else []
        research_results_removed += max(0, len(stored_results) - len(current_results))
        marginal_rows_removed += max(0, len(stored_replay_rows) - len(current_replay_rows))

    for stored in list(db[CATALOG_COLLECTION].find({})):
        if not isinstance(stored, dict):
            continue
        metrics = stored.get("latest_metrics") if isinstance(stored.get("latest_metrics"), dict) else {}
        persisted_view = {
            "history_window_complete": bool(stored.get("history_window_complete")),
            **metrics,
        }
        if _item_is_persistent_candidate(persisted_view):
            continue
        result = db[CATALOG_COLLECTION].delete_one({"_id": stored.get("_id")})
        catalog_records_removed += int(getattr(result, "deleted_count", 0) or 0)

    return {
        "research_documents_scanned": research_documents_scanned,
        "research_results_removed": research_results_removed,
        "marginal_rows_removed": marginal_rows_removed,
        "catalog_records_removed": catalog_records_removed,
    }


def _worker_alive() -> bool:
    with _worker_lock:
        return bool(_worker_thread and _worker_thread.is_alive())


def _utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _worker_heartbeat_fresh(document: dict[str, Any] | None) -> bool:
    if not isinstance(document, dict):
        return False
    heartbeat = _utc_datetime(document.get("worker_heartbeat_at"))
    if heartbeat is None:
        return False
    return max(0.0, (utc_now() - heartbeat).total_seconds()) <= WORKER_HEARTBEAT_STALE_SECONDS


def _heartbeat_worker(db: Database, run_id: str, worker_id: str, stop_event: threading.Event) -> None:
    while not stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS):
        now = utc_now()
        result = db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "status": {"$in": list(ACTIVE_STATUSES)}},
            {"$set": {
                "worker_id": worker_id,
                "worker_active": True,
                "worker_heartbeat_at": now,
                "updated_at": now,
            }},
        )
        if int(getattr(result, "matched_count", 0) or 0) == 0:
            return


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
    logger.info(
        "asset_discovery_event run_id=%s phase=%s message=%s",
        run_id,
        phase or "",
        str(message)[:500],
    )


def _set_stage_progress(
    db: Database,
    run_id: str,
    *,
    step: str,
    percent: float | None,
    label: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    changes: dict[str, Any] = {
        "progress_step": str(step or "")[:80],
        "stage_progress_percent": None if percent is None else round(max(0.0, min(100.0, float(percent))), 1),
        "current_stage": str(label or "")[:240],
        "stage_current": int(current) if current is not None else None,
        "stage_total": int(total) if total is not None else None,
        "updated_at": utc_now(),
    }
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {"$set": bson_value(changes)},
    )


def _ranker_progress_callback(db: Database, run_id: str) -> Any:
    last_percent = [-1.0]
    last_step = [""]

    def emit(percent: float, step: str) -> None:
        rounded = round(max(0.0, min(100.0, float(percent or 0.0))), 1)
        safe_step = str(step or "ranker_fit")[:80]
        if safe_step == last_step[0] and rounded < 100.0 and rounded - last_percent[0] < 1.0:
            return
        last_percent[0] = rounded
        last_step[0] = safe_step
        current = None
        total = None
        label = "Training Learning-to-Rank"
        if safe_step == "training_dataset":
            label = "Preparing Learning-to-Rank training dataset"
        elif safe_step.startswith("walk_forward:"):
            parts = safe_step.split(":")
            if len(parts) == 3:
                try:
                    current = int(parts[1])
                    total = int(parts[2])
                except ValueError:
                    current = None
                    total = None
            label = "Running purged walk-forward validation"
        elif safe_step == "final_refit":
            label = "Refitting final Learning-to-Rank model"
        elif safe_step == "ranker_completed":
            label = "Learning-to-Rank training completed"
        elif safe_step == "causal_refit":
            label = "Refitting historical causal-selection model"
        _set_stage_progress(
            db,
            run_id,
            step=safe_step.split(":", 1)[0],
            percent=rounded,
            label=label,
            current=current,
            total=total,
        )

    return emit


def _increment(db: Database, run_id: str, values: dict[str, int]) -> None:
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {"$inc": {key: int(value) for key, value in values.items()}, "$set": {"updated_at": utc_now()}},
    )


def _reject(db: Database, run_id: str, reason: str) -> None:
    safe_reason = str(reason or "unknown").strip().lower().replace(".", "_").replace("$", "_")[:80]
    _increment(db, run_id, {"rejected_count": 1, f"rejection_summary.{safe_reason}": 1})


def _discover_asset_metadata(db: Database) -> dict[str, dict[str, str | None]]:
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
    result: dict[str, dict[str, str | None]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        exchange = str(item.get("exchange") or "").strip().upper()
        if not symbol or not SYMBOL_PATTERN.fullmatch(symbol):
            continue
        if exchange not in SUPPORTED_EXCHANGES or not bool(item.get("tradable")):
            continue
        company_name = str(item.get("name") or "").strip() or None
        result[symbol] = {
            "symbol": symbol,
            "company_name": company_name,
            "exchange": exchange or None,
        }
    return result


def _discover_symbols(db: Database) -> list[str]:
    return sorted(_discover_asset_metadata(db))


def _backfill_company_metadata(db: Database, document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return document
    rows = document.get("results") if isinstance(document.get("results"), list) else []
    missing = [
        str(row.get("symbol") or "").strip().upper()
        for row in rows
        if isinstance(row, dict) and str(row.get("symbol") or "").strip() and not str(row.get("company_name") or "").strip()
    ]
    if not missing:
        return document
    try:
        metadata = _discover_asset_metadata(db)
    except Exception:
        return document
    changed = False
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            updated_rows.append(row)
            continue
        copy = dict(row)
        symbol = str(copy.get("symbol") or "").strip().upper()
        asset = metadata.get(symbol) or {}
        if not str(copy.get("company_name") or "").strip() and asset.get("company_name"):
            copy["company_name"] = asset.get("company_name")
            copy["exchange"] = asset.get("exchange")
            changed = True
        updated_rows.append(copy)
    if changed:
        db[COLLECTION].update_one(
            {"_id": document.get("_id", CURRENT_ID)},
            {"$set": {"results": bson_value(updated_rows), "updated_at": utc_now()}},
        )
        document = dict(document)
        document["results"] = updated_rows
    return document


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
    try:
        cleaned = validate_and_clean_bars(frame, candidate_config)
    except ValueError as exc:
        raise RuntimeError("insufficient_history") from exc
    coverage = _history_coverage_against_baseline(symbol, cleaned, candidate_config, required_sessions)
    return cleaned, coverage


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
        cleaned_downloaded = validate_and_clean_bars(downloaded, selected_config)
        _history_coverage_against_baseline(symbol, cleaned_downloaded, selected_config, required_sessions)
        complete_market_history(symbol, cleaned_downloaded, selected_config, provider="alpaca")
    except Exception as exc:
        reason = str(exc).strip().lower()
        if reason == "ticker_identity_discontinuity":
            raise AssetDiscoveryConflict(
                f"{symbol} contains a long internal historical gap consistent with ticker reuse and cannot be used for a comparable replay."
            ) from exc
        if reason == "discontinuous_history":
            raise AssetDiscoveryConflict(
                f"{symbol} does not cover every historical session required by the source Strategy and cannot be used for a comparable replay."
            ) from exc
        raise AssetDiscoveryConflict(
            f"{symbol} does not provide a complete clean historical window for the source Strategy: {str(exc)}"
        ) from exc
    collection = db[ALPACA_MARKET_BARS_COLLECTION]
    identity = _market_data_identity(symbol, selected_config)
    collection.delete_many(identity)
    _upsert_frame(collection, cleaned_downloaded, identity, selected_config.mongo_write_batch_size)

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
    return int(len(cleaned_downloaded))

def _candidate_frame(
    db: Database,
    symbol: str,
    config: Any,
    end_session: pd.Timestamp,
    *,
    credentials: dict[str, str] | None = None,
) -> pd.DataFrame:
    credentials = credentials or get_alpaca_credentials(db)
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


def _frame_through(frame: pd.DataFrame, end_session: str | pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    cutoff = pd.Timestamp(end_session)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    return frame.loc[index <= cutoff].copy()


def _causal_validation_window(required_sessions: pd.DatetimeIndex) -> dict[str, Any]:
    sessions = pd.DatetimeIndex(required_sessions).tz_localize(None).normalize().unique().sort_values()
    if CAUSAL_VALIDATION_SESSIONS + CAUSAL_CERTIFICATION_SESSIONS != CAUSAL_HOLDOUT_SESSIONS:
        raise RuntimeError("Asset Discovery causal holdout partition is inconsistent.")
    minimum = CAUSAL_MIN_TRAINING_SESSIONS + CAUSAL_HOLDOUT_SESSIONS + 1
    if len(sessions) < minimum:
        raise RuntimeError(
            "Asset Discovery causal validation requires enough history to reserve an untouched temporal holdout."
        )
    selection_index = len(sessions) - CAUSAL_HOLDOUT_SESSIONS - 1
    validation_start_index = selection_index + 1
    validation_end_index = validation_start_index + CAUSAL_VALIDATION_SESSIONS - 1
    certification_start_index = validation_end_index + 1
    certification_end_index = certification_start_index + CAUSAL_CERTIFICATION_SESSIONS - 1
    if certification_end_index != len(sessions) - 1:
        raise RuntimeError("Asset Discovery causal holdout does not end at the synchronized market cutoff.")

    selection_cutoff = pd.Timestamp(sessions[selection_index])
    validation_start = pd.Timestamp(sessions[validation_start_index])
    validation_end = pd.Timestamp(sessions[validation_end_index])
    certification_start = pd.Timestamp(sessions[certification_start_index])
    certification_end = pd.Timestamp(sessions[certification_end_index])
    return {
        "method": "historical_selection_then_validation_then_certification",
        "holdout_sessions": int(CAUSAL_HOLDOUT_SESSIONS),
        "validation_sessions": int(CAUSAL_VALIDATION_SESSIONS),
        "certification_sessions": int(CAUSAL_CERTIFICATION_SESSIONS),
        "selection_cutoff": selection_cutoff.date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "certification_start": certification_start.date().isoformat(),
        "certification_end": certification_end.date().isoformat(),
        # Compatibility aliases used by older UI/export consumers. They now refer only
        # to the candidate-screening validation slice, never to final certification.
        "evaluation_start": validation_start.date().isoformat(),
        "evaluation_end": validation_end.date().isoformat(),
        "selection_precedes_evaluation": bool(selection_cutoff < validation_start),
        "validation_precedes_certification": bool(validation_end < certification_start),
        "historical_gain_used_for_selection": False,
        "validation_gain_can_select_candidate": True,
        "promotion_uses_holdout_only": True,
        "promotion_uses_certification_only": True,
        "certification_reuse_policy": "non_overlapping_windows",
        "external_universe_snapshot": "current_active_alpaca_universe",
        "survivorship_bias_fully_eliminated": False,
        "historical_certification_is_protocol_isolated": True,
        "globally_unseen_to_project": False,
        "forward_confirmation_required_for_strongest_evidence": True,
    }


def _causal_sample_seed(strategy_hash: str, selection_cutoff: str) -> int:
    material = f"{str(strategy_hash or '').strip()}|{str(selection_cutoff or '').strip()}|asset-discovery-causal-v10"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _causal_sample_priority(strategy_hash: str, selection_cutoff: str, symbol: str) -> str:
    material = f"{str(strategy_hash or '').strip()}|{str(selection_cutoff or '').strip()}|{str(symbol or '').strip().upper()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _certification_window_status(db: Database, causal_window: dict[str, Any]) -> dict[str, Any]:
    certification_start = str(causal_window.get("certification_start") or "").strip()
    certification_end = str(causal_window.get("certification_end") or "").strip()
    latest = db[CERTIFICATION_LEDGER_COLLECTION].find_one(
        {"status": "consumed"},
        sort=[("certification_end", -1)],
    )
    last_end = str((latest or {}).get("certification_end") or "").strip()
    available = bool(certification_start and certification_end) and (not last_end or certification_start > last_end)
    return {
        "certification_available": available,
        "last_consumed_certification_end": last_end or None,
        "certification_reuse_policy": "non_overlapping_windows",
        "certification_block_reason": None if available else "certification_window_overlaps_previously_consumed_data",
    }


def _consume_certification_window(
    db: Database,
    *,
    run_id: str,
    validation_id: str,
    source_strategy_id: str,
    source_strategy_hash: str,
    selected_assets: list[str],
    causal_window: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    certification_start = str(causal_window.get("certification_start") or "").strip()
    certification_end = str(causal_window.get("certification_end") or "").strip()
    if not certification_start or not certification_end:
        raise AssetDiscoveryConflict("The causal certification window is incomplete.")
    status = _certification_window_status(db, causal_window)
    if not bool(status.get("certification_available")):
        raise AssetDiscoveryConflict(
            "This certification period overlaps data already exposed by a previous Asset Discovery certification. "
            "Wait for a new non-overlapping certification period before validating another promotion."
        )
    window_id = f"{certification_start}__{certification_end}"
    now = utc_now()
    document = {
        "_id": window_id,
        "status": "consumed",
        "certification_start": certification_start,
        "certification_end": certification_end,
        "selection_cutoff": causal_window.get("selection_cutoff"),
        "validation_start": causal_window.get("validation_start"),
        "validation_end": causal_window.get("validation_end"),
        "run_id": run_id,
        "validation_id": validation_id,
        "source_strategy_id": source_strategy_id,
        "source_strategy_hash": source_strategy_hash,
        "selected_assets": _selection_symbols(selected_assets),
        "decision": str(decision or "").upper(),
        "consumed_at": now,
        "updated_at": now,
    }
    try:
        db[CERTIFICATION_LEDGER_COLLECTION].insert_one(bson_value(document))
    except Exception as exc:
        # A duplicate window means another certification has already consumed it.
        if db[CERTIFICATION_LEDGER_COLLECTION].find_one({"_id": window_id}) is not None:
            raise AssetDiscoveryConflict(
                "This certification period has already been consumed and cannot be reused for another asset selection."
            ) from exc
        raise
    return document

def _annotate_causal_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        item for item in rows
        if isinstance(item.get("causal_selection"), dict)
        and bool(item["causal_selection"].get("available"))
        and _finite_number(item["causal_selection"].get("raw_score")) is not None
    ]
    eligible.sort(key=lambda item: float(item["causal_selection"]["raw_score"]), reverse=True)
    count = len(eligible)
    for index, item in enumerate(eligible, start=1):
        causal = dict(item.get("causal_selection") or {})
        causal["rank"] = index
        causal["rank_score"] = 1.0 if count <= 1 else float(1.0 - ((index - 1) / (count - 1)))
        item["causal_selection"] = causal
    return eligible


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


def _rank_all_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(item) for item in rows), key=lambda item: float(item["raw_score"]), reverse=True)
    count = len(ordered)
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
        row["rank_score"] = 1.0 if count <= 1 else float(1.0 - ((index - 1) / (count - 1)))
    return ordered


def _rank_results(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    ordered = _rank_all_results(rows)
    count = len(ordered)
    return ordered[: max(1, min(limit, count))] if count else []



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


def _prediction_sessions(results: list[Any]) -> pd.DatetimeIndex:
    session_sets: list[set[pd.Timestamp]] = []
    for result in results:
        predictions = getattr(result, "predictions", None)
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if isinstance(predictions.index, pd.DatetimeIndex):
            values = pd.DatetimeIndex(pd.to_datetime(predictions.index, utc=True))
        elif "timestamp" in predictions.columns:
            values = pd.DatetimeIndex(pd.to_datetime(predictions["timestamp"], utc=True))
        else:
            continue
        normalized = values.tz_convert("UTC").tz_localize(None).normalize().unique()
        session_sets.append(set(pd.DatetimeIndex(normalized)))
    if not session_sets:
        return pd.DatetimeIndex([])
    common = set.intersection(*session_sets)
    return pd.DatetimeIndex(sorted(common))


def _research_context_compatibility(
    baseline_sessions: pd.DatetimeIndex,
    candidate_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    baseline = pd.DatetimeIndex(baseline_sessions).tz_localize(None).normalize().unique().sort_values()
    candidate = pd.DatetimeIndex(candidate_sessions).tz_localize(None).normalize().unique().sort_values()
    candidate_set = set(candidate)
    missing = pd.DatetimeIndex([session for session in baseline if session not in candidate_set])
    return {
        "research_context_compatible": not bool(len(missing)),
        "research_context_baseline_sessions": int(len(baseline)),
        "research_context_candidate_sessions": int(len(candidate)),
        "research_context_missing_sessions": int(len(missing)),
        "research_context_first_missing_session": pd.Timestamp(missing[0]).date().isoformat() if len(missing) else None,
        "research_context_last_missing_session": pd.Timestamp(missing[-1]).date().isoformat() if len(missing) else None,
    }


def _monthly_replay_counts(result: Any, severe_threshold: float) -> tuple[int | None, int | None]:
    predictions = getattr(result, "predictions", None)
    if not isinstance(predictions, pd.DataFrame) or predictions.empty or "strategy_equity" not in predictions.columns:
        return None, None
    values = pd.to_numeric(predictions["strategy_equity"], errors="coerce").dropna()
    if values.empty:
        return None, None
    if not isinstance(values.index, pd.DatetimeIndex):
        if "timestamp" not in predictions.columns:
            return None, None
        timestamps = pd.to_datetime(predictions.loc[values.index, "timestamp"], utc=True, errors="coerce")
        values.index = pd.DatetimeIndex(timestamps)
    else:
        values.index = pd.DatetimeIndex(pd.to_datetime(values.index, utc=True))
    if values.index.tz is not None:
        values.index = values.index.tz_convert("UTC").tz_localize(None)
    monthly = values.groupby(values.index.to_period("M")).last().pct_change().dropna()
    if monthly.empty:
        return 0, 0
    return int((monthly < 0.0).sum()), int((monthly <= float(severe_threshold)).sum())


def _median_numbers(values: list[float | int | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and pd.notna(value))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _aggregate_rotation_replay(results: list[Any], *, severe_threshold: float = DEFAULT_SEVERE_MONTH_THRESHOLD) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Marginal Capital Replay produced no rotation result.")
    fold_returns: list[float] = []
    negative_month_counts: list[int | None] = []
    severe_month_counts: list[int | None] = []
    for result in results:
        folds = (result.metrics or {}).get("walk_forward_folds") or []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            value = _finite_number(fold.get("strategy_return"))
            if value is not None:
                fold_returns.append(value)
        negative_count, severe_count = _monthly_replay_counts(result, severe_threshold)
        negative_month_counts.append(negative_count)
        severe_month_counts.append(severe_count)
    sessions = _prediction_sessions(results)
    return {
        "ending_capital": _median_metric(results, "strategy_ending_capital"),
        "cagr": _median_metric(results, "strategy_cagr"),
        "sharpe": _median_metric(results, "strategy_sharpe"),
        "maximum_drawdown": _median_metric(results, "strategy_maximum_drawdown"),
        "market_exposure": _median_metric(results, "market_exposure"),
        "cash_days": _median_metric(results, "cash_days"),
        "switches": _median_metric(results, "capital_rotations"),
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "negative_months": _median_numbers(negative_month_counts),
        "severe_negative_months": _median_numbers(severe_month_counts),
        "severe_month_threshold": float(severe_threshold),
        "repetition_count": len(results),
        "decision_session_count": int(len(sessions)),
        "decision_session_start": pd.Timestamp(sessions[0]).date().isoformat() if len(sessions) else None,
        "decision_session_end": pd.Timestamp(sessions[-1]).date().isoformat() if len(sessions) else None,
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
    analysis_start_date: str | None = None,
    analysis_end_date: str | None = None,
) -> BacktestExecutionRequest:
    snapshot = get_strategy_model_snapshot(db, str(strategy.get("id") or ""))
    family = str(snapshot.get("family") or "lightgbm_utility")
    if family != "lightgbm_utility":
        raise RuntimeError("Asset Discovery requires a LightGBM Utility Strategy; XGBoost was retired in API v8.0.0.")
    settings = dict(snapshot.get("settings_snapshot") or {}) if isinstance(snapshot.get("settings_snapshot"), dict) else {}
    locked = base_config.model_copy(update={"assets": assets, "end_date": end_session})
    locked = apply_training_runtime_settings(db, locked)
    locked = apply_execution_profile(locked, family, settings)
    anchors = list(assets)
    return BacktestExecutionRequest.model_validate({
        **locked.model_dump(mode="python"),
        "analysis_start_date": analysis_start_date or locked.start_date,
        "analysis_end_date": analysis_end_date or end_session,
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
    severe_threshold: float = DEFAULT_SEVERE_MONTH_THRESHOLD,
) -> tuple[dict[str, Any], pd.DatetimeIndex]:
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
    return _aggregate_rotation_replay(results, severe_threshold=severe_threshold), _prediction_sessions(results)


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
                "progress_step": "marginal_replay",
                "stage_progress_percent": rounded,
                "current_stage": safe_stage or "Running Marginal Capital Replay",
                "stage_current": int(current_index),
                "stage_total": max(0, int(total_runs) - 1),
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


def _marginal_replay_is_persistent_candidate(replay: Any) -> bool:
    if not isinstance(replay, dict):
        return False
    if str(replay.get("status") or "").lower() != "completed":
        return False
    if str(replay.get("validation_method") or "") != "causal_temporal_validation":
        return False
    if not bool(replay.get("selection_precedes_evaluation")):
        return False
    if not bool(replay.get("research_context_compatible", True)):
        return False
    delta = _finite_number(replay.get("ending_capital_delta_rate"))
    return delta is not None and delta > MIN_PERSISTENT_MARGINAL_CAPITAL_DELTA_RATE


def _item_is_persistent_candidate(item: Any) -> bool:
    if not isinstance(item, dict) or not bool(item.get("history_window_complete")):
        return False
    return _marginal_replay_is_persistent_candidate(item.get("marginal_replay"))


def _selection_symbols(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value or "").strip().upper() for value in values if str(value or "").strip()))


def _selection_matches(left: Any, right: Any) -> bool:
    return sorted(_selection_symbols(list(left or []))) == sorted(_selection_symbols(list(right or [])))


def _evaluate_marginal_candidate(
    db: Database,
    *,
    item: dict[str, Any],
    config: BacktestRequest,
    strategy: dict[str, Any],
    winner_config: BacktestRequest,
    baseline_assets: list[str],
    validation_baseline_frames: dict[str, pd.DataFrame],
    baseline_metrics: dict[str, Any],
    baseline_decision_sessions: pd.DatetimeIndex,
    required_sessions: pd.DatetimeIndex,
    evaluation_start: str,
    evaluation_end: str,
    selection_cutoff: str,
    causal_window: dict[str, Any],
    candidate_frame_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    symbol = str(item.get("symbol") or "").strip().upper()
    row: dict[str, Any] = {"symbol": symbol, "status": "completed"}
    try:
        cached_frame = (candidate_frame_cache or {}).get(symbol)
        if cached_frame is not None and not cached_frame.empty:
            candidate_frame = _frame_through(cached_frame, evaluation_end)
            coverage = _history_coverage_against_baseline(symbol, candidate_frame, config, required_sessions)
        else:
            candidate_frame, coverage = _candidate_history_coverage(
                db, symbol, config, pd.Timestamp(evaluation_end), required_sessions
            )
        row["history_window_complete"] = bool(coverage.get("history_window_complete"))
        candidate_assets = list(dict.fromkeys([*baseline_assets, symbol]))
        candidate_request = _marginal_execution_request(
            db,
            config,
            strategy,
            winner_config,
            evaluation_end,
            assets=candidate_assets,
            reference_assets=baseline_assets,
            candidate_assets=[symbol],
            analysis_start_date=evaluation_start,
            analysis_end_date=evaluation_end,
        )
        candidate_frames = dict(validation_baseline_frames)
        candidate_frames[symbol] = candidate_frame
        candidate_metrics, candidate_decision_sessions = _run_rotation_replay(
            candidate_frames,
            candidate_request,
            progress_callback=None,
        )
        context_compatibility = _research_context_compatibility(
            baseline_decision_sessions, candidate_decision_sessions
        )
        if not bool(context_compatibility.get("research_context_compatible")):
            row.update({
                "status": "rejected",
                "rejection_reason": "research_context_incomplete",
                **context_compatibility,
            })
        else:
            row.update({
                **context_compatibility,
                "validation_method": "causal_temporal_validation",
                "selection_cutoff": selection_cutoff,
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "validation_sessions": int(causal_window.get("validation_sessions") or 0),
                "certification_start": causal_window.get("certification_start"),
                "certification_end": causal_window.get("certification_end"),
                "certification_sessions": int(causal_window.get("certification_sessions") or 0),
                "holdout_sessions": int(causal_window.get("holdout_sessions") or 0),
                "selection_precedes_evaluation": bool(causal_window.get("selection_precedes_evaluation")),
                "historical_gain_used_for_selection": False,
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
    except RuntimeError as exc:
        reason = str(exc).strip().lower()
        if reason in {
            "insufficient_history",
            "discontinuous_history",
            "ticker_identity_discontinuity",
            "price_filter",
            "liquidity_filter",
            "volume_quality_filter",
        }:
            row.update({"status": "rejected", "rejection_reason": reason, "history_window_complete": False})
        else:
            row.update({"status": "failed", "error": str(exc)[:700]})
    except Exception as exc:
        row.update({"status": "failed", "error": str(exc)[:700]})

    row["persistence_eligible"] = _marginal_replay_is_persistent_candidate(row)
    row["persistence_reason"] = (
        "positive_causal_validation_capital"
        if row["persistence_eligible"]
        else "causal_validation_not_positive"
    )
    return row


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
    causal_window: dict[str, Any],
    candidate_frame_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_assets = [str(symbol).strip().upper() for symbol in config.assets]
    evaluation_start = str(causal_window.get("validation_start") or causal_window.get("evaluation_start") or "").strip()
    evaluation_end = str(causal_window.get("validation_end") or causal_window.get("evaluation_end") or "").strip()
    selection_cutoff = str(causal_window.get("selection_cutoff") or "").strip()
    if not evaluation_start or not evaluation_end or not selection_cutoff:
        raise RuntimeError("Asset Discovery causal validation window is incomplete.")

    validation_baseline_frames = {
        symbol: _frame_through(frame, evaluation_end)
        for symbol, frame in baseline_frames.items()
    }
    baseline_request = _marginal_execution_request(
        db,
        config,
        strategy,
        winner_config,
        evaluation_end,
        assets=baseline_assets,
        reference_assets=baseline_assets,
        candidate_assets=[],
        analysis_start_date=evaluation_start,
        analysis_end_date=evaluation_end,
    )
    replay_workers = max(1, min(_replay_worker_count(), len(shortlist) or 1))
    _event(
        db,
        run_id,
        "Running the baseline replay once before parallel causal validation.",
        phase="marginal_replay",
        changes={
            "marginal_replay": {
                "status": "running",
                "total_count": len(shortlist),
                "completed_count": 0,
                "current_symbol": None,
                "current_index": 0,
                "current_stage": "Preparing baseline replay",
                "progress_percent": 0.0,
                "baseline": None,
                "results": [],
                "validation_method": "causal_temporal_validation",
                "selection_cutoff": selection_cutoff,
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "validation_sessions": int(causal_window.get("validation_sessions") or 0),
                "certification_start": causal_window.get("certification_start"),
                "certification_end": causal_window.get("certification_end"),
                "certification_sessions": int(causal_window.get("certification_sessions") or 0),
                "holdout_sessions": int(causal_window.get("holdout_sessions") or 0),
                "parallel_workers": replay_workers,
                "persistence_policy": "retained_candidates_and_aggregate_counts_only",
            },
            "results": [],
            "shortlisted_count": 0,
            "validation_candidate_count": len(shortlist),
        },
    )
    baseline_metrics, baseline_decision_sessions = _run_rotation_replay(
        validation_baseline_frames,
        baseline_request,
        progress_callback=_marginal_progress_callback(
            db,
            run_id,
            run_position=0,
            total_runs=max(1, len(shortlist) + 1),
            current_symbol="BASELINE",
            current_index=0,
            completed_count=0,
        ),
    )
    if baseline_decision_sessions.empty:
        raise RuntimeError("Asset Discovery baseline replay produced no decision-session context.")

    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {"$set": {
            "updated_at": utc_now(),
            "marginal_replay.progress_percent": round(100.0 / max(1, len(shortlist) + 1), 1),
            "marginal_replay.current_symbol": None,
            "marginal_replay.current_index": 0,
            "marginal_replay.current_stage": "Baseline replay completed; validating candidates in parallel",
            "marginal_replay.baseline": bson_value(baseline_metrics),
        }},
    )

    retained_results: list[dict[str, Any]] = []
    retained_replay_rows: list[dict[str, Any]] = []
    candidate_map = {
        str(item.get("symbol") or "").strip().upper(): dict(item)
        for item in shortlist
        if str(item.get("symbol") or "").strip()
    }
    completed_count = 0
    low_adherence_count = 0
    history_rejected_count = 0
    research_context_rejected_count = 0
    technical_failure_count = 0
    rankable_count = len(candidate_map)
    ordered_candidates = sorted(
        candidate_map.values(),
        key=lambda item: int((item.get("causal_selection") or {}).get("rank") or item.get("rank") or 999999),
    )

    for batch_index, batch_start in enumerate(range(0, len(ordered_candidates), replay_workers), start=1):
        if _should_stop_after_batch(db, run_id):
            break
        batch = ordered_candidates[batch_start: batch_start + replay_workers]
        _event(
            db,
            run_id,
            f"Parallel causal validation batch {batch_index} started with {len(batch)} candidates.",
            phase="marginal_replay",
            changes={
                "current_batch": batch_index,
                "current_symbol": None,
                "marginal_replay.current_symbol": None,
                "marginal_replay.current_stage": "Running candidate replays in parallel",
            },
        )
        with ThreadPoolExecutor(
            max_workers=max(1, min(replay_workers, len(batch))),
            thread_name_prefix="mct-asset-discovery-replay",
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_marginal_candidate,
                    db,
                    item=item,
                    config=config,
                    strategy=strategy,
                    winner_config=winner_config,
                    baseline_assets=baseline_assets,
                    validation_baseline_frames=validation_baseline_frames,
                    baseline_metrics=baseline_metrics,
                    baseline_decision_sessions=baseline_decision_sessions,
                    required_sessions=required_sessions,
                    evaluation_start=evaluation_start,
                    evaluation_end=evaluation_end,
                    selection_cutoff=selection_cutoff,
                    causal_window=causal_window,
                    candidate_frame_cache=candidate_frame_cache,
                )
                for item in batch
            ]
            for future in as_completed(futures):
                row = future.result()
                completed_count += 1
                symbol = str(row.get("symbol") or "").strip().upper()
                if bool(row.get("history_window_complete")):
                    _increment(db, run_id, {"adherence_validated_count": 1})

                if bool(row.get("persistence_eligible")):
                    retained_row = dict(row)
                    retained_replay_rows.append(retained_row)
                    source = dict(candidate_map.get(symbol) or {"symbol": symbol})
                    source["marginal_replay"] = retained_row
                    source["persistence_eligible"] = True
                    source["persistence_reason"] = "positive_causal_validation_capital"
                    retained_results.append(source)
                else:
                    rejection_reason = str(row.get("rejection_reason") or "").strip().lower()
                    if rejection_reason:
                        _reject(db, run_id, rejection_reason)
                        if rejection_reason == "research_context_incomplete":
                            research_context_rejected_count += 1
                        else:
                            history_rejected_count += 1
                    elif str(row.get("status") or "") == "completed":
                        _reject(db, run_id, "low_strategy_adherence")
                        low_adherence_count += 1
                    else:
                        _increment(db, run_id, {"technical_failure_count": 1})
                        technical_failure_count += 1

                retained_replay_rows.sort(
                    key=lambda candidate: float(candidate.get("ending_capital_delta_rate") or 0.0),
                    reverse=True,
                )
                for rank, retained in enumerate(retained_replay_rows, start=1):
                    retained["marginal_rank"] = rank
                retained_rank = {
                    str(item.get("symbol") or "").strip().upper(): index
                    for index, item in enumerate(retained_replay_rows, start=1)
                }
                for retained in retained_results:
                    replay = retained.get("marginal_replay") if isinstance(retained.get("marginal_replay"), dict) else None
                    symbol_key = str(retained.get("symbol") or "").strip().upper()
                    if replay is not None and symbol_key in retained_rank:
                        replay["marginal_rank"] = retained_rank[symbol_key]

                progress = round(100.0 * (completed_count + 1) / max(1, rankable_count + 1), 1)
                db[COLLECTION].update_one(
                    {"_id": CURRENT_ID, "run_id": run_id},
                    {"$set": bson_value({
                        "updated_at": utc_now(),
                        "marginal_replay.status": "running",
                        "marginal_replay.completed_count": completed_count,
                        "marginal_replay.current_symbol": None,
                        "marginal_replay.current_index": completed_count,
                        "marginal_replay.current_stage": "Running candidate replays in parallel",
                        "marginal_replay.progress_percent": progress,
                        "marginal_replay.results": retained_replay_rows,
                        "marginal_replay.persistent_candidate_count": len(retained_replay_rows),
                        "marginal_replay.low_adherence_count": low_adherence_count,
                        "marginal_replay.history_rejected_count": history_rejected_count,
                        "marginal_replay.research_context_rejected_count": research_context_rejected_count,
                        "marginal_replay.technical_failure_count": technical_failure_count,
                        "results": retained_results,
                        "shortlisted_count": len(retained_results),
                        "stage_progress_percent": progress,
                        "current_stage": "Running candidate replays in parallel",
                        "stage_current": completed_count,
                        "stage_total": rankable_count,
                        "current_symbol": None,
                    })},
                )

    retained_replay_rows.sort(
        key=lambda row: float(row.get("ending_capital_delta_rate") or 0.0),
        reverse=True,
    )
    for rank, row in enumerate(retained_replay_rows, start=1):
        row["marginal_rank"] = rank
    rank_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): rank
        for rank, row in enumerate(retained_replay_rows, start=1)
    }
    for item in retained_results:
        symbol = str(item.get("symbol") or "").strip().upper()
        replay = item.get("marginal_replay") if isinstance(item.get("marginal_replay"), dict) else None
        if replay is not None and symbol in rank_by_symbol:
            replay["marginal_rank"] = rank_by_symbol[symbol]

    stopped = completed_count < rankable_count
    replay_summary = {
        "status": "stopped" if stopped else "completed",
        "total_count": rankable_count,
        "completed_count": completed_count,
        "current_symbol": None,
        "current_index": completed_count,
        "current_stage": "Marginal Capital Replay stopped" if stopped else "Marginal Capital Replay completed",
        "progress_percent": round(100.0 * (completed_count + 1) / max(1, rankable_count + 1), 1) if stopped else 100.0,
        "baseline": baseline_metrics,
        "validation_method": "causal_temporal_validation",
        "selection_cutoff": selection_cutoff,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "validation_sessions": int(causal_window.get("validation_sessions") or 0),
        "certification_start": causal_window.get("certification_start"),
        "certification_end": causal_window.get("certification_end"),
        "certification_sessions": int(causal_window.get("certification_sessions") or 0),
        "holdout_sessions": int(causal_window.get("holdout_sessions") or 0),
        "selection_precedes_evaluation": bool(causal_window.get("selection_precedes_evaluation")),
        "historical_gain_used_for_selection": False,
        "parallel_workers": replay_workers,
        "persistence_policy": "retained_candidates_and_aggregate_counts_only",
        "results": retained_replay_rows,
        "eligible_count": len(retained_results),
        "persistent_candidate_count": len(retained_replay_rows),
        "low_adherence_count": low_adherence_count,
        "history_rejected_count": history_rejected_count,
        "research_context_rejected_count": research_context_rejected_count,
        "technical_failure_count": technical_failure_count,
    }
    return retained_results, replay_summary

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
    if isinstance(item.get("causal_selection"), dict):
        metrics["causal_selection"] = dict(item.get("causal_selection") or {})
    if isinstance(item.get("current_snapshot"), dict):
        metrics["current_snapshot"] = dict(item.get("current_snapshot") or {})
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
            "research_context_compatible": marginal.get("research_context_compatible"),
            "research_context_missing_sessions": marginal.get("research_context_missing_sessions"),
            "validation_method": marginal.get("validation_method"),
            "selection_cutoff": marginal.get("selection_cutoff"),
            "evaluation_start": marginal.get("evaluation_start"),
            "evaluation_end": marginal.get("evaluation_end"),
            "validation_sessions": marginal.get("validation_sessions"),
            "certification_start": marginal.get("certification_start"),
            "certification_end": marginal.get("certification_end"),
            "certification_sessions": marginal.get("certification_sessions"),
            "holdout_sessions": marginal.get("holdout_sessions"),
            "selection_precedes_evaluation": marginal.get("selection_precedes_evaluation"),
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
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if not _item_is_persistent_candidate(item):
            db[CATALOG_COLLECTION].delete_one({"_id": symbol})
            continue
        existing = db[CATALOG_COLLECTION].find_one({"_id": symbol}) or {}
        recent_run_ids = [str(value) for value in existing.get("recent_run_ids") or []]
        already_counted = run_id in recent_run_ids
        causal_selection = item.get("causal_selection") if isinstance(item.get("causal_selection"), dict) else {}
        rank = int(causal_selection.get("rank") or item.get("rank") or 0) or None
        raw_score = causal_selection.get("raw_score") if causal_selection else item.get("raw_score")
        discovery = {
            "run_id": run_id,
            "seen_at": now,
            "rank": rank,
            "raw_score": raw_score,
            "snapshot_end": baseline.get("market_snapshot_end"),
        }
        set_fields = {
            "symbol": symbol,
            "company_name": item.get("company_name") or existing.get("company_name"),
            "exchange": item.get("exchange") or existing.get("exchange"),
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
    for stored in list(db[CATALOG_COLLECTION].find({})):
        if not isinstance(stored, dict):
            continue
        metrics = stored.get("latest_metrics") if isinstance(stored.get("latest_metrics"), dict) else {}
        persisted_view = {
            "history_window_complete": bool(stored.get("history_window_complete")),
            **metrics,
        }
        if not _item_is_persistent_candidate(persisted_view):
            db[CATALOG_COLLECTION].delete_one({"_id": stored.get("_id")})
    missing_company = [
        str(item.get("symbol") or item.get("_id") or "").strip().upper()
        for item in db[CATALOG_COLLECTION].find({"$or": [{"company_name": {"$exists": False}}, {"company_name": None}, {"company_name": ""}]}, {"symbol": 1})
        if str(item.get("symbol") or item.get("_id") or "").strip()
    ]
    if missing_company:
        try:
            metadata = _discover_asset_metadata(db)
            now = utc_now()
            for symbol in missing_company:
                asset = metadata.get(symbol) or {}
                if asset.get("company_name"):
                    db[CATALOG_COLLECTION].update_one(
                        {"_id": symbol},
                        {"$set": {"company_name": asset.get("company_name"), "exchange": asset.get("exchange"), "updated_at": now}},
                    )
        except Exception:
            pass
    assets = [_public(item) for item in db[CATALOG_COLLECTION].find({}).sort([("last_seen_at", -1), ("times_discovered", -1), ("best_rank", 1)])]
    clean_assets = [item for item in assets if item is not None]
    return {
        "api_version": API_VERSION,
        "count": len(clean_assets),
        "assets": clean_assets,
        "persistence_policy": {
            "market_bars": "not_stored_by_catalog",
            "rejected_assets": "not_stored",
            "low_adherence_assets": "not_stored",
            "minimum_marginal_capital_delta_rate": MIN_PERSISTENT_MARGINAL_CAPITAL_DELTA_RATE,
            "validation_method": "causal_temporal_validation",
            "holdout_sessions": CAUSAL_HOLDOUT_SESSIONS,
            "validation_sessions": CAUSAL_VALIDATION_SESSIONS,
            "certification_sessions": CAUSAL_CERTIFICATION_SESSIONS,
            "certification_reuse_policy": "non_overlapping_windows",
            "historical_full_replay_not_eligible": True,
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
        "worker_active": False,
        "worker_finished_at": utc_now(),
        "stage_progress_percent": 100.0 if status == "completed" else None,
        "progress_step": "completed" if status == "completed" else status,
        "current_stage": "Asset Discovery completed" if status == "completed" else message,
        "stage_current": None,
        "stage_total": None,
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


def _run_worker(db: Database, run_id: str, worker_id: str) -> None:
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_worker,
        args=(db, run_id, worker_id, heartbeat_stop),
        name="asset-discovery-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        document = _campaign(db) or {}
        requested = int(document.get("research_size") or DEFAULT_RESEARCH_SIZE)
        _event(
            db,
            run_id,
            "Loading the Strategy Research baseline.",
            phase="baseline",
            changes={
                "status": "running",
                "started_at": utc_now(),
                "progress_step": "baseline",
                "stage_progress_percent": 0.0,
                "current_stage": "Preparing Strategy Research baseline",
                "stage_current": None,
                "stage_total": None,
            },
        )
        config, strategy = get_research_strategy_context(db)
        winner_config, winner_strategy = get_trader_winner_context(db)
        _event(
            db,
            run_id,
            "Synchronizing the Strategy Research market-data cache before freezing the Discovery snapshot.",
            phase="baseline_sync",
            changes={
                "progress_step": "baseline_sync",
                "stage_progress_percent": None,
                "current_stage": "Synchronizing Strategy Research market data",
                "stage_current": None,
                "stage_total": None,
            },
        )
        baseline_sync = refresh_market_data_to_live_cutoff(config)
        end_session = str(baseline_sync.get("live_market_cutoff") or "").strip()
        if not end_session:
            raise RuntimeError("Asset Discovery could not resolve the synchronized Strategy Research market cutoff.")
        safe_session = pd.Timestamp(end_session)
        baseline_frames = _baseline_frames(config, end_session)
        required_sessions = _baseline_required_sessions(baseline_frames, config, safe_session)
        causal_window = _causal_validation_window(required_sessions)
        causal_window.update(_certification_window_status(db, causal_window))
        validation_end_stamp = pd.Timestamp(str(causal_window["validation_end"]))
        validation_required_sessions = pd.DatetimeIndex(required_sessions)[
            pd.DatetimeIndex(required_sessions).tz_localize(None).normalize() <= validation_end_stamp.normalize()
        ]
        causal_cutoff = str(causal_window["selection_cutoff"])
        causal_seed = _causal_sample_seed(str(strategy.get("configuration_hash") or ""), causal_cutoff)
        causal_window["deterministic_sample_seed"] = int(causal_seed)
        causal_window["candidate_sample_policy"] = "stable_hash_order_for_strategy_and_cutoff"
        causal_baseline_frames = {
            symbol: _frame_through(frame, causal_cutoff)
            for symbol, frame in baseline_frames.items()
        }
        causal_baseline_returns = _baseline_recent_returns(causal_baseline_frames)
        _event(
            db,
            run_id,
            "Training the causal ranking model using only history available before the reserved validation and certification periods.",
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
                    "market_data_sync": {
                        "target_session": baseline_sync.get("target_session"),
                        "rows_refreshed": dict(baseline_sync.get("rows_refreshed") or {}),
                        "data_delay_minutes": baseline_sync.get("data_delay_minutes"),
                    },
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
                "causal_validation": causal_window,
                "progress_step": "training_dataset",
                "stage_progress_percent": 0.0,
                "current_stage": "Preparing Learning-to-Rank training dataset",
                "stage_current": None,
                "stage_total": None,
            },
        )
        try:
            bundle = train_ranker(
                causal_baseline_frames,
                random_state=causal_seed,
                stop_check=lambda: _should_stop_after_batch(db, run_id),
                progress_callback=_ranker_progress_callback(db, run_id),
            )
        except AssetDiscoveryRankerCancelled:
            _finish(
                db,
                run_id,
                "stopped",
                "Asset Discovery stopped during Learning-to-Rank at a safe model checkpoint.",
                results=[],
            )
            return
        _event(
            db,
            run_id,
            "Learning-to-Rank training completed.",
            phase="scanning",
            changes={
                "model": bundle.diagnostics,
                "causal_selection_model": {
                    **bundle.diagnostics,
                    "selection_cutoff": causal_window.get("selection_cutoff"),
                    "validation_start": causal_window.get("validation_start"),
                    "validation_end": causal_window.get("validation_end"),
                    "certification_start": causal_window.get("certification_start"),
                    "certification_end": causal_window.get("certification_end"),
                },
                "progress_step": "external_scan",
                "stage_progress_percent": 0.0,
                "current_stage": "Preparing external asset scan",
                "stage_current": 0,
                "stage_total": requested,
            },
        )
        if _should_stop_after_batch(db, run_id):
            _finish(db, run_id, "stopped", "Asset Discovery stopped after Learning-to-Rank training and before the first external batch.", results=[])
            return

        universe_metadata = _discover_asset_metadata(db)
        universe = sorted(universe_metadata)
        baseline_symbols = {str(item).upper() for item in config.assets}
        external = [symbol for symbol in universe if symbol not in baseline_symbols]
        external.sort(
            key=lambda symbol: _causal_sample_priority(
                str(strategy.get("configuration_hash") or ""),
                causal_cutoff,
                symbol,
            )
        )
        scan_budget = min(len(external), requested)
        selected = external[:scan_budget]
        causal_window["external_universe_count"] = len(external)
        causal_window["selected_sample_size"] = scan_budget
        causal_window["selected_sample_hash"] = hashlib.sha256("|".join(selected).encode("utf-8")).hexdigest()
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id},
            {"$set": {"causal_validation": bson_value(causal_window), "updated_at": utc_now()}},
        )
        _event(
            db,
            run_id,
            f"Scanning the deterministic external sample of {scan_budget} symbols frozen for this Strategy and historical cutoff.",
            changes={
                "universe_size": len(universe),
                "external_universe_size": len(external),
                "scan_budget": scan_budget,
                "causal_validation": causal_window,
            },
        )
        if _should_stop_after_batch(db, run_id):
            _finish(db, run_id, "stopped", "Asset Discovery stopped before the first external batch.", results=[])
            return

        evaluated: list[dict[str, Any]] = []
        scan_completed = 0
        scan_batch_size = _scan_batch_size()
        scan_workers = _scan_worker_count()
        for batch_index, batch_start in enumerate(range(0, len(selected), scan_batch_size), start=1):
            batch = selected[batch_start: batch_start + scan_batch_size]
            _event(
                db,
                run_id,
                f"Fast-scanning batch {batch_index} with up to {len(batch)} concurrent market-data requests.",
                changes={"current_batch": batch_index, "current_symbol": None},
            )

            scan_credentials = get_alpaca_credentials(db)
            with ThreadPoolExecutor(
                max_workers=max(1, min(scan_workers, len(batch))),
                thread_name_prefix="mct-asset-discovery-scan",
            ) as executor:
                futures = {
                    executor.submit(
                        _candidate_frame,
                        db,
                        symbol,
                        config,
                        safe_session,
                        credentials=scan_credentials,
                    ): symbol
                    for symbol in batch
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        frame = future.result()
                        causal_frame = _frame_through(frame, causal_window["selection_cutoff"])
                        causal_score = _score_candidate(
                            bundle, symbol, causal_frame, causal_baseline_returns
                        )
                        result = dict(causal_score)
                        result["causal_selection"] = {
                            "available": True,
                            "selection_cutoff": causal_window.get("selection_cutoff"),
                            "raw_score": causal_score.get("raw_score"),
                            "feature_at": causal_score.get("feature_at"),
                            "latest_close": causal_score.get("latest_close"),
                            "median_dollar_volume": causal_score.get("median_dollar_volume"),
                            "return_20": causal_score.get("return_20"),
                            "return_60": causal_score.get("return_60"),
                            "volatility_20": causal_score.get("volatility_20"),
                            "drawdown_60": causal_score.get("drawdown_60"),
                            "trend_efficiency_20": causal_score.get("trend_efficiency_20"),
                            "max_baseline_correlation_60": causal_score.get("max_baseline_correlation_60"),
                        }
                        asset_metadata = universe_metadata.get(symbol) or {}
                        result["company_name"] = asset_metadata.get("company_name")
                        result["exchange"] = asset_metadata.get("exchange")
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
                    finally:
                        scan_completed += 1
                        _set_stage_progress(
                            db,
                            run_id,
                            step="external_scan",
                            percent=(100.0 * scan_completed / max(1, scan_budget)),
                            label="Scanning external assets",
                            current=scan_completed,
                            total=scan_budget,
                        )

            if _should_stop_after_batch(db, run_id):
                _finish(
                    db,
                    run_id,
                    "stopped",
                    f"Asset Discovery stopped after fast-scan batch {batch_index}; unvalidated shortlist data was discarded.",
                    results=[],
                )
                return

        ranked_fast = _rank_all_results(evaluated)
        causal_ranked = _annotate_causal_ranks(ranked_fast)
        causal_ranked.sort(key=lambda item: int((item.get("causal_selection") or {}).get("rank") or 999999))
        causal_unavailable_count = max(0, scan_budget - len(causal_ranked))
        _event(
            db,
            run_id,
            f"Fast scan produced {len(causal_ranked)} historical causal candidates at {causal_window['selection_cutoff']} without using either reserved validation or certification data.",
            phase="marginal_replay",
            changes={
                "fast_evaluated_count": len(ranked_fast),
                "causal_ranked_count": len(causal_ranked),
                "causal_unavailable_count": causal_unavailable_count,
                "adherence_validated_count": 0,
                "validation_candidate_count": len(causal_ranked),
                "results": [],
                "shortlisted_count": 0,
                "progress_step": "marginal_replay",
                "stage_progress_percent": 0.0,
                "current_stage": "Preparing parallel causal validation",
                "stage_current": 0,
                "stage_total": len(causal_ranked),
            },
        )

        retained: list[dict[str, Any]] = []
        marginal_replay: dict[str, Any] = {
            "status": "completed",
            "total_count": 0,
            "completed_count": 0,
            "current_symbol": None,
            "baseline": None,
            "results": [],
        }
        if causal_ranked:
            retained, marginal_replay = _run_marginal_capital_replay(
                db,
                run_id,
                config=config,
                strategy=strategy,
                winner_config=winner_config,
                end_session=end_session,
                baseline_frames=baseline_frames,
                required_sessions=validation_required_sessions,
                shortlist=causal_ranked,
                causal_window=causal_window,
                candidate_frame_cache=None,
            )
            if _should_stop_after_batch(db, run_id):
                _event(
                    db,
                    run_id,
                    "Asset Discovery stopped after the current parallel validation batch; non-retained candidate details were discarded.",
                    changes={"marginal_replay": marginal_replay},
                )
                _finish(
                    db,
                    run_id,
                    "stopped",
                    f"Asset Discovery stopped after validating {marginal_replay.get('completed_count', 0)} of {len(causal_ranked)} causal candidates in bounded parallel batches.",
                    results=retained,
                )
                return
            _event(
                db,
                run_id,
                "Parallel causal validation completed. Only retained candidates and aggregate rejection counts were persisted.",
                changes={"marginal_replay": marginal_replay},
            )
        _finish(
            db,
            run_id,
            "completed",
            f"Asset Discovery causally evaluated {len(evaluated)} sampled assets and retained {len(retained)} validation candidates. Final Strategy certification remains separate and uses later non-overlapping data.",
            results=retained,
        )
    except Exception as exc:
        _finish(db, run_id, "failed", f"Asset Discovery failed: {str(exc)[:900]}")
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "worker_id": worker_id},
            {"$set": {"worker_active": False, "worker_finished_at": utc_now(), "updated_at": utc_now()}},
        )
        global _worker_thread
        with _worker_lock:
            _worker_thread = None


def get_asset_discovery_status(db: Database) -> dict[str, Any]:
    document = _sanitize_completed_campaign_persistence(db, _campaign(db))
    document = _backfill_company_metadata(db, document)
    if document and str(document.get("status") or "") in ACTIVE_STATUSES and not _worker_heartbeat_fresh(document):
        phase = str(document.get("phase") or "").strip().lower()
        stop_was_requested = bool(document.get("cancel_requested")) or str(document.get("status") or "") == "stopping"
        interrupted_marginal = phase == "marginal_replay"
        interrupted_full_validation = phase == "full_strategy_validation"
        changes: dict[str, Any] = {
            "status": "stopped" if stop_was_requested else "interrupted",
            "phase": "stopped" if stop_was_requested else (phase if (interrupted_marginal or interrupted_full_validation) else "interrupted"),
            "message": (
                "Asset Discovery stop completed; no active worker heartbeat remains."
                if stop_was_requested
                else (
                    "Marginal Capital Replay worker was interrupted. Run Marginal Capital Replay again to restart the replay."
                    if interrupted_marginal
                    else (
                        "Full Strategy validation was interrupted. Validate the exact selection again."
                        if interrupted_full_validation
                        else "The previous Asset Discovery worker is no longer active. Start a new manual campaign to continue research."
                    )
                )
            ),
            "worker_active": False,
            "worker_finished_at": utc_now(),
            "completed_at": utc_now(),
            "updated_at": utc_now(),
        }
        if interrupted_marginal and not stop_was_requested:
            changes.update({
                "marginal_replay.status": "interrupted",
                "marginal_replay.current_symbol": None,
                "marginal_replay.current_stage": "Replay interrupted",
            })
        if interrupted_full_validation and not stop_was_requested:
            changes.update({
                "full_strategy_validation.status": "interrupted",
                "full_strategy_validation.current_stage": "Validation interrupted",
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
        "batch_size": _scan_batch_size(),
        "scan_parallelism": _scan_worker_count(),
        "validation_parallelism": _replay_worker_count(),
        "research_size_default": DEFAULT_RESEARCH_SIZE,
        "research_size_unbounded_by_application": True,
        "baseline": selected_baseline,
        "persistence_policy": {
            "external_market_bars": "memory_only",
            "technical_failures": "aggregate_only",
            "rejected_assets": "aggregate_only",
            "stored_shortlist_limit": None,
            "marginal_replay": "all causal candidates are validated in bounded parallel batches; only retained candidates are persisted",
            "history": "latest_campaign_only",
            "discovery_catalog": "positive_causal_validation_candidates_only",
            "low_adherence_assets": "aggregate_only_not_visible_or_persisted",
            "candidate_test_details": "retained_candidates_only",
            "research_size_limit": "external_universe_only",
            "scan_parallelism": _scan_worker_count(),
            "validation_parallelism": _replay_worker_count(),
            "selection_policy": "historical_cutoff_then_126_session_validation_then_126_session_certification",
            "certification_reuse_policy": "non_overlapping_windows",
            "historical_full_replay_promotion": "disabled",
        },
        "campaign": _public(document),
    }


def start_asset_discovery(db: Database, *, research_size: int) -> dict[str, Any]:
    global _worker_thread
    requested = int(research_size)
    if requested < 1:
        raise AssetDiscoveryConflict("Research size must be at least 1 asset.")

    with _worker_lock:
        current = _campaign(db) or {}
        if _worker_thread and _worker_thread.is_alive():
            raise AssetDiscoveryConflict("An Asset Discovery campaign is already running.")
        if str(current.get("status") or "") in ACTIVE_STATUSES:
            if _worker_heartbeat_fresh(current):
                raise AssetDiscoveryConflict("An Asset Discovery campaign is already running.")
            db[COLLECTION].update_one(
                {"_id": CURRENT_ID},
                {"$set": {
                    "status": "interrupted",
                    "phase": "interrupted",
                    "worker_active": False,
                    "completed_at": utc_now(),
                    "updated_at": utc_now(),
                }},
            )

        run_id = f"asset-discovery-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        now = utc_now()
        document = {
            "_id": CURRENT_ID,
            "run_id": run_id,
            "schema_version": 4,
            "api_version": API_VERSION,
            "status": "queued",
            "phase": "queued",
            "mode": "manual",
            "research_size": requested,
            "batch_size": _scan_batch_size(),
            "scan_parallelism": _scan_worker_count(),
            "validation_parallelism": _replay_worker_count(),
            "scan_budget": requested,
            "attempted_count": 0,
            "evaluated_count": 0,
            "rejected_count": 0,
            "technical_failure_count": 0,
            "shortlisted_count": 0,
            "rejection_summary": {},
            "results": [],
            "marginal_replay": {"status": "pending", "total_count": 0, "completed_count": 0, "current_symbol": None, "baseline": None, "results": []},
            "full_strategy_validation": {"status": "idle", "selected_assets": [], "decision": None},
            "events": [{"at": utc_now(), "message": "Manual Asset Discovery campaign queued."}],
            "cancel_requested": False,
            "stop_requested_at": None,
            "worker_id": worker_id,
            "worker_active": True,
            "worker_heartbeat_at": now,
            "worker_started_at": now,
            "worker_finished_at": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "current_symbol": None,
            "current_batch": 0,
            "message": "Manual Asset Discovery campaign queued.",
            "progress_step": "queued",
            "stage_progress_percent": 0.0,
            "current_stage": "Asset Discovery queued",
            "stage_current": None,
            "stage_total": None,
        }
        db[COLLECTION].replace_one({"_id": CURRENT_ID}, bson_value(document), upsert=True)
        _worker_thread = threading.Thread(target=_run_worker, args=(db, run_id, worker_id), name="asset-discovery-ranker", daemon=True)
        _worker_thread.start()
        logger.info("asset_discovery_started run_id=%s worker_id=%s research_size=%s", run_id, worker_id, requested)
    return get_asset_discovery_status(db)


def stop_asset_discovery(db: Database) -> dict[str, Any]:
    document = _campaign(db)
    if not document or str(document.get("status") or "") not in ACTIVE_STATUSES:
        return get_asset_discovery_status(db)
    run_id = str(document.get("run_id") or "")
    now = utc_now()
    heartbeat_fresh = _worker_heartbeat_fresh(document)
    logger.info(
        "asset_discovery_stop_received run_id=%s status=%s phase=%s worker_id=%s heartbeat_fresh=%s local_worker_alive=%s",
        run_id,
        document.get("status"),
        document.get("phase"),
        document.get("worker_id"),
        heartbeat_fresh,
        _worker_alive(),
    )
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": run_id},
        {
            "$set": {
                "cancel_requested": True,
                "status": "stopping",
                "stop_requested_at": now,
                "message": "Stop requested. Active processing is being cancelled and no new batch will start.",
                "updated_at": now,
            },
            "$push": {"events": {"$each": [{"at": now, "message": "Stop requested; active processing will stop at the next safe checkpoint."}], "$slice": -24}},
        },
    )
    logger.info("asset_discovery_stop_persisted run_id=%s worker_id=%s", run_id, document.get("worker_id"))
    if not heartbeat_fresh:
        _finish(
            db,
            run_id,
            "stopped",
            "Asset Discovery stop completed immediately because no active worker heartbeat remains.",
            results=[],
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
        causal_window = document.get("causal_validation") if isinstance(document.get("causal_validation"), dict) else {}
        if str(causal_window.get("method") or "") != "historical_selection_then_validation_then_certification":
            raise AssetDiscoveryConflict(
                "This campaign predates nested causal Asset Discovery validation. Start a new campaign instead of replaying the legacy shortlist."
            )
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
            causal_window=causal_window,
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
                "full_strategy_validation": {
                    "status": "idle",
                    "selected_assets": [],
                    "decision": None,
                    "invalidated_reason": "marginal_replay_restarted",
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



def _current_research_source(db: Database) -> tuple[dict[str, Any], BacktestRequest]:
    config, public = get_research_strategy_context(db)
    source_id = str(public.get("id") or public.get("strategy_id") or "").strip()
    source = _raw_strategy(db, source_id)
    if source is None:
        raise AssetDiscoveryConflict("The current Strategy Research source is no longer available.")
    return source, BacktestRequest.model_validate(source.get("configuration") or config.model_dump(mode="python"))


def _discovery_metadata_for_symbols(
    db: Database,
    campaign_document: dict[str, Any],
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    del db
    shortlist = {
        str(item.get("symbol") or "").strip().upper(): item
        for item in campaign_document.get("results") or []
        if isinstance(item, dict)
    }
    metadata: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for symbol in symbols:
        if symbol in shortlist:
            metadata[symbol] = shortlist[symbol]
        else:
            missing.append(symbol)
    if missing:
        raise AssetDiscoveryConflict(
            "Final Strategy certification can use only candidates from the current causal Discovery campaign: "
            + ", ".join(missing)
            + ". Historical catalog entries are reference-only and cannot be promoted with a later campaign's certification data."
        )
    return metadata

def _require_persistent_candidate_selection(metadata: dict[str, dict[str, Any]], symbols: list[str]) -> None:
    low_adherence = [symbol for symbol in symbols if not _item_is_persistent_candidate(metadata.get(symbol))]
    if low_adherence:
        raise AssetDiscoveryConflict(
            "Assets without positive causal validation-period evidence cannot advance to final Strategy certification: "
            + ", ".join(low_adherence)
            + ". Run a new causal Discovery campaign if the evidence predates this validation method."
        )


def _severe_month_threshold(db: Database) -> float:
    try:
        snapshot = temporal_research_settings_snapshot(db)
        settings = snapshot.get("settings") if isinstance(snapshot.get("settings"), dict) else {}
        risk = settings.get("risk") if isinstance(settings.get("risk"), dict) else {}
        value = _finite_number(risk.get("severe_threshold"))
        if value is not None and -1.0 < value < 0.0:
            return float(value)
    except Exception:
        pass
    return DEFAULT_SEVERE_MONTH_THRESHOLD


def _full_strategy_validation_progress_callback(
    db: Database,
    run_id: str,
    validation_id: str,
    *,
    run_position: int,
    total_runs: int,
    label: str,
) -> Any:
    last_percent = -1.0
    last_stage = ""

    def emit(local_percent: float, stage: str, _completed: int) -> None:
        nonlocal last_percent, last_stage
        local = max(0.0, min(100.0, float(local_percent or 0.0)))
        global_percent = 100.0 * (float(run_position) + local / 100.0) / max(1, int(total_runs))
        rounded = round(global_percent, 1)
        safe_stage = str(stage or label or "").strip()[:240]
        if rounded < 100.0 and rounded - last_percent < 0.5 and safe_stage == last_stage:
            return
        last_percent = rounded
        last_stage = safe_stage
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "full_strategy_validation.validation_id": validation_id},
            {"$set": {
                "updated_at": utc_now(),
                "full_strategy_validation.progress_percent": rounded,
                "full_strategy_validation.current_stage": safe_stage,
            }},
        )

    return emit


def _full_strategy_validation_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    deltas = {
        "ending_capital_delta": _delta(candidate.get("ending_capital"), baseline.get("ending_capital")),
        "ending_capital_delta_rate": _capital_delta_rate(candidate.get("ending_capital"), baseline.get("ending_capital")),
        "cagr_delta": _delta(candidate.get("cagr"), baseline.get("cagr")),
        "sharpe_delta": _delta(candidate.get("sharpe"), baseline.get("sharpe")),
        "maximum_drawdown_delta": _delta(candidate.get("maximum_drawdown"), baseline.get("maximum_drawdown")),
        "worst_fold_return_delta": _delta(candidate.get("worst_fold_return"), baseline.get("worst_fold_return")),
        "negative_months_delta": _delta(candidate.get("negative_months"), baseline.get("negative_months")),
        "severe_negative_months_delta": _delta(candidate.get("severe_negative_months"), baseline.get("severe_negative_months")),
        "switches_delta": _delta(candidate.get("switches"), baseline.get("switches")),
    }
    capital_delta_rate = _finite_number(deltas.get("ending_capital_delta_rate"))
    cagr_delta = _finite_number(deltas.get("cagr_delta"))
    sharpe_delta = _finite_number(deltas.get("sharpe_delta"))
    maxdd_delta = _finite_number(deltas.get("maximum_drawdown_delta"))
    worst_fold_delta = _finite_number(deltas.get("worst_fold_return_delta"))
    baseline_severe = _finite_number(baseline.get("severe_negative_months"))
    candidate_severe = _finite_number(candidate.get("severe_negative_months"))
    gates = {
        "research_context": bool(context.get("research_context_compatible")),
        "causal_temporal_certification": (
            str(context.get("validation_method") or "") == "causal_temporal_certification"
            and bool(context.get("selection_precedes_evaluation"))
            and bool(context.get("validation_precedes_certification"))
            and not bool(context.get("certification_data_used_for_selection"))
        ),
        "capital_improves": capital_delta_rate is not None and capital_delta_rate > 0.0,
        "cagr_not_worse": cagr_delta is not None and cagr_delta >= -1e-12,
        "sharpe_not_worse": sharpe_delta is not None and sharpe_delta >= -1e-12,
        "max_drawdown_not_worse": maxdd_delta is not None and maxdd_delta >= -1e-12,
        "worst_fold_not_worse": worst_fold_delta is not None and worst_fold_delta >= -1e-12,
        "severe_months_not_worse": (
            baseline_severe is not None
            and candidate_severe is not None
            and candidate_severe <= baseline_severe
        ),
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    return deltas, gates, decision


def _update_catalog_full_validation(
    db: Database,
    symbols: list[str],
    validation: dict[str, Any],
) -> None:
    now = utc_now()
    summary = {
        "validation_id": validation.get("validation_id"),
        "decision": validation.get("decision"),
        "source_strategy_id": validation.get("source_strategy_id"),
        "source_strategy_revision": validation.get("source_strategy_revision"),
        "source_strategy_hash": validation.get("source_strategy_hash"),
        "source_model_family": validation.get("source_model_family"),
        "source_model_settings_hash": validation.get("source_model_settings_hash"),
        "source_model_settings_revision": validation.get("source_model_settings_revision"),
        "snapshot_end": validation.get("snapshot_end"),
        "causal_validation": validation.get("causal_validation"),
        "certification_ledger": validation.get("certification_ledger"),
        "deltas": validation.get("deltas"),
        "gates": validation.get("gates"),
        "completed_at": validation.get("completed_at"),
    }
    for symbol in symbols:
        db[CATALOG_COLLECTION].update_one(
            {"_id": symbol},
            {"$set": {"latest_full_strategy_validation": bson_value(summary), "updated_at": now}},
        )


def _run_full_strategy_validation_worker(db: Database, run_id: str, validation_id: str) -> None:
    certification_ledger: dict[str, Any] | None = None
    try:
        document = _campaign(db) or {}
        validation = document.get("full_strategy_validation") if isinstance(document.get("full_strategy_validation"), dict) else {}
        if str(document.get("run_id") or "") != run_id or str(validation.get("validation_id") or "") != validation_id:
            return

        selected_symbols = _selection_symbols(list(validation.get("selected_assets") or []))
        source_id = str(validation.get("source_strategy_id") or "").strip()
        source_raw = _raw_strategy(db, source_id)
        if source_raw is None:
            raise AssetDiscoveryConflict("The Strategy Research source selected for validation is no longer available.")
        if int(source_raw.get("revision") or 1) != int(validation.get("source_strategy_revision") or 1):
            raise AssetDiscoveryConflict("The Strategy Research source changed after validation was queued. Run validation again.")
        if str(source_raw.get("configuration_hash") or "") != str(validation.get("source_strategy_hash") or ""):
            raise AssetDiscoveryConflict("The Strategy Research source configuration changed. Run validation again.")
        current_model_snapshot = get_strategy_model_snapshot(db, source_id)
        if str(current_model_snapshot.get("family") or "") != str(validation.get("source_model_family") or ""):
            raise AssetDiscoveryConflict("The Strategy Research model changed after validation was queued. Run validation again.")
        if str(current_model_snapshot.get("settings_hash") or "") != str(validation.get("source_model_settings_hash") or ""):
            raise AssetDiscoveryConflict("The Strategy Research model settings changed after validation was queued. Run validation again.")

        source_config = BacktestRequest.model_validate(source_raw.get("configuration") or {})
        source_assets = [str(item).strip().upper() for item in source_config.assets]
        source_asset_set = set(source_assets)
        added_symbols = [symbol for symbol in selected_symbols if symbol not in source_asset_set]
        if not added_symbols:
            raise AssetDiscoveryConflict("The selected assets are already present in the current Strategy Research source.")

        snapshot_end = str(validation.get("snapshot_end") or "").strip()
        if not snapshot_end:
            snapshot_end = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()
        causal_window = validation.get("causal_validation") if isinstance(validation.get("causal_validation"), dict) else {}
        certification_start = str(causal_window.get("certification_start") or "").strip()
        certification_end = str(causal_window.get("certification_end") or "").strip()
        selection_cutoff = str(causal_window.get("selection_cutoff") or "").strip()
        if (
            str(causal_window.get("method") or "") != "historical_selection_then_validation_then_certification"
            or not certification_start
            or not certification_end
            or not selection_cutoff
            or not bool(causal_window.get("selection_precedes_evaluation"))
            or not bool(causal_window.get("validation_precedes_certification"))
        ):
            raise AssetDiscoveryConflict("Full Strategy validation requires the untouched causal certification slice after candidate validation.")

        # Claim the certification slice before loading or validating any candidate data
        # from it. Once exposed, this period cannot be reused to shop for another asset,
        # even if the candidate fails coverage or the certification result is FAIL.
        certification_ledger = _consume_certification_window(
            db,
            run_id=run_id,
            validation_id=validation_id,
            source_strategy_id=source_id,
            source_strategy_hash=str(validation.get("source_strategy_hash") or ""),
            selected_assets=selected_symbols,
            causal_window=causal_window,
            decision="PENDING",
        )
        baseline_frames = _baseline_frames(source_config, snapshot_end)
        required_sessions = _baseline_required_sessions(baseline_frames, source_config, snapshot_end)
        combined_frames = dict(baseline_frames)
        coverage_by_symbol: dict[str, Any] = {}
        for symbol in added_symbols:
            frame, coverage = _candidate_history_coverage(
                db, symbol, source_config, pd.Timestamp(snapshot_end), required_sessions
            )
            combined_frames[symbol] = frame
            coverage_by_symbol[symbol] = coverage

        severe_threshold = _severe_month_threshold(db)
        baseline_request = _marginal_execution_request(
            db,
            source_config,
            {"id": source_id},
            source_config,
            snapshot_end,
            assets=source_assets,
            reference_assets=source_assets,
            candidate_assets=[],
            analysis_start_date=certification_start,
            analysis_end_date=certification_end,
        )
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "full_strategy_validation.validation_id": validation_id},
            {"$set": {
                "status": "running",
                "phase": "full_strategy_validation",
                "updated_at": utc_now(),
                "full_strategy_validation.status": "running",
                "full_strategy_validation.current_stage": "Replaying current Strategy Research baseline",
                "full_strategy_validation.progress_percent": 0.0,
            }},
        )
        baseline_metrics, baseline_sessions = _run_rotation_replay(
            baseline_frames,
            baseline_request,
            progress_callback=_full_strategy_validation_progress_callback(
                db, run_id, validation_id, run_position=0, total_runs=2, label="Baseline replay"
            ),
            severe_threshold=severe_threshold,
        )
        if _should_stop_after_batch(db, run_id):
            stopped = {
                **validation,
                "status": "stopped",
                "current_stage": "Stopped after baseline replay",
                "progress_percent": 50.0,
                "baseline": baseline_metrics,
                "completed_at": utc_now(),
            }
            db[COLLECTION].update_one(
                {"_id": CURRENT_ID, "run_id": run_id},
                {"$set": {
                    "status": "stopped",
                    "phase": "full_strategy_validation",
                    "message": "Full Strategy validation stopped after the baseline replay.",
                    "full_strategy_validation": bson_value(stopped),
                    "updated_at": utc_now(),
                }},
            )
            return

        combined_assets = list(dict.fromkeys([*source_assets, *added_symbols]))
        combined_request = _marginal_execution_request(
            db,
            source_config,
            {"id": source_id},
            source_config,
            snapshot_end,
            assets=combined_assets,
            reference_assets=source_assets,
            candidate_assets=added_symbols,
            analysis_start_date=certification_start,
            analysis_end_date=certification_end,
        )
        candidate_metrics, candidate_sessions = _run_rotation_replay(
            combined_frames,
            combined_request,
            progress_callback=_full_strategy_validation_progress_callback(
                db, run_id, validation_id, run_position=1, total_runs=2, label="Selected-universe replay"
            ),
            severe_threshold=severe_threshold,
        )
        context = _research_context_compatibility(baseline_sessions, candidate_sessions)
        context.update({
            "validation_method": "causal_temporal_certification",
            "selection_cutoff": selection_cutoff,
            "validation_start": causal_window.get("validation_start"),
            "validation_end": causal_window.get("validation_end"),
            "certification_start": certification_start,
            "certification_end": certification_end,
            "evaluation_start": certification_start,
            "evaluation_end": certification_end,
            "selection_precedes_evaluation": bool(causal_window.get("selection_precedes_evaluation")),
            "validation_precedes_certification": bool(causal_window.get("validation_precedes_certification")),
            "certification_data_used_for_selection": False,
            "historical_gain_used_for_selection": False,
        })
        deltas, gates, decision = _full_strategy_validation_gates(baseline_metrics, candidate_metrics, context)
        completed_at = utc_now()
        db[CERTIFICATION_LEDGER_COLLECTION].update_one(
            {"_id": certification_ledger.get("_id"), "validation_id": validation_id},
            {"$set": {"decision": str(decision or "").upper(), "updated_at": completed_at, "completed_at": completed_at}},
        )
        certification_ledger = db[CERTIFICATION_LEDGER_COLLECTION].find_one({"_id": certification_ledger.get("_id")}) or certification_ledger
        completed_causal_window = {
            **causal_window,
            "certification_available": False,
            "last_consumed_certification_end": certification_ledger.get("certification_end"),
            "certification_block_reason": "certification_window_consumed",
        }
        completed = {
            **validation,
            "status": "completed",
            "current_stage": "Full Strategy validation completed",
            "progress_percent": 100.0,
            "source_asset_count": len(source_assets),
            "candidate_asset_count": len(combined_assets),
            "added_assets": added_symbols,
            "coverage": coverage_by_symbol,
            "severe_month_threshold": severe_threshold,
            "causal_validation": completed_causal_window,
            "certification_ledger": certification_ledger,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "context": context,
            "deltas": deltas,
            "gates": gates,
            "decision": decision,
            "completed_at": completed_at,
        }
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "full_strategy_validation.validation_id": validation_id},
            {"$set": {
                "status": "completed",
                "phase": "completed",
                "cancel_requested": False,
                "completed_at": completed_at,
                "updated_at": completed_at,
                "message": f"Full Strategy validation {decision} for {', '.join(added_symbols)}.",
                "causal_validation": bson_value(completed_causal_window),
                "full_strategy_validation": bson_value(completed),
            }},
        )
        _update_catalog_full_validation(db, selected_symbols, completed)
    except Exception as exc:
        now = utc_now()
        if certification_ledger and certification_ledger.get("_id"):
            db[CERTIFICATION_LEDGER_COLLECTION].update_one(
                {"_id": certification_ledger.get("_id")},
                {"$set": {
                    "decision": "ERROR_AFTER_EXPOSURE",
                    "error": str(exc)[:700],
                    "updated_at": now,
                    "completed_at": now,
                }},
            )
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": run_id, "full_strategy_validation.validation_id": validation_id},
            {"$set": {
                "status": "failed",
                "phase": "full_strategy_validation",
                "updated_at": now,
                "message": f"Full Strategy validation failed: {str(exc)[:700]}",
                "full_strategy_validation.status": "failed",
                "full_strategy_validation.current_stage": "Validation failed",
                "full_strategy_validation.error": str(exc)[:700],
                "full_strategy_validation.completed_at": now,
            }},
        )
    finally:
        global _worker_thread
        with _worker_lock:
            _worker_thread = None


def start_full_strategy_validation(
    db: Database,
    *,
    run_id: str | None,
    symbols: list[str],
) -> dict[str, Any]:
    global _worker_thread
    requested_symbols = _selection_symbols(symbols)
    if not requested_symbols:
        raise AssetDiscoveryConflict("Select at least one discovered asset.")

    with _worker_lock:
        document = _campaign(db) or {}
        if _worker_thread and _worker_thread.is_alive():
            raise AssetDiscoveryConflict("An Asset Discovery operation is already running.")
        current_run_id = str(document.get("run_id") or "").strip()
        if not current_run_id:
            raise AssetDiscoveryConflict("Complete an Asset Discovery search before validating a selection.")
        normalized_run_id = str(run_id or "").strip()
        if normalized_run_id and normalized_run_id != current_run_id:
            raise AssetDiscoveryConflict("The selected Asset Discovery run is no longer the current campaign.")

        metadata = _discovery_metadata_for_symbols(db, document, requested_symbols)
        _require_persistent_candidate_selection(metadata, requested_symbols)
        causal_window = document.get("causal_validation") if isinstance(document.get("causal_validation"), dict) else {}
        if str(causal_window.get("method") or "") != "historical_selection_then_validation_then_certification":
            raise AssetDiscoveryConflict(
                "This campaign predates nested causal Asset Discovery validation. Run a new Discovery campaign before Full Strategy validation."
            )
        if (
            not bool(causal_window.get("selection_precedes_evaluation"))
            or not bool(causal_window.get("validation_precedes_certification"))
            or not bool(causal_window.get("promotion_uses_certification_only"))
        ):
            raise AssetDiscoveryConflict("The Asset Discovery causal validation window is not promotion-safe.")
        certification_status = _certification_window_status(db, causal_window)
        causal_window = {**causal_window, **certification_status}
        if not bool(certification_status.get("certification_available")):
            last_end = certification_status.get("last_consumed_certification_end") or "another certification"
            raise AssetDiscoveryConflict(
                "The final certification period overlaps data already used by a previous Asset Discovery certification "
                f"(last consumed end: {last_end}). Wait for a new non-overlapping certification period before promoting another discovery."
            )
        source_raw, source_config = _current_research_source(db)
        source_id = str(source_raw.get("_id") or "")
        source_assets = [str(item).strip().upper() for item in source_config.assets]
        added_symbols = [symbol for symbol in requested_symbols if symbol not in set(source_assets)]
        if not added_symbols:
            raise AssetDiscoveryConflict("The selected assets are already present in the current Strategy Research source.")

        source_model_snapshot = get_strategy_model_snapshot(db, source_id)
        baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}
        snapshot_end = str(baseline.get("market_snapshot_end") or "").strip()
        if not snapshot_end:
            snapshot_end = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()
        validation_id = f"asset-full-{uuid4().hex[:12]}"
        validation = {
            "validation_id": validation_id,
            "status": "queued",
            "selected_assets": requested_symbols,
            "added_assets": added_symbols,
            "source_strategy_id": source_id,
            "source_strategy_sequence": source_raw.get("strategy_sequence"),
            "source_strategy_revision": int(source_raw.get("revision") or 1),
            "source_strategy_hash": str(source_raw.get("configuration_hash") or ""),
            "source_model_family": str(source_model_snapshot.get("family") or ""),
            "source_model_settings_hash": str(source_model_snapshot.get("settings_hash") or ""),
            "source_model_settings_revision": int(source_model_snapshot.get("settings_revision") or 0),
            "source_asset_count": len(source_assets),
            "snapshot_end": snapshot_end,
            "causal_validation": dict(causal_window),
            "current_stage": "Queued",
            "progress_percent": 0.0,
            "decision": None,
            "created_at": utc_now(),
        }
        db[COLLECTION].update_one(
            {"_id": CURRENT_ID, "run_id": current_run_id},
            {"$set": {
                "status": "queued",
                "phase": "full_strategy_validation",
                "cancel_requested": False,
                "completed_at": None,
                "updated_at": utc_now(),
                "message": "Full Strategy validation queued for the selected assets.",
                "full_strategy_validation": bson_value(validation),
            }},
        )
        _worker_thread = threading.Thread(
            target=_run_full_strategy_validation_worker,
            args=(db, current_run_id, validation_id),
            name="asset-discovery-full-strategy-validation",
            daemon=True,
        )
        _worker_thread.start()
    return get_asset_discovery_status(db)


def _validated_creation_source(
    db: Database,
    document: dict[str, Any],
    requested_symbols: list[str],
) -> tuple[dict[str, Any], BacktestRequest, dict[str, Any]]:
    validation = document.get("full_strategy_validation") if isinstance(document.get("full_strategy_validation"), dict) else {}
    if str(validation.get("status") or "").lower() != "completed" or str(validation.get("decision") or "").upper() != "PASS":
        raise AssetDiscoveryConflict("Run Full Strategy validation and obtain PASS before creating a Research Strategy.")
    if not _selection_matches(validation.get("selected_assets"), requested_symbols):
        raise AssetDiscoveryConflict("The selected assets changed after Full Strategy validation. Validate the exact selection again.")
    source_id = str(validation.get("source_strategy_id") or "").strip()
    source = _raw_strategy(db, source_id)
    if source is None:
        raise AssetDiscoveryConflict("The Strategy source used by Full Strategy validation is no longer available.")
    if int(source.get("revision") or 1) != int(validation.get("source_strategy_revision") or 1):
        raise AssetDiscoveryConflict("The Strategy source revision changed after Full Strategy validation. Validate again.")
    if str(source.get("configuration_hash") or "") != str(validation.get("source_strategy_hash") or ""):
        raise AssetDiscoveryConflict("The Strategy source configuration changed after Full Strategy validation. Validate again.")
    current_source, _current_config = _current_research_source(db)
    if str(current_source.get("_id") or "") != source_id:
        raise AssetDiscoveryConflict("The selected Strategy Research source changed after Full Strategy validation. Validate again.")
    if int(current_source.get("revision") or 1) != int(validation.get("source_strategy_revision") or 1):
        raise AssetDiscoveryConflict("The selected Strategy Research source revision changed after Full Strategy validation. Validate again.")
    if str(current_source.get("configuration_hash") or "") != str(validation.get("source_strategy_hash") or ""):
        raise AssetDiscoveryConflict("The selected Strategy Research source configuration changed after Full Strategy validation. Validate again.")
    current_model_snapshot = get_strategy_model_snapshot(db, source_id)
    if str(current_model_snapshot.get("family") or "") != str(validation.get("source_model_family") or ""):
        raise AssetDiscoveryConflict("The selected Strategy Research model changed after Full Strategy validation. Validate again.")
    if str(current_model_snapshot.get("settings_hash") or "") != str(validation.get("source_model_settings_hash") or ""):
        raise AssetDiscoveryConflict("The selected Strategy Research model settings changed after Full Strategy validation. Validate again.")
    config = BacktestRequest.model_validate(source.get("configuration") or {})
    return source, config, validation



def append_selected_assets_to_research_strategy(
    db: Database,
    *,
    run_id: str | None,
    symbols: list[str],
    actor_email: str | None,
) -> dict[str, Any]:
    """Append only the explicitly selected, fully certified assets to the current RESEARCH Strategy.

    Existing Strategy assets are preserved in their original order. Rejected/failed assets are never
    persisted here. Market history is persisted only for selected assets after an exact-selection
    final certification PASS.
    """
    current_document = _campaign(db) or {}
    current_run_id = str(current_document.get("run_id") or "").strip()
    normalized_run_id = str(run_id or "").strip()
    if normalized_run_id and normalized_run_id != current_run_id:
        raise AssetDiscoveryConflict("The selected Asset Discovery run is no longer the current campaign.")

    requested_symbols = _selection_symbols(symbols)
    if not requested_symbols:
        raise AssetDiscoveryConflict("Select at least one discovered asset.")

    discovery_metadata = _discovery_metadata_for_symbols(db, current_document, requested_symbols)
    _require_persistent_candidate_selection(discovery_metadata, requested_symbols)
    source_raw, source_config, validation = _validated_creation_source(db, current_document, requested_symbols)
    source_id = str(source_raw.get("_id") or "")
    source_assets = [str(item).strip().upper() for item in source_config.assets if str(item).strip()]
    source_asset_set = set(source_assets)
    added_symbols = [symbol for symbol in requested_symbols if symbol not in source_asset_set]
    already_present = [symbol for symbol in requested_symbols if symbol in source_asset_set]
    if not added_symbols:
        raise AssetDiscoveryConflict("All selected assets are already present in the current Strategy Research source.")

    snapshot_end = str(validation.get("snapshot_end") or "").strip()
    if not snapshot_end:
        snapshot_end = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()

    source_baseline_frames = _baseline_frames(source_config, snapshot_end)
    required_sessions = _baseline_required_sessions(source_baseline_frames, source_config, snapshot_end)
    persisted_history_rows: dict[str, int] = {}
    discarded_assets: list[dict[str, str]] = []
    for symbol in added_symbols:
        try:
            persisted_history_rows[symbol] = _persist_selected_asset_history(
                db, symbol, source_config, snapshot_end, required_sessions
            )
        except AssetDiscoveryConflict as exc:
            discarded_assets.append({"symbol": symbol, "reason": str(exc)})
            # If history cannot be persisted safely, the candidate no longer has promotion-quality evidence.
            db[CATALOG_COLLECTION].delete_one({"_id": symbol})

    if discarded_assets:
        discarded = ", ".join(item["symbol"] for item in discarded_assets)
        raise AssetDiscoveryConflict(
            "The certified selection changed during market-history persistence. "
            f"Discarded: {discarded}. Run final certification again."
        )

    combined_assets = list(dict.fromkeys([*source_assets, *added_symbols]))
    updated_configuration = source_config.model_copy(update={"assets": combined_assets})
    previous_revision = int(source_raw.get("revision") or 1)
    note = "Added Asset Discovery certified assets without removing existing Strategy assets: " + ", ".join(added_symbols)
    try:
        updated_strategy = update_strategy(
            db,
            source_id,
            configuration=updated_configuration,
            name=str(source_raw.get("name") or ""),
            description=str(source_raw.get("description") or ""),
            note=note,
            expected_revision=previous_revision,
            actor_email=actor_email,
        )
    except StrategyLabConflict as exc:
        raise AssetDiscoveryConflict(str(exc)) from exc

    now = utc_now()
    lightweight_lineage = {
        "run_id": current_run_id or None,
        "validation_id": validation.get("validation_id"),
        "added_assets": added_symbols,
        "previous_revision": previous_revision,
        "new_revision": updated_strategy.get("revision"),
        "updated_at": now,
        "updated_by": (actor_email or "").strip().lower() or None,
    }
    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": source_id, "revision": int(updated_strategy.get("revision") or previous_revision + 1)},
        {"$set": {"last_discovery_append": bson_value(lightweight_lineage)}},
    )
    updated_strategy = get_strategy(db, source_id)

    for symbol in added_symbols:
        db[CATALOG_COLLECTION].update_one(
            {"_id": symbol},
            {
                "$set": {
                    "status": "added_to_research_strategy",
                    "last_strategy_id": source_id,
                    "last_strategy_added_at": now,
                    "updated_at": now,
                },
                "$inc": {"research_strategy_added_count": 1},
            },
        )

    append_record = {
        "run_id": current_run_id or None,
        "validation_id": validation.get("validation_id"),
        "decision": validation.get("decision"),
        "strategy_id": source_id,
        "strategy_sequence": updated_strategy.get("strategy_sequence"),
        "previous_revision": previous_revision,
        "new_revision": updated_strategy.get("revision"),
        "selected_assets": requested_symbols,
        "added_assets": added_symbols,
        "already_present_assets": already_present,
        "asset_count_before": len(source_assets),
        "asset_count_after": len(combined_assets),
        "persisted_history_rows": persisted_history_rows,
        "updated_at": now,
        "updated_by": (actor_email or "").strip().lower() or None,
    }
    db[COLLECTION].update_one(
        {"_id": CURRENT_ID, "run_id": current_run_id},
        {"$set": {"research_strategy_append": bson_value(append_record), "updated_at": now}},
    )

    return {
        "research_strategy": updated_strategy,
        "selected_assets": requested_symbols,
        "added_assets": added_symbols,
        "already_present_assets": already_present,
        "asset_count_before": len(source_assets),
        "asset_count_after": len(combined_assets),
        "persisted_history_rows": persisted_history_rows,
        "full_strategy_validation": {
            "validation_id": validation.get("validation_id"),
            "decision": validation.get("decision"),
            "deltas": validation.get("deltas"),
            "gates": validation.get("gates"),
        },
    }


def create_research_strategy_from_discovery(
    db: Database,
    *,
    run_id: str | None,
    symbols: list[str],
    actor_email: str | None,
) -> dict[str, Any]:
    current_document = _campaign(db) or {}
    current_run_id = str(current_document.get("run_id") or "").strip()
    normalized_run_id = str(run_id or "").strip()
    if normalized_run_id and normalized_run_id != current_run_id:
        raise AssetDiscoveryConflict("The selected Asset Discovery run is no longer the current campaign.")

    requested_symbols = _selection_symbols(symbols)
    if not requested_symbols:
        raise AssetDiscoveryConflict("Select at least one discovered asset.")

    discovery_metadata = _discovery_metadata_for_symbols(db, current_document, requested_symbols)
    _require_persistent_candidate_selection(discovery_metadata, requested_symbols)
    source_raw, source_config, validation = _validated_creation_source(db, current_document, requested_symbols)
    source_id = str(source_raw.get("_id") or "")
    template_id = source_id
    source_resolution = "full_strategy_validated_current_research_strategy"
    source_assets = [str(item).strip().upper() for item in source_config.assets]
    source_asset_set = set(source_assets)
    added_symbols = [symbol for symbol in requested_symbols if symbol not in source_asset_set]
    if not added_symbols:
        raise AssetDiscoveryConflict("The selected assets are already present in the validated Strategy Research source.")

    snapshot_end = str(validation.get("snapshot_end") or "").strip()
    if not snapshot_end:
        snapshot_end = pd.Timestamp(latest_safe_completed_xnys_session()).date().isoformat()

    source_baseline_frames = _baseline_frames(source_config, snapshot_end)
    required_sessions = _baseline_required_sessions(source_baseline_frames, source_config, snapshot_end)
    persisted_history_rows: dict[str, int] = {}
    valid_symbols: list[str] = []
    discarded_assets: list[dict[str, str]] = []
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

    if discarded_assets:
        discarded = ", ".join(item["symbol"] for item in discarded_assets)
        raise AssetDiscoveryConflict(
            "The validated selection changed during market-history persistence. "
            f"Discarded: {discarded}. Run Full Strategy validation again."
        )

    combined_assets = list(dict.fromkeys([*source_assets, *valid_symbols]))
    updated_configuration = source_config.model_copy(update={"assets": combined_assets})
    configuration_payload = updated_configuration.model_dump(mode="json")

    created = create_strategy(
        db,
        name="Asset Discovery Research Strategy",
        description="Research Strategy created from Asset Discovery selected assets after Full Strategy validation.",
        clone_from_strategy_id=template_id,
        actor_email=actor_email,
    )
    created_id = str(created.get("id") or "")

    try:
        source_model_snapshot = get_strategy_model_snapshot(db, source_id)
    except Exception:
        source_model_snapshot = source_raw.get("research_model_snapshot") if isinstance(source_raw.get("research_model_snapshot"), dict) else None

    campaign_baseline = current_document.get("baseline") if isinstance(current_document.get("baseline"), dict) else {}
    winner = current_document.get("winner_source") if isinstance(current_document.get("winner_source"), dict) else {}
    origin = {
        "run_id": current_run_id or None,
        "source_resolution": source_resolution,
        "source_strategy_id": source_id,
        "source_strategy_revision": source_raw.get("revision"),
        "source_strategy_hash": source_raw.get("configuration_hash"),
        "source_strategy_kind": source_raw.get("strategy_kind"),
        "campaign_winner_strategy_id": winner.get("strategy_id"),
        "discovery_baseline_strategy_id": campaign_baseline.get("strategy_id"),
        "discovery_baseline_hash": campaign_baseline.get("configuration_hash"),
        "discovery_snapshot_end": snapshot_end,
        "selected_assets": valid_symbols,
        "added_assets": [symbol for symbol in valid_symbols if symbol not in source_asset_set],
        "discarded_assets": discarded_assets,
        "persisted_history_rows": persisted_history_rows,
        "full_strategy_validation": {
            "validation_id": validation.get("validation_id"),
            "decision": validation.get("decision"),
            "source_strategy_id": validation.get("source_strategy_id"),
            "source_strategy_revision": validation.get("source_strategy_revision"),
            "source_strategy_hash": validation.get("source_strategy_hash"),
            "source_model_family": validation.get("source_model_family"),
            "source_model_settings_hash": validation.get("source_model_settings_hash"),
            "source_model_settings_revision": validation.get("source_model_settings_revision"),
            "snapshot_end": validation.get("snapshot_end"),
            "causal_validation": validation.get("causal_validation"),
            "certification_ledger": validation.get("certification_ledger"),
            "baseline": validation.get("baseline"),
            "candidate": validation.get("candidate"),
            "deltas": validation.get("deltas"),
            "gates": validation.get("gates"),
            "completed_at": validation.get("completed_at"),
        },
        "ranked_assets": [
            {
                "symbol": symbol,
                "causal_rank": (discovery_metadata[symbol].get("causal_selection") or {}).get("rank")
                if isinstance(discovery_metadata[symbol].get("causal_selection"), dict) else discovery_metadata[symbol].get("rank"),
                "causal_raw_score": (discovery_metadata[symbol].get("causal_selection") or {}).get("raw_score")
                if isinstance(discovery_metadata[symbol].get("causal_selection"), dict) else discovery_metadata[symbol].get("raw_score"),
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
        "last_change_note": "Created from Asset Discovery after Full Strategy validation PASS.",
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
        "full_strategy_validation": {
            "validation_id": validation.get("validation_id"),
            "decision": validation.get("decision"),
            "deltas": validation.get("deltas"),
            "gates": validation.get("gates"),
        },
        "selected_assets": valid_symbols,
        "discarded_assets": discarded_assets,
        "persisted_history_rows": persisted_history_rows,
        "asset_count": len(combined_assets),
        "added_asset_count": len([symbol for symbol in valid_symbols if symbol not in source_asset_set]),
    }


def export_asset_discovery(db: Database, *, front_version: str | None = None) -> dict[str, Any]:
    document = _sanitize_completed_campaign_persistence(db, _campaign(db))
    if not document:
        raise AssetDiscoveryConflict("There is no Asset Discovery campaign to export.")
    payload = _public(document) or {}
    return {
        "schema_version": 4,
        "package_kind": "market_cycle_trader_asset_discovery_research",
        "api_version": API_VERSION,
        "front_version": str(front_version or "") or None,
        "generated_at": utc_now(),
        "storage_policy": {
            "raw_external_market_data_persisted": False,
            "rejected_symbols_persisted": False,
            "technical_failure_symbols_persisted": False,
            "low_adherence_symbols_persisted": False,
            "campaign_history_persisted": False,
            "discovery_catalog_persisted": True,
            "historical_full_replay_can_promote": False,
            "causal_nested_validation_required": True,
            "validation_sessions": CAUSAL_VALIDATION_SESSIONS,
            "certification_sessions": CAUSAL_CERTIFICATION_SESSIONS,
            "certification_windows_non_overlapping": True,
            "research_size_limit": "external_universe_only",
            "scan_parallelism": _scan_worker_count(),
            "validation_parallelism": _replay_worker_count(),
            "rejected_candidate_details_persisted": False,
            "research_strategy_update_mode": "append_explicitly_selected_certified_assets_only",
            "existing_research_strategy_assets_preserved": True,
        },
        "campaign": payload,
    }
