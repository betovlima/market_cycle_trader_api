from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ..infrastructure.market_data.alpaca import download_stock_bars
from .market_data_snapshot import (
    TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
    decode_market_frame,
)
from ..infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION,
    create_client,
    get_alpaca_credentials,
    get_database,
)

BAR_COLUMNS = ("open", "high", "low", "close", "volume", "vwap", "trade_count")
REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def effective_execution_end_date(config: Any) -> str | None:
    

    analysis_end = getattr(config, "analysis_end_date", None)
    if analysis_end:
        return str(analysis_end)
    locked_end = getattr(config, "end_date", None)
    return str(locked_end) if locked_end else None


EASTERN = ZoneInfo("America/New_York")
SAFE_DAILY_BAR_DELAY_MINUTES = 20


def normalize_end_date(value: str | None) -> str | None:
    

    if not value:
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid end date: {value}")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.strftime("%Y-%m-%d")


def inclusive_end_exclusive_boundary(value: str | None) -> pd.Timestamp | None:
    

    normalized = normalize_end_date(value)
    if normalized is None:
        return None
    return pd.Timestamp(normalized, tz="UTC") + pd.Timedelta(days=1)


def latest_completed_xnys_session(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    

    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    local_day = pd.Timestamp(stamp.tz_convert(EASTERN).date())
    calendar = xcals.get_calendar("XNYS")
    if calendar.is_session(local_day):
        close_at = calendar.session_close(local_day)
        if stamp >= close_at:
            return local_day
        return pd.Timestamp(calendar.previous_session(local_day))
    return pd.Timestamp(calendar.date_to_session(local_day, direction="previous"))


def latest_safe_completed_xnys_session(
    now: datetime | pd.Timestamp | None = None,
    *,
    data_delay_minutes: int = SAFE_DAILY_BAR_DELAY_MINUTES,
) -> pd.Timestamp:
    """Return the latest XNYS session whose daily bar is safe to consume.

    A session becomes operationally complete only after the regular close plus a
    small data-availability buffer.  This prevents the Trader or a new research
    campaign from treating an in-flight/just-closed daily candle as final.
    """
    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    local_day = pd.Timestamp(stamp.tz_convert(EASTERN).date())
    calendar = xcals.get_calendar("XNYS")
    delay = pd.Timedelta(minutes=max(0, int(data_delay_minutes)))
    if calendar.is_session(local_day):
        close_at = calendar.session_close(local_day)
        if stamp >= close_at + delay:
            return local_day
        return pd.Timestamp(calendar.previous_session(local_day))
    return pd.Timestamp(calendar.date_to_session(local_day, direction="previous"))


def _market_data_identity(symbol: str, config: Any) -> dict[str, Any]:
    return {
        "symbol": str(symbol).strip().upper(),
        "interval": config.timeframe,
        "feed": config.alpaca_historical_feed,
        "adjustment": config.alpaca_adjustment,
    }


def _cache_has_session(collection: Any, identity: dict[str, Any], session: pd.Timestamp) -> bool:
    start = pd.Timestamp(session.date(), tz="UTC")
    end = start + pd.Timedelta(days=1)
    return collection.find_one(
        {
            **identity,
            "timestamp": {"$gte": start.to_pydatetime(), "$lt": end.to_pydatetime()},
        },
        {"_id": 1},
    ) is not None


def _cache_has_identity(collection: Any, identity: dict[str, Any]) -> bool:
    return collection.find_one(identity, {"_id": 1}) is not None


def _latest_cached_session_on_or_before(
    collection: Any,
    identity: dict[str, Any],
    session: pd.Timestamp,
    calendar: Any,
) -> pd.Timestamp | None:
    boundary = pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(days=1)
    document = collection.find_one(
        {**identity, "timestamp": {"$lt": boundary.to_pydatetime()}},
        {"timestamp": 1, "_id": 0},
        sort=[("timestamp", -1)],
    )
    if not document or document.get("timestamp") is None:
        return None
    timestamp = _utc_timestamp(document["timestamp"])
    return pd.Timestamp(
        calendar.date_to_session(pd.Timestamp(timestamp.date()), direction="previous")
    )


def _latest_common_cached_session(
    collection: Any,
    config: Any,
    target: pd.Timestamp,
    calendar: Any,
) -> pd.Timestamp:
    identities = [
        _market_data_identity(symbol, config)
        for symbol in list(getattr(config, "assets", []) or [])
    ]
    existing = [identity for identity in identities if _cache_has_identity(collection, identity)]
    if not existing:
        return target

    candidate = target
    start_date = normalize_end_date(getattr(config, "start_date", None))
    minimum_session = (
        pd.Timestamp(calendar.date_to_session(pd.Timestamp(start_date), direction="next"))
        if start_date
        else None
    )

    while True:
        latest_sessions = [
            _latest_cached_session_on_or_before(collection, identity, candidate, calendar)
            for identity in existing
        ]
        if any(session is None for session in latest_sessions):
            raise RuntimeError(
                "MongoDB has no common cached market session inside the configured backtest window."
            )
        candidate = min([candidate, *latest_sessions])
        if minimum_session is not None and candidate < minimum_session:
            raise RuntimeError(
                "MongoDB has no common cached market session inside the configured backtest window."
            )
        if all(_cache_has_session(collection, identity, candidate) for identity in existing):
            return candidate
        candidate = pd.Timestamp(calendar.previous_session(candidate))


def resolve_backtest_analysis_end_date(
    config: Any,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> str:
    calendar = xcals.get_calendar("XNYS")
    latest_closed = latest_completed_xnys_session(now)
    requested = normalize_end_date(getattr(config, "end_date", None))
    if requested:
        requested_session = pd.Timestamp(
            calendar.date_to_session(pd.Timestamp(requested), direction="previous")
        )
        target = min(requested_session, latest_closed)
    else:
        target = latest_closed

    client = create_client()
    try:
        collection = get_database(client)[ALPACA_MARKET_BARS_COLLECTION]
        target = _latest_common_cached_session(collection, config, target, calendar)
    finally:
        client.close()

    return target.date().isoformat()


def resolve_live_market_cutoff(
    config: Any,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> str:
    """Resolve the latest common cached session for the operational Winner.

    Unlike the certified backtest cutoff, this deliberately ignores config.end_date.
    The live Winner advances with completed XNYS sessions while its model/parameters
    remain immutable.
    """
    calendar = xcals.get_calendar("XNYS")
    target = latest_safe_completed_xnys_session(now)
    client = create_client()
    try:
        collection = get_database(client)[ALPACA_MARKET_BARS_COLLECTION]
        target = _latest_common_cached_session(collection, config, target, calendar)
    finally:
        client.close()
    return target.date().isoformat()


def refresh_market_data_to_live_cutoff(
    config: Any,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Refresh the Winner universe through the latest SIP-safe XNYS session.

    This function is intentionally called only at operational/research boundaries.
    Model tuning continues to use a frozen MongoDB snapshot and never reaches Alpaca.
    """
    calendar = xcals.get_calendar("XNYS")
    target = latest_safe_completed_xnys_session(now)
    target_date = target.date().isoformat()
    assets = [str(item).strip().upper() for item in list(getattr(config, "assets", []) or []) if str(item).strip()]
    if not assets:
        raise RuntimeError("The live Winner has no assets to refresh.")

    client = create_client()
    rows_by_symbol: dict[str, int] = {}
    try:
        db = get_database(client)
        collection = db[ALPACA_MARKET_BARS_COLLECTION]
        from pymongo import ASCENDING

        collection.create_index(
            [
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
                ("feed", ASCENDING),
                ("adjustment", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
            name="uq_alpaca_market_bar",
        )

        for symbol in assets:
            identity = _market_data_identity(symbol, config)
            latest = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", -1)])
            latest_session = None
            if latest and latest.get("timestamp") is not None:
                latest_stamp = _utc_timestamp(latest["timestamp"])
                latest_session = pd.Timestamp(
                    calendar.date_to_session(pd.Timestamp(latest_stamp.date()), direction="previous")
                )
            if latest_session is not None and latest_session >= target and _cache_has_session(collection, identity, target):
                rows_by_symbol[symbol] = 0
                continue

            if latest_session is None:
                refresh_start = str(getattr(config, "start_date", None) or target_date)
            else:
                # Re-fetch a small tail so the latest daily bar can be safely replaced
                # if the provider revised it after the first observation.
                refresh_start = max(
                    pd.Timestamp(str(getattr(config, "start_date", None) or latest_session.date().isoformat())),
                    latest_session - pd.Timedelta(days=7),
                ).date().isoformat()
            downloaded = _download_alpaca_bars(symbol, config, refresh_start, target_date)
            if downloaded is not None and not downloaded.empty:
                _upsert_frame(collection, downloaded, identity, config.mongo_write_batch_size)
                rows_by_symbol[symbol] = int(len(downloaded))
            else:
                rows_by_symbol[symbol] = 0

        missing = [
            symbol
            for symbol in assets
            if not _cache_has_session(collection, _market_data_identity(symbol, config), target)
        ]
        if missing:
            raise RuntimeError(
                "LiveMarketDataIncomplete: latest completed XNYS session "
                f"{target_date} is missing for: {', '.join(missing)}."
            )
        common = _latest_common_cached_session(collection, config, target, calendar)
        if common != target:
            raise RuntimeError(
                f"LiveMarketDataIncomplete: common market cutoff is {common.date().isoformat()}, expected {target_date}."
            )
    finally:
        client.close()

    return {
        "live_market_cutoff": target_date,
        "target_session": target_date,
        "rows_refreshed": rows_by_symbol,
        "data_delay_minutes": SAFE_DAILY_BAR_DELAY_MINUTES,
    }


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid timestamp: {value}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _optional_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return _utc_timestamp(value)
    except (TypeError, ValueError):
        return None


def filter_non_trading_rows(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result.attrs.update(getattr(frame, "attrs", {}))
    if timeframe == "1Day" and isinstance(result.index, pd.DatetimeIndex):
        result = result[result.index.dayofweek < 5]
    return result


def trim_downloaded_range(
    frame: pd.DataFrame,
    requested_start: str,
    requested_end: str | None,
    timeframe: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    start = _utc_timestamp(requested_start)
    result = result.loc[result.index >= start]
    end = inclusive_end_exclusive_boundary(requested_end)
    if end is not None:
        result = result.loc[result.index < end]
    return filter_non_trading_rows(result, timeframe)


def _normalize_timestamp(value: Any) -> datetime:
    return _utc_timestamp(value).to_pydatetime()


def _chunked(items: list[Any], batch_size: int):
    size = max(1, int(batch_size))
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _row_document(
    row: Any,
    identity: dict[str, Any],
    updated_at: datetime,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        **identity,
        "timestamp": _normalize_timestamp(row.Index),
        "updated_at": updated_at,
    }
    for column in BAR_COLUMNS:
        value = getattr(row, column, None)
        if value is not None and pd.notna(value):
            document[column] = float(value)
    return document


def _upsert_frame(
    collection: Any,
    frame: pd.DataFrame,
    identity: dict[str, Any],
    batch_size: int,
) -> dict[str, int]:
    from pymongo import UpdateOne

    totals = {"processed": 0, "inserted": 0, "updated": 0, "matched": 0}
    if frame.empty:
        return totals
    now = datetime.now(timezone.utc)
    operations = []
    for row in frame.itertuples():
        document = _row_document(row, identity, now)
        operations.append(
            UpdateOne(
                {**identity, "timestamp": document["timestamp"]},
                {
                    "$set": document,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        )
    for batch in _chunked(operations, batch_size):
        result = collection.bulk_write(batch, ordered=False)
        totals["processed"] += len(batch)
        totals["inserted"] += result.upserted_count
        totals["updated"] += result.modified_count
        totals["matched"] += result.matched_count
    return totals


def _read_frame(
    collection: Any,
    identity: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    query: dict[str, Any] = {
        **identity,
        "timestamp": {"$gte": start.to_pydatetime()},
    }
    if end is not None:
        query["timestamp"]["$lt"] = end.to_pydatetime()
    projection = {
        "_id": 0,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "vwap": 1,
        "trade_count": 1,
    }
    rows = list(collection.find(query, projection).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    columns = [column for column in BAR_COLUMNS if column in frame.columns]
    return frame[columns].sort_index()


def _history_tolerance(config: Any) -> pd.Timedelta:
    return pd.Timedelta(
        int(getattr(config, "market_data_history_start_tolerance_days", 0)),
        unit="D",
    )


def _history_is_complete(frame: pd.DataFrame, config: Any) -> bool:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return False
    first = _optional_utc_timestamp(frame.index.min())
    if first is None:
        return False
    requested = _utc_timestamp(config.start_date).normalize()
    return first.normalize() <= requested + _history_tolerance(config)


def _range_is_cached(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    config: Any,
) -> bool:
    if frame is None or frame.empty:
        return False
    first = _optional_utc_timestamp(frame.index.min())
    last = _optional_utc_timestamp(frame.index.max())
    if first is None or last is None:
        return False
    tolerance = _history_tolerance(config)
    if first.normalize() > start.normalize() + tolerance:
        return False
    if end is not None and last.normalize() < end.normalize() - tolerance:
        return False
    return True




def _market_data_provenance(
    *,
    symbol: str,
    config: Any,
    frame: pd.DataFrame,
    provider: str,
    initial_rows: int,
    history_backfill_rows: int = 0,
) -> dict[str, Any]:
    first = _optional_utc_timestamp(frame.index.min()) if not frame.empty else None
    last = _optional_utc_timestamp(frame.index.max()) if not frame.empty else None
    requested = _utc_timestamp(config.start_date)
    return {
        "symbol": symbol,
        "requested_start": requested.isoformat(),
        "actual_start": first.isoformat() if first is not None else None,
        "actual_end": last.isoformat() if last is not None else None,
        "history_complete": _history_is_complete(frame, config),
        "provider": provider,
        "effective_provider": provider,
        "historical_feed": str(config.alpaca_historical_feed),
        "live_feed": str(config.alpaca_live_feed),
        "adjustment": str(config.alpaca_adjustment),
        "initial_rows": int(initial_rows),
        "history_backfill_provider": provider if history_backfill_rows > 0 else None,
        "history_backfill_rows": int(history_backfill_rows),
        "total_rows": int(len(frame)),
    }

def _attach_provenance(frame: pd.DataFrame, provenance: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result.attrs["market_data_provenance"] = provenance
    return result




ALPACA_DAILY_HISTORY_CHUNK_DAYS = 730
ALPACA_INTRADAY_HISTORY_CHUNK_DAYS = 30


def complete_market_history(
    symbol: str,
    frame: pd.DataFrame,
    config: Any,
    *,
    provider: str = "alpaca",
    initial_rows: int | None = None,
    history_backfill_rows: int = 0,
) -> pd.DataFrame:
    

    effective_frame = filter_non_trading_rows(frame, config.timeframe)
    effective_frame = effective_frame[
        ~effective_frame.index.duplicated(keep="last")
    ].sort_index()
    provenance = _market_data_provenance(
        symbol=symbol,
        config=config,
        frame=effective_frame,
        provider=provider,
        initial_rows=len(effective_frame) if initial_rows is None else initial_rows,
        history_backfill_rows=history_backfill_rows,
    )
    effective_frame = _attach_provenance(effective_frame, provenance)

    require_complete = bool(
        getattr(config, "market_data_require_complete_history", True)
    )
    if not provenance["history_complete"] and require_complete:
        raise RuntimeError(
            f"Incomplete MongoDB market history for {symbol}: requested "
            f"{config.start_date}, but the earliest available session is "
            f"{provenance['actual_start'] or 'unavailable'}. "
            f"Historical feed={config.alpaca_historical_feed}; adjustment={config.alpaca_adjustment}. "
            "Existing cached assets are not backfilled from Alpaca by research executions."
        )
    return effective_frame


def _download_alpaca_bars(
    symbol: str,
    config: Any,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    






    credentials = get_alpaca_credentials()
    requested_start = _utc_timestamp(start_date)
    normalized_end = normalize_end_date(end_date)
    if normalized_end is None:
        normalized_end = latest_completed_xnys_session().date().isoformat()
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(
        calendar.date_to_session(pd.Timestamp(normalized_end), direction="previous")
    )
    requested_end = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    if requested_end <= requested_start:
        return pd.DataFrame()

    chunk_days = (
        ALPACA_DAILY_HISTORY_CHUNK_DAYS
        if config.timeframe == "1Day"
        else ALPACA_INTRADAY_HISTORY_CHUNK_DAYS
    )
    frames: list[pd.DataFrame] = []
    cursor = requested_start
    while cursor < requested_end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), requested_end)
        frame = download_stock_bars(
            api_key_id=credentials["api_key_id"],
            secret_key=credentials["secret_key"],
            symbol=symbol,
            timeframe=config.timeframe,
            start=cursor.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            feed=config.alpaca_historical_feed,
            adjustment=config.alpaca_adjustment,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
        cursor = chunk_end

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return trim_downloaded_range(
        combined,
        start_date,
        end_date,
        config.timeframe,
    )


def _end_is_complete(frame: pd.DataFrame, config: Any) -> bool:
    cutoff = normalize_end_date(effective_execution_end_date(config))
    if cutoff is None:
        return True
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return False
    last = _optional_utc_timestamp(frame.index.max())
    if last is None:
        return False
    return last.date() >= date.fromisoformat(cutoff)




def _load_frozen_tuning_snapshot_bars(symbol: str, config: Any) -> pd.DataFrame:
    snapshot_id = str(getattr(config, "research_market_data_snapshot_id", None) or "").strip().lower()
    if not snapshot_id:
        raise RuntimeError("Frozen tuning snapshot id is missing.")
    client = create_client()
    try:
        db = get_database(client)
        collection = db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION]
        document = collection.find_one(
            {
                "snapshot_id": snapshot_id,
                "kind": "symbol",
                "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
                "symbol": str(symbol).strip().upper(),
                "interval": config.timeframe,
                "feed": config.alpaca_historical_feed,
                "adjustment": config.alpaca_adjustment,
            },
            {"_id": 0},
        )
        if document is None:
            raise RuntimeError(
                f"FrozenTuningMarketDataMissing: snapshot {snapshot_id} has no data for {symbol}."
            )
        frame = decode_market_frame(document.get("payload") or b"", list(document.get("columns") or []))
        start = pd.Timestamp(config.start_date, tz="UTC")
        end = inclusive_end_exclusive_boundary(effective_execution_end_date(config))
        frame = frame.loc[frame.index >= start]
        if end is not None:
            frame = frame.loc[frame.index < end]
        provenance = dict(document.get("provenance") or {})
        provenance["research_access_path"] = "frozen_tuning_snapshot"
        provenance["market_data_snapshot_id"] = snapshot_id
        provenance["requested_end"] = normalize_end_date(effective_execution_end_date(config))
        provenance["end_complete"] = _end_is_complete(frame, config)
        frame = _attach_provenance(frame, provenance)
        if frame.empty:
            raise RuntimeError(
                f"FrozenTuningMarketDataMissing: snapshot {snapshot_id} is empty for {symbol}."
            )
        return frame
    finally:
        client.close()

def load_mongo_market_bars(symbol: str, config: Any) -> pd.DataFrame:
    

    if str(getattr(config, "research_market_data_snapshot_id", None) or "").strip():
        return _load_frozen_tuning_snapshot_bars(symbol, config)

    if not bool(getattr(config, "mongo_cache_enabled", True)):
        raise RuntimeError(
            "Research market data is MongoDB-only. Enable the MongoDB market-data cache for backtests and tuning."
        )

    execution_end = effective_execution_end_date(config)
    start = pd.Timestamp(config.start_date, tz="UTC")
    end = inclusive_end_exclusive_boundary(execution_end)
    access_mode = str(getattr(config, "research_market_data_mode", "database_only"))
    allow_bootstrap = access_mode == "backtest_bootstrap_missing"

    client = create_client()
    try:
        db = get_database(client)
        collection = db[ALPACA_MARKET_BARS_COLLECTION]
        from pymongo import ASCENDING

        collection.create_index(
            [
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
                ("feed", ASCENDING),
                ("adjustment", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
            name="uq_alpaca_market_bar",
        )
        identity = _market_data_identity(symbol, config)
        first = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", 1)])
        bootstrapped_rows = 0

        if first is None:
            if not allow_bootstrap:
                raise RuntimeError(
                    f"MarketDataMissingInMongoDB: {symbol} has no cached {config.timeframe} "
                    f"bars for feed={config.alpaca_historical_feed}, adjustment={config.alpaca_adjustment}. "
                    "Model tuning and parameter optimization are database-only and never download market data."
                )
            downloaded = _download_alpaca_bars(
                symbol,
                config,
                config.start_date,
                execution_end,
            )
            if downloaded.empty:
                raise RuntimeError(
                    f"MarketDataBootstrapFailed: Alpaca returned no historical bars for missing asset {symbol}."
                )
            _upsert_frame(collection, downloaded, identity, config.mongo_write_batch_size)
            bootstrapped_rows = len(downloaded)

        cached = _read_frame(collection, identity, start, end)
        if cached.empty:
            raise RuntimeError(
                f"MarketDataMissingInMongoDB: no cached bars for {symbol} inside the locked research window."
            )

        result = complete_market_history(
            symbol,
            cached,
            config,
            provider="alpaca",
            initial_rows=len(cached),
            history_backfill_rows=0,
        )
        provenance = dict(result.attrs.get("market_data_provenance", {}))
        provenance["research_access_path"] = (
            "alpaca_bootstrap_then_mongodb" if bootstrapped_rows else "mongodb_only"
        )
        provenance["cache_bootstrap_rows"] = int(bootstrapped_rows)
        provenance["requested_end"] = normalize_end_date(execution_end)
        provenance["end_complete"] = _end_is_complete(result, config)
        result = _attach_provenance(result, provenance)
        if not provenance["end_complete"]:
            last = provenance.get("actual_end") or "unavailable"
            raise RuntimeError(
                f"MarketDataIncomplete: MongoDB market data for {symbol} ends at {last}; "
                f"the frozen research cutoff is {provenance.get('requested_end')}. "
                "Existing cached assets are never refreshed from Alpaca by a backtest or tuning run."
            )
        return result
    finally:
        client.close()


def load_market_bars(symbol: str, config: Any) -> pd.DataFrame:
    if config.market_data_provider != "alpaca":
        raise ValueError("This release supports Alpaca-origin market data stored in MongoDB.")
    return load_mongo_market_bars(symbol, config)


def validate_and_clean_bars(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    source_attrs = dict(getattr(bars, "attrs", {}))
    bars = filter_non_trading_rows(bars, config.timeframe)
    if bars.empty:
        raise ValueError("The OHLCV dataset is empty.")
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    result = bars.copy()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(REQUIRED_BAR_COLUMNS)
    )
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    result.attrs.update(source_attrs)

    minimum = (
        int(config.rotation_minimum_training_rows)
        + max(int(item) for item in config.rotation_target_horizons)
        + int(config.rotation_purge_days)
    )
    if len(result) < minimum:
        raise ValueError(
            f"Only {len(result)} valid {config.timeframe} bars were loaded; "
            f"at least {minimum} are required by the locked training, horizon and purge settings."
        )

    if bool(getattr(config, "market_data_require_complete_history", True)) and not _history_is_complete(result, config):
        provenance = result.attrs.get("market_data_provenance", {})
        raise ValueError(
            "The cleaned market data does not reach the locked historical start. "
            f"Requested={config.start_date}; actual={provenance.get('actual_start') or result.index.min()}."
        )
    return result
