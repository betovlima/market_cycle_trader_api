from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
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


def normalize_end_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid end date: {value}")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    if parsed.normalize() >= pd.Timestamp(date.today()):
        return None
    return parsed.strftime("%Y-%m-%d")


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
    normalized_end = normalize_end_date(requested_end)
    if normalized_end is not None:
        end = pd.Timestamp(normalized_end, tz="UTC")
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
            f"Incomplete Alpaca market history for {symbol}: requested "
            f"{config.start_date}, but the earliest available session is "
            f"{provenance['actual_start'] or 'unavailable'}. "
            f"Historical feed={config.alpaca_historical_feed}; adjustment={config.alpaca_adjustment}. "
            "The backtest was stopped instead of silently changing the promoted strategy result."
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
    requested_end = (
        pd.Timestamp(normalized_end, tz="UTC")
        if normalized_end
        else pd.Timestamp(date.today(), tz="UTC")
    )
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


def load_alpaca_bars(symbol: str, config: Any) -> pd.DataFrame:
    start = pd.Timestamp(config.start_date, tz="UTC")
    execution_end = effective_execution_end_date(config)
    normalized_end = normalize_end_date(execution_end)
    end = pd.Timestamp(normalized_end, tz="UTC") if normalized_end else None

    if not config.mongo_cache_enabled:
        downloaded = _download_alpaca_bars(
            symbol,
            config,
            config.start_date,
            execution_end,
        )
        return complete_market_history(
            symbol,
            downloaded,
            config,
            provider="alpaca",
            initial_rows=len(downloaded),
        )

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
        identity = {
            "symbol": symbol,
            "interval": config.timeframe,
            "feed": config.alpaca_historical_feed,
            "adjustment": config.alpaca_adjustment,
        }
        first = collection.find_one(
            identity,
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", 1)],
        )
        last = collection.find_one(
            identity,
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", -1)],
        )

        history_backfill_rows = 0
        initial_rows = 0
        if first is None:
            downloaded = _download_alpaca_bars(
                symbol,
                config,
                config.start_date,
                execution_end,
            )
            initial_rows = len(downloaded)
            _upsert_frame(
                collection,
                downloaded,
                identity,
                config.mongo_write_batch_size,
            )
        else:
            first_ts = _utc_timestamp(first["timestamp"])
            last_ts = _utc_timestamp(last["timestamp"])
            cached_before = _read_frame(collection, identity, start, end)
            initial_rows = len(cached_before)

            if (
                bool(config.market_data_history_backfill_enabled)
                and start.normalize() < first_ts.normalize()
            ):
                historical = _download_alpaca_bars(
                    symbol,
                    config,
                    config.start_date,
                    first_ts.isoformat(),
                )
                history_backfill_rows = len(historical)
                _upsert_frame(
                    collection,
                    historical,
                    identity,
                    config.mongo_write_batch_size,
                )

            refresh = max(
                start,
                last_ts
                - pd.Timedelta(int(config.mongo_refresh_overlap_days), unit="D"),
            )
            if end is None or refresh < end:
                recent = _download_alpaca_bars(
                    symbol,
                    config,
                    refresh.isoformat(),
                    execution_end,
                )
                _upsert_frame(
                    collection,
                    recent,
                    identity,
                    config.mongo_write_batch_size,
                )

        cached = _read_frame(collection, identity, start, end)
        if cached.empty:
            raise RuntimeError("Alpaca MongoDB cache returned no bars")
        return complete_market_history(
            symbol,
            cached,
            config,
            provider="alpaca",
            initial_rows=initial_rows or len(cached),
            history_backfill_rows=history_backfill_rows,
        )
    finally:
        client.close()


def load_market_bars(symbol: str, config: Any) -> pd.DataFrame:
    if config.market_data_provider != "alpaca":
        raise ValueError(
            "This release supports Alpaca as the only market data provider."
        )
    return load_alpaca_bars(symbol, config)

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
        + int(config.rotation_horizon_days)
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
