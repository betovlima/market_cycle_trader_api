from __future__ import annotations

import warnings
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.persistence.mongo_repository import ALPACA_MARKET_BARS_COLLECTION, MARKET_BARS_COLLECTION, create_client, get_alpaca_credentials, get_database

MINIMUM_BARS_BY_TIMEFRAME = {"1Day": 800, "15Min": 800}


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


def filter_non_trading_rows(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if timeframe == "1Day" and isinstance(result.index, pd.DatetimeIndex):
        result = result[result.index.dayofweek < 5]
    return result


def trim_downloaded_range(frame: pd.DataFrame, requested_start: str, requested_end: str | None, timeframe: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    start = pd.Timestamp(requested_start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    result = result.loc[result.index >= start]
    normalized_end = normalize_end_date(requested_end)
    if normalized_end is not None:
        end = pd.Timestamp(normalized_end, tz="UTC")
        result = result.loc[result.index < end]
    return filter_non_trading_rows(result, timeframe)


def normalize_yfinance_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        selected = None
        for level in range(result.columns.nlevels):
            if symbol in set(map(str, result.columns.get_level_values(level))):
                try:
                    selected = result.xs(symbol, axis=1, level=level, drop_level=True)
                    break
                except Exception:
                    pass
        result = selected if selected is not None else result.set_axis(result.columns.get_level_values(0), axis=1)
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    result = result.rename(columns={"adj_close": "close"})
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in result.columns for column in required):
        return pd.DataFrame()
    result = result[required].copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce", utc=True)
    result.index.name = "timestamp"
    return result[~result.index.isna()].dropna().sort_index()


def download_yfinance_bars(symbol: str, config: Any, start_date: str | None = None, end_date: str | None = None, allow_empty: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed.") from exc
    interval = "1d" if config.timeframe == "1Day" else "15m"
    requested_start = start_date or config.start_date
    requested_end = end_date if end_date is not None else config.end_date
    normalized_end = normalize_end_date(requested_end)
    download_end = None
    if normalized_end:
        buffer_days = 7 if config.timeframe == "1Day" else 1
        download_end = (pd.Timestamp(normalized_end) + pd.Timedelta(buffer_days, unit="D")).strftime("%Y-%m-%d")
    errors: list[str] = []
    history_kwargs: dict[str, Any] = {"start": requested_start, "interval": interval, "auto_adjust": config.yfinance_auto_adjust, "repair": config.yfinance_repair, "actions": False, "timeout": config.yfinance_timeout, "raise_errors": False}
    if download_end:
        history_kwargs["end"] = download_end
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.Ticker(symbol).history(**history_kwargs)
        frame = trim_downloaded_range(normalize_yfinance_frame(frame, symbol), requested_start, requested_end, config.timeframe)
        if not frame.empty:
            return frame
        errors.append("Ticker.history returned no rows")
    except Exception as exc:
        errors.append(f"Ticker.history failed: {exc}")
    kwargs: dict[str, Any] = {"tickers": symbol, "start": requested_start, "interval": interval, "auto_adjust": config.yfinance_auto_adjust, "repair": config.yfinance_repair, "progress": False, "threads": False, "timeout": config.yfinance_timeout, "group_by": "column"}
    if download_end:
        kwargs["end"] = download_end
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.download(**kwargs)
        frame = trim_downloaded_range(normalize_yfinance_frame(frame, symbol), requested_start, requested_end, config.timeframe)
        if not frame.empty:
            return frame
        errors.append("yf.download returned no rows")
    except Exception as exc:
        errors.append(f"yf.download failed: {exc}")
    if config.timeframe == "1Day":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = yf.Ticker(symbol).history(period=config.yfinance_fallback_period, interval="1d", auto_adjust=config.yfinance_auto_adjust, repair=config.yfinance_repair, actions=False, timeout=config.yfinance_timeout, raise_errors=False)
            frame = trim_downloaded_range(normalize_yfinance_frame(frame, symbol), requested_start, requested_end, config.timeframe)
            if not frame.empty:
                return frame
            errors.append("period fallback returned no rows")
        except Exception as exc:
            errors.append(f"period fallback failed: {exc}")
    if allow_empty:
        return pd.DataFrame()
    raise RuntimeError(f"yfinance returned no trading bars for {symbol}. Attempts: {' | '.join(errors)}")


def _normalize_timestamp(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _chunked(items: list[Any], batch_size: int):
    size = max(1, int(batch_size))
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _row_document(row: Any, identity: dict[str, Any], updated_at: datetime) -> dict[str, Any]:
    document: dict[str, Any] = {**identity, "timestamp": _normalize_timestamp(row.Index), "updated_at": updated_at}
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        value = getattr(row, column, None)
        if value is not None and pd.notna(value):
            document[column] = float(value)
    return document


def _upsert_frame(collection: Any, frame: pd.DataFrame, identity: dict[str, Any], batch_size: int) -> dict[str, int]:
    from pymongo import UpdateOne
    totals = {"processed": 0, "inserted": 0, "updated": 0, "matched": 0}
    if frame.empty:
        return totals
    now = datetime.now(timezone.utc)
    operations = []
    for row in frame.itertuples():
        document = _row_document(row, identity, now)
        operations.append(UpdateOne({**identity, "timestamp": document["timestamp"]}, {"$set": document, "$setOnInsert": {"created_at": now}}, upsert=True))
    for batch in _chunked(operations, batch_size):
        result = collection.bulk_write(batch, ordered=False)
        totals["processed"] += len(batch)
        totals["inserted"] += result.upserted_count
        totals["updated"] += result.modified_count
        totals["matched"] += result.matched_count
    return totals


def _read_frame(collection: Any, identity: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    query: dict[str, Any] = {**identity, "timestamp": {"$gte": start.to_pydatetime()}}
    if end is not None:
        query["timestamp"]["$lt"] = end.to_pydatetime()
    projection = {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "vwap": 1, "trade_count": 1}
    rows = list(collection.find(query, projection).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    columns = [column for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count") if column in frame.columns]
    return frame[columns].sort_index()


def load_yfinance_bars(symbol: str, config: Any) -> pd.DataFrame:
    start = pd.Timestamp(config.start_date, tz="UTC")
    normalized_end = normalize_end_date(config.end_date)
    end = pd.Timestamp(normalized_end, tz="UTC") if normalized_end else None
    if not config.mongo_cache_enabled:
        return download_yfinance_bars(symbol, config)
    client = create_client()
    try:
        db = get_database(client)
        collection = db[MARKET_BARS_COLLECTION]
        from pymongo import ASCENDING
        collection.create_index([("symbol", ASCENDING), ("interval", ASCENDING), ("timestamp", ASCENDING)], unique=True, name="uq_market_bar")
        identity = {"symbol": symbol, "interval": "1d" if config.timeframe == "1Day" else "15m"}
        first = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", 1)])
        last = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", -1)])
        if first is None:
            _upsert_frame(collection, download_yfinance_bars(symbol, config), identity, config.mongo_write_batch_size)
        else:
            first_ts = pd.Timestamp(first["timestamp"])
            first_ts = first_ts.tz_localize("UTC") if first_ts.tzinfo is None else first_ts.tz_convert("UTC")
            last_ts = pd.Timestamp(last["timestamp"])
            last_ts = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
            if start < first_ts:
                historical = download_yfinance_bars(symbol, config, config.start_date, first_ts.strftime("%Y-%m-%d"), allow_empty=True)
                _upsert_frame(collection, historical, identity, config.mongo_write_batch_size)
            refresh = max(start, last_ts - pd.Timedelta(int(config.mongo_refresh_overlap_days), unit="D"))
            if end is None or refresh < end:
                recent = download_yfinance_bars(symbol, config, refresh.strftime("%Y-%m-%d"), config.end_date)
                _upsert_frame(collection, recent, identity, config.mongo_write_batch_size)
        cached = _read_frame(collection, identity, start, end)
        if cached.empty:
            raise RuntimeError("Yahoo MongoDB cache returned no bars")
        return cached
    except Exception as exc:
        print(f"Yahoo cache unavailable: {exc}. Downloading directly.")
        return download_yfinance_bars(symbol, config)
    finally:
        client.close()


def _download_alpaca_bars(symbol: str, config: Any, start_date: str, end_date: str | None) -> pd.DataFrame:
    client = create_client()
    try:
        credentials = get_alpaca_credentials(get_database(client))
    finally:
        client.close()
    frame = download_stock_bars(api_key_id=credentials["api_key_id"], secret_key=credentials["secret_key"], symbol=symbol, timeframe=config.timeframe, start=start_date, end=normalize_end_date(end_date), feed=config.alpaca_feed, adjustment=config.alpaca_adjustment)
    return trim_downloaded_range(frame, start_date, end_date, config.timeframe)


def load_alpaca_bars(symbol: str, config: Any) -> pd.DataFrame:
    start = pd.Timestamp(config.start_date, tz="UTC")
    normalized_end = normalize_end_date(config.end_date)
    end = pd.Timestamp(normalized_end, tz="UTC") if normalized_end else None
    if not config.mongo_cache_enabled:
        return _download_alpaca_bars(symbol, config, config.start_date, config.end_date)
    client = create_client()
    try:
        db = get_database(client)
        collection = db[ALPACA_MARKET_BARS_COLLECTION]
        from pymongo import ASCENDING
        collection.create_index([("symbol", ASCENDING), ("interval", ASCENDING), ("feed", ASCENDING), ("adjustment", ASCENDING), ("timestamp", ASCENDING)], unique=True, name="uq_alpaca_market_bar")
        identity = {"symbol": symbol, "interval": config.timeframe, "feed": config.alpaca_feed, "adjustment": config.alpaca_adjustment}
        first = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", 1)])
        last = collection.find_one(identity, {"timestamp": 1, "_id": 0}, sort=[("timestamp", -1)])
        if first is None:
            _upsert_frame(collection, _download_alpaca_bars(symbol, config, config.start_date, config.end_date), identity, config.mongo_write_batch_size)
        else:
            first_ts = pd.Timestamp(first["timestamp"])
            first_ts = first_ts.tz_localize("UTC") if first_ts.tzinfo is None else first_ts.tz_convert("UTC")
            last_ts = pd.Timestamp(last["timestamp"])
            last_ts = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
            if start.normalize() < first_ts.normalize():
                historical = _download_alpaca_bars(symbol, config, config.start_date, first_ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
                _upsert_frame(collection, historical, identity, config.mongo_write_batch_size)
            refresh = max(start, last_ts - pd.Timedelta(int(config.mongo_refresh_overlap_days), unit="D"))
            if end is None or refresh < end:
                recent = _download_alpaca_bars(symbol, config, refresh.isoformat(), config.end_date)
                _upsert_frame(collection, recent, identity, config.mongo_write_batch_size)
        cached = _read_frame(collection, identity, start, end)
        if cached.empty:
            raise RuntimeError("Alpaca MongoDB cache returned no bars")
        return cached
    finally:
        client.close()


def load_market_bars(symbol: str, config: Any) -> pd.DataFrame:
    if config.market_data_provider == "alpaca":
        return load_alpaca_bars(symbol, config)
    if config.market_data_provider == "yahoo":
        return load_yfinance_bars(symbol, config)
    raise ValueError(f"Unsupported market data provider: {config.market_data_provider}")


def validate_and_clean_bars(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    bars = filter_non_trading_rows(bars, config.timeframe)
    if bars.empty:
        raise ValueError("The OHLCV dataset is empty.")
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    result = bars.copy()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    minimum = MINIMUM_BARS_BY_TIMEFRAME[config.timeframe]
    if len(result) < minimum:
        raise ValueError(f"Only {len(result)} valid {config.timeframe} bars were loaded; at least {minimum} are required.")
    return result
