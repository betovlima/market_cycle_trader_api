from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any

import pandas as pd


SUPPORTED_FEEDS = {"iex", "sip"}
SUPPORTED_ADJUSTMENTS = {"raw", "split", "dividend", "all"}


def _require_alpaca():
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise RuntimeError(
            "alpaca-py is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return Adjustment, DataFeed, StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit


def _feed_value(feed: str):
    Adjustment, DataFeed, *_ = _require_alpaca()
    normalized = str(feed or "iex").strip().lower()
    if normalized not in SUPPORTED_FEEDS:
        raise ValueError(f"Unsupported Alpaca stock feed: {normalized}")
    return DataFeed.IEX if normalized == "iex" else DataFeed.SIP


def _adjustment_value(adjustment: str):
    Adjustment, *_ = _require_alpaca()
    normalized = str(adjustment or "all").strip().lower()
    mapping = {
        "raw": Adjustment.RAW,
        "split": Adjustment.SPLIT,
        "dividend": Adjustment.DIVIDEND,
        "all": Adjustment.ALL,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Alpaca adjustment: {normalized}")
    return mapping[normalized]


def _timeframe_value(timeframe: str):
    *_, TimeFrame, TimeFrameUnit = _require_alpaca()
    mapping = {
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported Alpaca timeframe: {timeframe}")
    return mapping[timeframe]


def _utc_datetime(value: str | datetime | pd.Timestamp | None, *, default_now: bool = False) -> datetime | None:
    if value is None:
        return datetime.now(timezone.utc) if default_now else None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def normalize_alpaca_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex):
        names = list(result.index.names)
        if "symbol" in names:
            try:
                result = result.xs(symbol, level="symbol", drop_level=True)
            except KeyError:
                return pd.DataFrame()
        elif len(names) >= 2:
            try:
                result = result.xs(symbol, level=0, drop_level=True)
            except KeyError:
                return pd.DataFrame()

    result.columns = [str(column).strip().lower() for column in result.columns]
    rename = {
        "trade_count": "trade_count",
        "vwap": "vwap",
    }
    result = result.rename(columns=rename)

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise RuntimeError(
            f"Alpaca returned an unexpected bar schema for {symbol}; missing {missing}."
        )

    result.index = pd.to_datetime(result.index, utc=True)
    result.index.name = "timestamp"
    keep = [
        column
        for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count")
        if column in result.columns
    ]
    result = result[keep].sort_index()
    result = result[~result.index.duplicated(keep="last")]
    for column in keep:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=required)


def download_stock_bars(
    *,
    api_key_id: str,
    secret_key: str,
    symbol: str,
    timeframe: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp | None,
    feed: str,
    adjustment: str,
) -> pd.DataFrame:
    Adjustment, DataFeed, StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit = _require_alpaca()

    api_key_id = str(api_key_id or "").strip()
    secret_key = str(secret_key or "").strip()
    if not api_key_id or not secret_key:
        raise RuntimeError("Alpaca API credentials are not configured.")

    client = StockHistoricalDataClient(api_key_id, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_timeframe_value(timeframe),
        start=_utc_datetime(start),
        end=_utc_datetime(end),
        feed=_feed_value(feed),
        adjustment=_adjustment_value(adjustment),
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            bars = client.get_stock_bars(request)
            return normalize_alpaca_frame(bars.df, symbol)
        except Exception as exc:  
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Alpaca historical data request failed for {symbol} after 3 attempts: {last_error}"
    ) from last_error


def test_connection(
    *,
    api_key_id: str,
    secret_key: str,
    feed: str,
    symbol: str = "SPY",
) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    frame = download_stock_bars(
        api_key_id=api_key_id,
        secret_key=secret_key,
        symbol=symbol,
        timeframe="15Min",
        start=start,
        end=end,
        feed=feed,
        adjustment="all",
    )
    if frame.empty:
        raise RuntimeError(
            f"Alpaca authentication succeeded but no {symbol} 15-minute bars were returned for the last 7 days."
        )
    last = frame.iloc[-1]
    return {
        "ok": True,
        "symbol": symbol,
        "feed": str(feed).lower(),
        "bars": int(len(frame)),
        "last_timestamp": pd.Timestamp(frame.index[-1]).isoformat(),
        "last_close": float(last["close"]),
    }
