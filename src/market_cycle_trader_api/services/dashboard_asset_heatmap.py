"""Read-only daily price heatmap for the configured immutable Winner universe.

Only completed XNYS daily bars are shown. No training, paper account,
intraday quote, order execution, strategy mutation, or broker account lookup.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
import threading
import time
from typing import Any

import pandas as pd

from ..engine.market_data import latest_safe_completed_xnys_session
from ..infrastructure.persistence.mongo_repository import get_alpaca_credentials
from .strategy_lab import get_trader_winner_context

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,12}$")
_CACHE_TTL_SECONDS = 600
_CHUNK_SIZE = 25
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _fetch_completed_bars(
    symbols: list[str], *, start: datetime, end: datetime,
    credentials: dict[str, str],
) -> pd.DataFrame:
    """One batched, read-only Alpaca historical request per small symbol group."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        credentials["api_key_id"], credentials["secret_key"],
    )
    frames: list[pd.DataFrame] = []
    failures = 0
    for index in range(0, len(symbols), _CHUNK_SIZE):
        chunk = symbols[index:index + _CHUNK_SIZE]
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            feed=DataFeed.SIP, adjustment=Adjustment.RAW,
            limit=10000,
        )
        try:
            response = client.get_stock_bars(request)
            batch = response.df
            if isinstance(batch, pd.DataFrame) and not batch.empty:
                frames.append(batch)
        except Exception:
            # Do not hide an outage as a zero return; missing tiles remain N/D.
            failures += 1
    if failures == math.ceil(len(symbols) / _CHUNK_SIZE):
        raise RuntimeError("No Alpaca daily market data batch is available.")
    return pd.concat(frames) if frames else pd.DataFrame()


def _close_by_symbol(raw: pd.DataFrame, symbols: list[str], as_of: str) -> dict[str, dict[str, float]]:
    """Normalize Alpaca's (symbol, timestamp) MultiIndex without guessing bars."""
    out: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}
    if raw is None or raw.empty or "close" not in raw.columns:
        return out
    if not isinstance(raw.index, pd.MultiIndex):
        return out
    names = list(raw.index.names)
    if "symbol" not in names or "timestamp" not in names:
        return out
    for item in raw.reset_index().itertuples(index=False):
        # Never trust arbitrary provider columns or take the last unverified day.
        symbol = str(getattr(item, "symbol", "")).upper()
        if symbol not in out:
            continue
        price = _finite_float(getattr(item, "close", None))
        if price is None:
            continue
        try:
            date = pd.Timestamp(getattr(item, "timestamp")).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            continue
        if date <= as_of:
            out[symbol][date] = price
    return out


def _percent_change(closes: list[float], sessions: int) -> float | None:
    if len(closes) <= sessions:
        return None
    older, latest = closes[-sessions - 1], closes[-1]
    if older <= 0:
        return None
    result = (latest / older - 1.0) * 100.0
    return round(result, 4) if math.isfinite(result) else None


def dashboard_asset_heatmap(db: Any) -> dict[str, Any]:
    """Summarize the latest complete daily close for each configured Winner asset."""
    config, profile = get_trader_winner_context(db)
    symbols = list(dict.fromkeys(
        str(asset).strip().upper() for asset in (config.assets or [])
        if _SYMBOL_RE.fullmatch(str(asset).strip().upper())
    ))
    if not symbols:
        raise ValueError("Winner strategy does not contain registered symbols.")
    now = datetime.now(timezone.utc)
    cutoff = latest_safe_completed_xnys_session(now=now)
    safe_date = pd.Timestamp(cutoff).date().isoformat()
    identity = (
        str(profile.get("id") or ""), str(profile.get("revision") or ""),
        tuple(symbols), safe_date,
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(identity)
        if cached and time.monotonic() < cached[0]:
            return cached[1]
        start = (pd.Timestamp(safe_date) - pd.Timedelta(days=62)).to_pydatetime().replace(tzinfo=timezone.utc)
        end = (pd.Timestamp(safe_date) + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
        credentials = get_alpaca_credentials()
        market = _fetch_completed_bars(symbols, start=start, end=end, credentials=credentials)
        closes_by_symbol = _close_by_symbol(market, symbols, safe_date)
        rows: list[dict[str, Any]] = []
        available = 0
        for symbol in symbols:
            dated = closes_by_symbol[symbol]
            sorted_dates = sorted(dated)
            # A symbol without a bar on the requested completed session is STALE.
            current = sorted_dates[-1] if sorted_dates else None
            fresh = current == safe_date
            prices = [dated[date] for date in sorted_dates] if fresh else []
            if fresh:
                available += 1
            rows.append({
                "symbol": symbol,
                "date": current,
                "stale": not fresh,
                "price": round(prices[-1], 6) if fresh else None,
                "change_1d": _percent_change(prices, 1),
                "change_5d": _percent_change(prices, 5),
                "change_20d": _percent_change(prices, 20),
            })

        response = {
            "universe": "trader_winner",
            "strategy_name": str(profile.get("name") or "Winner"),
            "strategy_sequence": int(profile.get("strategy_sequence") or 0),
            "as_of": safe_date,
            "generated_at": now.isoformat(),
            "source": "alpaca_sip_completed_daily_raw",
            "timeframes": [1, 5, 20],
            "total_assets": len(rows),
            "available_assets": available,
            "items": rows,
        }
        _CACHE[identity] = (time.monotonic() + _CACHE_TTL_SECONDS, response)
        # Keep the cache small across daily/strategy changes.
        if len(_CACHE) > 8:
            for key in list(_CACHE)[:-4]:
                _CACHE.pop(key, None)
        return response
