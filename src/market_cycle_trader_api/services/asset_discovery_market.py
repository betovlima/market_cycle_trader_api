from __future__ import annotations

import os
import re
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..engine.market_data import REQUIRED_BAR_COLUMNS, load_market_bars
from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.persistence.mongo_repository import get_alpaca_credentials
from .asset_discovery_behavior import behavior_risk_profile

SUPPORTED_EXCHANGES = frozenset({"AMEX", "ARCA", "BATS", "NASDAQ", "NYSE"})
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
RECENT_PREFILTER_DAYS = 120
BEHAVIOR_PREFILTER_DAYS = 1_125
SIP_DELAY_BUFFER_MINUTES = 20
EASTERN = ZoneInfo("America/New_York")


class NoRecentMarketData(RuntimeError):
    pass


class NoHistoricalMarketData(RuntimeError):
    pass


class MarketDataAccessBlocked(RuntimeError):
    pass


def _alpaca_headers(credentials: dict[str, str]) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": credentials["api_key_id"],
        "APCA-API-SECRET-KEY": credentials["secret_key"],
    }


def _trading_base_url() -> str:
    return str(os.getenv("ALPACA_TRADING_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")


def _session_close_utc(session: dict[str, Any]) -> datetime | None:
    date_text = str(session.get("date") or "").strip()
    close_text = str(session.get("close") or "").strip()
    if not date_text or not close_text:
        return None
    try:
        session_date = date.fromisoformat(date_text[:10])
        close_clock = dt_time.fromisoformat(close_text)
    except ValueError:
        return None
    local_close = datetime.combine(session_date, close_clock, tzinfo=EASTERN)
    return local_close.astimezone(timezone.utc)


def _latest_safe_completed_session_end(credentials: dict[str, str]) -> datetime:
    

    now = datetime.now(timezone.utc)
    sip_cutoff = now - timedelta(minutes=SIP_DELAY_BUFFER_MINUTES)
    response = requests.get(
        f"{_trading_base_url()}/v2/calendar",
        params={
            "start": (now.date() - timedelta(days=14)).isoformat(),
            "end": now.date().isoformat(),
        },
        headers=_alpaca_headers(credentials),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Alpaca returned an unexpected market-calendar payload.")

    completed = [
        close_at
        for item in payload
        if isinstance(item, dict)
        for close_at in [_session_close_utc(item)]
        if close_at is not None and close_at <= sip_cutoff
    ]
    if not completed:
        raise RuntimeError("Alpaca calendar did not return a completed SIP-safe trading session.")
    return max(completed)


def _raise_if_global_market_data_error(exc: Exception) -> None:
    message = str(exc)
    lowered = message.lower()
    global_markers = (
        "subscription does not permit querying recent sip data",
        "alpaca api credentials are not configured",
        "unauthorized",
        "forbidden",
        "status code 401",
        "status code 403",
    )
    if any(marker in lowered for marker in global_markers):
        raise MarketDataAccessBlocked(message) from exc


def discover_alpaca_symbols() -> list[str]:
    credentials = get_alpaca_credentials()
    base_url = _trading_base_url()
    response = requests.get(
        f"{base_url}/v2/assets",
        params={"status": "active", "asset_class": "us_equity"},
        headers=_alpaca_headers(credentials),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Alpaca returned an unexpected assets payload.")

    symbols: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        exchange = str(item.get("exchange") or "").strip().upper()
        if not symbol or not SYMBOL_PATTERN.fullmatch(symbol):
            continue
        if exchange not in SUPPORTED_EXCHANGES:
            continue
        if not bool(item.get("tradable")):
            continue
        symbols.append(symbol)
    return sorted(set(symbols))


def resolve_completed_market_data_end() -> datetime:
    credentials = get_alpaca_credentials()
    try:
        return _latest_safe_completed_session_end(credentials)
    except Exception as exc:
        raise MarketDataAccessBlocked(
            f"Unable to resolve a completed Alpaca trading session: {exc}"
        ) from exc


def _recent_market_frame(symbol: str, config: Any, *, end: datetime) -> pd.DataFrame:
    credentials = get_alpaca_credentials()
    start = end - timedelta(days=RECENT_PREFILTER_DAYS)
    try:
        return download_stock_bars(
            api_key_id=credentials["api_key_id"],
            secret_key=credentials["secret_key"],
            symbol=symbol,
            timeframe="1Day",
            start=start,
            end=end,
            feed=config.alpaca_historical_feed,
            adjustment=config.alpaca_adjustment,
        )
    except Exception as exc:
        _raise_if_global_market_data_error(exc)
        raise




def _behavior_market_frame(symbol: str, config: Any, *, end: datetime, settings: dict[str, Any]) -> pd.DataFrame:
    

    credentials = get_alpaca_credentials()
    lookback_days = int(settings.get("behavior_lookback_days", BEHAVIOR_PREFILTER_DAYS))
    start = end - timedelta(days=lookback_days)
    try:
        frame = download_stock_bars(
            api_key_id=credentials["api_key_id"],
            secret_key=credentials["secret_key"],
            symbol=symbol,
            timeframe="1Day",
            start=start,
            end=end,
            feed=config.alpaca_historical_feed,
            adjustment=config.alpaca_adjustment,
        )
    except Exception as exc:
        _raise_if_global_market_data_error(exc)
        raise
    return _basic_clean_history(frame)

def _quality_metrics(frame: pd.DataFrame) -> tuple[float, float, float]:
    if frame is None or frame.empty:
        raise NoRecentMarketData("No usable recent daily market data was returned.")
    recent = frame.tail(min(63, len(frame)))
    close = pd.to_numeric(recent.get("close"), errors="coerce")
    volume = pd.to_numeric(recent.get("volume"), errors="coerce")
    valid_close = close.dropna()
    if valid_close.empty:
        raise NoRecentMarketData("Recent daily market data has no usable close values.")
    latest_close = float(valid_close.iloc[-1])
    dollar_volume = close * volume
    median_dollar_volume = float(dollar_volume.dropna().median()) if not dollar_volume.dropna().empty else 0.0
    nonzero_volume_ratio = float((volume > 0).mean()) if len(volume) else 0.0
    return latest_close, median_dollar_volume, nonzero_volume_ratio


def _basic_clean_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise NoHistoricalMarketData("No historical daily bars were returned.")
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise NoHistoricalMarketData("Historical daily bars are missing required OHLCV fields.")
    result = frame.copy()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=list(REQUIRED_BAR_COLUMNS))
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[pd.to_numeric(result["volume"], errors="coerce") >= 0]
    if result.empty:
        raise NoHistoricalMarketData("Historical daily bars were present but none were usable after cleaning.")
    return result


def _minimum_model_sessions(config: Any) -> int:
    return (
        int(config.rotation_minimum_training_rows)
        + max(int(item) for item in config.rotation_target_horizons)
        + int(config.rotation_purge_days)
    )


def _history_profile(frame: pd.DataFrame, config: Any) -> tuple[str, bool]:
    first = pd.Timestamp(frame.index.min())
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    else:
        first = first.tz_convert("UTC")
    requested = pd.Timestamp(config.start_date, tz="UTC")
    tolerance = pd.Timedelta(
        int(getattr(config, "market_data_history_start_tolerance_days", 0)),
        unit="D",
    )
    model_ready = len(frame) >= _minimum_model_sessions(config)
    if first.normalize() <= requested.normalize() + tolerance:
        return "full_history", model_ready
    return ("limited_history" if model_ready else "young_history"), model_ready


def _available_history(symbol: str, config: Any) -> pd.DataFrame:
    
    
    
    
    discovery_config = config.model_copy(
        update={
            "end_date": None,
            "mongo_cache_enabled": True,
            "market_data_require_complete_history": False,
            "market_data_history_backfill_enabled": False,
        }
    )
    try:
        return _basic_clean_history(load_market_bars(symbol, discovery_config))
    except RuntimeError as exc:
        if "cache returned no bars" in str(exc).lower():
            raise NoHistoricalMarketData("No historical daily bars were returned.") from exc
        raise


def market_quality_snapshot(
    symbol: str,
    config: Any,
    settings: dict[str, Any],
    *,
    recent_end: datetime,
) -> dict[str, Any]:
    recent = _recent_market_frame(symbol, config, end=recent_end)
    latest_close, median_dollar_volume, nonzero_volume_ratio = _quality_metrics(recent)
    checks = {
        "price_ready": latest_close >= float(settings["min_price"]),
        "liquidity_ready": median_dollar_volume >= float(settings["min_median_dollar_volume"]),
        "volume_quality_ready": nonzero_volume_ratio >= float(settings["min_nonzero_volume_ratio"]),
    }
    reason_codes = [name for name, passed in checks.items() if passed]
    reason_codes.extend(f"{name}_failed" for name, passed in checks.items() if not passed)

    if not all(checks.values()):
        return {
            "status": "rejected",
            "historical_cache_ready": False,
            "history_profile": None,
            "model_ready": False,
            "history_start": None,
            "history_end": None,
            "history_sessions": None,
            "latest_close": latest_close,
            "median_dollar_volume_63d": median_dollar_volume,
            "nonzero_volume_ratio": nonzero_volume_ratio,
            "reason_codes": reason_codes,
        }

    
    
    
    
    behavior_frame = _behavior_market_frame(symbol, config, end=recent_end, settings=settings)
    behavior = behavior_risk_profile(behavior_frame, settings)
    if behavior.get("sample_ready") and not behavior.get("passed"):
        return {
            "status": "rejected",
            "historical_cache_ready": False,
            "history_profile": None,
            "model_ready": False,
            "history_start": None,
            "history_end": None,
            "history_sessions": None,
            "latest_close": latest_close,
            "median_dollar_volume_63d": median_dollar_volume,
            "nonzero_volume_ratio": nonzero_volume_ratio,
            "behavior_profile": behavior,
            "reason_codes": reason_codes + list(behavior.get("reason_codes") or []),
        }

    
    
    
    
    frame = _available_history(symbol, config)
    history_profile, model_ready = _history_profile(frame, config)
    status = "candidate" if model_ready else "watchlist"
    readiness_code = "model_ready" if model_ready else "model_not_ready"
    return {
        "status": status,
        "historical_cache_ready": True,
        "history_profile": history_profile,
        "model_ready": model_ready,
        "history_start": pd.Timestamp(frame.index.min()).isoformat(),
        "history_end": pd.Timestamp(frame.index.max()).isoformat(),
        "history_sessions": int(len(frame)),
        "latest_close": latest_close,
        "median_dollar_volume_63d": median_dollar_volume,
        "nonzero_volume_ratio": nonzero_volume_ratio,
        "behavior_profile": behavior,
        "reason_codes": reason_codes + list(behavior.get("reason_codes") or []) + [history_profile, readiness_code, "historical_cache_ready"],
    }
