from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import time
from typing import Any

import pandas as pd
import requests

from ...core.environment import load_project_environment


DEFAULT_BASE_URL = "https://api.tiingo.com"
SUPPORTED_TIMEFRAMES = {"1Day"}
SUPPORTED_ADJUSTMENTS = {"raw", "all"}


@dataclass(frozen=True)
class TiingoSettings:
    api_key: str
    base_url: str
    timeout_seconds: float
    max_retries: int


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}.")
    return value


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}.")
    return value


def get_tiingo_settings() -> TiingoSettings:
    """Read Tiingo connection settings exclusively from the process/.env."""
    load_project_environment()

    api_key = str(os.getenv("TIINGO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Tiingo API credentials are not configured. "
            "Set TIINGO_API_KEY in the server .env file."
        )

    base_url = str(os.getenv("TIINGO_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url:
        raise RuntimeError("TIINGO_BASE_URL cannot be empty.")

    return TiingoSettings(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=_env_float("TIINGO_TIMEOUT_SECONDS", 30.0, minimum=1.0),
        max_retries=_env_int("TIINGO_MAX_RETRIES", 3, minimum=1),
    )


def _normalize_date(value: str | datetime | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid Tiingo date: {value}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.date().isoformat()


def _request_json(
    url: str,
    *,
    settings: TiingoSettings,
    params: dict[str, Any] | None = None,
) -> Any:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {settings.api_key}",
        "User-Agent": "market-cycle-trader-api/tiingo",
    }

    last_error: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=settings.timeout_seconds,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and str(retry_after).replace(".", "", 1).isdigit()
                    else float(2**attempt)
                )
                if attempt + 1 < settings.max_retries:
                    time.sleep(max(1.0, delay))
                    continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < settings.max_retries:
                time.sleep(float(2**attempt))

    raise RuntimeError(
        f"Tiingo request failed after {settings.max_retries} attempts: {last_error}"
    ) from last_error


def normalize_tiingo_eod_rows(
    rows: Any,
    *,
    symbol: str,
    adjustment: str,
) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    normalized_adjustment = str(adjustment or "all").strip().lower()
    if normalized_adjustment not in SUPPORTED_ADJUSTMENTS:
        raise ValueError(
            "Tiingo EOD integration supports adjustment='raw' or adjustment='all'. "
            f"Received: {normalized_adjustment}."
        )

    source_columns = (
        {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        if normalized_adjustment == "raw"
        else {
            "adjOpen": "open",
            "adjHigh": "high",
            "adjLow": "low",
            "adjClose": "close",
            "adjVolume": "volume",
        }
    )

    records: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        date_value = item.get("date")
        if not date_value:
            continue

        record: dict[str, Any] = {"timestamp": date_value}
        missing = False
        for source, target in source_columns.items():
            value = item.get(source)
            if value is None:
                missing = True
                break
            record[target] = value
        if missing:
            continue

        # Keep corporate-action metadata for diagnostics. The rotation engine
        # consumes only OHLCV, but these fields make source verification easier.
        record["split_factor"] = item.get("splitFactor")
        record["div_cash"] = item.get("divCash")
        records.append(record)

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
    frame.index.name = "timestamp"

    required = ["open", "high", "low", "close", "volume"]
    for column in required + ["split_factor", "div_cash"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def download_stock_bars(
    *,
    symbol: str,
    timeframe: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp | None,
    adjustment: str,
    settings: TiingoSettings | None = None,
) -> pd.DataFrame:
    normalized_timeframe = str(timeframe or "").strip()
    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "This isolated Tiingo test branch supports only 1Day bars. "
            f"Received timeframe={normalized_timeframe}."
        )

    active_settings = settings or get_tiingo_settings()
    ticker = str(symbol or "").strip().upper()
    if not ticker:
        raise ValueError("Tiingo symbol cannot be empty.")

    params: dict[str, Any] = {
        "startDate": _normalize_date(start),
        "resampleFreq": "daily",
    }
    normalized_end = _normalize_date(end)
    if normalized_end:
        params["endDate"] = normalized_end

    rows = _request_json(
        f"{active_settings.base_url}/tiingo/daily/{ticker}/prices",
        settings=active_settings,
        params=params,
    )
    return normalize_tiingo_eod_rows(
        rows,
        symbol=ticker,
        adjustment=adjustment,
    )


def test_connection(symbol: str = "SPY") -> dict[str, Any]:
    settings = get_tiingo_settings()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=10)
    frame = download_stock_bars(
        symbol=symbol,
        timeframe="1Day",
        start=start,
        end=end,
        adjustment="all",
        settings=settings,
    )
    if frame.empty:
        raise RuntimeError(
            f"Tiingo authentication succeeded but no daily bars were returned for {symbol}."
        )
    return {
        "ok": True,
        "provider": "tiingo",
        "symbol": symbol,
        "bars": int(len(frame)),
        "last_timestamp": pd.Timestamp(frame.index[-1]).isoformat(),
        "last_close": float(frame.iloc[-1]["close"]),
        "base_url": settings.base_url,
    }
