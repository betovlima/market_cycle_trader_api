from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pymongo.database import Database

from ...infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    MARKET_BARS_COLLECTION,
)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _diagnostic_trade_frame(
    trade_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not trade_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(trade_rows).copy()
    if "timestamp" not in frame:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame = frame.dropna(subset=["timestamp"]).sort_values(
        ["timestamp", "sequence"] if "sequence" in frame else ["timestamp"]
    )
    return frame


def _classify_relative_episode(
    *,
    benchmark_return: float,
    strategy_return: float,
    cash_share: float,
    rotations: int,
    duration_sessions: int,
    average_holding: float | None,
) -> str:
    active_rotation_threshold = max(3, int(math.ceil(duration_sessions / 5)))

    if benchmark_return >= 0.08:
        if cash_share >= 0.15:
            return "CASH_DRAG"
        if (
            rotations >= active_rotation_threshold
            and average_holding is not None
            and average_holding <= 3.0
        ):
            return "OVER_ROTATION"
        if average_holding is not None and average_holding <= 3.0:
            return "EARLY_EXIT_RISK"
        return "TREND_CAPTURE_GAP"

    if benchmark_return < 0 and strategy_return < benchmark_return:
        return "DOWNSIDE_CONTROL_FAILURE"

    if rotations >= active_rotation_threshold:
        return "OVER_ROTATION"

    return "RELATIVE_UNDERPERFORMANCE"


def _empty_market_series() -> pd.Series:
    
    return pd.Series(
        data=[],
        index=pd.DatetimeIndex([], tz="UTC"),
        dtype=float,
    )


def _market_close_series(
    db: Database,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    timestamp_filter = {
        "$gte": start.to_pydatetime(),
        "$lte": end.to_pydatetime(),
    }
    projection = {
        "_id": 0,
        "timestamp": 1,
        "close": 1,
        "updated_at": 1,
    }
    sources = (
        (
            ALPACA_MARKET_BARS_COLLECTION,
            {
                "symbol": symbol,
                "interval": "1Day",
                "timestamp": timestamp_filter,
            },
        ),
        (
            MARKET_BARS_COLLECTION,
            {
                "symbol": symbol,
                "interval": "1d",
                "timestamp": timestamp_filter,
            },
        ),
    )

    rows: list[dict[str, Any]] = []
    for collection_name, query in sources:
        rows = list(
            db[collection_name]
            .find(query, projection)
            .sort([("timestamp", 1), ("updated_at", 1)])
        )
        if rows:
            break

    if not rows:
        return _empty_market_series()

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"])
    if frame.empty:
        return _empty_market_series()

    series = pd.Series(
        frame["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"], name="timestamp"),
        dtype=float,
        name="close",
    )
    return series.loc[~series.index.duplicated(keep="last")].sort_index()


def _future_market_prices(
    prices: pd.Series | None,
    sold_at: Any,
) -> pd.Series:
    






    if prices is None or prices.empty:
        return _empty_market_series()

    if not isinstance(prices.index, pd.DatetimeIndex):
        return _empty_market_series()

    try:
        exit_timestamp = pd.Timestamp(sold_at)
    except (TypeError, ValueError):
        return _empty_market_series()

    if pd.isna(exit_timestamp):
        return _empty_market_series()

    if exit_timestamp.tzinfo is None:
        exit_timestamp = exit_timestamp.tz_localize("UTC")
    else:
        exit_timestamp = exit_timestamp.tz_convert("UTC")

    normalized_index = pd.to_datetime(
        prices.index,
        utc=True,
        errors="coerce",
    )
    numeric_values = pd.to_numeric(prices.to_numpy(), errors="coerce")
    normalized = pd.DataFrame(
        {
            "timestamp": normalized_index,
            "close": numeric_values,
        }
    ).dropna(subset=["timestamp", "close"])

    if normalized.empty:
        return _empty_market_series()

    normalized = normalized.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    series = pd.Series(
        normalized["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(normalized["timestamp"]),
        dtype=float,
    )
    return series.loc[series.index > exit_timestamp]
