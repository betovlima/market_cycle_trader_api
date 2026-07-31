from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pymongo.database import Database

from ...infrastructure.persistence.mongo_repository import MARKET_BARS_COLLECTION


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


def _market_close_series(
    db: Database,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    query = {
        "symbol": symbol,
        "interval": "1d",
        "timestamp": {
            "$gte": start.to_pydatetime(),
            "$lte": end.to_pydatetime(),
        },
    }
    rows = list(
        db[MARKET_BARS_COLLECTION]
        .find(query, {"_id": 0, "timestamp": 1, "close": 1})
        .sort("timestamp", 1)
    )
    if not rows:
        return pd.Series(dtype=float)

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"])
    if frame.empty:
        return pd.Series(dtype=float)

    return pd.Series(
        frame["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"]),
        dtype=float,
    ).sort_index()
