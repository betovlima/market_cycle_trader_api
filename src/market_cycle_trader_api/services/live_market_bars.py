"""Live-only current-session market data, independent of the Paper pipeline.

All eligible symbols use the same provider feed and as-of timestamp.
Only completed one-minute bars are aggregated into a scoring observation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.trading.alpaca_live import live_credentials

EASTERN = ZoneInfo("America/New_York")
_LIVE_SIP_DELAY_MINUTES = 20
_REQUIRED = ("open", "high", "low", "close", "volume")


def load_live_intraday_session(
    historical: dict[str, pd.DataFrame],
    config: Any,
    *,
    session: str,
    now: datetime | pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build a consistent, current-session observation for the Live Winner."""
    stamp = pd.Timestamp(now)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp.tz_convert(EASTERN).date().isoformat() != session:
        raise RuntimeError("LiveIntradaySessionMismatch: broker clock/session disagreement.")
    start = pd.Timestamp(f"{session} 09:30:00", tz=EASTERN).tz_convert("UTC")
    completed_cutoff = stamp.floor("min")
    if completed_cutoff <= start:
        raise RuntimeError("LiveIntradayCoverageNotReady: no completed regular-session bar.")
    preferred = str(getattr(config, "alpaca_live_feed", "iex") or "iex").strip().lower()
    if preferred not in {"iex", "sip"}:
        raise RuntimeError("LiveIntradayFeedInvalid: feed must be IEX or SIP.")
    key, secret = live_credentials()
    if not historical:
        raise RuntimeError("LiveIntradayCoverageNotReady: no historical universe.")

    def full_universe(feed: str, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        if end <= start:
            raise RuntimeError("LiveIntradayCoverageNotReady: delayed SIP begins before the opening.")
        result: dict[str, pd.DataFrame] = {}
        for symbol, daily in historical.items():
            bars = download_stock_bars(
                api_key_id=key, secret_key=secret, symbol=symbol,
                timeframe="1Min", start=start.to_pydatetime(),
                end=end.to_pydatetime(), feed=feed, adjustment="raw",
            )
            if bars is None or bars.empty:
                raise RuntimeError(f"LiveIntradayCoverageNotReady: {symbol} has no {feed.upper()} bars.")
            bars = bars.copy()
            bars.index = pd.to_datetime(bars.index, utc=True)
            bars = bars.loc[
                (bars.index >= start) & (bars.index < end) &
                (bars.index.tz_convert(EASTERN).date == pd.Timestamp(session).date())
            ].sort_index()
            if bars.empty or any(column not in bars for column in _REQUIRED):
                raise RuntimeError(f"LiveIntradayCoverageNotReady: incomplete {feed.upper()} bars for {symbol}.")
            vol = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
            row = {
                "open": float(bars["open"].iloc[0]),
                "high": float(bars["high"].max()),
                "low": float(bars["low"].min()),
                "close": float(bars["close"].iloc[-1]),
                "volume": float(vol.sum()),
            }
            if "trade_count" in bars:
                row["trade_count"] = float(pd.to_numeric(bars["trade_count"], errors="coerce").fillna(0).sum())
            if "vwap" in bars and vol.sum() > 0:
                row["vwap"] = float((pd.to_numeric(bars["vwap"], errors="coerce").fillna(bars["close"]) * vol).sum() / vol.sum())
            else:
                row["vwap"] = row["close"]
            if not all(pd.notna(value) for value in row.values()) or min(
                row["open"], row["high"], row["low"], row["close"]
            ) <= 0:
                raise RuntimeError(f"LiveIntradayCoverageNotReady: invalid OHLCV for {symbol}.")
            current = pd.DataFrame([row], index=pd.DatetimeIndex(
                [pd.Timestamp(session, tz=EASTERN).tz_convert("UTC")], name="timestamp"
            ))
            previous = daily.copy()
            previous.index = pd.to_datetime(previous.index, utc=True)
            previous = previous.loc[previous.index.tz_convert(EASTERN).date != pd.Timestamp(session).date()]
            merged = pd.concat([previous, current]).sort_index()
            merged = merged.loc[~merged.index.duplicated(keep="last")]
            merged.attrs.update(dict(getattr(daily, "attrs", {}) or {}))
            merged.attrs.update({
                "current_session_intraday": True, "current_session": session,
                "current_session_live_feed": feed,
                "current_session_market_data_cutoff_utc": end.isoformat(),
            })
            result[symbol] = merged
        return result

    try:
        augmented = full_universe(preferred, completed_cutoff)
        feed = preferred
        delay = 0
    except (RuntimeError, ValueError) as preferred_error:
        if preferred != "iex" or not str(preferred_error).startswith("LiveIntradayCoverageNotReady:"):
            raise
        try:
            augmented = full_universe("sip", completed_cutoff)
            feed = "sip"
            delay = 0
        except Exception as sip_error:
            entitlement = "subscription does not permit querying recent sip data" in str(sip_error).lower()
            if not entitlement:
                raise RuntimeError(
                    f"LiveIntradayCoverageNotReady: IEX={preferred_error}; SIP={sip_error}"
                ) from sip_error
            delayed = (stamp - pd.Timedelta(minutes=_LIVE_SIP_DELAY_MINUTES)).floor("min")
            augmented = full_universe("sip", delayed)
            feed = "sip"
            delay = _LIVE_SIP_DELAY_MINUTES
            completed_cutoff = delayed

    return augmented, {
        "analysis_mode": "live_current_session_intraday",
        "live_session": session,
        "live_feed": feed,
        "intraday_asset_count": len(augmented),
        "intraday_market_data_cutoff_utc": completed_cutoff.isoformat(),
        "intraday_data_delay_minutes": delay,
    }
