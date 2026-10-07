from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from market_cycle_trader_api.engine import (
    tcc_u67_operational_market_data as market_data,
)


def _daily_frame(symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1_000.0,
            }
        ],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-10-06T04:00:00Z")],
            name="timestamp",
        ),
    )
    frame.attrs["symbol"] = symbol
    return frame


def _intraday_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open": 101.0,
                "high": 102.0,
                "low": 100.5,
                "close": 101.5,
                "volume": 100.0,
                "vwap": 101.4,
                "trade_count": 10.0,
            },
            {
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "volume": 200.0,
                "vwap": 102.0,
                "trade_count": 20.0,
            },
        ],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-10-07T13:30:00Z"),
                pd.Timestamp("2026-10-07T13:31:00Z"),
            ],
            name="timestamp",
        ),
    )


def test_current_session_bar_aggregates_intraday_ohlcv() -> None:
    bar = market_data._current_session_intraday_bar(
        _intraday_frame(),
        session="2026-10-07",
    )

    assert list(bar.index) == [pd.Timestamp("2026-10-07T04:00:00Z")]
    assert float(bar.iloc[0]["open"]) == 101.0
    assert float(bar.iloc[0]["high"]) == 103.0
    assert float(bar.iloc[0]["low"]) == 100.5
    assert float(bar.iloc[0]["close"]) == 102.5
    assert float(bar.iloc[0]["volume"]) == 300.0
    assert float(bar.iloc[0]["trade_count"]) == 30.0
    assert abs(float(bar.iloc[0]["vwap"]) - 101.8) < 1e-12


def test_append_current_session_uses_live_feed_and_today_only() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"XSD": _daily_frame("XSD")}

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={
                "api_key_id": "key",
                "secret_key": "secret",
            },
        ),
        patch.object(
            market_data,
            "download_stock_bars",
            return_value=_intraday_frame(),
        ) as download,
    ):
        augmented, metadata = (
            market_data.append_u67_current_session_intraday(
                frames,
                config,
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        )

    request = download.call_args.kwargs
    assert request["timeframe"] == "1Min"
    assert request["feed"] == "iex"
    assert request["adjustment"] == "raw"
    assert request["end"] == pd.Timestamp(
        "2026-10-07T17:00:00Z"
    ).to_pydatetime()

    result = augmented["XSD"]
    assert list(result.index) == [
        pd.Timestamp("2026-10-06T04:00:00Z"),
        pd.Timestamp("2026-10-07T04:00:00Z"),
    ]
    assert float(result.iloc[-1]["close"]) == 102.5
    assert metadata["analysis_mode"] == "current_session_intraday"
    assert metadata["live_session"] == "2026-10-07"
    assert metadata["live_feed"] == "iex"
    assert metadata["live_timeframe"] == "1Min"
    assert metadata["intraday_asset_count"] == 1


def test_append_current_session_fails_closed_without_live_bars() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"XSD": _daily_frame("XSD")}

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={
                "api_key_id": "key",
                "secret_key": "secret",
            },
        ),
        patch.object(
            market_data,
            "download_stock_bars",
            return_value=pd.DataFrame(),
        ),
    ):
        try:
            market_data.append_u67_current_session_intraday(
                frames,
                config,
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayDataMissing" in str(exc)
        else:
            raise AssertionError(
                "Expected missing current-session data to fail closed."
            )
