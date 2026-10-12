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


def test_iex_missing_symbol_recovers_from_delayed_sip() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"AMS": _daily_frame("AMS")}
    requests = []

    def fetch(**kwargs):
        requests.append(kwargs)
        if kwargs["feed"] == "iex":
            return pd.DataFrame()
        assert kwargs["feed"] == "sip"
        return _intraday_frame()

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
        ),
        patch.object(market_data, "download_stock_bars", side_effect=fetch),
    ):
        augmented, metadata = market_data.append_u67_current_session_intraday(
            frames,
            config,
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
        )

    assert list(augmented) == ["AMS"]
    assert len(requests) == 2
    assert requests[0]["feed"] == "iex"
    assert requests[1]["feed"] == "sip"
    assert requests[1]["end"] == pd.Timestamp(
        "2026-10-07T16:40:00Z"
    ).to_pydatetime()
    assert float(augmented["AMS"]["close"].iloc[-1]) == 102.5
    assert augmented["AMS"].attrs["current_session_live_feed"] == "sip"
    assert metadata["intraday_fallback_assets"] == ["AMS"]
    assert metadata["intraday_missing_assets"] == []
    assert metadata["live_feed"] == "mixed_iex_sip"
    assert metadata["per_asset"]["AMS"]["feed"] == "sip"


def test_no_iex_or_sip_data_omits_only_nonheld_asset() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"AMS": _daily_frame("AMS"), "NVDA": _daily_frame("NVDA")}

    def fetch(**kwargs):
        if kwargs["symbol"] == "AMS":
            return pd.DataFrame()
        return _intraday_frame()

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
        ),
        patch.object(market_data, "download_stock_bars", side_effect=fetch),
    ):
        augmented, metadata = market_data.append_u67_current_session_intraday(
            frames,
            config,
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            current_asset="NVDA",
            per_asset_pause_seconds=0,
        )

    assert set(augmented) == {"NVDA"}
    assert metadata["intraday_asset_count"] == 1
    assert metadata["intraday_missing_assets"] == ["AMS"]
    assert "no IEX current-session bars" in metadata[
        "intraday_missing_reasons"
    ]["AMS"]
    assert "AMS" not in augmented
    assert "AMS" in frames


def test_held_asset_without_iex_or_sip_does_not_fabricate_quote() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"AMS": _daily_frame("AMS"), "NVDA": _daily_frame("NVDA")}

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
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
                current_asset="AMS",
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayDataMissing" in str(exc)
            assert "currently held asset" in str(exc)
            assert "AMS" in str(exc)
        else:
            raise AssertionError("Held asset cannot use a fabricated price.")


def test_sip_permission_failure_does_not_hide_other_available_assets() -> None:
    config = SimpleNamespace(alpaca_live_feed="iex")
    frames = {"AMS": _daily_frame("AMS"), "NVDA": _daily_frame("NVDA")}

    def fetch(**kwargs):
        if kwargs["symbol"] == "AMS" and kwargs["feed"] == "sip":
            raise PermissionError("SIP entitlement unavailable")
        if kwargs["symbol"] == "AMS":
            return pd.DataFrame()
        return _intraday_frame()

    with (
        patch.object(
            market_data,
            "get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
        ),
        patch.object(market_data, "download_stock_bars", side_effect=fetch),
    ):
        augmented, metadata = market_data.append_u67_current_session_intraday(
            frames,
            config,
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
        )

    assert list(augmented) == ["NVDA"]
    assert metadata["intraday_missing_assets"] == ["AMS"]
    assert "PermissionError" in metadata["intraday_missing_reasons"]["AMS"]
