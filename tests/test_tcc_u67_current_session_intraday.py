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
            assert "CurrentSessionIntradayCoverageUnavailable" in str(exc)
            assert "SIP retry failed" in str(exc)
        else:
            raise AssertionError(
                "Expected missing current-session data to fail closed."
            )


def test_missing_iex_ams_retries_entire_universe_on_uniform_sip():
    frames = {
        "MAN": _daily_frame("MAN"),
        "AMS": _daily_frame("AMS"),
        "CAPR": _daily_frame("CAPR"),
    }
    calls = []

    def fake_bars(**kwargs):
        calls.append((kwargs["symbol"], kwargs["feed"]))
        if kwargs["symbol"] == "AMS" and kwargs["feed"] == "iex":
            return pd.DataFrame()
        return _intraday_frame()

    progress = []
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        result, metadata = market_data.append_u67_current_session_intraday(
            frames, SimpleNamespace(alpaca_live_feed="iex"),
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
            progress_callback=lambda stage, n, total, sym: progress.append(
                (stage, n, total, sym)
            ),
        )
    assert calls == [
        ("MAN", "iex"), ("AMS", "iex"),
        ("MAN", "sip"), ("AMS", "sip"), ("CAPR", "sip"),
    ]
    assert metadata["live_feed"] == "sip"
    assert "AMS" in metadata["live_feed_fallback_reason"]
    assert metadata["intraday_asset_count"] == 3
    assert list(result) == ["MAN", "AMS", "CAPR"]
    assert all(frame.attrs["current_session_live_feed"] == "sip"
               for frame in result.values())
    assert any(stage == "intraday_sip" for stage, _, _, _ in progress)


def test_missing_iex_sip_unauthorized_fails_without_returning_partial_data():
    frames = {"MAN": _daily_frame("MAN"), "AMS": _daily_frame("AMS")}
    calls = []

    def fake_bars(**kwargs):
        calls.append((kwargs["symbol"], kwargs["feed"]))
        if kwargs["feed"] == "sip":
            raise RuntimeError("403 forbidden: SIP entitlement required")
        return _intraday_frame() if kwargs["symbol"] == "MAN" else pd.DataFrame()

    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        try:
            market_data.append_u67_current_session_intraday(
                frames, SimpleNamespace(alpaca_live_feed="iex"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            message = str(exc)
            assert "CurrentSessionIntradayCoverageUnavailable" in message
            assert "AMS" in message
            assert "SIP entitlement" in message
        else:
            raise AssertionError("The live Winner must not use a partial feed.")
    assert calls == [("MAN", "iex"), ("AMS", "iex"), ("MAN", "sip")]


def test_no_sip_fallback_for_other_iex_data_failures():
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars",
                     side_effect=RuntimeError("IEX provider unavailable")) as download,
    ):
        try:
            market_data.append_u67_current_session_intraday(
                {"MAN": _daily_frame("MAN")},
                SimpleNamespace(alpaca_live_feed="iex"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert str(exc) == "IEX provider unavailable"
        else:
            raise AssertionError("Unexpected automatic feed substitution.")
    assert download.call_count == 1
    assert download.call_args.kwargs["feed"] == "iex"


def test_configured_sip_never_uses_iex():
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars",
                     return_value=_intraday_frame()) as download,
    ):
        _, metadata = market_data.append_u67_current_session_intraday(
            {"MAN": _daily_frame("MAN")},
            SimpleNamespace(alpaca_live_feed="sip"),
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
        )
    assert metadata["live_feed"] == "sip"
    assert metadata["live_feed_fallback_reason"] is None
    assert download.call_args.kwargs["feed"] == "sip"


def test_recent_sip_subscription_denial_retries_delayed_entire_universe():
    frames = {"MAN": _daily_frame("MAN"), "AMS": _daily_frame("AMS")}
    calls = []
    progress = []
    historical_cutoff = pd.Timestamp("2026-10-07T16:40:00Z").to_pydatetime()
    recent_cutoff = pd.Timestamp("2026-10-07T17:00:00Z").to_pydatetime()

    def fake_bars(**kwargs):
        calls.append((kwargs["symbol"], kwargs["feed"], kwargs["end"]))
        if kwargs["feed"] == "iex" and kwargs["symbol"] == "AMS":
            return pd.DataFrame()
        if kwargs["feed"] == "sip" and kwargs["end"] == recent_cutoff:
            raise RuntimeError(
                'Alpaca historical data request failed for MAN after 3 attempts: '
                '{"message":"subscription does not permit querying recent SIP data"}'
            )
        return _intraday_frame()

    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        result, metadata = market_data.append_u67_current_session_intraday(
            frames, SimpleNamespace(alpaca_live_feed="iex"),
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
            progress_callback=lambda *args: progress.append(args),
        )

    assert calls == [
        ("MAN", "iex", recent_cutoff),
        ("AMS", "iex", recent_cutoff),
        ("MAN", "sip", recent_cutoff),
        ("MAN", "sip", historical_cutoff),
        ("AMS", "sip", historical_cutoff),
    ]
    assert metadata["live_feed"] == "sip"
    assert metadata["intraday_feed_mode"] == "delayed_sip"
    assert metadata["intraday_data_delay_minutes"] == 20
    assert metadata["intraday_market_data_cutoff_utc"] == "2026-10-07T16:40:00+00:00"
    assert metadata["intraday_asset_count"] == 2
    assert all(x.attrs["current_session_live_feed"] == "sip" for x in result.values())
    assert all(x.attrs["current_session_market_data_cutoff_utc"]
               == metadata["intraday_market_data_cutoff_utc"] for x in result.values())
    assert any(row[0] == "intraday_sip_delayed" for row in progress)


def test_delayed_sip_still_fails_closed_when_any_asset_has_no_bars():
    frames = {"MAN": _daily_frame("MAN"), "AMS": _daily_frame("AMS")}
    def fake_bars(**kwargs):
        if kwargs["feed"] == "iex" and kwargs["symbol"] == "AMS":
            return pd.DataFrame()
        if kwargs["feed"] == "sip" and kwargs["end"] == pd.Timestamp(
            "2026-10-07T17:00:00Z"
        ).to_pydatetime():
            raise RuntimeError(
                "subscription does not permit querying recent SIP data"
            )
        if kwargs["feed"] == "sip" and kwargs["symbol"] == "AMS":
            return pd.DataFrame()
        return _intraday_frame()

    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        try:
            market_data.append_u67_current_session_intraday(
                frames, SimpleNamespace(alpaca_live_feed="iex"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayCoverageUnavailable" in str(exc)
            assert "AMS" in str(exc)
            assert "Delayed complete-universe SIP retry failed" in str(exc)
        else:
            raise AssertionError("Partial delayed SIP universe must not score.")


def test_delayed_sip_not_allowed_before_enough_session_history():
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars",
                     side_effect=RuntimeError(
                         "subscription does not permit querying recent SIP data"
                     )),
    ):
        try:
            market_data.append_u67_current_session_intraday(
                {"MAN": _daily_frame("MAN")},
                SimpleNamespace(alpaca_live_feed="sip"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T13:45:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayDelayedSIPNotReady" in str(exc)
        else:
            raise AssertionError("Premature delayed SIP scoring should be blocked.")


def test_configured_sip_recent_entitlement_uses_delayed_cutoff():
    end_recent = pd.Timestamp("2026-10-07T17:00:00Z").to_pydatetime()
    calls = []

    def fake_bars(**kwargs):
        calls.append(kwargs)
        if kwargs["end"] == end_recent:
            raise RuntimeError(
                "subscription does not permit querying recent SIP data"
            )
        return _intraday_frame()

    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        _, meta = market_data.append_u67_current_session_intraday(
            {"MAN": _daily_frame("MAN")},
            SimpleNamespace(alpaca_live_feed="sip"),
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T17:00:00Z"),
            per_asset_pause_seconds=0,
        )
    assert [x["feed"] for x in calls] == ["sip", "sip"]
    assert meta["intraday_data_delay_minutes"] == 20


def test_delayed_sip_does_not_use_any_bar_at_or_after_cutoff():
    today_after_cutoff_only = pd.DataFrame(
        [{
            "open": 1.0, "high": 1.0, "low": 1.0,
            "close": 1.0, "volume": 100.0,
        }],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-10-07T16:41:00Z")],
            name="timestamp",
        ),
    )
    calls = []
    def fake_bars(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "subscription does not permit querying recent SIP data"
            )
        return today_after_cutoff_only
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars", side_effect=fake_bars),
    ):
        try:
            market_data.append_u67_current_session_intraday(
                {"MAN": _daily_frame("MAN")},
                SimpleNamespace(alpaca_live_feed="sip"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T17:00:00Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayDataMissing" in str(exc)
        else:
            raise AssertionError("Out-of-cutoff future bar must not score.")


def test_subminute_clock_excludes_still_open_one_minute_bar():
    """09:31:37 cannot consume a bar beginning at 09:31."""
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars",
                     return_value=_intraday_frame()) as download,
    ):
        augmented, metadata = market_data.append_u67_current_session_intraday(
            {"MAN": _daily_frame("MAN")},
            SimpleNamespace(alpaca_live_feed="iex"),
            session="2026-10-07",
            now=pd.Timestamp("2026-10-07T13:31:37Z"),
            per_asset_pause_seconds=0,
        )
    assert download.call_args.kwargs["end"] == pd.Timestamp(
        "2026-10-07T13:31:00Z"
    ).to_pydatetime()
    assert metadata["intraday_market_data_cutoff_utc"] == "2026-10-07T13:31:00+00:00"
    assert metadata["intraday_asset_count"] == 1
    assert augmented["MAN"].iloc[-1]["close"] == 101.5
    assert augmented["MAN"].attrs["current_session_market_data_cutoff_utc"] == (
        "2026-10-07T13:31:00+00:00"
    )


def test_no_intraday_query_before_one_completed_regular_minute():
    with (
        patch.object(market_data, "get_alpaca_credentials",
                     return_value={"api_key_id": "key", "secret_key": "secret"}),
        patch.object(market_data, "download_stock_bars") as download,
    ):
        try:
            market_data.append_u67_current_session_intraday(
                {"MAN": _daily_frame("MAN")},
                SimpleNamespace(alpaca_live_feed="iex"),
                session="2026-10-07",
                now=pd.Timestamp("2026-10-07T13:30:46Z"),
                per_asset_pause_seconds=0,
            )
        except RuntimeError as exc:
            assert "CurrentSessionIntradayNotStarted" in str(exc)
        else:
            raise AssertionError("Current minute is not a completed bar.")
        download.assert_not_called()
