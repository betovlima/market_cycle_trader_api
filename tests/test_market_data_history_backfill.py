from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from market_cycle_trader_api.engine.market_data import (
    _download_alpaca_bars,
    complete_market_history,
    inclusive_end_exclusive_boundary,
    trim_downloaded_range,
)
from market_cycle_trader_api.schemas.requests import BacktestRequest


def _config() -> BacktestRequest:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "winner-v1.13.2.json"
    )
    return BacktestRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _frame(start: str, periods: int, base: float) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    values = [base + offset for offset in range(periods)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [value + 1 for value in values],
            "low": [value - 1 for value in values],
            "close": values,
            "volume": [1000.0] * periods,
        },
        index=index,
    )


def test_complete_alpaca_history_is_accepted() -> None:
    config = _config()
    frame = _frame("2016-01-04", 1200, 100.0)

    result = complete_market_history(
        "AAPL",
        frame,
        config,
        provider="alpaca",
        initial_rows=len(frame),
    )

    provenance = result.attrs["market_data_provenance"]
    assert provenance["history_complete"] is True
    assert provenance["provider"] == "alpaca"
    assert provenance["effective_provider"] == "alpaca"
    assert provenance["history_backfill_provider"] is None
    assert provenance["history_backfill_rows"] == 0


def test_incomplete_alpaca_history_stops_without_secondary_provider() -> None:
    config = _config()
    incomplete = _frame("2020-07-27", 500, 200.0)

    with pytest.raises(RuntimeError, match="Incomplete Alpaca market history"):
        complete_market_history(
            "MSFT",
            incomplete,
            config,
            provider="alpaca",
            initial_rows=len(incomplete),
        )


def test_long_alpaca_history_is_downloaded_in_date_chunks() -> None:
    config = _config()

    def fake_download(**kwargs):
        start = pd.Timestamp(kwargs["start"])
        end = pd.Timestamp(kwargs["end"])
        midpoint = start + (end - start) / 2
        return _frame(midpoint.strftime("%Y-%m-%d"), 1, 100.0)

    with (
        patch(
            "market_cycle_trader_api.engine.market_data.get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
        ),
        patch(
            "market_cycle_trader_api.engine.market_data.download_stock_bars",
            side_effect=fake_download,
        ) as downloader,
    ):
        result = _download_alpaca_bars(
            "NVDA",
            config,
            "2016-01-01",
            "2026-01-01",
        )

    assert downloader.call_count > 1
    assert not result.empty
    for call in downloader.call_args_list:
        assert call.kwargs["feed"] == config.alpaca_historical_feed
        assert call.kwargs["adjustment"] == config.alpaca_adjustment
        assert call.kwargs["symbol"] == "NVDA"


def test_historical_end_date_is_inclusive_for_daily_bars() -> None:
    frame = _frame("2024-01-08", 3, 100.0)

    result = trim_downloaded_range(
        frame,
        requested_start="2024-01-08",
        requested_end="2024-01-09",
        timeframe="1Day",
    )

    assert list(result.index.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-09"]
    assert inclusive_end_exclusive_boundary("2024-01-09") == pd.Timestamp("2024-01-10", tz="UTC")


def test_alpaca_download_uses_day_after_inclusive_historical_end() -> None:
    config = _config()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return _frame("2024-01-09", 1, 100.0)

    with (
        patch(
            "market_cycle_trader_api.engine.market_data.get_alpaca_credentials",
            return_value={"api_key_id": "key", "secret_key": "secret"},
        ),
        patch(
            "market_cycle_trader_api.engine.market_data.download_stock_bars",
            side_effect=fake_download,
        ),
    ):
        result = _download_alpaca_bars(
            "NVDA",
            config,
            "2024-01-08",
            "2024-01-09",
        )

    assert calls
    assert pd.Timestamp(calls[-1]["end"]) == pd.Timestamp("2024-01-10", tz="UTC")
    assert list(result.index.strftime("%Y-%m-%d")) == ["2024-01-09"]
