from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from market_cycle_trader_api.engine.market_data import (
    _download_alpaca_bars,
    complete_market_history,
    inclusive_end_exclusive_boundary,
    latest_completed_xnys_session,
    load_mongo_market_bars,
    resolve_backtest_analysis_end_date,
    trim_downloaded_range,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest


def _config() -> BacktestRequest:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "winner-v1.13.2.json"
    )
    return BacktestRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _execution_config(*, mode: str, end_date: str = "2024-01-09") -> BacktestExecutionRequest:
    base = _config()
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "end_date": None,
            "analysis_start_date": base.start_date,
            "analysis_end_date": end_date,
            "calendar_anchor_assets": list(base.assets),
            "research_reference_assets": list(base.assets),
            "research_candidate_assets": [],
            "research_model_family": "lightgbm_utility",
            "research_model_settings": {},
            "research_market_data_mode": mode,
        }
    )
    return BacktestExecutionRequest.model_validate(payload)


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


class _FakeCollection:
    def __init__(self, *, first=None):
        self.first = first

    def create_index(self, *args, **kwargs):
        return None

    def find_one(self, *args, **kwargs):
        return self.first




class _SessionCollection:
    def __init__(self, documents):
        self.documents = list(documents)

    def find_one(self, query, projection=None, sort=None):
        matches = []
        for document in self.documents:
            matched = True
            for key, expected in query.items():
                actual = document.get(key)
                if key == "timestamp" and isinstance(expected, dict):
                    if "$gte" in expected and not (actual >= expected["$gte"]):
                        matched = False
                    if "$lt" in expected and not (actual < expected["$lt"]):
                        matched = False
                elif actual != expected:
                    matched = False
                if not matched:
                    break
            if matched:
                matches.append(document)
        if sort and matches:
            key, direction = sort[0]
            matches.sort(key=lambda item: item[key], reverse=direction < 0)
        if not matches:
            return None
        result = dict(matches[0])
        if projection is not None:
            included = {key for key, value in projection.items() if value and key != "_id"}
            if included:
                result = {key: value for key, value in result.items() if key in included}
        return result


def _session_config(assets):
    return SimpleNamespace(
        assets=list(assets),
        timeframe="1Day",
        alpaca_historical_feed="sip",
        alpaca_adjustment="all",
        start_date="2016-01-01",
        end_date=None,
    )


def _session_document(symbol: str, day: str):
    return {
        "symbol": symbol,
        "interval": "1Day",
        "feed": "sip",
        "adjustment": "all",
        "timestamp": pd.Timestamp(day, tz="UTC").to_pydatetime(),
    }


class _FakeClient:
    def close(self):
        return None


class _FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        return self.collection


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


def test_incomplete_mongodb_history_stops_without_remote_backfill() -> None:
    config = _config()
    incomplete = _frame("2020-07-27", 500, 200.0)

    with pytest.raises(RuntimeError, match="Incomplete MongoDB market history"):
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
            "2026-01-02",
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


def test_alpaca_bootstrap_stops_at_resolved_session_close_without_safety_lookahead() -> None:
    config = _config()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return _frame("2024-01-09", 2, 100.0)

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
    assert pd.Timestamp(calls[-1]["end"]) == pd.Timestamp("2024-01-09T21:00:00Z")
    assert list(result.index.strftime("%Y-%m-%d")) == ["2024-01-09"]




def test_backtest_cutoff_falls_back_to_latest_common_cached_session() -> None:
    config = _session_config(["NVDA", "AAPL"])
    collection = _SessionCollection(
        [
            _session_document("NVDA", "2026-08-11"),
            _session_document("AAPL", "2026-08-11"),
        ]
    )
    with (
        patch("market_cycle_trader_api.engine.market_data.create_client", return_value=_FakeClient()),
        patch("market_cycle_trader_api.engine.market_data.get_database", return_value=_FakeDatabase(collection)),
    ):
        result = resolve_backtest_analysis_end_date(
            config,
            now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        )

    assert result == "2026-08-11"


def test_backtest_cutoff_keeps_latest_closed_session_when_cache_is_current() -> None:
    config = _session_config(["NVDA", "AAPL"])
    collection = _SessionCollection(
        [
            _session_document("NVDA", "2026-08-12"),
            _session_document("AAPL", "2026-08-12"),
        ]
    )
    with (
        patch("market_cycle_trader_api.engine.market_data.create_client", return_value=_FakeClient()),
        patch("market_cycle_trader_api.engine.market_data.get_database", return_value=_FakeDatabase(collection)),
    ):
        result = resolve_backtest_analysis_end_date(
            config,
            now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        )

    assert result == "2026-08-12"


def test_backtest_cutoff_does_not_let_a_new_missing_asset_force_the_cutoff_back() -> None:
    config = _session_config(["NVDA", "NEW"])
    collection = _SessionCollection([_session_document("NVDA", "2026-08-11")])
    with (
        patch("market_cycle_trader_api.engine.market_data.create_client", return_value=_FakeClient()),
        patch("market_cycle_trader_api.engine.market_data.get_database", return_value=_FakeDatabase(collection)),
    ):
        result = resolve_backtest_analysis_end_date(
            config,
            now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        )

    assert result == "2026-08-11"


def test_latest_completed_session_never_uses_current_open_session() -> None:
    
    result = latest_completed_xnys_session(datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc))
    assert result.date().isoformat() == "2026-08-11"


def test_tuning_database_only_missing_asset_never_calls_alpaca() -> None:
    config = _execution_config(mode="database_only")
    collection = _FakeCollection(first=None)
    with (
        patch("market_cycle_trader_api.engine.market_data.create_client", return_value=_FakeClient()),
        patch("market_cycle_trader_api.engine.market_data.get_database", return_value=_FakeDatabase(collection)),
        patch("market_cycle_trader_api.engine.market_data._download_alpaca_bars") as downloader,
    ):
        with pytest.raises(RuntimeError, match="Model tuning and parameter optimization are database-only"):
            load_mongo_market_bars("HD", config)
    downloader.assert_not_called()


def test_normal_backtest_bootstraps_only_a_completely_missing_asset_then_reads_mongodb() -> None:
    config = _execution_config(mode="backtest_bootstrap_missing")
    collection = _FakeCollection(first=None)
    downloaded = _frame("2016-01-04", 2092, 100.0)
    
    downloaded = downloaded.loc[downloaded.index < pd.Timestamp("2024-01-10", tz="UTC")]

    with (
        patch("market_cycle_trader_api.engine.market_data.create_client", return_value=_FakeClient()),
        patch("market_cycle_trader_api.engine.market_data.get_database", return_value=_FakeDatabase(collection)),
        patch("market_cycle_trader_api.engine.market_data._download_alpaca_bars", return_value=downloaded) as downloader,
        patch("market_cycle_trader_api.engine.market_data._upsert_frame") as upsert,
        patch("market_cycle_trader_api.engine.market_data._read_frame", return_value=downloaded),
    ):
        result = load_mongo_market_bars("HD", config)

    downloader.assert_called_once()
    upsert.assert_called_once()
    assert result.attrs["market_data_provenance"]["research_access_path"] == "alpaca_bootstrap_then_mongodb"
    assert result.attrs["market_data_provenance"]["cache_bootstrap_rows"] == len(downloaded)
