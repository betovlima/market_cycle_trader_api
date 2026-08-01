from __future__ import annotations

import pandas as pd

from market_cycle_trader_api.services.diagnostics.common import (
    _empty_market_series,
    _future_market_prices,
)


def test_empty_market_series_uses_datetime_index() -> None:
    series = _empty_market_series()

    assert series.empty
    assert isinstance(series.index, pd.DatetimeIndex)
    assert str(series.index.tz) == "UTC"


def test_empty_default_series_does_not_compare_range_index_to_timestamp() -> None:
    future = _future_market_prices(
        pd.Series(dtype=float),
        pd.Timestamp("2026-08-01", tz="UTC"),
    )

    assert future.empty
    assert isinstance(future.index, pd.DatetimeIndex)


def test_future_prices_are_filtered_after_utc_normalization() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.DatetimeIndex(
            ["2026-07-30", "2026-07-31", "2026-08-01"]
        ),
        dtype=float,
    )

    future = _future_market_prices(prices, "2026-07-31")

    assert future.tolist() == [102.0]
    assert isinstance(future.index, pd.DatetimeIndex)
    assert str(future.index.tz) == "UTC"


def test_non_datetime_index_is_ignored_safely() -> None:
    future = _future_market_prices(
        pd.Series([100.0, 101.0]),
        pd.Timestamp("2026-08-01", tz="UTC"),
    )

    assert future.empty
    assert isinstance(future.index, pd.DatetimeIndex)


class _EmptyCursor(list):
    def sort(self, *args, **kwargs):
        return self


class _EmptyCollection:
    def find(self, *args, **kwargs):
        return _EmptyCursor()


class _EmptyDatabase:
    def __getitem__(self, name: str):
        return _EmptyCollection()


def test_performance_diagnostics_skip_exit_without_future_bars() -> None:
    from market_cycle_trader_api.services.diagnostics.performance import (
        build_performance_diagnostics,
    )

    prediction_rows = [
        {
            "timestamp": "2026-07-30T00:00:00Z",
            "strategy_equity": 10_000.0,
            "buy_hold_equity": 10_000.0,
        },
        {
            "timestamp": "2026-07-31T00:00:00Z",
            "strategy_equity": 10_100.0,
            "buy_hold_equity": 10_050.0,
        },
    ]
    trade_rows = [
        {
            "timestamp": "2026-07-31T00:00:00Z",
            "sequence": 1,
            "action": "SELL",
            "asset": "NVDA",
            "execution_price": 175.0,
        }
    ]

    diagnostics = build_performance_diagnostics(
        _EmptyDatabase(),
        prediction_rows,
        trade_rows,
        {"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"},
    )

    assert diagnostics["exit_diagnostics"] == []
