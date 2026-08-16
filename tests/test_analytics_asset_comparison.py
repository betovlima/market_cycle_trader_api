from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi import HTTPException

from market_cycle_trader_api.services.analytics import asset_strategy_comparison


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @staticmethod
    def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            actual = row.get(key)
            if isinstance(expected, dict):
                minimum = expected.get("$gte")
                maximum = expected.get("$lte")
                if minimum is not None and (actual is None or actual < minimum):
                    return False
                if maximum is not None and (actual is None or actual > maximum):
                    return False
                continue
            if actual != expected:
                return False
        return True

    @staticmethod
    def _project(row: dict[str, Any], projection: dict[str, int] | None) -> dict[str, Any]:
        if not projection:
            return dict(row)
        included = {key for key, enabled in projection.items() if enabled and key != "_id"}
        return {key: row.get(key) for key in included if key in row}

    def find(self, query: dict[str, Any], projection: dict[str, int] | None = None):
        return [self._project(row, projection) for row in self.rows if self._matches(row, query)]

    def find_one(self, query: dict[str, Any], projection: dict[str, int] | None = None):
        rows = self.find(query, projection)
        return rows[0] if rows else None


class FakeDatabase(dict[str, FakeCollection]):
    def __getitem__(self, key: str) -> FakeCollection:
        return super().__getitem__(key)


def _database() -> FakeDatabase:
    start = datetime(2025, 1, 2, tzinfo=timezone.utc)
    job_id = "job-asset-chart"
    backend = "lightgbm_utility"
    predictions = []
    market = []
    for index in range(5):
        timestamp = start + timedelta(days=index)
        predictions.append({
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": backend,
            "timestamp": timestamp,
            "strategy_equity": 100000 + index * 5000,
            "buy_hold_equity": 100000 + index * 2500,
            "selected_asset": "AAPL" if index in {1, 2, 3} else "CASH",
            "portfolio_weights": {"AAPL": 1.0 if index in {1, 2, 3} else 0.0, "MSFT": 0.0},
        })
        market.append({
            "symbol": "AAPL",
            "interval": "1Day",
            "feed": "sip",
            "adjustment": "all",
            "timestamp": timestamp,
            "close": 200 + index * 10,
        })
    trades = [
        {
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": backend,
            "timestamp": start + timedelta(days=1),
            "sequence": 1,
            "action": "BUY",
            "asset": "AAPL",
            "execution_price": 211,
            "quantity": 10,
            "total_fee": 1,
        },
        {
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": backend,
            "timestamp": start + timedelta(days=3),
            "sequence": 2,
            "action": "SELL",
            "asset": "AAPL",
            "execution_price": 229,
            "quantity": 10,
            "holding_bars": 2,
            "position_return": 0.085,
            "realized_pnl": 180,
            "total_fee": 1,
        },
    ]
    return FakeDatabase({
        "backtest_jobs": FakeCollection([{
            "id": job_id,
            "status": "completed",
            "created_at": start,
            "finished_at": start + timedelta(days=5),
        }]),
        "backtest_comparisons": FakeCollection([{
            "job_id": job_id,
            "effective_config": {
                "assets": ["AAPL", "MSFT"],
                "timeframe": "1Day",
                "alpaca_historical_feed": "sip",
                "alpaca_adjustment": "all",
            },
            "results": [{"portfolio_rotation": True, "backend": backend}],
        }]),
        "backtest_runs": FakeCollection([{"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend}]),
        "backtest_predictions": FakeCollection(predictions),
        "backtest_trades": FakeCollection(trades),
        "alpaca_market_bars": FakeCollection(market),
        "market_bars": FakeCollection([]),
    })


def test_asset_strategy_comparison_aligns_strategy_market_and_decisions() -> None:
    payload = asset_strategy_comparison(_database(), "job-asset-chart", "aapl")

    assert payload["asset"] == "AAPL"
    assert payload["available_assets"] == ["AAPL", "MSFT"]
    assert len(payload["series"]) == 5
    assert payload["series"][0]["strategy_index"] == pytest.approx(100.0)
    assert payload["series"][0]["asset_index"] == pytest.approx(100.0)
    assert payload["series"][2]["strategy_weight"] == pytest.approx(1.0)
    assert payload["summary"]["strategy_return"] == pytest.approx(0.20)
    assert payload["summary"]["asset_return"] == pytest.approx(0.20)
    assert payload["summary"]["closed_positions"] == 1
    assert payload["summary"]["profitable_positions"] == 1
    assert payload["summary"]["win_rate"] == pytest.approx(1.0)
    assert [event["outcome"] for event in payload["events"]] == ["entry", "positive"]


def test_asset_strategy_comparison_rejects_asset_outside_backtest_universe() -> None:
    with pytest.raises(HTTPException) as error:
        asset_strategy_comparison(_database(), "job-asset-chart", "NVDA")
    assert error.value.status_code == 404
