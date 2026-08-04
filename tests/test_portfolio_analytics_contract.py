from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any

from market_cycle_trader_api.services.analytics import portfolio_analytics


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @staticmethod
    def _project(row: dict[str, Any], projection: dict[str, int] | None) -> dict[str, Any]:
        if not projection:
            return dict(row)
        included = {key for key, enabled in projection.items() if enabled and key != "_id"}
        if included:
            return {key: row.get(key) for key in included if key in row}
        return {key: value for key, value in row.items() if projection.get(key, 1)}

    def find(self, query: dict[str, Any], projection: dict[str, int] | None = None):
        del query
        return [self._project(row, projection) for row in self.rows]


class FakeDatabase(dict[str, FakeCollection]):
    def __getitem__(self, key: str) -> FakeCollection:
        return super().__getitem__(key)


def _database() -> FakeDatabase:
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    snapshots = [
        {
            "recorded_at": start + timedelta(days=index),
            "portfolio_value": 100000 + index * 1000,
            "strategy_cash": 25000,
            "market_value": 75000 + index * 1000,
            "total_pnl": index * 1000,
            "total_return": index / 100,
            "managed_symbol": "AAPL",
        }
        for index in range(35)
    ]
    orders = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "status": "filled",
            "quantity": 10,
            "filled_quantity": 10,
            "filled_average_price": 180,
            "submitted_at": start,
            "filled_at": start + timedelta(seconds=2),
            "created_at": start,
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "status": "rejected",
            "quantity": 10,
            "submitted_at": start + timedelta(days=10),
            "created_at": start + timedelta(days=10),
        },
    ]
    return FakeDatabase({
        "paper_portfolio_snapshots": FakeCollection(snapshots),
        "paper_trade_orders": FakeCollection(orders),
    })


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_portfolio_analytics_uses_live_snapshot_and_sanitized_history(monkeypatch) -> None:
    fake_module = types.ModuleType("market_cycle_trader_api.services.public_paper_portfolio")
    fake_module.public_paper_portfolio_snapshot = lambda db: {
        "status": "ready",
        "recorded_at": datetime(2026, 8, 4, tzinfo=timezone.utc),
        "initial_capital": 100000,
        "strategy_cash": 25000,
        "market_value": 109000,
        "portfolio_value": 134000,
        "realized_pnl": 20000,
        "unrealized_pnl": 14000,
        "total_pnl": 34000,
        "total_return": .34,
        "position": {"symbol": "AAPL", "quantity": 10},
        "history": [{"effective_config": "must-not-leak"}],
        "recent_orders": [{"client_order_id": "must-not-leak"}],
    }
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    payload = portfolio_analytics(_database())

    assert payload["connection"]["status"] == "ready"
    assert payload["summary"]["portfolio_value"] == 134000.0
    assert payload["summary"]["return_30_days"] is not None
    assert payload["order_analytics"]["total_orders"] == 2
    assert payload["order_analytics"]["filled_orders"] == 1
    assert payload["order_analytics"]["rejected_orders"] == 1
    forbidden = {"effective_config", "client_order_id", "plan_id", "random_seed", "backend"}
    assert not (_keys(payload) & forbidden)
