from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from market_cycle_trader_api.services.analytics import backtest_analytics


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @staticmethod
    def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(row.get(key) == value for key, value in query.items())

    @staticmethod
    def _project(row: dict[str, Any], projection: dict[str, int] | None) -> dict[str, Any]:
        if not projection:
            return dict(row)
        included = {key for key, enabled in projection.items() if enabled and key != "_id"}
        if included:
            return {key: row.get(key) for key in included if key in row}
        return {key: value for key, value in row.items() if projection.get(key, 1)}

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
    job_id = "job-analytics"
    backend = "protected-backend"
    predictions = []
    for index, value in enumerate((100000, 104000, 99000, 110000, 116000, 113000, 125000)):
        predictions.append({
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": backend,
            "timestamp": start + timedelta(days=index * 35),
            "strategy_equity": value,
            "buy_hold_equity": 100000 + index * 2500,
            "q_final_action": 999,
        })
    trades = [
        {
            "job_id": job_id, "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": start + timedelta(days=40), "sequence": 1,
            "action": "SELL", "asset": "AAPL", "rotation_id": "r1",
            "rotation_from_asset": "AAPL", "rotation_to_asset": "NVDA",
            "holding_bars": 8, "position_return": 0.08, "realized_pnl": 8000,
            "total_fee": 10, "random_seed": 7,
        },
        {
            "job_id": job_id, "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": start + timedelta(days=40), "sequence": 2,
            "action": "BUY", "asset": "NVDA", "rotation_id": "r1",
            "rotation_from_asset": "AAPL", "rotation_to_asset": "NVDA",
            "holding_bars": 0, "position_return": 0, "realized_pnl": 0,
            "total_fee": 8,
        },
        {
            "job_id": job_id, "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": start + timedelta(days=95), "sequence": 3,
            "action": "SELL", "asset": "NVDA", "rotation_id": "r2",
            "rotation_from_asset": "NVDA", "rotation_to_asset": "MSFT",
            "holding_bars": 18, "position_return": -0.03, "realized_pnl": -3000,
            "total_fee": 9,
        },
        {
            "job_id": job_id, "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": start + timedelta(days=95), "sequence": 4,
            "action": "BUY", "asset": "MSFT", "rotation_id": "r2",
            "rotation_from_asset": "NVDA", "rotation_to_asset": "MSFT",
            "holding_bars": 0, "position_return": 0, "realized_pnl": 0,
            "total_fee": 7,
        },
    ]
    return FakeDatabase({
        "backtest_jobs": FakeCollection([{"id": job_id, "status": "completed", "created_at": start, "finished_at": start + timedelta(days=220)}]),
        "backtest_comparisons": FakeCollection([{"job_id": job_id, "results": [{
            "portfolio_rotation": True, "backend": backend,
            "initial_capital": 100000, "strategy_ending_capital": 125000,
            "strategy_return": .25, "buy_hold_ending_capital": 115000,
            "buy_hold_return": .15, "strategy_cagr": .2, "buy_hold_cagr": .12,
            "strategy_sharpe": 1.1, "buy_hold_sharpe": .7,
            "strategy_maximum_drawdown": -.08, "buy_hold_maximum_drawdown": -.12,
            "capital_rotations": 2, "average_holding_days": 13,
        }]}]),
        "backtest_runs": FakeCollection([{"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend}]),
        "backtest_predictions": FakeCollection(predictions),
        "backtest_trades": FakeCollection(trades),
    })


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_backtest_analytics_is_useful_and_strategy_neutral() -> None:
    payload = backtest_analytics(_database(), "job-analytics")
    assert payload["rotation_summary"]["total_rotations"] == 2
    assert payload["asset_attribution"][0]["asset"] == "AAPL"
    assert len(payload["transition_matrix"]) == 2
    assert len(payload["holding_buckets"]) == 4
    assert payload["trade_dependency"]["total_realized_pnl"] == 5000.0
    assert payload["monthly_returns"]
    forbidden = {"backend", "random_seed", "q_final_action", "effective_config", "decision_score"}
    assert not (_keys(payload) & forbidden)
