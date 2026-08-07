from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from market_cycle_trader_api.services.dashboard import dashboard_job_detail, dashboard_summary


class FakeCursor(list[dict[str, Any]]):
    def sort(self, key: str, direction: int) -> "FakeCursor":
        return FakeCursor(
            sorted(
                self,
                key=lambda item: item.get(key) or datetime.min.replace(tzinfo=timezone.utc),
                reverse=direction < 0,
            )
        )

    def limit(self, value: int) -> "FakeCursor":
        return FakeCursor(self[:value])


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @staticmethod
    def _matches(row: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            actual = row.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _project(row: dict[str, Any], projection: dict[str, int] | None) -> dict[str, Any]:
        if not projection:
            return dict(row)
        included = {key for key, enabled in projection.items() if enabled and key != "_id"}
        if included:
            return {key: row.get(key) for key in included if key in row}
        excluded = {key for key, enabled in projection.items() if not enabled}
        return {key: value for key, value in row.items() if key not in excluded}

    def find(self, query: dict[str, Any], projection: dict[str, int] | None = None) -> FakeCursor:
        return FakeCursor([
            self._project(row, projection)
            for row in self.rows
            if self._matches(row, query)
        ])

    def find_one(self, query: dict[str, Any], projection: dict[str, int] | None = None) -> dict[str, Any] | None:
        cursor = self.find(query, projection)
        return cursor[0] if cursor else None

    def count_documents(self, query: dict[str, Any]) -> int:
        return sum(1 for row in self.rows if self._matches(row, query))


class FakeDatabase(dict[str, FakeCollection]):
    def __getitem__(self, key: str) -> FakeCollection:
        return super().__getitem__(key)


def _database() -> FakeDatabase:
    now = datetime(2026, 8, 2, 20, 0, tzinfo=timezone.utc)
    job_id = "20260802T200000-abcdef12"
    metrics = {
        "portfolio_rotation": True,
        "initial_capital": 10000.0,
        "strategy_ending_capital": 15750.0,
        "strategy_return": 0.575,
        "buy_hold_ending_capital": 14300.0,
        "buy_hold_return": 0.43,
        "strategy_cagr": 0.091,
        "buy_hold_cagr": 0.075,
        "strategy_sharpe": 1.24,
        "buy_hold_sharpe": 0.93,
        "strategy_maximum_drawdown": -0.18,
        "buy_hold_maximum_drawdown": -0.22,
        "market_exposure": 0.61,
        "session_win_rate": 0.57,
        "capital_rotations": 42,
        "average_holding_days": 8.5,
        # These internal values must never leave the new dashboard contract.
        "assets": ["PRIVATE"],
        "random_seed": 42,
        "model_family": "private-model",
        "strategy_configuration_sha256": "secret-hash",
        "decision_horizon_days": 40,
        "backend": "internal-backend",
    }
    return FakeDatabase({
        "backtest_jobs": FakeCollection([
            {
                "id": job_id,
                "status": "completed",
                "stage": "Completed",
                "progress": 100.0,
                "created_at": now,
                "updated_at": now + timedelta(minutes=45),
                "started_at": now + timedelta(minutes=1),
                "finished_at": now + timedelta(minutes=45),
                "request": {"private": True},
                "strategy_profile_name": "Drawdown Reduction Test A2 - Cash 0.005",
            },
            {
                "id": "failed-job",
                "status": "failed",
                "stage": "Backtest failed",
                "progress": 22,
                "created_at": now - timedelta(days=1),
                "updated_at": now - timedelta(days=1) + timedelta(minutes=4),
            },
        ]),
        "backtest_comparisons": FakeCollection([
            {"job_id": job_id, "results": [metrics], "effective_config": {"private": True}},
        ]),
        "backtest_runs": FakeCollection([
            {"job_id": job_id, "symbol": "PORTFOLIO", "backend": "internal-backend"},
        ]),
        "strategy_control": FakeCollection([
            {"_id": "default", "research_strategy_id": "drawdown-test-a2"},
        ]),
        "strategy_profiles": FakeCollection([
            {"_id": "drawdown-test-a2", "name": "Drawdown Reduction Test A2 - Cash 0.005"},
        ]),
        "backtest_predictions": FakeCollection([
            {
                "job_id": job_id,
                "symbol": "PORTFOLIO",
                "backend": "internal-backend",
                "timestamp": now,
                "strategy_equity": 10000.0,
                "buy_hold_equity": 10000.0,
                "private_signal": 99,
            },
            {
                "job_id": job_id,
                "symbol": "PORTFOLIO",
                "backend": "internal-backend",
                "timestamp": now + timedelta(days=1),
                "strategy_equity": 10100.0,
                "buy_hold_equity": 10050.0,
                "private_signal": 100,
            },
        ]),
    })


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_dashboard_summary_is_additive_and_strategy_neutral() -> None:
    payload = dashboard_summary(_database(), limit=10)

    assert payload["total_backtests"] == 2
    assert payload["completed_backtests"] == 1
    assert payload["failed_backtests"] == 1
    assert payload["best_performance"]["metrics"]["simulation_return"] == 0.575
    assert payload["average_sharpe"] == 1.24
    assert payload["profitable_backtest_rate"] == 1.0
    assert payload["selected_backtest_strategy_name"] == "Drawdown Reduction Test A2 - Cash 0.005"
    assert payload["recent_backtests"][0]["strategy_profile_name"] == "Drawdown Reduction Test A2 - Cash 0.005"

    forbidden = {
        "assets",
        "random_seed",
        "model_family",
        "backend",
        "strategy_configuration_sha256",
        "decision_horizon_days",
        "effective_config",
        "request",
    }
    assert not (_all_keys(payload) & forbidden)


def test_dashboard_job_detail_contains_only_public_metrics_and_series() -> None:
    payload = dashboard_job_detail(_database(), "20260802T200000-abcdef12")

    assert payload["metrics"]["ending_capital"] == 15750.0
    assert payload["metrics"]["position_changes"] == 42.0
    assert payload["strategy_profile_name"] == "Drawdown Reduction Test A2 - Cash 0.005"
    assert payload["series"] == [
        {
            "timestamp": "2026-08-02T20:00:00+00:00",
            "simulation_equity": 10000.0,
            "reference_equity": 10000.0,
        },
        {
            "timestamp": "2026-08-03T20:00:00+00:00",
            "simulation_equity": 10100.0,
            "reference_equity": 10050.0,
        },
    ]
    assert "private_signal" not in _all_keys(payload)
    assert "internal-backend" not in str(payload)
