from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from market_cycle_trader_api.services.admin_rotations import admin_job_rotations


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
        excluded = {key for key, enabled in projection.items() if not enabled}
        return {key: value for key, value in row.items() if key not in excluded}

    def find(self, query: dict[str, Any], projection: dict[str, int] | None = None):
        return [
            self._project(row, projection)
            for row in self.rows
            if self._matches(row, query)
        ]

    def find_one(self, query: dict[str, Any], projection: dict[str, int] | None = None):
        rows = self.find(query, projection)
        return rows[0] if rows else None


class FakeDatabase(dict[str, FakeCollection]):
    def __getitem__(self, key: str) -> FakeCollection:
        return super().__getitem__(key)


def _database() -> FakeDatabase:
    at = datetime(2026, 8, 3, 15, 30, tzinfo=timezone.utc)
    job_id = "job-1"
    backend = "private-seed-backend"
    return FakeDatabase({
        "backtest_jobs": FakeCollection([
            {"id": job_id, "status": "completed"},
        ]),
        "backtest_comparisons": FakeCollection([
            {
                "job_id": job_id,
                "results": [
                    {
                        "portfolio_rotation": True,
                        "backend": backend,
                        "strategy_return": 0.20,
                    }
                ],
            }
        ]),
        "backtest_runs": FakeCollection([
            {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend},
        ]),
        "backtest_trades": FakeCollection([
            {
                "job_id": job_id,
                "symbol": "PORTFOLIO",
                "backend": backend,
                "timestamp": at,
                "sequence": 1,
                "action": "SELL",
                "asset": "AAPL",
                "rotation_id": "rotation-1",
                "rotation_from_asset": "AAPL",
                "rotation_to_asset": "NVDA",
                "holding_bars": 8,
                "position_return": 0.12,
                "realized_pnl": 120.0,
                "total_fee": 1.25,
                "execution_price": 210.5,
                "reason": "ROTATE",
                "q_delta_final_vs_current": 99,
                "random_seed": 42,
            },
            {
                "job_id": job_id,
                "symbol": "PORTFOLIO",
                "backend": backend,
                "timestamp": at,
                "sequence": 2,
                "action": "BUY",
                "asset": "NVDA",
                "rotation_id": "rotation-1",
                "rotation_from_asset": "AAPL",
                "rotation_to_asset": "NVDA",
                "holding_bars": 0,
                "position_return": 0,
                "realized_pnl": 0,
                "total_fee": 1.75,
                "execution_price": 121.25,
                "reason": "ROTATE",
                "subsequent_holding_days": 12,
                "subsequent_position_return": 0.15,
                "chosen_market_return": 0.14,
                "counterfactual_previous_asset_return": 0.05,
                "rotation_value_added": 0.09,
                "rotation_regret": 0.0,
                "best_alternative_asset": "MSFT",
                "best_alternative_return": 0.18,
                "opportunity_cost": 0.04,
                "maximum_favorable_excursion": 0.22,
                "maximum_adverse_excursion": -0.06,
                "profit_capture_ratio": 0.68,
                "q_delta_final_vs_current": 100,
                "random_seed": 43,
            },
        ]),
    })


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_admin_rotation_payload_is_useful_and_strategy_neutral() -> None:
    payload = admin_job_rotations(_database(), "job-1")

    assert payload["summary"]["total_rotations"] == 1
    assert payload["summary"]["asset_to_asset_rotations"] == 1
    assert payload["summary"]["market_to_cash_moves"] == 0
    assert payload["summary"]["cash_to_market_moves"] == 0
    assert payload["summary"]["profitable_rotations"] == 1
    assert payload["summary"]["total_realized_pnl"] == 120.0
    assert payload["summary"]["total_transaction_fees"] == 3.0
    assert payload["rotations"] == [
        {
            "executed_at": "2026-08-03T15:30:00+00:00",
            "from_asset": "AAPL",
            "to_asset": "NVDA",
            "holding_days": 8.0,
            "position_return": 0.12,
            "realized_pnl": 120.0,
            "transaction_fees": 3.0,
            "sell_execution_price": 210.5,
            "buy_execution_price": 121.25,
            "sell_reason": "ROTATE",
            "buy_reason": "ROTATE",
            "subsequent_holding_days": 12.0,
            "subsequent_position_return": 0.15,
            "chosen_market_return": 0.14,
            "counterfactual_previous_asset_return": 0.05,
            "rotation_value_added": 0.09,
            "rotation_regret": 0.0,
            "best_alternative_asset": "MSFT",
            "best_alternative_return": 0.18,
            "opportunity_cost": 0.04,
            "maximum_favorable_excursion": 0.22,
            "maximum_adverse_excursion": -0.06,
            "profit_capture_ratio": 0.68,
        }
    ]
    assert payload["summary"]["diagnosed_rotations"] == 1
    assert payload["summary"]["positive_value_added_rate"] == 1.0
    assert payload["summary"]["average_opportunity_cost"] == 0.04

    forbidden = {
        "backend",
        "random_seed",
        "q_delta_final_vs_current",
        "q_current_position",
        "q_final_action",
        "strategy_configuration_sha256",
        "effective_config",
    }
    assert not (_all_keys(payload) & forbidden)


def test_admin_rotation_payload_includes_cash_transitions_and_reconstructs_legacy_ids() -> None:
    db = _database()
    at_exit = datetime(2026, 8, 4, 15, 30, tzinfo=timezone.utc)
    at_entry = datetime(2026, 8, 5, 15, 30, tzinfo=timezone.utc)
    backend = "private-seed-backend"
    db["backtest_trades"].rows.extend([
        {
            "job_id": "job-1", "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": at_exit, "sequence": 3, "action": "SELL", "asset": "NVDA",
            "reason": "MOVE_TO_CASH", "holding_bars": 4, "position_return": -0.03,
            "realized_pnl": -30.0, "total_fee": 0.50,
        },
        {
            "job_id": "job-1", "symbol": "PORTFOLIO", "backend": backend,
            "timestamp": at_entry, "sequence": 4, "action": "BUY", "asset": "MSFT",
            "reason": "BEST_CAPITAL_UTILITY", "holding_bars": 0, "position_return": 0.0,
            "realized_pnl": 0.0, "total_fee": 0.40,
        },
    ])

    payload = admin_job_rotations(db, "job-1")
    assert payload["summary"]["total_rotations"] == 3
    assert payload["summary"]["asset_to_asset_rotations"] == 1
    assert payload["summary"]["market_to_cash_moves"] == 1
    assert payload["summary"]["cash_to_market_moves"] == 1
    assert [(row["from_asset"], row["to_asset"]) for row in payload["rotations"]] == [
        ("AAPL", "NVDA"),
        ("NVDA", "CASH"),
        ("CASH", "MSFT"),
    ]
    assert payload["rotations"][-1]["realized_pnl"] is None
