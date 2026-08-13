from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import market_cycle_trader_api.services.dashboard as dashboard_service
from market_cycle_trader_api.services.dashboard import (
    dashboard_strategy_intelligence,
    dashboard_tuning_candidate_detail,
)
from market_cycle_trader_api.services.model_tuning import _candidate_equity_preview


class FakeCursor(list[dict[str, Any]]):
    def sort(self, key: str, direction: int) -> "FakeCursor":
        return FakeCursor(sorted(self, key=lambda item: item.get(key) or datetime.min.replace(tzinfo=timezone.utc), reverse=direction < 0))

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
        return FakeCursor([self._project(row, projection) for row in self.rows if self._matches(row, query)])

    def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, int] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> dict[str, Any] | None:
        cursor = self.find(query, projection)
        if sort:
            for key, direction in reversed(sort):
                cursor = cursor.sort(key, direction)
        return cursor[0] if cursor else None


class FakeDatabase(dict[str, FakeCollection]):
    def __getitem__(self, key: str) -> FakeCollection:
        return super().__getitem__(key)


def test_protected_dashboard_exposes_strategy_forecast_and_risk_off_diagnostics(monkeypatch) -> None:
    now = datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc)
    job_id = "risk-off-job"
    db = FakeDatabase({
        "strategy_control": FakeCollection([{
            "_id": "default",
            "research_strategy_id": "research-risk-off",
            "trader_winner_strategy_id": "winner-risk-off",
        }]),
        "paper_trade_plans": FakeCollection([
            {
                "plan_id": "old-winner-plan",
                "status": "prepared",
                "winner_strategy_id": "former-winner",
                "winner_strategy_name": "Former Winner",
                "created_at": now + timedelta(minutes=2),
                "utilities": {"NVDA": 99.0},
                "cash_edges": {"NVDA": 99.0},
                "winner_assets": ["NVDA"],
            },
            {
                "plan_id": "current-winner-plan",
                "status": "prepared",
                "winner_strategy_id": "winner-risk-off",
                "winner_strategy_name": "Risk-Off Winner",
                "winner_strategy_revision": 7,
                "winner_model_family": "lightgbm_utility",
                "winner_assets": ["NVDA", "MSFT"],
                "decision_date": "2026-08-12",
                "expected_market_open": "2026-08-13T13:30:00+00:00",
                "execution_session": "2026-08-13",
                "current_asset": "NVDA",
                "target_asset": "CASH",
                "raw_best_asset": "MSFT",
                "action": "sell_to_cash",
                "selected_utility": 0.27,
                "utilities": {"CASH": 0.0, "NVDA": 0.22, "MSFT": 0.27},
                "cash_edges": {"NVDA": -0.015, "MSFT": -0.006},
                "opportunity_probability": 0.72,
                "opportunity_confidence": 0.84,
                "opportunity_threshold": 0.61,
                "opportunity_accepted": True,
                "effective_switch_margin": 0.02,
                "calibrated_candidate_margin": 0.02,
                "calibration_score": 1.4,
                "random_state": 42,
                "training_end": "2026-08-11",
                "calibration_start": "2026-01-01",
                "calibration_end": "2026-07-31",
                "final_fit_end": "2026-08-11",
                "created_at": now,
            },
        ]),
        "backtest_jobs": FakeCollection([{
            "id": job_id,
            "status": "completed",
            "strategy_profile_id": "research-risk-off",
            "strategy_profile_name": "Risk-Off Candidate",
        }]),
        "backtest_comparisons": FakeCollection([{
            "job_id": job_id,
            "results": [{
                "portfolio_rotation": True,
                "backend": "lightgbm_utility",
                "strategy_ending_capital": 150000.0,
            }],
        }]),
        "backtest_runs": FakeCollection([{
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": "lightgbm_utility",
        }]),
        "backtest_predictions": FakeCollection([{
            "job_id": job_id,
            "symbol": "PORTFOLIO",
            "backend": "lightgbm_utility",
            "timestamp": now,
            "strategy_equity": 100000.0,
            "buy_hold_equity": 100000.0,
            "current_asset": "NVDA",
            "best_asset": "MSFT",
            "best_score": 0.27,
            "current_cash_edge": -0.015,
            "best_cash_edge": -0.006,
            "cash_exit_threshold": 0.0,
            "cash_entry_threshold": 0.01,
            "final_action_asset": "CASH",
            "decision_reason": "CASH_THRESHOLD",
        }]),
        "model_tuning_runs": FakeCollection([]),
    })

    strategies = {
        "research-risk-off": {
            "id": "research-risk-off",
            "name": "Risk-Off Candidate",
            "status": "candidate",
            "revision": 3,
            "last_backtest_id": job_id,
            "configuration": {
                "strategy_mode": "COMPOUND_ROTATION_SWING_RISK_OFF",
                "rotation_cash_threshold": 0.0,
                "rotation_min_expected_edge": 0.01,
            },
            "research_model_configuration": {"family": "lightgbm_utility", "settings": {"num_leaves": 31}},
        },
        "winner-risk-off": {
            "id": "winner-risk-off",
            "name": "Risk-Off Winner",
            "status": "winner",
            "revision": 7,
            "configuration": {
                "strategy_mode": "COMPOUND_ROTATION_SWING_RISK_OFF",
                "rotation_cash_threshold": 0.0,
                "rotation_min_expected_edge": 0.01,
            },
            "research_model_configuration": {"family": "lightgbm_utility", "settings": {"num_leaves": 31}},
        },
    }
    monkeypatch.setattr(dashboard_service, "_strategy_profile_detail", lambda _db, strategy_id: strategies.get(strategy_id))

    payload = dashboard_strategy_intelligence(db, job_id=job_id)

    assert payload["research_strategy"]["configuration"]["strategy_mode"] == "COMPOUND_ROTATION_SWING_RISK_OFF"
    assert payload["research_strategy"]["research_model_configuration"]["settings"]["num_leaves"] == 31
    assert payload["forecast"]["plan_id"] == "current-winner-plan"
    assert payload["forecast"]["target_asset"] == "CASH"
    assert payload["forecast"]["asset_forecast"][0]["asset"] == "MSFT"
    assert payload["forecast"]["asset_forecast"][0]["ranking_utility"] == 0.27
    assert payload["forecast"]["asset_forecast"][0]["cash_edge"] == -0.006
    assert payload["forecast"]["cash_exit_threshold"] == 0.0
    assert payload["forecast"]["cash_entry_threshold"] == 0.01
    assert payload["forecast"]["opportunity_probability"] == 0.72
    assert payload["forecast"]["opportunity_confidence"] == 0.84
    assert payload["forecast"]["opportunity_threshold"] == 0.61
    assert payload["forecast"]["opportunity_accepted"] is True
    row = payload["decision_history"]["rows"][0]
    assert row["current_cash_edge"] == -0.015
    assert row["best_cash_edge"] == -0.006
    assert row["final_action_asset"] == "CASH"
    assert row["decision_reason"] == "CASH_THRESHOLD"


def test_candidate_detail_returns_retained_preview_and_hyperparameters() -> None:
    db = FakeDatabase({
        "model_tuning_runs": FakeCollection([{
            "id": "lhs-1",
            "candidates": [{
                "candidate_id": 4,
                "kind": "latin_hypercube",
                "status": "completed",
                "settings": {"num_leaves": 47, "learning_rate": 0.04},
                "metrics": {"ending_capital": 180000.0, "sharpe": 1.8},
                "equity_preview": [{
                    "timestamp": "2026-08-11T00:00:00+00:00",
                    "simulation_equity": 180000.0,
                    "reference_equity": 125000.0,
                    "selected_asset": "CASH",
                    "cash_edge": -0.01,
                }],
            }],
        }]),
    })

    payload = dashboard_tuning_candidate_detail(db, "lhs-1", 4)

    assert payload["settings"]["num_leaves"] == 47
    assert payload["metrics"]["ending_capital"] == 180000.0
    assert payload["equity_preview"][0]["selected_asset"] == "CASH"
    assert payload["equity_preview"][0]["cash_edge"] == -0.01


def test_candidate_equity_preview_survives_dtype_independent_dashboard_summary() -> None:
    now = datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc)
    db = FakeDatabase({
        "backtest_runs": FakeCollection([{"job_id": "candidate-job", "symbol": "PORTFOLIO", "backend": "lightgbm_utility"}]),
        "backtest_predictions": FakeCollection([
            {
                "job_id": "candidate-job",
                "symbol": "PORTFOLIO",
                "backend": "lightgbm_utility",
                "timestamp": now,
                "strategy_equity": 100000,
                "buy_hold_equity": 100000,
                "selected_asset": "CASH",
                "trade_action": "stay_in_cash",
                "final_action_cash_edge": -0.004,
            },
            {
                "job_id": "candidate-job",
                "symbol": "PORTFOLIO",
                "backend": "lightgbm_utility",
                "timestamp": now + timedelta(days=1),
                "strategy_equity": 100000,
                "buy_hold_equity": 101000,
                "selected_asset": "NVDA",
                "trade_action": "buy",
                "final_action_cash_edge": 0.021,
            },
        ]),
    })

    preview = _candidate_equity_preview(db, "candidate-job")

    assert len(preview) == 2
    assert preview[0]["selected_asset"] == "CASH"
    assert preview[0]["cash_edge"] == -0.004
    assert preview[1]["trade_action"] == "buy"
    assert preview[1]["simulation_equity"] == 100000.0


def test_strategy_intelligence_routes_require_trader_or_admin_session() -> None:
    source = Path("src/market_cycle_trader_api/api/routers/dashboard.py").read_text(encoding="utf-8")
    assert '@router.get("/strategy-intelligence")' in source
    assert '@router.get("/strategy-intelligence/tuning/{run_id}/candidates/{candidate_id}")' in source
    assert source.count("Depends(require_portfolio_session)") >= 2


def test_strategy_intelligence_service_has_no_alpaca_or_network_dependency() -> None:
    source = Path("src/market_cycle_trader_api/services/dashboard.py").read_text(encoding="utf-8").lower()
    import_lines = "\n".join(line for line in source.splitlines() if line.startswith("from ") or line.startswith("import "))
    assert "alpaca" not in import_lines
    assert "requests" not in import_lines
    assert "httpx" not in import_lines
