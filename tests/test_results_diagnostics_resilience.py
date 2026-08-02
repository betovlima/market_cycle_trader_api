from __future__ import annotations

from unittest.mock import patch

from market_cycle_trader_api.services import results


class _Cursor(list):
    def sort(self, *args, **kwargs):
        return self


class _Collection:
    def __init__(self, rows):
        self.rows = rows

    def find(self, *args, **kwargs):
        return _Cursor(self.rows)


class _Database:
    def __init__(self):
        self.collections = {
            "backtest_predictions": _Collection([
                {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "strategy_equity": 10000.0,
                    "buy_hold_equity": 10000.0,
                }
            ]),
            "backtest_trades": _Collection([]),
        }

    def __getitem__(self, name):
        return self.collections[name]


def test_optional_diagnostics_do_not_hide_completed_run(monkeypatch) -> None:
    monkeypatch.setattr(results, "database", lambda: _Database())
    run = {
        "job_id": "job",
        "symbol": "PORTFOLIO",
        "backend": "xgboost_utility",
        "metrics": {},
        "summary": "ok",
    }
    with patch.object(
        results,
        "build_performance_diagnostics",
        side_effect=TypeError("diagnostic failure"),
    ):
        payload = results.build_run_payload(run)

    assert payload["summary"] == "ok"
    assert payload["diagnostics"]["status"] == "unavailable"
    assert payload["diagnostics"]["error_type"] == "TypeError"
