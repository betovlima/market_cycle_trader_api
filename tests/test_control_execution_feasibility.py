"""Offline Control execution-feasibility safeguards.

These tests do not download Alpaca data, submit orders or alter production.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import StartControlExecutionRequest
from market_cycle_trader_api.engine import control_execution_feasibility as execution
from market_cycle_trader_api.services import control_shadow_execution_jobs as service

SOURCE_VALIDATION = "control-validation-7821002400424ccc"
SHA = "6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


def _fees(side, shares, price, _config):
    return dict(
        commission_fee=0.0, sec_fee=0.0, taf_fee=0.0,
        cat_fee=0.0, total_fee=0.0,
    )


class ControlExecutionFeasibilityTests(TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-07-01", periods=42, tz="UTC")
        self.frames = {}
        for symbol in ("AAA", "BBB"):
            self.frames[symbol] = pd.DataFrame(
                {"open": 10.0, "high": 11.0, "low": 9.0,
                 "close": 10.0, "volume": 100.0},
                index=self.dates,
            )
        self.config = SimpleNamespace(
            initial_capital=1000.0,
            rotation_model_repetitions=1,
            rotation_downside_penalty=0.2,
            rotation_drawdown_penalty=0.3,
        )

    def _run(self, callback, frames=None):
        return execution.simulate_feasible_control(
            "lightgbm_utility", callback, frames or self.frames,
            ["AAA", "BBB"], self.dates[23:40], self.config,
            _fees, lambda p, side, cfg: p,
        )

    def test_state_aware_partial_fills_no_overlapping_positions_or_debt(self):
        calls = []
        def policy(date, held, days):
            calls.append((date, held, days))
            return (1 if date < self.dates[29] else 2), 0.1

        with patch.object(execution, "_equal_weight_benchmark",
                          side_effect=lambda _frames, _symbols, dates, *args:
                          pd.Series(1000.0, index=dates)):
            run = self._run(policy)
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(run.predictions), 16)
        self.assertTrue((run.predictions["cash"] >= -1e-8).all())
        self.assertTrue((run.trades["quantity"] > 0).all())
        self.assertTrue((run.trades["realized_volume_participation"] <= .10000000001).all())
        self.assertTrue((run.trades["quantity"] <= run.trades["maximum_fill_quantity"]).all())
        self.assertTrue(run.metrics["terminal_mark_to_market_only"])
        self.assertGreater(run.metrics["partial_sessions"], 0)
        self.assertGreater(run.metrics["cash_weight_mean"], 0)
        self.assertTrue(run.predictions["actual_position_used_in_policy"].all())
        # Actual constrained position remains AAA after the target changes to
        # BBB until old shares are fully sold. Never hold both simultaneously.
        changed = [held for date, held, _ in calls if date >= self.dates[29]]
        self.assertEqual(changed[0], 1)
        bbuys = run.trades.loc[
            (run.trades["asset"] == "BBB") & (run.trades["action"] == "BUY")
        ]
        if not bbuys.empty:
            first_buy = bbuys.iloc[0]["timestamp"]
            self.assertFalse(
                ((run.trades["asset"] == "AAA")
                 & (run.trades["action"] == "SELL")
                 & (run.trades["timestamp"] > first_buy)).any()
            )

    def test_zero_realized_volume_prevents_sell_even_when_forecast_positive(self):
        frame = self.frames["AAA"].copy()
        frame.loc[self.dates[29], "volume"] = 0.0
        forecast = execution._volume_capacity(frame, self.dates[29], execution.SCENARIO)
        self.assertGreater(forecast["expected_capacity_shares"], 0)
        self.assertEqual(forecast["ex_post_fill_capacity_shares"], 0)
        frames = {**self.frames, "AAA": frame}
        with patch.object(execution, "_equal_weight_benchmark",
                          side_effect=lambda _frames, _symbols, dates, *args:
                          pd.Series(1000.0, index=dates)):
            run = self._run(
                lambda date, held, days: (1 if date < self.dates[28] else 2, 0.1),
                frames,
            )
        blocked = run.predictions.loc[self.dates[29]]
        self.assertEqual(blocked["selected_asset"], "AAA")
        self.assertEqual(blocked["signal_asset"], "BBB")
        self.assertEqual(blocked["trade_reason"], "SELL_BLOCKED")
        self.assertGreater(run.metrics["zero_volume_block_sessions"], 0)
        self.assertEqual(run.metrics["stale_mark_sessions"], 0)

    def test_modeled_price_adverse_and_fill_above_actual_volume_forbidden(self):
        price, cost = execution._fill_price(
            10.0, "BUY", 10, 100, execution.SCENARIO,
        )
        self.assertGreater(price, 10.0)
        self.assertGreater(cost, 0.0)
        sell, _ = execution._fill_price(10.0, "SELL", 10, 100, execution.SCENARIO)
        self.assertLess(sell, 10.0)
        with self.assertRaisesRegex(ValueError, "above capped"):
            execution._fill_price(10.0, "BUY", 11, 100, execution.SCENARIO)

    def test_buy_affordability_includes_fee_and_impact(self):
        def expensive_fee(side, shares, price, config):
            return {"total_fee": 10.0 if shares else 0.0}
        qty, price, _, fees = execution._affordable_shares(
            105.0, 10, 10.0, 100.0, execution.SCENARIO,
            expensive_fee, self.config,
        )
        self.assertLessEqual(qty * price + fees["total_fee"], 105.000000001)
        self.assertLess(qty, 10)

    def test_endpoint_requires_verified_validation_hash_and_confirmation(self):
        model = StartControlExecutionRequest.model_validate({
            "confirm": "SIMULATE_CONTROL_EXECUTION_FEASIBILITY_NO_ORDERS",
            "source_validation_job_id": SOURCE_VALIDATION,
            "expected_snapshot_sha256": SHA,
        })
        self.assertEqual(model.source_validation_job_id, SOURCE_VALIDATION)
        for wrong in ("no", None):
            with self.assertRaises(ValidationError):
                StartControlExecutionRequest.model_validate({
                    "confirm": wrong,
                    "source_validation_job_id": SOURCE_VALIDATION,
                    "expected_snapshot_sha256": SHA,
                })
        with self.assertRaises(ValidationError):
            StartControlExecutionRequest.model_validate({
                "confirm": "SIMULATE_CONTROL_EXECUTION_FEASIBILITY_NO_ORDERS",
                "source_validation_job_id": SOURCE_VALIDATION,
                "expected_snapshot_sha256": SHA,
                "data_directory": "C:/user-selected-unsafe-path",
            })

    def test_cannot_queue_unverified_or_different_source(self):
        db = MagicMock()
        db[service.VALIDATION_COLLECTION].find_one.return_value = {
            "status": "completed",
            "snapshot_sha256": SHA,
            "source_job_id": "control-shadow-870fb66e1bdc4fd0",
            "result": {"source_snapshot_sha256": SHA,
                       "original_shadow": {"reproduced": False}},
        }
        with patch.object(service, "_require_enabled"):
            with self.assertRaises(service.ExecutionInvalid):
                service.start_execution_feasibility(
                    db, source_validation_job_id=SOURCE_VALIDATION,
                    expected_snapshot_sha256=SHA,
                )
        db[service.COLLECTION].insert_one.assert_not_called()

    def test_valid_job_never_downloads_or_places_orders(self):
        db = MagicMock()
        db[service.VALIDATION_COLLECTION].find_one.return_value = {
            "status": "completed",
            "snapshot_sha256": SHA,
            "source_job_id": "control-shadow-870fb66e1bdc4fd0",
            "result": {
                "source_unchanged": True,
                "source_snapshot_sha256": SHA,
                "order_submission": "never",
                "numeric_input_integrity": {"status": "verified", "checked_assets": 55},
                "original_shadow": {"reproduced": True},
                "oos": {"strategy_ending_capital": 5887904.3815},
            },
        }
        with (patch.object(service, "_require_enabled"),
              patch.object(service.threading, "Thread") as thread):
            queued = service.start_execution_feasibility(
                db, source_validation_job_id=SOURCE_VALIDATION,
                expected_snapshot_sha256=SHA,
            )
        self.assertEqual(queued["status"], "queued")
        self.assertFalse(queued["order_eligible"])
        self.assertEqual(queued["source_download"], "never")
        self.assertEqual(queued["order_submission"], "never")
        thread.return_value.start.assert_called_once()
        service._THREADS.pop(queued["job_id"], None)


if __name__ == "__main__":
    import unittest
    unittest.main()
