"""Fixed Control execution sensitivity: scientific-source and no-order guards."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import ValidationError

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import StartControlSensitivityRequest
from market_cycle_trader_api.engine import control_execution_feasibility as feasibility
from market_cycle_trader_api.engine import control_execution_adapter as adapter
from market_cycle_trader_api.engine import control_execution_sensitivity as research
from market_cycle_trader_api.services import control_shadow_sensitivity_jobs as jobs
from market_cycle_trader_api.tcc_v106_reference import research_challengers as scientific

VAL = "control-validation-7821002400424ccc"
EXEC = "control-execution-c38169f6c6fc4ea0"
SNAP = "control-shadow-870fb66e1bdc4fd0"
SHA = "6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


def fees(side, qty, price, config):
    return {"total_fee": 0.0}


def fake_scientific_runner(bars, config, fees, slippage, *,
                           progress_callback, trade_callback,
                           progress_detail_callback, technical_log_callback):
    first = next(iter(bars.values()))
    return [_simulate_exact(
        "lightgbm_utility",
        lambda timestamp, position, holding: (1, .1),
        bars, sorted(bars), first.index[23:40], config, fees, slippage,
    )]


class ControlSensitivityTests(TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-07-01", periods=42, tz="UTC")
        self.frame = pd.DataFrame({
            "open": 10.0, "high": 11.0, "low": 9.0,
            "close": 10.0, "volume": 100.0,
        }, index=self.dates)
        self.config = SimpleNamespace(
            initial_capital=10000.0,
            rotation_model_repetitions=1,
            rotation_downside_penalty=.2,
            rotation_drawdown_penalty=.3,
        )

    def test_exact_five_fixed_scenarios_no_oos_parameter_search(self):
        self.assertEqual(
            [name for name, _ in adapter.SENSITIVITY_SCENARIOS],
            ["cap10_cost", "no_cap_no_cost", "cap10_no_cost",
             "cap05_cost", "cap01_cost"],
        )
        self.assertEqual(
            len({name for name, _ in adapter.SENSITIVITY_SCENARIOS}), 5
        )
        self.assertEqual(feasibility.SCENARIO["participation_rate"], .10)
        self.assertFalse(feasibility.SCENARIO["unlimited_capacity"])

    def test_sensitivity_returns_same_default_v1042_10percent_accounting(self):
        with patch.object(feasibility, "_equal_weight_benchmark",
                          side_effect=lambda frames, symbols, dates, *args:
                          pd.Series(10000., index=dates)):
            original = feasibility.simulate_feasible_control(
                "lightgbm_utility", lambda d, p, h: (1, .1),
                {"AAA": self.frame}, ["AAA"], self.dates[23:40],
                self.config, fees, lambda p, side, c: p,
            )
            duplicate = feasibility.simulate_feasible_control(
                "lightgbm_utility", lambda d, p, h: (1, .1),
                {"AAA": self.frame}, ["AAA"], self.dates[23:40],
                self.config, fees, lambda p, side, c: p,
                scenario_override=dict(adapter.SENSITIVITY_SCENARIOS[0][1]),
            )
        self.assertEqual(original.metrics["strategy_ending_capital"],
                         duplicate.metrics["strategy_ending_capital"])
        self.assertEqual(original.metrics["partial_sessions"],
                         duplicate.metrics["partial_sessions"])
        self.assertEqual(original.trades["quantity"].tolist(),
                         duplicate.trades["quantity"].tolist())

    def test_idealized_no_cap_explicitly_can_exceed_volume_and_zero_volume(self):
        frame = self.frame.copy()
        frame.loc[self.dates[27], "volume"] = 0.0
        scenario = dict(adapter.SENSITIVITY_SCENARIOS[1][1])
        with patch.object(feasibility, "_equal_weight_benchmark",
                          side_effect=lambda frames, symbols, dates, *args:
                          pd.Series(10000., index=dates)):
            result = feasibility.simulate_feasible_control(
                "lightgbm_utility", lambda d, p, h: (1, .1),
                {"AAA": frame}, ["AAA"], self.dates[23:40],
                self.config, fees, lambda p, side, c: p,
                scenario_override=scenario,
            )
        self.assertTrue(result.metrics["unlimited_capacity_is_idealized_not_executable"])
        self.assertTrue((result.trades["quantity"] >
                         result.trades["realized_daily_volume"]).any())
        self.assertTrue((result.predictions["cash"] >= 0).all())

    def test_capacity_limits_and_no_price_cost_scenario(self):
        scenario = dict(adapter.SENSITIVITY_SCENARIOS[2][1])
        price, bps = feasibility._fill_price(10.0, "BUY", 10, 100.0, scenario)
        self.assertEqual(price, 10.0)
        self.assertEqual(bps, 0.0)
        with self.assertRaisesRegex(ValueError, "above capped"):
            feasibility._fill_price(10.0, "BUY", 11, 100.0, scenario)
        cap_5 = feasibility._volume_capacity(
            self.frame, self.dates[30], dict(adapter.SENSITIVITY_SCENARIOS[3][1],
                                              prior_volume_lookback=20,
                                              minimum_prior_volume_observations=5),
        )
        cap_1 = feasibility._volume_capacity(
            self.frame, self.dates[30], dict(adapter.SENSITIVITY_SCENARIOS[4][1],
                                              prior_volume_lookback=20,
                                              minimum_prior_volume_observations=5),
        )
        self.assertEqual(cap_5["ex_post_fill_capacity_shares"], 5)
        self.assertEqual(cap_1["ex_post_fill_capacity_shares"], 1)

    def test_only_isolated_runner_changes_simulator_and_trains_once(self):
        original = scientific._simulate_exact
        cloned_fake = FunctionType(
            fake_scientific_runner.__code__,
            {"_simulate_exact": original},
        )
        seen = []
        def fake_simulator(*args, **kwargs):
            seen.append(dict(kwargs["scenario_override"]))
            return SimpleNamespace(
                backend="lightgbm_utility", predictions=pd.DataFrame(),
                trades=pd.DataFrame(), metrics={},
            )
        with (
            patch.object(scientific, "_run_lightgbm", cloned_fake),
            patch.object(scientific, "allocation_execution_enabled", return_value=False),
            patch.object(adapter, "simulate_feasible_control", side_effect=fake_simulator),
        ):
            cases, primary = adapter.run_sensitivity_lightgbm(
                {"AAA": self.frame}, self.config, fees, lambda p, side, c: p,
            )
        self.assertEqual(tuple(cases),
                         tuple(name for name, _ in adapter.SENSITIVITY_SCENARIOS))
        self.assertIs(primary, cases["cap10_cost"])
        self.assertEqual(len(seen), 5)
        self.assertIs(scientific._simulate_exact, original)
        self.assertIs(cloned_fake.__globals__["_simulate_exact"], original)

    def test_admin_request_is_fixed_and_rejects_user_chosen_parameters(self):
        good = {
            "confirm": "COMPARE_FIXED_CONTROL_EXECUTION_SCENARIOS_NO_ORDERS",
            "source_validation_job_id": VAL,
            "source_execution_job_id": EXEC,
            "expected_snapshot_sha256": SHA,
        }
        self.assertEqual(
            StartControlSensitivityRequest.model_validate(good).source_execution_job_id,
            EXEC,
        )
        with self.assertRaises(ValidationError):
            StartControlSensitivityRequest.model_validate({
                **good, "participation_rate": .8,
            })
        with self.assertRaises(ValidationError):
            StartControlSensitivityRequest.model_validate({
                **good, "confirm": "TRADE_NOW",
            })

    def test_job_does_not_queue_mismatched_research_or_download(self):
        db = MagicMock()
        db[jobs.VALIDATION_COLLECTION].find_one.return_value = {
            "status": "completed", "source_job_id": SNAP,
            "snapshot_sha256": SHA,
            "result": {
                "original_shadow": {"reproduced": True},
                "source_snapshot_sha256": SHA,
                "numeric_input_integrity": {"status": "verified"},
                "order_submission": "never",
            },
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value = {
            "status": "completed", "source_job_id": "control-shadow-aaaaaaaaaaaaaaaa",
            "source_validation_job_id": VAL, "snapshot_sha256": SHA,
            "result": {"research_kind": "control_execution_feasibility_scenario"},
        }
        with patch.object(jobs, "_require_enabled"):
            with self.assertRaises(jobs.SensitivityInvalid):
                jobs.start_control_sensitivity(
                    db, source_validation_job_id=VAL,
                    source_execution_job_id=EXEC,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()

    def test_report_replays_exact_reference_and_keeps_snapshot_immutable(self):
        original_folds = []
        for fold, date in enumerate(self.dates[23:26], 1):
            original_folds.append({
                "fold_id": fold, "test_start": date.isoformat(),
                "test_end": date.isoformat(), "sessions": 1,
            })
        baseline41 = {
            "source_snapshot_sha256": SHA, "source_job_id": SNAP,
            "source_unchanged": True, "snapshot_assets": 55,
            "calendar_sessions": 2500, "order_submission": "never",
            "original_shadow": {"reproduced": True},
            "numeric_input_integrity": {"status": "verified", "checked_assets": 55},
            "oos": {
                "strategy_ending_capital": 2000.,
                "walk_forward_fold_count": 3,
                "walk_forward_folds": original_folds,
            },
        }
        expected = {key: 0.0 for key in research.METRICS}
        expected.update({
            "strategy_ending_capital": 1001.,
            "session_count": 3,
            "terminal_holdings_asset": "AAA",
        })
        baseline42 = {
            "source_snapshot_sha256": SHA, "source_job_id": SNAP,
            "source_validation_job_id": VAL, "source_unchanged": True,
            "order_submission": "never",
            "numeric_input_integrity": {"status": "verified", "assets": 55},
            "source_verified_control": {"strategy_ending_capital": 2000.},
            "execution_scenario": {
                "participation_rate": .1,
                "assumed_full_spread_bps": 15.,
                "assumed_impact_coefficient_bps": 20.,
            },
            "feasible_oos": {
                **expected, "walk_forward_fold_count": 3,
            },
        }
        prediction = pd.DataFrame({
            "strategy_equity": [1000., 1000., 1001.],
            "buy_hold_equity": [1000., 1000., 1000.],
            "walk_forward_fold": [1, 2, 3],
            "cash_weight": [0., 0., 0.],
        }, index=self.dates[23:26])
        cases = {}
        for name, override in adapter.SENSITIVITY_SCENARIOS:
            cases[name] = SimpleNamespace(
                predictions=prediction.copy(), trades=pd.DataFrame(),
                metrics={
                    **expected,
                    "execution_scenario": {**feasibility.SCENARIO, **override},
                },
            )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / SNAP
            source.mkdir()
            manifest = source / "manifest.json"
            manifest.write_text("source must never be overwritten", encoding="utf-8")
            with (
                patch.object(research, "read_verified_control_snapshot",
                             return_value=(
                                 {f"S{i:02d}": pd.DataFrame() for i in range(55)},
                                 {"completed_session": "2026-09-28"}, source,
                             )),
                patch.object(research, "prepare_operational_control_panel",
                             return_value=(
                                 {}, self.dates,
                                 SimpleNamespace(calendar_sessions=2500),
                             )),
                patch.object(research, "run_sensitivity_lightgbm",
                             return_value=(cases, cases["cap10_cost"])),
            ):
                report = research.run_control_execution_sensitivity(
                    source_job_id=SNAP, validation_job_id=VAL,
                    feasibility_job_id=EXEC,
                    sensitivity_job_id="control-sensitivity-aaaaaaaaaaaaaaaa",
                    expected_sha256=SHA,
                    baseline41=baseline41, baseline42=baseline42,
                )
            self.assertEqual(manifest.read_text(), "source must never be overwritten")
            self.assertEqual(report["v1042_scenario_regression"], "verified")
            self.assertEqual(len(report["scenario_comparison"]), 5)
            self.assertEqual(len(report["fold_comparison"]), 15)
            self.assertTrue(report["comparisons_are_not_causal_attributions"])
            self.assertEqual(report["order_submission"], "never")
            for filename in report["artifacts"]:
                self.assertTrue((Path(report["report_directory"]) / filename).exists())

    def test_predeclared_comparison_reports_no_causal_attribution(self):
        self.assertEqual(len(research.METRICS), len(set(research.METRICS)))
        self.assertIn("strategy_ending_capital", research.PARITY_METRICS)
        self.assertNotIn("winner", research.METRICS)


if __name__ == "__main__":
    import unittest
    unittest.main()
