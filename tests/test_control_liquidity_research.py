"""Offline, fail-closed original Control versus liquidity-aware policy tests."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import StartControlLiquidityResearchRequest
from market_cycle_trader_api.engine import control_execution_feasibility as execution
from market_cycle_trader_api.engine import control_liquidity_policy as policy
from market_cycle_trader_api.engine import control_liquidity_research as research
from market_cycle_trader_api.services import control_shadow_liquidity_jobs as jobs
from market_cycle_trader_api.tcc_v106_reference import research_challengers as scientific

VALIDATION = "control-validation-7821002400424ccc"
EXECUTION = "control-execution-c38169f6c6fc4ea0"
SENSITIVITY = "control-sensitivity-1c8b4690d80d4145"
SNAP = "control-shadow-870fb66e1bdc4fd0"
SHA = "6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


def fees(side, amount, price, config):
    return {
        "commission_fee": 0.0, "sec_fee": 0.0,
        "taf_fee": 0.0, "cat_fee": 0.0, "total_fee": 0.0,
    }


def fake_runner(bars, config, fee, slippage, *,
                progress_callback, trade_callback,
                progress_detail_callback, technical_log_callback):
    symbols = sorted(bars)
    dates = bars[symbols[0]].index[23:40]
    for _ in range(3):
        cache, profile = _precompute_model_utilities(
            {}, bars, symbols, dates, config,
        )
    def current_policy(date, holding, duration):
        values = cache.get(pd.Timestamp(date))
        picked = int(np.argmax(values))
        return picked, float(values[picked])
    output = _simulate_exact(
        "lightgbm_utility", current_policy, bars, symbols, dates,
        config, fee, slippage,
    )
    return [output]


class ControlLiquidityAwareTests(TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-07-01", periods=42, tz="UTC")
        self.frames = {
            "AAA": pd.DataFrame({
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.0, "volume": 100.0,
            }, index=self.dates),
            "BBB": pd.DataFrame({
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.0, "volume": 1000.0,
            }, index=self.dates),
        }
        self.config = SimpleNamespace(
            initial_capital=10000.0, rotation_model_repetitions=1,
            rotation_downside_penalty=.2, rotation_drawdown_penalty=.3,
        )

    def _cache(self, raw=None):
        raw = [0.0, .30, .10] if raw is None else raw
        key = self.dates[26]
        account = {
            "enabled": True, "audit": {},
            "decision_timestamp": key,
            "equity": 10000.0, "cash": 10000.0,
            "position": 0, "shares": 0,
        }
        return policy._CapitalAwareUtilityCache(
            {key: np.array(raw, dtype=float)}, self.frames,
            ["AAA", "BBB"], account,
        ), account

    def test_candidate_utility_uses_only_completed_close_and_known_capital(self):
        cache, account = self._cache()
        actual = cache.get(self.dates[26])
        self.assertAlmostEqual(actual[1], .003)
        self.assertAlmostEqual(actual[2], .01)
        self.assertEqual(account["audit"][self.dates[26]]["raw_best_asset"], "AAA")
        self.assertEqual(account["audit"][self.dates[26]]["adjusted_best_asset"], "BBB")
        self.assertEqual(float(dict.get(cache, self.dates[26])[1]), .30)
        self.assertEqual(actual[0], 0.0)

    def test_future_volume_price_mutations_cannot_change_prior_close_candidate_utility(self):
        before, _ = self._cache()
        expected = before.get(self.dates[26]).copy()
        later = {name: frame.copy() for name, frame in self.frames.items()}
        for frame in later.values():
            frame.loc[self.dates[27]:, "volume"] = 0.0
            frame.loc[self.dates[27]:, "close"] = 90000.0
            frame.loc[self.dates[27]:, "open"] = 90000.0
        cache, account = self._cache()
        cache._frames = later
        np.testing.assert_array_equal(cache.get(self.dates[26]), expected)
        self.assertNotIn(self.dates[27], account["audit"])

    def test_incumbent_hold_is_not_penalized_and_insufficient_exit_reduces_switch(self):
        cache, account = self._cache()
        account.update({"position": 1, "shares": 900, "cash": 1000.0})
        adjusted = cache.get(self.dates[26])
        self.assertEqual(adjusted[1], .30)
        self.assertAlmostEqual(
            adjusted[2], .10 * min(.10, (100 * .10 * 10) / (900*10)),
        )
        self.assertLess(adjusted[2], .01)

    def test_account_date_mismatch_aborts(self):
        cache, account = self._cache()
        account["decision_timestamp"] = self.dates[25]
        with self.assertRaisesRegex(ValueError, "stale"):
            cache.get(self.dates[26])

    def test_true_training_single_fit_and_tcc_original_module_untouched(self):
        original_sim = scientific._simulate_exact
        original_pre = scientific._precompute_model_utilities
        def fake_pre(models, frames, symbols, timestamps, cfg):
            return (
                {
                    pd.Timestamp(d): np.array([0.0, .30, .10], dtype=float)
                    for d in timestamps
                }, {"model_role": "weighted_utility"},
            )
        cloned = FunctionType(
            fake_runner.__code__,
            {
                "_simulate_exact": original_sim,
                "_precompute_model_utilities": fake_pre,
                "np": np, "pd": pd,
            },
        )
        with (
            patch.object(scientific, "_run_lightgbm", cloned),
            patch.object(scientific, "_precompute_model_utilities", fake_pre),
            patch.object(scientific, "allocation_execution_enabled", return_value=False),
            patch.object(execution, "_equal_weight_benchmark",
                         side_effect=lambda bars, symbols, dates, *args:
                         pd.Series(self.config.initial_capital, index=dates)),
        ):
            paths, returned, audit = policy.run_control_liquidity_pair(
                self.frames, self.config, fees, lambda p, s, c: p,
            )
            self.assertIs(returned, paths["control_reference"])
            self.assertTrue(
                (paths["control_reference"].predictions["signal_asset"] == "AAA").all()
            )
            self.assertTrue(
                (paths["liquidity_aware"].predictions["signal_asset"] == "BBB").all()
            )
            self.assertEqual(len(audit), 16)
            self.assertEqual(
                paths["control_reference"].metrics["execution_scenario"],
                paths["liquidity_aware"].metrics["execution_scenario"],
            )
            self.assertIs(scientific._simulate_exact, original_sim)
            self.assertIs(cloned.__globals__["_simulate_exact"], original_sim)
            self.assertIs(cloned.__globals__["_precompute_model_utilities"], fake_pre)
        self.assertIs(scientific._simulate_exact, original_sim)
        self.assertIs(scientific._precompute_model_utilities, original_pre)

    def test_research_request_fixed_no_external_parameters_or_orders(self):
        body = {
            "confirm": "RESEARCH_CONTROL_PRIOR_CLOSE_LIQUIDITY_NO_ORDERS",
            "source_validation_job_id": VALIDATION,
            "source_execution_job_id": EXECUTION,
            "source_sensitivity_job_id": SENSITIVITY,
            "expected_snapshot_sha256": SHA,
        }
        model = StartControlLiquidityResearchRequest.model_validate(body)
        self.assertEqual(model.source_sensitivity_job_id, SENSITIVITY)
        with self.assertRaises(ValidationError):
            StartControlLiquidityResearchRequest.model_validate({
                **body, "participation_rate": .8,
            })
        with self.assertRaises(ValidationError):
            StartControlLiquidityResearchRequest.model_validate({
                **body, "confirm": "SEND_REAL_ORDER",
            })
        self.assertEqual(policy.POLICY_SPEC["calibrated_on_oos"], False)
        self.assertEqual(research.RESEARCH_STRATEGY_MODE,
                         "MCT_RESEARCH_CONTROL_LIQUIDITY_AWARE_V1")

    def test_real_report_contract_pairs_previous_close_to_next_execution_date(self):
        full_dates = pd.bdate_range("2020-07-21", periods=1555, tz="UTC")
        oos_dates = full_dates[1:]
        fold_ids = [1] * 504 + [2] * 504 + [3] * 546
        original_folds = []
        for i, (first, last) in enumerate(((0, 503), (504, 1007), (1008, 1553)), 1):
            original_folds.append({
                "fold_id": i, "sessions": last - first + 1,
                "test_start": oos_dates[first].isoformat(),
                "test_end": oos_dates[last].isoformat(),
            })
        reference_metric = {
            key: 0.0 for key in research.REPORT_METRICS
        }
        reference_metric.update({
            "initial_capital": 10000.0,
            "strategy_ending_capital": 10001.0,
            "session_count": 1554,
            "terminal_holdings_asset": "S00",
            "execution_scenario": dict(execution.SCENARIO),
        })
        values = np.linspace(10000, 10001, 1554)
        def curve(target):
            return pd.DataFrame({
                "strategy_equity": values,
                "cash_weight": np.zeros(1554),
                "walk_forward_fold": fold_ids,
                "decision_date": full_dates[:-1],
                "signal_asset": [target] * 1554,
            }, index=oos_dates)
        runs = {
            "control_reference": SimpleNamespace(
                predictions=curve("S00"), trades=pd.DataFrame(),
                metrics=reference_metric,
            ),
            "liquidity_aware": SimpleNamespace(
                predictions=curve("S01"), trades=pd.DataFrame(),
                metrics=dict(reference_metric),
            ),
        }
        candidate = {
            f"S{i:02d}": {
                "capacity_dollars": 100.0,
                "capacity_fraction": .1,
                "raw_utility": .2,
                "effective_utility": .02,
            } for i in range(55)
        }
        details = {
            full_dates[i]: {
                "equity_at_decision": 10000.0,
                "cash_at_decision": 9000.0,
                "held_asset_at_decision": "S00",
                "incumbent_exit_fraction": .1,
                "raw_best_asset": "S00",
                "adjusted_best_asset": "S01",
                "candidate_liquidity_detail": candidate,
            } for i in range(1554)
        }
        base41 = {
            "source_snapshot_sha256": SHA, "source_job_id": SNAP,
            "source_unchanged": True, "order_submission": "never",
            "original_shadow": {"reproduced": True},
            "snapshot_assets": 55, "calendar_sessions": 2500,
            "numeric_input_integrity": {"status": "verified", "checked_assets": 55},
            "oos": {
                "walk_forward_fold_count": 3,
                "walk_forward_folds": original_folds,
            },
        }
        base42 = {
            "source_snapshot_sha256": SHA, "source_job_id": SNAP,
            "source_unchanged": True, "order_submission": "never",
            "source_validation_job_id": VALIDATION,
            "numeric_input_integrity": {"status": "verified", "assets": 55},
            "execution_scenario": dict(execution.SCENARIO),
            "feasible_oos": {
                **reference_metric, "walk_forward_fold_count": 3,
            },
        }
        base43 = {
            "source_snapshot_sha256": SHA, "source_job_id": SNAP,
            "source_unchanged": True, "order_submission": "never",
            "source_validation_job_id": VALIDATION,
            "source_execution_job_id": EXECUTION,
            "v1042_scenario_regression": "verified",
            "numeric_input_integrity": {"status": "verified", "assets": 55},
            "scenario_comparison": [{
                "scenario": name,
                "strategy_ending_capital": 10001.0,
            } for name in (
                "cap10_cost", "no_cap_no_cost", "cap10_no_cost",
                "cap05_cost", "cap01_cost",
            )],
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / SNAP
            folder.mkdir()
            manifest = folder / "manifest.json"
            manifest.write_text("immutable sentinel", encoding="utf8")
            with (
                patch.object(research, "read_verified_control_snapshot",
                             return_value=(
                                 {f"S{i:02d}": pd.DataFrame() for i in range(55)},
                                 {"completed_session": "2026-09-28"}, folder,
                             )),
                patch.object(research, "prepare_operational_control_panel",
                             return_value=(
                                 {}, full_dates,
                                 SimpleNamespace(calendar_sessions=2500),
                             )),
                patch.object(research, "run_control_liquidity_pair",
                             return_value=(runs, runs["control_reference"], details)),
            ):
                report = research.run_control_liquidity_research(
                    source_job_id=SNAP, validation_job_id=VALIDATION,
                    execution_job_id=EXECUTION,
                    sensitivity_job_id=SENSITIVITY,
                    research_job_id="control-liquidity-aaaaaaaaaaaaaaaa",
                    expected_sha256=SHA,
                    baseline41=base41, baseline42=base42, baseline43=base43,
                )
            self.assertEqual(manifest.read_text(), "immutable sentinel")
            self.assertEqual(report["control_reference_parity"], "verified")
            self.assertEqual(report["diagnostics"]["candidate_score_rows"], 1554*55)
            self.assertEqual(report["diagnostics"]["policy_target_disagreement_sessions"], 1554)
            self.assertFalse(report["order_eligible"])
            output = Path(report["report_directory"])
            self.assertTrue(all((output / name).is_file() for name in report["artifacts"]))
            audit = pd.read_csv(output / "liquidity_decision_audit.csv")
            self.assertEqual(audit.iloc[0]["timestamp"], oos_dates[0].isoformat())
            self.assertEqual(audit.iloc[0]["decision_timestamp"], full_dates[0].isoformat())

    def test_rejects_mismatched_sources_before_starting_thread(self):
        db = MagicMock()
        db[jobs.VALIDATION_COLLECTION].find_one.return_value = {
            "_id": VALIDATION, "status": "completed",
            "source_job_id": SNAP, "snapshot_sha256": SHA,
            "result": {
                "original_shadow": {"reproduced": True},
                "numeric_input_integrity": {"status": "verified"},
                "order_submission": "never",
                "source_snapshot_sha256": SHA,
            },
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value = {
            "_id": EXECUTION, "status": "completed",
            "source_job_id": "control-shadow-aaaaaaaaaaaaaaaa",
            "source_validation_job_id": VALIDATION,
            "snapshot_sha256": SHA,
            "result": {
                "source_snapshot_sha256": SHA,
                "numeric_input_integrity": {"status": "verified"},
            },
        }
        db[jobs.SENSITIVITY_COLLECTION].find_one.return_value = {
            "_id": SENSITIVITY, "status": "completed",
            "source_job_id": SNAP, "snapshot_sha256": SHA,
            "result": {
                "source_snapshot_sha256": SHA,
                "numeric_input_integrity": {"status": "verified"},
            },
        }
        with patch.object(jobs, "_require_enabled"):
            with self.assertRaises(jobs.LiquidityInvalid):
                jobs.start_liquidity_research(
                    db,
                    source_validation_job_id=VALIDATION,
                    source_execution_job_id=EXECUTION,
                    source_sensitivity_job_id=SENSITIVITY,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
