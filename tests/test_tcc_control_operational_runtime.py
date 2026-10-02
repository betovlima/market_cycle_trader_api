"""Regression contract for the operational TCC v1.0.6 Control mode."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.config import TCC_CONTROL_OPERATIONAL_MODE
from market_cycle_trader_api.engine import live_model_signal
from market_cycle_trader_api.engine import tcc_control_operational_runtime as runtime
from market_cycle_trader_api.services import paper_trading, strategy_lab
from market_cycle_trader_api.services.model_research import (
    execution_settings_from_values,
    model_execution_snapshot,
)


def _contract_config(**overrides):
    values = runtime.tcc_control_strategy_updates()
    values.update({
        "research_model_family": "lightgbm_utility",
        "research_market_data_protocol": "raw_total_causal_v1",
    })
    values.update(overrides)
    return SimpleNamespace(**values)


def _contract_model_snapshot():
    settings = execution_settings_from_values(
        "lightgbm_utility",
        runtime.tcc_control_model_values(),
        settings_revision=1,
        profile_id="tcc-v1.0.6-control",
    )
    snapshot = model_execution_snapshot("lightgbm_utility", settings)
    snapshot["source"] = "tcc_v106_operational_contract"
    return snapshot


class TCCControlOperationalRuntimeTests(TestCase):
    def test_contract_profile_matches_frozen_tcc_values(self):
        config = _contract_config()
        self.assertEqual(runtime.tcc_control_contract_issues(config), [])
        self.assertEqual(
            tuple(config.assets),
            runtime.TCC_CONTROL_REQUESTED_ASSETS,
        )
        self.assertEqual(config.rotation_target_horizons, [5, 10, 20, 40, 60])
        self.assertEqual(
            config.rotation_target_horizon_weights,
            [0.10, 0.15, 0.20, 0.30, 0.25],
        )
        self.assertEqual(config.rotation_switch_margin, 0.0005)
        self.assertEqual(
            config.rotation_switch_margin_candidates,
            [0.0, 0.0025, 0.005, 0.01],
        )

    def test_contract_rejects_parameter_drift(self):
        config = _contract_config(rotation_switch_margin=0.02)
        issues = runtime.tcc_control_contract_issues(config)
        self.assertTrue(
            any("rotation_switch_margin:" in issue for issue in issues)
        )

    def test_model_snapshot_matches_frozen_lightgbm(self):
        snapshot = _contract_model_snapshot()
        self.assertEqual(
            runtime.tcc_control_model_snapshot_issues(snapshot),
            [],
        )
        modified = _contract_model_snapshot()
        modified["settings_snapshot"]["lightgbm"]["n_estimators"] = 1
        issues = runtime.tcc_control_model_snapshot_issues(modified)
        self.assertTrue(
            any("lightgbm.n_estimators:" in issue for issue in issues)
        )

    def test_live_router_bypasses_generic_lightgbm_engine(self):
        sentinel = object()
        config = SimpleNamespace(strategy_mode=TCC_CONTROL_OPERATIONAL_MODE)
        with (
            patch.object(
                live_model_signal,
                "build_live_tcc_control_decision",
                return_value=sentinel,
            ) as tcc_runner,
            patch.object(
                live_model_signal,
                "build_live_lightgbm_decision",
            ) as generic_runner,
        ):
            result = live_model_signal.build_live_model_decision(
                {},
                config,
                model_family="lightgbm_utility",
                current_asset="CASH",
                holding_sessions=0,
            )
        self.assertIs(result, sentinel)
        tcc_runner.assert_called_once()
        generic_runner.assert_not_called()

    def test_backtest_runner_calls_vendored_tcc_control(self):
        index = pd.date_range("2020-01-01", periods=3, tz="UTC")
        bars = {
            "NVDA": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=index,
            ),
            "AAPL": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1.0, 1.0, 1.0],
                },
                index=index,
            ),
        }
        fake = SimpleNamespace(metrics={})
        with patch.object(
            runtime,
            "run_research_challenger",
            return_value=[fake],
        ) as scientific:
            result = runtime.run_tcc_control_operational_backtest(
                bars,
                SimpleNamespace(analysis_end_date=None, end_date=None),
            )
        self.assertEqual(result, [fake])
        args = scientific.call_args.args
        self.assertEqual(args[0], "lightgbm_utility")
        scientific_config = args[2]
        self.assertEqual(
            scientific_config.strategy_mode,
            "COMPOUND_ROTATION_SWING_LIGHTGBM",
        )
        self.assertFalse(
            scientific_config.research_model_settings["soft_horizon_consensus"][
                "enabled"
            ]
        )
        self.assertEqual(
            fake.metrics["operational_contract"],
            "tcc_v1.0.6_control",
        )

    def test_strategy_catalog_requires_exact_tcc_contract(self):
        snapshot = _contract_model_snapshot()
        document = {
            "configuration": runtime.tcc_control_strategy_updates(),
            "research_model_snapshot": snapshot,
            "strategy_kind": "standard",
        }
        result = strategy_lab._trader_runtime_compatibility(document)
        self.assertTrue(result["eligible"])
        self.assertEqual(
            result["code"],
            "tcc_control_v106_live_runtime_ready",
        )

        broken = {
            **document,
            "configuration": {
                **document["configuration"],
                "rotation_switch_margin": 0.02,
            },
        }
        result = strategy_lab._trader_runtime_compatibility(broken)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["code"], "tcc_control_contract_mismatch")

    def test_paper_runtime_uses_raw_sip_control_snapshot(self):
        source = inspect.getsource(paper_trading.prepare_next_paper_plan)
        self.assertIn("download_current_control_snapshot(", source)
        self.assertIn("tcc_control_v106_raw_sip_snapshot", source)
        self.assertIn("tcc_control_snapshot_sha256", source)

    def test_strategy_save_normalizes_tcc_profile_and_model(self):
        source = inspect.getsource(strategy_lab.update_strategy)
        self.assertIn("tcc_control_strategy_updates()", source)
        self.assertIn("tcc_control_model_values()", source)
        self.assertIn('"tcc_v106_operational_contract"', source)


if __name__ == "__main__":
    import unittest
    unittest.main()
