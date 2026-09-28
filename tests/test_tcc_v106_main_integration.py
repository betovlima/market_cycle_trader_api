"""Integration guard: retain main's progress heartbeat and TCC reference research."""

from __future__ import annotations

import inspect
import unittest

from market_cycle_trader_api.core.config import API_VERSION
from market_cycle_trader_api.engine import compound_rotation_backtest
from market_cycle_trader_api.engine.tcc_frozen_reference_source import DEFAULT_SOURCE
from market_cycle_trader_api.services.strategy_lab import (
    TCC_V106_BACKTEST_ENGINE_BINDING,
    _trader_runtime_compatibility,
)


class MainReferenceIntegrationTests(unittest.TestCase):
    def test_distinct_integrated_api_version(self) -> None:
        self.assertEqual(API_VERSION, "10.8.37")

    def test_main_heartbeat_and_tcc_data_path_coexist(self) -> None:
        source = inspect.getsource(compound_rotation_backtest.run_job)
        self.assertTrue(callable(compound_rotation_backtest._start_progress_heartbeat))
        self.assertIn("load_research_market_bars(", source)
        self.assertIn("_start_progress_heartbeat()", source)
        self.assertIn("heartbeat_stop.set()", source)
        self.assertIn("heartbeat_thread.join(", source)
        self.assertIn("effective_config,", source)
        self.assertIn("structural_exclusions", source)

    def test_default_reference_source_is_mct_current(self) -> None:
        self.assertEqual(DEFAULT_SOURCE, "mct_current")

    def test_reference_strategy_cannot_use_live_trader(self) -> None:
        result = _trader_runtime_compatibility({
            "strategy_kind": "standard",
            "backtest_engine_binding": TCC_V106_BACKTEST_ENGINE_BINDING,
            "research_model_snapshot": {"family": "lightgbm_utility"},
        })
        self.assertFalse(result["eligible"])


if __name__ == "__main__":
    unittest.main()
