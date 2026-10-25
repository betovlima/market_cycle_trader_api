"""Regression tests: capture U67/U77 replay diagnostics without policy changes."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.tcc_u67_operational_runtime import (
    _replay_ranked_policy,
    run_tcc_u67_operational_backtest,
)


class FrozenPolicyReplayScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = pd.Timestamp("2026-07-14", tz="UTC")
        self.symbols = ["CEF", "AVGO", "MSFT", "LLY", "NVDA", "SPY"]
        self.cache = {
            self.day: np.array([0.0, 0.30, 0.70, 0.50, 0.20, 0.40, np.nan]),
        }

    def test_captures_rank_four_and_five_and_does_not_change_action(self) -> None:
        diagnostics = {}
        calls = []

        def frozen_policy(day, position, holding):
            calls.append((day, position, holding))
            diagnostics[pd.Timestamp(day)] = {
                "top_1_asset": "AVGO", "top_1_score": 0.70,
                "top_2_asset": "MSFT", "top_2_score": 0.50,
                "top_3_asset": "NVDA", "top_3_score": 0.40,
                "decision_reason": "ROTATE_TO_BEST_ASSET",
                "effective_switch_margin": 0.01,
            }
            return 2, 0.70

        replay_policy = _replay_ranked_policy(
            frozen_policy, diagnostics=diagnostics,
            utility_cache=self.cache, symbols=self.symbols,
        )
        self.assertEqual(replay_policy(self.day, 1, 4), (2, 0.70))
        self.assertEqual(calls, [(self.day, 1, 4)])
        self.assertEqual(diagnostics[self.day]["top_4_asset"], "CEF")
        self.assertEqual(diagnostics[self.day]["top_4_score"], 0.30)
        self.assertEqual(diagnostics[self.day]["top_5_asset"], "LLY")
        self.assertEqual(diagnostics[self.day]["top_5_score"], 0.20)
        self.assertEqual(diagnostics[self.day]["decision_reason"], "ROTATE_TO_BEST_ASSET")

    def test_refuses_to_mix_mismatched_rankings(self) -> None:
        diagnostics = {}

        def frozen_policy(day, position, holding):
            diagnostics[pd.Timestamp(day)] = {
                "top_1_asset": "CEF", "top_1_score": 0.70,
            }
            return position, 0.12

        policy = _replay_ranked_policy(
            frozen_policy, diagnostics=diagnostics,
            utility_cache=self.cache, symbols=self.symbols,
        )
        self.assertEqual(policy(self.day, 1, 2), (1, 0.12))
        self.assertNotIn("top_4_asset", diagnostics[self.day])

    def test_no_scores_are_invented_when_frozen_policy_has_no_diagnostics(self) -> None:
        diagnostics = {}
        def frozen_policy(day, position, holding):
            return position, 0.0
        policy = _replay_ranked_policy(
            frozen_policy, diagnostics=diagnostics,
            utility_cache=self.cache, symbols=self.symbols,
        )
        self.assertEqual(policy(self.day, 1, 2), (1, 0.0))
        self.assertEqual(diagnostics, {})

    def test_operational_adapter_passes_recorded_diagnostics_to_simulator(self) -> None:
        source = inspect.getsource(run_tcc_u67_operational_backtest)
        self.assertIn("decision_diagnostics=replay_decision_diagnostics", source)
        self.assertIn("policy_decision_diagnostics=replay_decision_diagnostics", source)
        self.assertIn("_replay_ranked_policy(", source)


if __name__ == "__main__":
    unittest.main()
