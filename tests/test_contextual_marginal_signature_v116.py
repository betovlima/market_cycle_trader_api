from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v116 as experiment  # noqa: E402


class ContextualMarginalSignatureV116Tests(unittest.TestCase):
    def test_identity(self) -> None:
        self.assertEqual(experiment.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.16")
        self.assertEqual(
            experiment.EXPERIMENT_NAME,
            "contextual_marginal_signature_paired_action_advantage",
        )

    def test_utc_normalizes_naive_and_aware(self) -> None:
        naive = experiment._utc("2026-01-02")
        aware = experiment._utc("2026-01-02T05:00:00-03:00")
        self.assertEqual(str(naive.tz), "UTC")
        self.assertEqual(str(aware.tz), "UTC")
        self.assertEqual(aware.hour, 8)

    def test_forced_first_action_then_resumes_policy(self) -> None:
        original_simulator = experiment._ORIGINAL_SIMULATE_EXACT
        original_context = experiment.frozen._CURRENT_CONTEXT
        calls: list[tuple[pd.Timestamp, int, int]] = []
        observed: dict[str, object] = {}

        def policy(timestamp, current_position, holding_days):
            calls.append((pd.Timestamp(timestamp), int(current_position), int(holding_days)))
            return (1, 0.5)

        def fake_simulator(
            backend,
            wrapped_policy,
            frames,
            symbols,
            decision_dates,
            config,
            fee_calculator,
            slippage,
            **kwargs,
        ):
            first_target, _ = wrapped_policy(decision_dates[0], 0, 0)
            second_target, _ = wrapped_policy(decision_dates[1], first_target, 1)
            observed["first_target"] = first_target
            observed["second_target"] = second_target
            return SimpleNamespace(metrics={})

        try:
            experiment._ORIGINAL_SIMULATE_EXACT = fake_simulator
            experiment.frozen._CURRENT_CONTEXT = {"candidate": "BBB"}
            result = experiment._forced_first_action_simulator(
                "lightgbm_utility",
                policy,
                {},
                ["AAA", "BBB"],
                pd.DatetimeIndex(["2025-12-31T05:00:00Z", "2026-01-02T05:00:00Z"]),
                SimpleNamespace(strategy_mode=experiment.LEGACY_MODE),
                lambda *args, **kwargs: {},
                lambda value, side, config: value,
                policy_decision_diagnostics={},
            )
        finally:
            experiment._ORIGINAL_SIMULATE_EXACT = original_simulator
            experiment.frozen._CURRENT_CONTEXT = original_context

        self.assertEqual(observed["first_target"], 2)
        self.assertEqual(observed["second_target"], 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], 2)
        self.assertTrue(result.metrics["research_forced_first_action"])
        self.assertEqual(result.metrics["research_forced_action_asset"], "BBB")
        self.assertEqual(result.metrics["research_policy_action_before_force"], "AAA")

    def test_rejects_stateful_strategy_mode(self) -> None:
        original_context = experiment.frozen._CURRENT_CONTEXT
        try:
            experiment.frozen._CURRENT_CONTEXT = {"candidate": "BBB"}
            with self.assertRaisesRegex(RuntimeError, "stateless legacy rotation policy"):
                experiment._forced_first_action_simulator(
                    "lightgbm_utility",
                    lambda timestamp, position, holding: (1, 0.0),
                    {},
                    ["AAA", "BBB"],
                    pd.DatetimeIndex(["2025-12-31T05:00:00Z", "2026-01-02T05:00:00Z"]),
                    SimpleNamespace(strategy_mode="COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE"),
                    lambda *args, **kwargs: {},
                    lambda value, side, config: value,
                )
        finally:
            experiment.frozen._CURRENT_CONTEXT = original_context


if __name__ == "__main__":
    unittest.main()
