from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v115 as experiment  # noqa: E402


class ContextualMarginalSignatureV115Tests(unittest.TestCase):
    def test_identity_and_feature_contract(self) -> None:
        self.assertEqual(experiment.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.15")
        self.assertEqual(tuple(experiment.STATIC_FEATURES), tuple(experiment.v103.MODEL_FEATURES))
        self.assertEqual(len(experiment.POLICY_FEATURES), 5)
        self.assertIn("baseline_effective_switch_margin", experiment.POLICY_FEATURES)
        self.assertEqual(
            len(experiment.M0_FEATURES),
            len(experiment.STATIC_FEATURES) + len(experiment.POLICY_FEATURES),
        )
        self.assertEqual(len(experiment.LEVEL2_FEATURES), 6)
        self.assertIn("logsig2__candidate__universe", experiment.LEVEL2_FEATURES)
        self.assertNotIn("logsig1__candidate", experiment.LEVEL2_FEATURES)

    def test_screen_gate_requires_positive_incremental_ranking(self) -> None:
        baseline = {
            "pooled_spearman": -0.2,
            "mean_context_spearman": -0.1,
            "top1_negative_contexts": 0,
            "top1_positive_contexts": 2,
            "mean_top1_direct_log": 0.01,
        }
        improved = {
            "pooled_spearman": 0.3,
            "mean_context_spearman": 0.2,
            "top1_negative_contexts": 0,
            "top1_positive_contexts": 3,
            "mean_top1_direct_log": 0.02,
        }
        self.assertTrue(experiment._passes(baseline, improved))
        improved["mean_context_spearman"] = -0.01
        self.assertFalse(experiment._passes(baseline, improved))


if __name__ == "__main__":
    unittest.main()
