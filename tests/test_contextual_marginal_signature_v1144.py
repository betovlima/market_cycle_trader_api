from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1144 as runner  # noqa: E402
import research_contextual_marginal_signature_v1144_analysis as analysis  # noqa: E402


class ContextualMarginalSignatureV1144Tests(unittest.TestCase):
    def test_campaign_identity(self) -> None:
        self.assertEqual(runner.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14.4")
        self.assertEqual(analysis.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14.4")

    def test_extracts_fold_specific_margin_schedule(self) -> None:
        result = SimpleNamespace(
            metrics={
                "repetition_index": 1,
                "walk_forward_fold_count": 3,
                "walk_forward_folds": [
                    {
                        "fold_id": 2,
                        "calibrated_candidate_margin": 0.0,
                        "effective_switch_margin": 0.0005,
                    },
                    {
                        "fold_id": 3,
                        "calibrated_candidate_margin": 0.01,
                        "effective_switch_margin": 0.01,
                    },
                ],
            }
        )
        schedule = runner._extract_margin_schedule([result])
        self.assertEqual(len(schedule["repetitions"]), 1)
        self.assertEqual(schedule["repetitions"][0]["fold_count"], 3)
        self.assertEqual(schedule["relevant"][(1, 2)]["calibrated_switch_margin"], 0.0)
        self.assertEqual(schedule["relevant"][(1, 3)]["calibrated_switch_margin"], 0.01)

    def test_scheduled_candidates_freeze_only_relevant_folds(self) -> None:
        schedule = {
            "repetitions": [
                {
                    "repetition_index": 1,
                    "fold_count": 3,
                    "relevant_folds": {
                        2: {
                            "calibrated_switch_margin": 0.0,
                            "effective_switch_margin": 0.0005,
                        },
                        3: {
                            "calibrated_switch_margin": 0.01,
                            "effective_switch_margin": 0.01,
                        },
                    },
                }
            ],
            "relevant": {},
        }
        candidates = runner._FoldScheduledMarginCandidates(
            [0.0, 0.0025, 0.005, 0.01],
            schedule,
        )
        self.assertEqual(tuple(candidates), (0.0, 0.0025, 0.005, 0.01))
        self.assertEqual(tuple(candidates), (0.0,))
        self.assertEqual(tuple(candidates), (0.01,))
        self.assertEqual(candidates.consumed_steps, 3)
        self.assertEqual(candidates.expected_steps, 3)
        with self.assertRaises(RuntimeError):
            tuple(candidates)

    def test_relevant_schedule_comparison_accepts_matching_folds(self) -> None:
        expected = {
            "relevant": {
                (1, 2): {"calibrated_switch_margin": 0.0, "effective_switch_margin": 0.0005},
                (1, 3): {"calibrated_switch_margin": 0.01, "effective_switch_margin": 0.01},
            }
        }
        observed = {
            "relevant": {
                (1, 2): {"calibrated_switch_margin": 0.0, "effective_switch_margin": 0.0005},
                (1, 3): {"calibrated_switch_margin": 0.01, "effective_switch_margin": 0.01},
            }
        }
        runner._compare_relevant_schedules(expected, observed, "XSD")


if __name__ == "__main__":
    unittest.main()
