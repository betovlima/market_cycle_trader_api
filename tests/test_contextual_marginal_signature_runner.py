from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature as runner  # noqa: E402


class ContextualMarginalSignatureRunnerTests(unittest.TestCase):
    def test_stable_entrypoint_and_export_names_have_no_version_suffix(self) -> None:
        self.assertEqual(runner.EXPORT_FOLDER_NAME, "contextual_marginal_signature")
        self.assertEqual(runner.EXPORT_ZIP_NAME, "contextual_marginal_signature.zip")
        self.assertNotIn("v1", runner.EXPORT_FOLDER_NAME)
        self.assertNotIn("v1", runner.EXPORT_ZIP_NAME)
        self.assertEqual(runner.HORIZON_SESSIONS, 40)
        self.assertEqual(len(runner.DECISION_DATES), 6)
        self.assertEqual(len(runner.CANDIDATES), 7)
        self.assertEqual(len(runner.UNIVERSES), 2)

    def test_readiness_requires_diverse_nonzero_both_signs_and_2026(self) -> None:
        rows = []
        candidates = ["A", "B", "C"]
        dates = ["2024-01-02", "2025-01-02", "2026-01-02"]
        for index in range(12):
            rows.append(
                {
                    "candidate": candidates[index % len(candidates)],
                    "decision_date": dates[index % len(dates)],
                    "action_advantage_log": 0.01 if index % 2 == 0 else -0.01,
                }
            )
        result = runner._readiness(pd.DataFrame(rows))
        self.assertTrue(result["ready"])
        self.assertTrue(result["both_signs"])
        self.assertTrue(result["validation_2026"])

    def test_readiness_rejects_one_sided_targets(self) -> None:
        rows = [
            {
                "candidate": ["A", "B", "C"][index % 3],
                "decision_date": ["2024-01-02", "2025-01-02", "2026-01-02"][index % 3],
                "action_advantage_log": 0.01,
            }
            for index in range(12)
        ]
        result = runner._readiness(pd.DataFrame(rows))
        self.assertFalse(result["ready"])
        self.assertFalse(result["both_signs"])


if __name__ == "__main__":
    unittest.main()
