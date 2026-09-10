from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_exact_marginal_capital_search_v103 as hotfix  # noqa: E402


class ExactMarginalCapitalSearchV103Tests(unittest.TestCase):
    def setUp(self) -> None:
        hotfix._SELECTED_CANDIDATES = ["LB", "VSAT"]
        hotfix._IDENTITY_CACHE = None
        hotfix._IDENTITY_ERROR = None

    def test_identity_rejection_skips_expensive_candidate_replay(self) -> None:
        rejected = {
            "LB": {
                "status": "rejected",
                "checked": True,
                "source": "alpaca_corporate_actions_v1",
                "reason": "economic_identity_discontinuity",
                "event_count": 1,
                "comparability_break_count": 1,
                "comparability_breaks": [{"type": "name_change", "old_symbol": "LB", "new_symbol": "BBWI"}],
            }
        }
        with patch.object(hotfix, "_identity_integrity_snapshot", return_value=rejected), patch.object(
            hotfix.previous, "_candidate_evaluation_v102"
        ) as exact:
            result = hotfix._candidate_evaluation_v103(
                db=object(),
                symbol="LB",
                history_start=pd.Timestamp("2016-01-01"),
                snapshot_end=pd.Timestamp("2026-09-04"),
            )

        self.assertEqual(result["evaluation_status"], "context_rejected")
        self.assertEqual(result["rejection_reason"], "economic_identity_discontinuity")
        self.assertEqual(result["identity_integrity_break_count"], 1)
        self.assertIsNone(result["exact_replay_seconds"])
        exact.assert_not_called()

    def test_identity_pass_delegates_to_v102_exact_evaluation(self) -> None:
        passed = {
            "VSAT": {
                "status": "passed",
                "checked": True,
                "source": "alpaca_corporate_actions_v1",
                "reason": "no_identity_break_detected",
                "event_count": 0,
                "comparability_break_count": 0,
                "comparability_breaks": [],
            }
        }
        with patch.object(hotfix, "_identity_integrity_snapshot", return_value=passed), patch.object(
            hotfix.previous,
            "_candidate_evaluation_v102",
            return_value={"symbol": "VSAT", "evaluation_status": "completed", "ending_capital_delta_rate": -0.1},
        ) as exact:
            result = hotfix._candidate_evaluation_v103(
                db=object(),
                symbol="VSAT",
                history_start=pd.Timestamp("2016-01-01"),
                snapshot_end=pd.Timestamp("2026-09-04"),
            )

        self.assertEqual(result["evaluation_status"], "completed")
        self.assertEqual(result["identity_integrity_status"], "passed")
        exact.assert_called_once()

    def test_candidate_capture_resets_identity_cache(self) -> None:
        hotfix._IDENTITY_CACHE = {"OLD": {"status": "passed"}}
        hotfix._IDENTITY_ERROR = "old error"
        with patch.object(
            hotfix,
            "_ORIGINAL_CANDIDATE_SYMBOLS",
            return_value=["CCS", "RARE"],
        ):
            values = hotfix._candidate_symbols_v103(object(), {}, ["NVDA"], ["CCS", "RARE"])

        self.assertEqual(values, ["CCS", "RARE"])
        self.assertEqual(hotfix._SELECTED_CANDIDATES, ["CCS", "RARE"])
        self.assertIsNone(hotfix._IDENTITY_CACHE)
        self.assertIsNone(hotfix._IDENTITY_ERROR)


if __name__ == "__main__":
    unittest.main()
