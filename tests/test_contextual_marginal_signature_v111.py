from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v111 as research  # noqa: E402


class ContextualMarginalSignatureV111Tests(unittest.TestCase):
    def test_cases_spec_is_explicit_and_rejects_duplicates(self) -> None:
        payload = {
            "universes": ["U0", "U1"],
            "cases": [
                {"decision_date": "2026-01-02", "candidates": ["apd", "vnce"]},
                {"decision_date": "2026-03-02", "candidates": ["APD"]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            universes, cases = research._load_cases(path)
        self.assertEqual(universes, ["U0", "U1"])
        self.assertEqual(cases[0]["candidates"], ["APD", "VNCE"])

        payload["cases"].append({"decision_date": "2026-01-02", "candidates": ["APD"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Duplicate candidate/date case"):
                research._load_cases(path)

    def test_snapshot_hash_is_order_invariant(self) -> None:
        a = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "timestamp": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
                "open": [1.0, 2.0],
                "high": [1.2, 2.2],
                "low": [0.9, 1.9],
                "close": [1.1, 2.1],
                "volume": [100.0, 200.0],
            }
        )
        b = a.iloc[::-1].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self.assertEqual(research._snapshot_hash(a.sort_values(["symbol", "timestamp"])), research._snapshot_hash(b))

    def test_first_divergence_reports_decision_column(self) -> None:
        left = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
                "selected_asset": ["A", "A"],
                "strategy_equity": [100.0, 101.0],
            }
        )
        right = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
                "selected_asset": ["A", "B"],
                "strategy_equity": [100.0, 101.0],
            }
        )
        result = research._first_divergence(left, right)
        self.assertEqual(result["first_divergence"], "2026-01-05T00:00:00+00:00")
        self.assertIn("selected_asset", result["changed_columns"])

    def test_pairwise_decomposition_obeys_delta_identity(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "decision_date": "2026-01-02",
                    "candidate": "APD",
                    "universe_name": "U0",
                    "baseline_ending_capital": 100.0,
                    "candidate_ending_capital": 110.0,
                    "delta_log_capital": __import__("math").log(1.1),
                },
                {
                    "decision_date": "2026-01-02",
                    "candidate": "APD",
                    "universe_name": "U1",
                    "baseline_ending_capital": 90.0,
                    "candidate_ending_capital": 99.0,
                    "delta_log_capital": __import__("math").log(1.1),
                },
            ]
        )
        result = research._pairwise_decomposition(frame, ["U0", "U1"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0]["identity_error"]), 0.0, places=12)
        self.assertAlmostEqual(float(result.iloc[0]["delta_marginal"]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
