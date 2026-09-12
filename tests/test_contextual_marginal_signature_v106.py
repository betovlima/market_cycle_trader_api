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

import research_contextual_marginal_signature_v106 as research  # noqa: E402


class ContextualMarginalSignatureV106Tests(unittest.TestCase):
    def test_universe_spec_requires_distinct_explicit_universes(self) -> None:
        payload = {
            "universes": [
                {"name": " U-A ", "assets": ["aapl", "msft", "AAPL"]},
                {"name": "U-B", "assets": ["NVDA", "META"]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universes.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            universes = research.load_universe_spec(path)
        self.assertEqual([item["name"] for item in universes], ["U-A", "U-B"])
        self.assertEqual(universes[0]["assets"], ["AAPL", "MSFT"])
        self.assertEqual(universes[0]["size"], 2)
        self.assertNotEqual(universes[0]["hash"], universes[1]["hash"])

    def test_candidates_must_stay_outside_every_universe(self) -> None:
        universes = [
            {"name": "A", "assets": ["AAPL", "MSFT"]},
            {"name": "B", "assets": ["NVDA", "META"]},
        ]
        research.validate_candidate_universe_separation(universes, ["GKOS", "DNN"])
        with self.assertRaisesRegex(RuntimeError, "Candidates must remain outside every universe"):
            research.validate_candidate_universe_separation(universes, ["GKOS", "MSFT"])

    def _row(self, universe: str, candidate: str, decision_date: str, y: float, corr: float) -> dict:
        values = {feature: 0.0 for feature in research.MODEL_FEATURES}
        values["corr_20_to_universe"] = corr
        return {
            "universe_name": universe,
            "universe_size": 10 if universe == "U1" else 12,
            "universe_hash": universe.lower(),
            "decision_date": decision_date,
            "horizon_end": "2025-03-31",
            "candidate": candidate,
            "evaluation_status": "completed",
            "ending_capital_delta_rate": float(__import__("math").expm1(y)),
            "delta_log_capital": y,
            **values,
        }

    def test_pairwise_difference_holds_candidate_and_date_fixed(self) -> None:
        frame = pd.DataFrame([
            self._row("U1", "APD", "2025-02-03", -0.02, 0.10),
            self._row("U2", "APD", "2025-02-03", 0.03, 0.40),
            self._row("U1", "CCK", "2025-02-03", 0.00, 0.20),
            self._row("U2", "CCK", "2025-02-03", 0.00, 0.25),
        ])
        pairwise = research.build_pairwise_differences(frame, ["U1", "U2"])
        self.assertEqual(len(pairwise), 2)
        apd = pairwise.loc[pairwise["candidate"] == "APD"].iloc[0]
        self.assertAlmostEqual(float(apd["delta_delta_log_capital"]), 0.05)
        self.assertAlmostEqual(float(apd["delta__corr_20_to_universe"]), 0.30)
        self.assertTrue(bool(apd["sign_changed"]))
        self.assertTrue(bool(apd["strict_sign_flip"]))
        self.assertFalse(bool(apd["activation_changed"]))
        cck = pairwise.loc[pairwise["candidate"] == "CCK"].iloc[0]
        self.assertFalse(bool(cck["sign_changed"]))

    def test_processor_is_standalone_from_previous_research_versions(self) -> None:
        source = (SCRIPT_ROOT / "research_contextual_marginal_signature_v106.py").read_text(encoding="utf-8")
        self.assertNotIn("research_contextual_marginal_signature_v103", source)
        self.assertNotIn("research_contextual_marginal_signature_v104", source)
        self.assertNotIn("research_contextual_marginal_signature_v105", source)
        self.assertIn("delta_delta_log_capital", source)
        self.assertIn("select_shared_executable_decision_sessions", source)


if __name__ == "__main__":
    unittest.main()
