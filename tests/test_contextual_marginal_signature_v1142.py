from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1142 as runner  # noqa: E402
import research_contextual_marginal_signature_v1142_analysis as analysis  # noqa: E402
import research_contextual_marginal_signature_v1142_snapshot as snapshot  # noqa: E402


class ContextualMarginalSignatureV1142Tests(unittest.TestCase):
    def test_campaign_identity(self) -> None:
        self.assertEqual(runner.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14.2")
        self.assertEqual(analysis.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14.2")
        self.assertEqual(snapshot.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14.2")

    def test_campaign_collects_universe_and_candidate_symbols(self) -> None:
        universe_spec = PROJECT_ROOT / "scripts" / "research_contextual_marginal_signature_v109_universe_spec.json"
        cases_spec = PROJECT_ROOT / "scripts" / "research_contextual_marginal_signature_v114_cases.json"
        symbols, universes, cases = snapshot._campaign(universe_spec, cases_spec)
        for symbol in ["XSD", "MKSI", "GKOS", "CLMT", "CORT", "APD", "CCK", "ADM"]:
            self.assertIn(symbol, symbols)
        self.assertEqual([item["name"] for item in universes], ["Original25", "Original24_MinusADM"])
        self.assertEqual(len(cases), 6)


if __name__ == "__main__":
    unittest.main()
