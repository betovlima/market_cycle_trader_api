from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1143 as runner  # noqa: E402
import research_contextual_marginal_signature_v1143_analysis as analysis  # noqa: E402
import research_contextual_marginal_signature_v1143_snapshot as snapshot  # noqa: E402


class DummyMongoRepository:
    MONGO_URI = ""
    MONGO_DATABASE = ""


class ContextualMarginalSignatureV1143Tests(unittest.TestCase):
    def test_campaign_identity(self) -> None:
        expected = "contextual-marginal-signature-v1.0.14.3"
        self.assertEqual(runner.SCRIPT_VERSION, expected)
        self.assertEqual(analysis.SCRIPT_VERSION, expected)
        self.assertEqual(snapshot.SCRIPT_VERSION, expected)

    def test_runtime_mongo_settings_are_resolved_after_environment_load(self) -> None:
        args = argparse.Namespace(mongo_uri=None, database=None)
        module = DummyMongoRepository()
        with patch.dict(
            os.environ,
            {"MONGO_URL": "mongodb://localhost:27017", "MONGO_DATABASE": "market_cycle_trader"},
            clear=False,
        ):
            uri, database = snapshot._runtime_mongo_settings(args, module)
        self.assertEqual(uri, "mongodb://localhost:27017")
        self.assertEqual(database, "market_cycle_trader")
        self.assertEqual(module.MONGO_URI, uri)
        self.assertEqual(module.MONGO_DATABASE, database)

    def test_explicit_mongo_arguments_take_precedence(self) -> None:
        args = argparse.Namespace(
            mongo_uri="mongodb://127.0.0.1:27018",
            database="research_db",
        )
        module = DummyMongoRepository()
        with patch.dict(
            os.environ,
            {"MONGO_URL": "mongodb://localhost:27017", "MONGO_DATABASE": "default_db"},
            clear=False,
        ):
            uri, database = snapshot._runtime_mongo_settings(args, module)
        self.assertEqual(uri, "mongodb://127.0.0.1:27018")
        self.assertEqual(database, "research_db")

    def test_campaign_collects_required_symbols(self) -> None:
        universe_spec = PROJECT_ROOT / "scripts" / "research_contextual_marginal_signature_v109_universe_spec.json"
        cases_spec = PROJECT_ROOT / "scripts" / "research_contextual_marginal_signature_v114_cases.json"
        symbols, universes, cases = snapshot._campaign(universe_spec, cases_spec)
        for symbol in ["XSD", "MKSI", "GKOS", "CLMT", "CORT", "APD", "CCK", "ADM"]:
            self.assertIn(symbol, symbols)
        self.assertEqual([item["name"] for item in universes], ["Original25", "Original24_MinusADM"])
        self.assertEqual(len(cases), 6)


if __name__ == "__main__":
    unittest.main()
