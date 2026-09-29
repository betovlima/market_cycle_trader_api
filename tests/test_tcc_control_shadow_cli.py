"""CLI smoke test: frozen SHA audit and local-only preview output."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPEC = importlib.util.spec_from_file_location(
    "control_shadow_preview_cli", ROOT / "scripts" / "control_shadow_preview.py"
)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ControlShadowCliTests(TestCase):
    def test_cli_never_prepares_orders_and_writes_only_requested_local_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "shadow.json"
            mock_preview = {
                "status": "shadow_only",
                "order_eligible": False,
                "order_submission": "never",
                "decision_date": "2026-09-17",
                "current_asset": "CASH",
                "target_asset": "AAPL",
                "effective_switch_margin": 0.0005,
                "input_audit": {},
            }
            with (
                patch.object(cli, "ASSETS", ("AAPL", "DOC")),
                patch.object(cli, "validate_frozen_tcc_main",
                             return_value={"assets": ["AAPL", "DOC"]}) as validate,
                patch.object(cli, "load_frozen_tcc_main_symbol",
                             side_effect=[object(), cli.StructuralResearchAssetExclusion({"symbol": "DOC", "action_type": "merger", "acquirer_symbol": "PEAK"})]) as load,
                patch.object(cli, "build_control_shadow_decision",
                             return_value=mock_preview) as preview,
            ):
                result = cli.run(
                    frozen_root=root,
                    completed_session="2026-09-17",
                    current_asset="CASH",
                    holding_sessions=0,
                    output=output,
                )
            validate.assert_called_once()
            self.assertEqual(load.call_count, 2)
            preview.assert_called_once()
            self.assertTrue(output.exists())
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "shadow_only")
            self.assertIs(data["order_eligible"], False)
            self.assertEqual(data["order_submission"], "never")
            self.assertEqual(
                data["input_audit"]["structural_exclusions"],
                [{"symbol": "DOC", "reason": "DOC: structural identity change (merger) to PEAK; excluded from the research universe."}],
            )
            self.assertEqual(data["input_audit"]["eligible_assets"], 1)
            self.assertEqual(result["input_audit"]["source_kind"],
                             "frozen_scientific_reference_not_current_alpaca")


if __name__ == "__main__":
    import unittest
    unittest.main()
