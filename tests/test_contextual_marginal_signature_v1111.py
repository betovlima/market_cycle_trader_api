from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1111 as research  # noqa: E402


class ContextualMarginalSignatureV1111Tests(unittest.TestCase):
    def _result(self) -> types.SimpleNamespace:
        predictions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
                "selected_asset": ["A", "B"],
                "strategy_equity": [100.0, 101.0],
            }
        ).set_index("timestamp")
        trades = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-05"], utc=True),
                "symbol": ["B"],
                "action": ["BUY"],
            }
        )
        return types.SimpleNamespace(
            backend="lightgbm_utility",
            predictions=predictions,
            trades=trades,
            metrics={"strategy_ending_capital": 101.0},
            summary="summary",
        )

    def test_compact_capture_does_not_create_backend_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "traces" / "2026-01-02" / "Original23_MinusADM_ADI" / "BASELINE"
            captured = research._compact_save_capture(base_path, [self._result()])

            self.assertIn("lightgbm_utility_00", captured)
            self.assertFalse((base_path / "lightgbm_utility_00").exists())
            self.assertTrue((base_path / "b00_p.csv").exists())
            self.assertTrue((base_path / "b00_t.csv").exists())
            self.assertTrue((base_path / "b00_m.json").exists())
            self.assertTrue((base_path / "b00_s.txt").exists())
            self.assertTrue((base_path / "b00_c.json").exists())
            self.assertTrue((base_path / "trace_index.json").exists())

    def test_compact_capture_preserves_prediction_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "capture"
            captured = research._compact_save_capture(base_path, [self._result()])
            columns = list(captured["lightgbm_utility_00"]["predictions"].columns)
            self.assertIn("timestamp", columns)
            self.assertIn("selected_asset", columns)
            self.assertIn("strategy_equity", columns)

    def test_script_version_is_patch_release(self) -> None:
        self.assertEqual(research.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.11.1")


if __name__ == "__main__":
    unittest.main()
