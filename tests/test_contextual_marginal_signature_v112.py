from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v112 as research  # noqa: E402


@dataclass
class Result:
    predictions: pd.DataFrame


class Config:
    def __init__(self, values: dict | None = None) -> None:
        self.values = dict(values or {})

    def model_copy(self, *, update: dict):
        merged = dict(self.values)
        merged.update(update)
        return Config(merged)


class ContextualMarginalSignatureV112Tests(unittest.TestCase):
    def setUp(self) -> None:
        research._BASELINE_MARGIN_BY_KEY.clear()
        research._CURRENT_CONTEXT = None

    def test_extract_margin_requires_one_constant_pair(self) -> None:
        result = Result(
            predictions=pd.DataFrame(
                {
                    "calibrated_switch_margin": [0.005, 0.005],
                    "effective_switch_margin": [0.005, 0.005],
                }
            )
        )
        calibrated, effective = research._extract_margin([result])
        self.assertEqual(calibrated, 0.005)
        self.assertEqual(effective, 0.005)

        bad = Result(
            predictions=pd.DataFrame(
                {
                    "calibrated_switch_margin": [0.005, 0.01],
                    "effective_switch_margin": [0.005, 0.01],
                }
            )
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            research._extract_margin([bad])

    def test_context_key_is_stable(self) -> None:
        key = research._context_key(
            pd.Timestamp("2026-01-02", tz="UTC"),
            ["nvda", "aapl", "adm"],
        )
        self.assertEqual(key, ("2026-01-02", ("NVDA", "AAPL", "ADM")))

    def test_challenger_config_uses_baseline_calibrated_margin(self) -> None:
        key = ("2026-01-02", ("NVDA", "AAPL"))
        research._BASELINE_MARGIN_BY_KEY[key] = {
            "calibrated_switch_margin": 0.005,
            "effective_switch_margin": 0.005,
        }
        captured = {}
        original = research._ORIGINAL_WINDOW_REQUEST

        def fake_window_request(**kwargs):
            captured.update(kwargs)
            return kwargs

        research._ORIGINAL_WINDOW_REQUEST = fake_window_request
        try:
            result = research._window_request_frozen(
                db=None,
                config=Config({"rotation_switch_margin_candidates": [0.0, 0.005, 0.01]}),
                strategy_id="strategy",
                assets=["NVDA", "AAPL", "APD"],
                reference_assets=["NVDA", "AAPL"],
                candidate_assets=["APD"],
                decision_session=pd.Timestamp("2026-01-02", tz="UTC"),
                horizon_end=pd.Timestamp("2026-03-02", tz="UTC"),
            )
        finally:
            research._ORIGINAL_WINDOW_REQUEST = original

        self.assertEqual(result["config"].values["rotation_switch_margin_candidates"], [0.005])
        self.assertEqual(captured["candidate_assets"], ["APD"])
        self.assertFalse(bool(research._CURRENT_CONTEXT["is_baseline"]))


if __name__ == "__main__":
    unittest.main()
