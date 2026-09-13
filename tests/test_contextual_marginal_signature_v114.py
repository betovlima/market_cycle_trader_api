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

import research_contextual_marginal_signature_v114 as runner  # noqa: E402
import research_contextual_marginal_signature_v114_analysis as analysis  # noqa: E402


class ContextualMarginalSignatureV114Tests(unittest.TestCase):
    def test_campaign_identity_is_frozen_margin_extension(self) -> None:
        self.assertEqual(runner.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.14")
        self.assertEqual(runner.EXPERIMENT_NAME, "contextual_marginal_signature_direct_effect_prevalence")

    def test_analysis_readiness_requires_cross_candidate_and_cross_date_effects(self) -> None:
        rows = []
        candidates = ["A", "B", "C"]
        dates = ["2024-01-02", "2025-01-02", "2026-01-02"]
        universes = ["U0", "U1"]
        for date in dates:
            for candidate in candidates:
                for universe in universes:
                    rows.append({
                        "decision_date": date,
                        "universe_name": universe,
                        "candidate": candidate,
                        "delta_log_capital": 0.01,
                    })
        # 18 non-zero rows satisfy the explicit readiness rule.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_dir = root / "frozen"
            output_dir = root / "research_output" / "contextual_marginal_signature_v114_test"
            frozen_dir.mkdir()
            pd.DataFrame(rows).to_csv(frozen_dir / "trace_aggregate_dataset.csv", index=False)
            (frozen_dir / "trace_manifest.json").write_text(
                json.dumps({
                    "script_version": "contextual-marginal-signature-v1.0.14",
                    "market_snapshot_hash": "market",
                    "strategy_configuration_hash": "strategy",
                }),
                encoding="utf-8",
            )
            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "analysis",
                    "--frozen-dir", str(frozen_dir),
                    "--output-dir", str(output_dir),
                    "--fresh-run",
                ]
                self.assertEqual(analysis.main(), 0)
            finally:
                sys.argv = original_argv
            summary = json.loads((output_dir / "direct_effect_prevalence_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["predictive_signature_dataset_ready"])
            self.assertEqual(summary["nonzero_direct_candidates"], 3)
            self.assertEqual(summary["nonzero_direct_dates"], 3)


if __name__ == "__main__":
    unittest.main()
