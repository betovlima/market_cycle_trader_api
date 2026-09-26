"""Guard the XNYS cutoff handoff from MCT job to frozen TCC v1.0.6 policy."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from market_cycle_trader_api.engine import tcc_v106_reference_backtest as reference
from market_cycle_trader_api.tcc_v106_reference.config import (
    ASSETS, CONFIG as SCIENTIFIC_TCC,
)


def _eligible_frames():
    # Frame values are not read while constructing the two execution configs.
    return {
        symbol: pd.DataFrame()
        for symbol in ASSETS
        if symbol != "DOC"
    }


class ReferenceCutoffPropagationTests(unittest.TestCase):
    def test_current_mct_cutoff_propagates_to_both_variants(self):
        frames = _eligible_frames()
        control, soft = reference._tcc_variant_configs(
            frames, analysis_end_date="2026-09-25",
        )
        self.assertEqual(control.analysis_end_date, "2026-09-25")
        self.assertEqual(soft.analysis_end_date, "2026-09-25")
        self.assertEqual(len(control.assets), 55)
        self.assertEqual(len(soft.assets), 55)
        self.assertEqual(control.assets, soft.assets)
        self.assertEqual(control.rotation_target_horizons, SCIENTIFIC_TCC.rotation_target_horizons)
        self.assertEqual(
            control.research_model_settings["soft_horizon_consensus"],
            {"enabled": False},
        )
        self.assertEqual(
            soft.research_model_settings["soft_horizon_consensus"],
            {"enabled": True, "penalty_strength": 1.0},
        )
        self.assertEqual(SCIENTIFIC_TCC.analysis_end_date, "2026-09-17")

    def test_frozen_snapshot_remains_pinned_without_mutating_base(self):
        control, soft = reference._tcc_variant_configs(
            _eligible_frames(), analysis_end_date="2026-09-17",
        )
        self.assertEqual(control.analysis_end_date, "2026-09-17")
        self.assertEqual(soft.analysis_end_date, "2026-09-17")
        self.assertEqual(SCIENTIFIC_TCC.analysis_end_date, "2026-09-17")

    def test_cutoff_is_required_not_implicitly_inherited_from_frozen_config(self):
        with self.assertRaisesRegex(ValueError, "resolved analysis_end_date"):
            reference._tcc_variant_configs(
                _eligible_frames(), analysis_end_date="",
            )

    def test_job_uses_resolved_cutoff_in_both_runs_and_result_metrics(self):
        fake_config = SimpleNamespace(
            analysis_end_date="2026-09-25",
            end_date=None,
            mongo_write_batch_size=1000,
            tcc_reference_input_source="mct_current",
        )
        observed = []
        def fake_run_variant(**kwargs):
            observed.append((kwargs["label"], kwargs["config"].analysis_end_date))
            return SimpleNamespace(
                backend=kwargs["backend"],
                metrics={
                    "strategy_ending_capital": 10000.0,
                    "strategy_return": 0.0,
                },
                summary="Test",
                predictions=pd.DataFrame(),
                trades=pd.DataFrame(),
            )
        with (
            patch.object(reference, "_load_mct_market_frames", return_value=(
                _eligible_frames(), [{"symbol": "DOC", "reason": "structural_identity_change"}], fake_config
            )),
            patch.object(reference, "build_reproducibility_manifest", return_value={
                "market_data_signature_sha256": "test-hash"
            }),
            patch.object(reference, "_run_variant", side_effect=fake_run_variant),
            patch.object(reference, "_fold_rows", return_value=[]),
            patch.object(reference, "replace_run_result") as persisted,
            patch.object(reference, "emit_progress"),
        ):
            comparison, returned = reference.run_reference_job(
                "job-test", fake_config, db=object(),
            )
        self.assertIs(returned, fake_config)
        self.assertEqual(observed, [
            ("CONTROL", "2026-09-25"),
            ("SOFT_HORIZON_CONSENSUS", "2026-09-25"),
        ])
        self.assertEqual(persisted.call_count, 2)
        for call in persisted.call_args_list:
            metrics = call.kwargs["metrics"]
            self.assertEqual(metrics["requested_analysis_cutoff"], "2026-09-25")
            self.assertEqual(metrics["effective_analysis_cutoff"], "2026-09-25")
            self.assertEqual(metrics["input_source"], "mct_current")
        self.assertEqual(len(comparison), 2)
        for row in comparison:
            self.assertEqual(row["effective_analysis_cutoff"], "2026-09-25")


if __name__ == "__main__":
    unittest.main()
