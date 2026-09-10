"""Exercise the research runner with real models and deterministic synthetic bars.

Only the Mongo input boundary is replaced. This tests execution and artifact
contracts; it makes no statement about the profitability of real market data.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import market_cycle_trader_api.services  # Import service registrations before engine modules.
import research_pooled_candidate_episode_advantage as research
from market_cycle_trader_api.services.model_research import _LIGHTGBM_DEFAULTS, model_execution_snapshot
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (
    LABEL_AVAILABLE_COLUMN, OBSERVED_TARGET_COLUMN, TARGET_COLUMN,
)


class PCEAPipelineIntegrationTests(unittest.TestCase):
    def test_complete_runner_with_real_lightgbm_replays_and_bayesian_crossfit(self):
        dates = research.common._expected_sessions(
            pd.Timestamp("2017-01-03"), pd.Timestamp("2021-12-31")
        )[:1040].tz_localize("UTC")
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        rng = np.random.default_rng(912)
        bars = {}
        for symbol in symbols:
            closes = 100 * np.exp(np.cumsum(rng.normal(.001, .02, len(dates))))
            opens = np.r_[closes[0], closes[:-1]] * np.exp(rng.normal(0, .005, len(dates)))
            bars[symbol] = pd.DataFrame({
                "open": opens, "close": closes,
                "high": np.maximum(opens, closes) * 1.01,
                "low": np.minimum(opens, closes) * .99,
                "volume": rng.integers(100000, 1000000, len(dates)),
            }, index=dates)
        values = json.loads((ROOT / "src/market_cycle_trader_api/parameterizations/winner-v1.13.2.json").read_text())
        values.update(
            assets=symbols[:2], start_date=str(dates[0].date()), end_date=str(dates[-1].date()),
            rotation_minimum_training_rows=300, rotation_walk_forward_calibration_days=40,
            rotation_walk_forward_test_days=80, rotation_walk_forward_min_test_days=20,
            rotation_purge_days=10, rotation_horizon_days=10, rotation_target_horizons=[5, 10],
            rotation_target_horizon_weights=[.5, .5], rotation_switch_margin=0.,
            rotation_switch_margin_candidates=[0.], rotation_min_expected_edge=0.,
            xgb_n_jobs=1, deterministic_execution=True,
        )
        settings = {"lightgbm": {
            **_LIGHTGBM_DEFAULTS, "n_estimators": 10, "n_jobs": 1,
            "min_child_samples": 5, "min_child_weight": .001,
        }}
        strategy = {
            "strategy_sequence": 10, "revision": 1, "configuration": values,
            "configuration_hash": "synthetic-test-configuration",
            "research_model_snapshot": model_execution_snapshot("lightgbm_utility", settings),
        }
        mongo = MagicMock()
        mongo.__getitem__.return_value.__getitem__.return_value.find_one.return_value = strategy
        mongo.__getitem__.return_value.__getitem__.return_value.distinct.return_value = symbols

        def load_frames(collection, assets, identity, start, end):
            cutoff = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("UTC")
            return {s: bars[s].loc[bars[s].index < cutoff].copy() for s in assets}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / ("experiment_" * 12) / ("selection_" * 12)
            arguments = [
                "research_pooled_candidate_episode_advantage.py", "--strategy-sequence", "10",
                "--history-start", str(dates[0].date()), "--snapshot-end", str(dates[-1].date()),
                "--validation-sessions", "80", "--mongo-uri", "mongodb://localhost:27017",
                "--database", "synthetic_test_only", "--output-dir", str(output),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(research, "MongoClient", return_value=mongo),
                patch.object(research.common, "load_project_environment"),
                patch.object(research.timing, "_load_frames_allow_incomplete", side_effect=load_frames),
            ):
                self.assertEqual(research.main(), 0)
            result = research.file_io.read_json(output / "pcea_result.json")
            manifest = research.file_io.read_json(output / "experiment_manifest.json")
            self.assertEqual(result["schema_version"], 2)
            self.assertFalse(result["full_stateful_overlay_backtest_run"])
            self.assertFalse(result["holdout_is_untouched"])
            self.assertEqual(result["candidate_pool_count"], 2)
            self.assertEqual(result["validation_sessions"], 80)
            self.assertEqual(len(manifest["selection_market_data_sha256_by_asset"]), 4)
            self.assertIsNotNone(manifest["code_identity"]["source_sha256"])
            for stage in result["prevalidation_crossfit_stages"]:
                self.assertLess(
                    pd.Timestamp(stage["training_last_label_available_at"]),
                    pd.Timestamp(stage["training_cutoff_exclusive"]),
                )
            for prefix in ("pcea_prevalidation", "pcea_holdout"):
                episodes = research.file_io.read_csv(output / f"{prefix}_all_episodes.csv")
                samples = research.file_io.read_csv(output / f"{prefix}_episode_samples.csv")
                self.assertEqual(len(samples), len(episodes))
                open_samples = samples.loc[samples.right_censored]
                self.assertTrue(open_samples[TARGET_COLUMN].isna().all())
                self.assertTrue(open_samples[LABEL_AVAILABLE_COLUMN].isna().all())
                self.assertTrue(np.isfinite(pd.to_numeric(open_samples[OBSERVED_TARGET_COLUMN])).all())
            print(json.dumps({
                "validation_kind": "synthetic execution only; no real market performance claim",
                "raw_sessions": len(dates), "assets": len(symbols),
                "prevalidation_folds": result["prevalidation_fold_count"],
                "training_rows": result["holdout_model_diagnostics"]["training_rows"],
                "holdout_episode_diagnostics": result["holdout_episode_diagnostics"],
                "holdout_summary": result["holdout_summary"],
            }, indent=2))
            research.file_io.remove_tree(output.parent)


if __name__ == "__main__":
    unittest.main()
