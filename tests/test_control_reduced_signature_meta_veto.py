"""Safeguards for v10.8.58 worst-regime temporal Meta-Veto."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import ValidationError

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import (
    StartControlReducedMetaVetoRequest,
)
from market_cycle_trader_api.engine.control_reduced_rollout_signature import (
    REDUCED_FEATURES,
)
from market_cycle_trader_api.engine.control_reduced_signature_meta_veto import (
    MODEL_C,
    MIN_CALIBRATION_SAMPLES,
    MIN_CALIBRATION_BALANCED_ACCURACY,
    MIN_CALIBRATION_ROC_AUC,
    VETO_PROBABILITY_MAX,
    train_fold_models,
)
from market_cycle_trader_api.services import (
    control_shadow_reduced_meta_veto_jobs as jobs,
)
from market_cycle_trader_api.engine.control_expected_advantage_meta_veto import (
    RIDGE_ALPHA,
    MIN_CALIBRATION_SIGN_BALANCED_ACCURACY,
    MIN_CALIBRATION_SPEARMAN,
)
from market_cycle_trader_api.engine import control_consensus_meta_veto as consensus_engine
from market_cycle_trader_api.engine import control_temporal_ensemble_meta_veto as temporal_engine

REDUCED="control-reduced-b9404f6e1f694e38"
SIGNATURE="control-signature-5dedf5eee0f94e8a"
ROLLOUT="control-rollout-03253d310ef445b2"
SNAP="control-shadow-870fb66e1bdc4fd0"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


class ReducedSignatureMetaVetoTests(TestCase):
    def test_frozen_contract(self):
        self.assertEqual(MODEL_C,0.25)
        self.assertEqual(MIN_CALIBRATION_SAMPLES,20)
        self.assertEqual(MIN_CALIBRATION_BALANCED_ACCURACY,0.52)
        self.assertEqual(MIN_CALIBRATION_ROC_AUC,0.52)
        self.assertEqual(VETO_PROBABILITY_MAX,0.35)
        self.assertEqual(len(REDUCED_FEATURES),9)

    def test_expected_advantage_contract_is_fixed(self):
        self.assertEqual(RIDGE_ALPHA,1.0)
        self.assertEqual(MIN_CALIBRATION_SIGN_BALANCED_ACCURACY,0.52)
        self.assertEqual(MIN_CALIBRATION_SPEARMAN,0.0)

    def test_expected_advantage_uses_zero_as_economic_break_even(self):
        import inspect
        from market_cycle_trader_api.engine import control_expected_advantage_meta_veto as engine

        source=inspect.getsource(engine.run_expected_advantage_meta_veto_pair)
        self.assertIn("if predicted_delta <= 0.0:",source)
        self.assertIn("VETO_NONPOSITIVE_EXPECTED_ADVANTAGE",source)

    def test_temporal_ensemble_supports_fixed_worst_regime_aggregation(self):
        import inspect

        source=inspect.getsource(temporal_engine.run_temporal_ensemble_meta_veto_pair)
        self.assertIn('aggregation not in {"mean", "min"}',source)
        self.assertIn("np.min(component_probabilities)",source)
        self.assertIn("WORST_REGIME_VETO",source)

    def test_temporal_ensemble_contract_preserves_threshold_and_mean(self):
        import inspect

        source=inspect.getsource(temporal_engine.run_temporal_ensemble_meta_veto_pair)
        self.assertIn("np.mean(component_probabilities)",source)
        self.assertIn("probability <= VETO_PROBABILITY_MAX",source)
        self.assertIn("TEMPORAL_ENSEMBLE_VETO",source)

    def test_v1058_service_exposes_runner_identity(self):
        import inspect

        source=inspect.getsource(jobs)
        self.assertIn('EXPECTED_API_VERSION = "10.8.58"',source)
        self.assertIn('RESEARCH_RUNNER = "worst-regime-temporal-veto-v1058"',source)

    def test_consensus_cannot_create_new_veto(self):
        import inspect

        source=inspect.getsource(consensus_engine.run_consensus_meta_veto_pair)
        self.assertIn("classifier_veto_candidate",source)
        self.assertIn("elif regressor.enabled:",source)
        self.assertIn("REGRESSION_CANCELLED_CLASSIFIER_VETO",source)
        self.assertIn("CLASSIFIER_VETO_REGRESSOR_UNAVAILABLE",source)
        self.assertNotIn("VETO_NONPOSITIVE_EXPECTED_ADVANTAGE",source)

    def test_liquidity_overlay_is_explicitly_enabled_in_replay(self):
        import inspect
        from market_cycle_trader_api.engine import control_reduced_signature_meta_veto as engine

        source=inspect.getsource(engine.run_reduced_signature_meta_veto_pair)
        self.assertIn('{"enabled": True, "audit": {}}',source)

    def test_service_exposes_runner_and_runtime_guard(self):
        import inspect

        source=inspect.getsource(jobs)
        self.assertIn('EXPECTED_API_VERSION = "10.8.57"',source)
        self.assertIn('RESEARCH_RUNNER = "temporal-ensemble-meta-veto-v1057"',source)
        self.assertIn('"api_version": record.get("api_version")',source)
        self.assertIn('"research_runner": record.get("research_runner")',source)

    def test_endpoint_rejects_tuning_or_order_fields(self):
        body={
            "confirm":"RESEARCH_CONTROL_REDUCED_META_VETO_NO_ORDERS",
            "source_reduced_job_id":REDUCED,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlReducedMetaVetoRequest.model_validate(body)
        self.assertEqual(parsed.source_reduced_job_id,REDUCED)
        for extra in (
            {"C":1.0},
            {"threshold":0.40},
            {"minimum_auc":0.50},
            {"features":["incumbent__return_20"]},
            {"grid_search":True},
            {"ridge_alpha":0.1},
            {"regression_threshold":-0.01},
            {"send_order":True},
            {"winner_strategy_id":"x"},
        ):
            with self.assertRaises(ValidationError):
                StartControlReducedMetaVetoRequest.model_validate({**body,**extra})

    def test_fold_one_is_disabled_and_only_fully_mature_prior_rows_are_eligible(self):
        rows=[]
        base=pd.Timestamp("2020-01-01",tz="UTC")
        for i in range(30):
            fold=1 if i<10 else (2 if i<20 else 3)
            decision=base+pd.Timedelta(days=i)
            rollout_end=decision+pd.Timedelta(days=2)
            if i in (18,19):
                rollout_end=base+pd.Timedelta(days=25)
            row={
                "decision_date":decision,
                "rollout_end_date":rollout_end,
                "source_fold_id":fold,
                "rotate_better":i%2,
            }
            row.update({feature:float(i+j) for j,feature in enumerate(REDUCED_FEATURES)})
            rows.append(row)
        dataset=pd.DataFrame(rows)
        folds=[
            {"fold_id":1,"test_start":base},
            {"fold_id":2,"test_start":base+pd.Timedelta(days=10)},
            {"fold_id":3,"test_start":base+pd.Timedelta(days=20)},
        ]
        models,reports=train_fold_models(dataset,folds)
        self.assertFalse(models[1].enabled)
        self.assertEqual(reports[0]["disable_reason"],"NO_PRIOR_OOS_FOLD")
        self.assertEqual(reports[1]["eligible_prior_rows"],8)
        self.assertEqual(reports[2]["eligible_prior_rows"],18)

    def test_one_shot_veto_forces_next_policy_call_back_to_control(self):
        import inspect
        from market_cycle_trader_api.engine import control_reduced_signature_meta_veto as engine

        source=inspect.getsource(engine.run_reduced_signature_meta_veto_pair)
        self.assertIn("force_control_next = [False]",source)
        self.assertIn("if force_control_next[0]:",source)
        self.assertIn('reason = "CONTROL_AFTER_ONE_SHOT_VETO"',source)
        self.assertIn("force_control_next[0] = True",source)

    def test_unconfirmed_v1050_is_rejected_before_thread(self):
        db=MagicMock()
        db[jobs.REDUCED_COLLECTION].find_one.return_value={
            "_id":REDUCED,
            "status":"completed",
            "error":None,
            "source_job_id":SNAP,
            "source_rollout_job_id":ROLLOUT,
            "source_signature_job_id":SIGNATURE,
            "snapshot_sha256":SHA,
            "result":{
                "research_kind":"control_reduced_rollout_signature_confirmation",
                "reduced_job_id":REDUCED,
                "source_signature_job_id":SIGNATURE,
                "source_rollout_job_id":ROLLOUT,
                "source_job_id":SNAP,
                "source_snapshot_sha256":SHA,
                "source_unchanged":True,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "reduced_signature":{"reduced_signature_confirmed":False},
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.MetaVetoInvalid):
                jobs.start_reduced_signature_meta_veto_research(
                    db,
                    source_reduced_job_id=REDUCED,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__=="__main__":
    import unittest
    unittest.main()
