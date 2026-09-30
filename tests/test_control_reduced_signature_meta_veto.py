"""Safeguards for v10.8.51 reduced-signature meta-veto research."""
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

    def test_liquidity_overlay_is_explicitly_enabled_in_replay(self):
        import inspect
        from market_cycle_trader_api.engine import control_reduced_signature_meta_veto as engine

        source=inspect.getsource(engine.run_reduced_signature_meta_veto_pair)
        self.assertIn('{"enabled": True, "audit": {}}',source)

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
            {"send_order":True},
            {"winner_strategy_id":"x"},
        ):
            with self.assertRaises(ValidationError):
                StartControlReducedMetaVetoRequest.model_validate({**body,**extra})

    def test_fold_one_is_always_disabled_and_future_fold_rows_are_excluded(self):
        rows=[]
        for i in range(30):
            fold=1 if i<10 else (2 if i<20 else 3)
            row={
                "decision_date":pd.Timestamp("2020-01-01",tz="UTC")+pd.Timedelta(days=i),
                "source_fold_id":fold,
                "rotate_better":i%2,
            }
            row.update({feature:float(i+j) for j,feature in enumerate(REDUCED_FEATURES)})
            rows.append(row)
        dataset=pd.DataFrame(rows)
        folds=[
            {"fold_id":1},
            {"fold_id":2},
            {"fold_id":3},
        ]
        models,reports=train_fold_models(dataset,folds)
        self.assertFalse(models[1].enabled)
        self.assertEqual(reports[0]["disable_reason"],"NO_PRIOR_OOS_FOLD")
        self.assertEqual(reports[1]["eligible_prior_rows"],10)
        self.assertEqual(reports[2]["eligible_prior_rows"],20)

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
