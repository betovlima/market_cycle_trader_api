"""Safeguards for v10.8.50 reduced rollout signature confirmation."""
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
    StartControlReducedSignatureRequest,
)
from market_cycle_trader_api.engine.control_reduced_rollout_signature import (
    REDUCED_FEATURES,
    evaluate_reduced_signature,
)
from market_cycle_trader_api.services import (
    control_shadow_reduced_signature_jobs as jobs,
)

SIGNATURE="control-signature-5dedf5eee0f94e8a"
ROLLOUT="control-rollout-03253d310ef445b2"
SNAP="control-shadow-870fb66e1bdc4fd0"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


class ReducedRolloutSignatureTests(TestCase):
    def test_feature_contract_is_frozen_to_nine_features(self):
        self.assertEqual(len(REDUCED_FEATURES),9)
        self.assertEqual(REDUCED_FEATURES,(
            "incumbent__return_20",
            "incumbent__return_60",
            "incumbent__ema_distance_20",
            "incumbent__ema_distance_50",
            "incumbent__rsi_14",
            "incumbent__channel_position_50",
            "incumbent_capacity_equity_ratio",
            "state_shares",
            "candidate__channel_position_50",
        ))

    def test_endpoint_rejects_tuning_policy_or_feature_override(self):
        body={
            "confirm":"RESEARCH_CONTROL_REDUCED_ROLLOUT_SIGNATURE_NO_ORDERS",
            "source_signature_job_id":SIGNATURE,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlReducedSignatureRequest.model_validate(body)
        self.assertEqual(parsed.source_signature_job_id,SIGNATURE)
        for extra in (
            {"features":["return_5"]},
            {"C":10.0},
            {"threshold":.40},
            {"grid_search":True},
            {"create_policy":True},
            {"send_order":True},
        ):
            with self.assertRaises(ValidationError):
                StartControlReducedSignatureRequest.model_validate({**body,**extra})

    def test_exact_321_rows_are_required(self):
        frame=pd.DataFrame({
            "source_fold_id":[1,2,3],
            "rotate_better":[0,1,0],
            **{feature:[0.1,0.2,0.3] for feature in REDUCED_FEATURES},
        })
        with self.assertRaisesRegex(ValueError,"exact 321-event"):
            evaluate_reduced_signature(frame)

    def test_unverified_or_signal_negative_v1049_is_rejected_before_thread(self):
        db=MagicMock()
        db[jobs.SIGNATURE_COLLECTION].find_one.return_value={
            "_id":SIGNATURE,
            "status":"completed",
            "error":None,
            "source_job_id":SNAP,
            "source_rollout_job_id":ROLLOUT,
            "snapshot_sha256":SHA,
            "result":{
                "research_kind":"control_rollout_decision_signature_diagnostic",
                "signature_job_id":SIGNATURE,
                "source_rollout_job_id":ROLLOUT,
                "source_job_id":SNAP,
                "source_snapshot_sha256":SHA,
                "source_unchanged":True,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "diagnostic_summary":{"predictive_signal_detected":False},
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.ReducedSignatureInvalid):
                jobs.start_reduced_rollout_signature_research(
                    db,
                    source_signature_job_id=SIGNATURE,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__=="__main__":
    import unittest
    unittest.main()
