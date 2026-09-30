"""Safeguards for v10.8.49 rollout decision signature diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import (
    StartControlRolloutSignatureRequest,
)
from market_cycle_trader_api.services import (
    control_shadow_rollout_signature_jobs as jobs,
)

ROLLOUT="control-rollout-03253d310ef445b2"
SNAP="control-shadow-870fb66e1bdc4fd0"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


class RolloutDecisionSignatureTests(TestCase):
    def test_endpoint_has_no_model_or_tuning_parameters(self):
        body={
            "confirm":"RESEARCH_CONTROL_ROLLOUT_DECISION_SIGNATURE_NO_ORDERS",
            "source_rollout_job_id":ROLLOUT,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlRolloutSignatureRequest.model_validate(body)
        self.assertEqual(parsed.source_rollout_job_id,ROLLOUT)
        for extra in (
            {"model":"xgboost"},
            {"max_depth":10},
            {"threshold":.40},
            {"grid_search":True},
            {"send_order":True},
            {"create_policy":True},
        ):
            with self.assertRaises(ValidationError):
                StartControlRolloutSignatureRequest.model_validate({**body,**extra})

    def test_unverified_rollout_source_is_rejected_before_thread(self):
        db=MagicMock()
        db[jobs.ROLLOUT_COLLECTION].find_one.return_value={
            "_id":ROLLOUT,
            "status":"completed",
            "error":None,
            "source_job_id":SNAP,
            "snapshot_sha256":SHA,
            "result":{
                "research_kind":"control_policy_rollout_advantage_meta_veto",
                "rollout_job_id":ROLLOUT,
                "source_job_id":SNAP,
                "source_snapshot_sha256":SHA,
                "source_unchanged":True,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "v1044_parity":{"status":"failed"},
                "paired_rollout_count":321,
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.SignatureInvalid):
                jobs.start_rollout_signature_research(
                    db,
                    source_rollout_job_id=ROLLOUT,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__=="__main__":
    import unittest
    unittest.main()
