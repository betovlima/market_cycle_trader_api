"""Endpoint and fail-closed service tests for v10.8.47."""
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
    StartControlCounterfactualAdvantageRequest,
)
from market_cycle_trader_api.services import (
    control_shadow_counterfactual_advantage_jobs as jobs,
)

VAL="control-validation-7821002400424ccc"
EXEC="control-execution-c38169f6c6fc4ea0"
LIQ="control-liquidity-dbe4ec6f52c74694"
TCN="control-tcn-b7d0247118314107"
RANK="control-rank-c67e91abb3914f26"
SNAP="control-shadow-870fb66e1bdc4fd0"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


class CounterfactualAdvantageEndpointTests(TestCase):
    def test_request_is_fixed_and_rejects_tuning_or_order_fields(self):
        body={
            "confirm":"RESEARCH_CONTROL_COUNTERFACTUAL_ADVANTAGE_NO_ORDERS",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "source_tcn_job_id":TCN,
            "source_ranking_job_id":RANK,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlCounterfactualAdvantageRequest.model_validate(body)
        self.assertEqual(parsed.source_ranking_job_id,RANK)
        for extra in (
            {"veto_probability":.9},
            {"minimum_accuracy":.1},
            {"epochs":100},
            {"send_order":True},
            {"confirm":"PROMOTE_WINNER"},
        ):
            with self.assertRaises(ValidationError):
                StartControlCounterfactualAdvantageRequest.model_validate(
                    {**body,**extra}
                )

    def test_mismatched_chain_is_rejected_before_thread_creation(self):
        db=MagicMock()
        common={
            "status":"completed","error":None,
            "source_job_id":SNAP,"snapshot_sha256":SHA,
        }
        verified={
            "source_snapshot_sha256":SHA,
            "source_job_id":SNAP,
            "source_unchanged":True,
            "order_submission":"never",
            "numeric_input_integrity":{"status":"verified"},
        }
        db[jobs.VALIDATION_COLLECTION].find_one.return_value={
            **common,
            "result":{**verified,"original_shadow":{"reproduced":True}},
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "result":{**verified,"research_kind":"control_execution_feasibility_scenario"},
        }
        db[jobs.LIQUIDITY_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "result":{
                **verified,
                "research_kind":"control_liquidity_aware_fixed_hypothesis",
                "control_reference_parity":"verified",
            },
        }
        db[jobs.TCN_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "result":{
                **verified,
                "research_kind":"control_tcn_fixed_temporal_baseline",
            },
        }
        # Deliberately point v10.8.46 at another TCN job.
        db[jobs.RANK_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "source_tcn_job_id":"control-tcn-aaaaaaaaaaaaaaaa",
            "result":{
                **verified,
                "research_kind":"control_deep_pairwise_rank_hybrid",
                "source_liquidity_job_id":LIQ,
                "source_tcn_job_id":"control-tcn-aaaaaaaaaaaaaaaa",
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.AdvantageInvalid):
                jobs.start_counterfactual_advantage_research(
                    db,
                    source_validation_job_id=VAL,
                    source_execution_job_id=EXEC,
                    source_liquidity_job_id=LIQ,
                    source_tcn_job_id=TCN,
                    source_ranking_job_id=RANK,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__=="__main__":
    import unittest
    unittest.main()
