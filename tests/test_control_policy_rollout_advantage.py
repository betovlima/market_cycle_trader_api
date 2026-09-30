"""Unit and endpoint safeguards for v10.8.48 policy-rollout advantage."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import (
    StartControlPolicyRolloutRequest,
)
from market_cycle_trader_api.engine import control_execution_feasibility as execution
from market_cycle_trader_api.engine import control_policy_rollout_advantage as rollout
from market_cycle_trader_api.services import control_shadow_policy_rollout_jobs as jobs

VAL="control-validation-7821002400424ccc"
EXEC="control-execution-c38169f6c6fc4ea0"
LIQ="control-liquidity-dbe4ec6f52c74694"
TCN="control-tcn-b7d0247118314107"
RANK="control-rank-c67e91abb3914f26"
ADV="control-advantage-53fc67bd758f443a"
SNAP="control-shadow-870fb66e1bdc4fd0"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"


def _fees(side, shares, price, _config):
    return {
        "commission_fee":0.0,"sec_fee":0.0,"taf_fee":0.0,
        "cat_fee":0.0,"total_fee":0.0,
    }


class PolicyRolloutAdvantageTests(TestCase):
    def test_request_is_fixed_and_rejects_rollout_tuning(self):
        body={
            "confirm":"RESEARCH_CONTROL_POLICY_ROLLOUT_ADVANTAGE_NO_ORDERS",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "source_tcn_job_id":TCN,
            "source_ranking_job_id":RANK,
            "source_advantage_job_id":ADV,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlPolicyRolloutRequest.model_validate(body)
        self.assertEqual(parsed.source_advantage_job_id,ADV)
        for extra in (
            {"rollout_horizon":60},
            {"train_fraction":.9},
            {"veto_probability":.5},
            {"minimum_accuracy":.1},
            {"epochs":100},
            {"send_order":True},
        ):
            with self.assertRaises(ValidationError):
                StartControlPolicyRolloutRequest.model_validate({**body,**extra})

    def test_fold_one_is_always_disabled_and_future_fold_labels_are_excluded(self):
        dates=pd.bdate_range("2024-01-02",periods=180,tz="UTC")
        frames={}
        for n,symbol in enumerate(("AAA","BBB")):
            frame=pd.DataFrame(index=dates)
            frame["close"]=np.linspace(10+n,20+n,len(dates))
            for j,name in enumerate(rollout._pair_sample.__globals__["PairSample"].__module__ and __import__(
                "market_cycle_trader_api.engine.control_counterfactual_advantage",
                fromlist=["FEATURES"]
            ).FEATURES):
                frame[name]=np.linspace(.01+j*.001,.2+j*.001,len(dates))+n*.001
            frames[symbol]=frame
        folds=[
            {"fold_id":1,"test_start":dates[60],"test_end":dates[99]},
            {"fold_id":2,"test_start":dates[100],"test_end":dates[139]},
            {"fold_id":3,"test_start":dates[140],"test_end":dates[179]},
        ]
        rows=[]
        for i in range(75):
            source_fold=1 if i<50 else 2
            date=dates[20+i]
            rows.append({
                "decision_date":date,
                "rollout_end_date":date+pd.Timedelta(days=1),
                "incumbent_asset":"AAA",
                "control_target_asset":"BBB",
                "source_fold_id":source_fold,
                "delta_capital_usd":10.0 if i%2 else -5.0,
            })
        meta,reports=rollout._train_meta_by_fold(
            frames=frames,folds=folds,rollout_rows=rows,
        )
        self.assertFalse(meta[1]["enabled"])
        self.assertEqual(reports[0]["disable_reason"],"NO_PRIOR_OOS_FOLD")
        # Fold 2 is never allowed to use source fold 2 rows.
        self.assertLessEqual(reports[1]["eligible_prior_rollouts"],50)

    def test_execution_replay_can_start_from_reconciled_historical_state(self):
        dates=pd.bdate_range("2026-07-01",periods=30,tz="UTC")
        frames={
            "AAA":pd.DataFrame({
                "open":10.0,"high":11.0,"low":9.0,
                "close":10.0,"volume":1000.0,
            },index=dates),
            "BBB":pd.DataFrame({
                "open":20.0,"high":21.0,"low":19.0,
                "close":20.0,"volume":1000.0,
            },index=dates),
        }
        cfg=SimpleNamespace(
            initial_capital=1000.0,rotation_model_repetitions=1,
            rotation_downside_penalty=.2,rotation_drawdown_penalty=.3,
        )
        initial={
            "cash":500.0,"position":1,"shares":50,
            "holding_days":3,"equity":1000.0,
        }
        seen=[]
        def policy(date,position,holding):
            seen.append((position,holding))
            return position,0.1
        with patch.object(
            execution,"_equal_weight_benchmark",
            side_effect=lambda _f,_s,d,*a: pd.Series(1000.0,index=d),
        ):
            run=execution.simulate_feasible_control(
                "rollout",policy,frames,["AAA","BBB"],dates[10:15],
                cfg,_fees,lambda p,s,c:p,initial_state=initial,
            )
        self.assertEqual(seen[0],(1,3))
        self.assertGreater(run.metrics["strategy_ending_capital"],0)

    def test_invalid_historical_state_fails_closed(self):
        dates=pd.bdate_range("2026-07-01",periods=25,tz="UTC")
        frames={"AAA":pd.DataFrame({
            "open":10.0,"high":11.0,"low":9.0,
            "close":10.0,"volume":1000.0,
        },index=dates)}
        cfg=SimpleNamespace(
            initial_capital=1000.0,rotation_model_repetitions=1,
            rotation_downside_penalty=.2,rotation_drawdown_penalty=.3,
        )
        with self.assertRaisesRegex(ValueError,"equity does not reconcile"):
            execution.simulate_feasible_control(
                "rollout",lambda d,p,h:(1,.1),frames,["AAA"],
                dates[10:13],cfg,_fees,lambda p,s,c:p,
                initial_state={
                    "cash":500.0,"position":1,"shares":50,
                    "holding_days":3,"equity":999.0,
                },
            )

    def test_mismatched_v47_chain_rejected_before_thread(self):
        db=MagicMock()
        common={
            "status":"completed","error":None,
            "source_job_id":SNAP,"snapshot_sha256":SHA,
        }
        verified={
            "source_snapshot_sha256":SHA,"source_job_id":SNAP,
            "source_unchanged":True,"order_submission":"never",
            "numeric_input_integrity":{"status":"verified"},
        }
        db[jobs.VALIDATION_COLLECTION].find_one.return_value={
            **common,"result":{**verified,"original_shadow":{"reproduced":True}},
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "result":{**verified,"research_kind":"control_execution_feasibility_scenario"},
        }
        db[jobs.LIQUIDITY_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,"source_execution_job_id":EXEC,
            "result":{**verified,"control_reference_parity":"verified"},
        }
        db[jobs.TCN_COLLECTION].find_one.return_value={
            **common,"source_liquidity_job_id":LIQ,
            "result":{**verified,"research_kind":"control_tcn_fixed_temporal_baseline"},
        }
        db[jobs.RANK_COLLECTION].find_one.return_value={
            **common,"source_liquidity_job_id":LIQ,"source_tcn_job_id":TCN,
            "result":{**verified,"research_kind":"control_deep_pairwise_rank_hybrid"},
        }
        db[jobs.ADVANTAGE_COLLECTION].find_one.return_value={
            **common,"source_liquidity_job_id":LIQ,"source_tcn_job_id":TCN,
            "source_ranking_job_id":"control-rank-aaaaaaaaaaaaaaaa",
            "result":{
                **verified,
                "research_kind":"control_counterfactual_advantage_meta_veto",
                "source_liquidity_job_id":LIQ,
                "source_tcn_job_id":TCN,
                "source_ranking_job_id":"control-rank-aaaaaaaaaaaaaaaa",
                "v1044_parity":{"status":"verified"},
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.RolloutInvalid):
                jobs.start_policy_rollout_advantage_research(
                    db,
                    source_validation_job_id=VAL,
                    source_execution_job_id=EXEC,
                    source_liquidity_job_id=LIQ,
                    source_tcn_job_id=TCN,
                    source_ranking_job_id=RANK,
                    source_advantage_job_id=ADV,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()


if __name__=="__main__":
    import unittest
    unittest.main()
