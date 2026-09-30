"""Temporal isolation and ranking-objective tests for v10.8.46."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import StartControlDeepRankingRequest
from market_cycle_trader_api.engine import control_deep_ranking as ranker
from market_cycle_trader_api.engine import control_deep_ranking_research as research
from market_cycle_trader_api.services import control_shadow_deep_rank_jobs as jobs

VAL="control-validation-7821002400424ccc"
EXEC="control-execution-c38169f6c6fc4ea0"
LIQ="control-liquidity-dbe4ec6f52c74694"
TCN="control-tcn-b7d0247118314107"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"
SNAP="control-shadow-870fb66e1bdc4fd0"


class PairwiseDeepRankingTests(TestCase):
    def test_pairwise_loss_rewards_correct_order_not_target_mean(self):
        targets=torch.tensor([.8,.3,-.2],dtype=torch.float32)
        correct=torch.tensor([2.,1.,0.],dtype=torch.float32)
        reversed_=torch.tensor([0.,1.,2.],dtype=torch.float32)
        self.assertLess(
            float(ranker._pairwise_loss(correct,targets)),
            float(ranker._pairwise_loss(reversed_,targets)),
        )
        good,total=ranker._ranking_accuracy(correct,targets)
        bad,_=ranker._ranking_accuracy(reversed_,targets)
        self.assertEqual((good,total),(3,3))
        self.assertEqual(bad,0)
        shifted=correct+1000.
        self.assertAlmostEqual(
            float(ranker._pairwise_loss(correct,targets)),
            float(ranker._pairwise_loss(shifted,targets)),
            places=5,
        )

    def test_ranking_groups_require_full_label_maturity_before_cutoff(self):
        dates=pd.bdate_range("2024-01-02",periods=170,tz="UTC")
        frames={}
        for i in range(25):
            frame=pd.DataFrame(index=dates)
            for j,name in enumerate(ranker.FEATURES):
                frame[name]=np.linspace(.01+j*.001,.1+j*.001,len(dates))
            frame[ranker.TARGET]=np.linspace(-.2,.3,len(dates))+i*.001
            frames[f"S{i:02d}"]=frame
        groups=ranker._date_groups(
            frames,sorted(frames),dates[:140],
            maturity_before=dates[140],
            date_stride=1,
        )
        self.assertTrue(groups)
        for date,members in groups:
            self.assertGreaterEqual(len(members),ranker.MIN_ASSETS_PER_DATE)
            for symbol,loc,_ in members:
                self.assertLess(frames[symbol].index[loc+60],dates[140])
                self.assertLessEqual(frames[symbol].index[loc],date)

    def test_scaler_ignores_all_future_feature_mutations(self):
        dates=pd.bdate_range("2024-01-02",periods=100,tz="UTC")
        frame=pd.DataFrame(index=dates)
        for j,name in enumerate(ranker.FEATURES):
            frame[name]=np.linspace(.01+j*.01,.2+j*.01,len(dates))
        frame[ranker.TARGET]=.1
        original=ranker._scale_from_dates({"AAA":frame},["AAA"],dates[:70])
        mutated=frame.copy()
        mutated.loc[dates[70]:,list(ranker.FEATURES)]=999999.
        current=ranker._scale_from_dates({"AAA":mutated},["AAA"],dates[:70])
        np.testing.assert_array_equal(original.mean,current.mean)
        np.testing.assert_array_equal(original.std,current.std)

    def test_hybrid_policy_deep_orders_but_lightgbm_controls_absolute_gate(self):
        dates=pd.bdate_range("2025-01-02",periods=30,tz="UTC")
        frames={}
        for symbol in ("AAA","BBB","CCC"):
            frames[symbol]=pd.DataFrame({
                "close":10.,"volume":100000.,
            },index=dates)
        date=dates[-1]
        # Ranker likes AAA most, but its Control utility cannot enter.
        utilities={date:np.array([0.,.0002,.02,.01],dtype=float)}
        ranks={date:np.array([-np.inf,9.,8.,7.],dtype=float)}
        account={
            "enabled":True,"audit":{},"decision_timestamp":date,
            "equity":10000.,"cash":10000.,"position":0,"shares":0,
        }
        cfg=SimpleNamespace(
            rotation_cash_threshold=0.,
            rotation_min_expected_edge=.001,
            rotation_switch_margin=.0005,
            rotation_min_holding_days=2,
        )
        policy=ranker._deep_rank_policy(
            original_utility_cache=utilities,rank_cache=ranks,
            frames=frames,symbols=["AAA","BBB","CCC"],config=cfg,
            switch_margin=.0025,account=account,
            diagnostics={},fold_id=1,
        )
        target,score=policy(date,0,0)
        # We intentionally do NOT skip to a lower-ranked candidate after the
        # ranker's top choice fails the original absolute Control entry gate.
        self.assertEqual(target,0)
        self.assertEqual(score,0.)

        ranks[date]=np.array([-np.inf,7.,9.,8.],dtype=float)
        account["decision_timestamp"]=date
        target,score=policy(date,0,0)
        self.assertEqual(target,2)
        self.assertAlmostEqual(score,.02)

    def test_request_schema_is_fixed_and_rejects_hyperparameter_tuning(self):
        body={
            "confirm":"RESEARCH_PAIRWISE_DEEP_RANK_NO_ORDERS",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "source_tcn_job_id":TCN,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlDeepRankingRequest.model_validate(body)
        self.assertEqual(parsed.source_tcn_job_id,TCN)
        for extra in (
            {"epochs":100},
            {"loss":"listwise"},
            {"learning_rate":.1},
            {"confirm":"PROMOTE_WINNER"},
        ):
            with self.assertRaises(ValidationError):
                StartControlDeepRankingRequest.model_validate({**body,**extra})

    def test_server_rejects_mismatched_chain_before_thread(self):
        db=MagicMock()
        common={
            "status":"completed","source_job_id":SNAP,
            "snapshot_sha256":SHA,
        }
        db[jobs.VALIDATION_COLLECTION].find_one.return_value={
            **common,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "original_shadow":{"reproduced":True},
            },
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_execution_feasibility_scenario",
            },
        }
        db[jobs.LIQUIDITY_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_liquidity_aware_fixed_hypothesis",
                "control_reference_parity":"verified",
            },
        }
        db[jobs.TCN_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":"control-liquidity-aaaaaaaaaaaaaaaa",
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_tcn_fixed_temporal_baseline",
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.DeepRankInvalid):
                jobs.start_deep_ranking_research(
                    db,source_validation_job_id=VAL,
                    source_execution_job_id=EXEC,
                    source_liquidity_job_id=LIQ,
                    source_tcn_job_id=TCN,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()

    def test_metric_reports_pairwise_and_top_realized_percentile(self):
        frame=pd.DataFrame({
            "decision_date":[pd.Timestamp("2025-01-01",tz="UTC")]*3,
            "rank_score":[3.,2.,1.],
            "realized_utility_if_mature":[.9,.2,-.1],
        })
        result=research._ranking_metrics(frame,"x")
        self.assertEqual(result["pairwise_accuracy"],1.)
        self.assertEqual(result["mean_predicted_top_realized_percentile"],1.)
        self.assertEqual(result["mature_rows"],3)


if __name__=="__main__":
    import unittest
    unittest.main()
