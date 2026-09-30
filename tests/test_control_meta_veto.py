"""Safety and chronology tests for Control-first meta-veto v10.8.47."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.api.routers.control_shadow import StartControlMetaVetoRequest
from market_cycle_trader_api.engine import control_meta_veto as veto
from market_cycle_trader_api.services import control_shadow_meta_veto_jobs as jobs

VAL="control-validation-7821002400424ccc"
EXEC="control-execution-c38169f6c6fc4ea0"
LIQ="control-liquidity-dbe4ec6f52c74694"
TCN="control-tcn-b7d0247118314107"
RANK="control-rank-c67e91abb3914f26"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"
SNAP="control-shadow-870fb66e1bdc4fd0"


class MetaVetoTests(TestCase):
    def test_threshold_requires_precision_and_minimum_sample_count(self):
        probabilities=[.05]*30+[.25]*70+[.8]*100
        labels=[0]*25+[1]*5+[0]*35+[1]*35+[1]*100
        threshold,rows=veto._choose_threshold(probabilities,labels)
        # 0.10 has good precision but too few examples; 0.30 has 60% bad
        # precision over 100 examples and is allowed.
        self.assertEqual(threshold,.30)
        selected=next(x for x in rows if x["threshold"]==.30)
        self.assertTrue(selected["qualified"])

    def test_weak_validation_skill_disables_all_intervention(self):
        probabilities=np.linspace(.05,.95,200).tolist()
        labels=[i%2 for i in range(200)]
        threshold,rows=veto._choose_threshold(probabilities,labels)
        self.assertIsNone(threshold)
        self.assertFalse(any(row["qualified"] for row in rows))

    def test_meta_model_is_pair_direction_sensitive_and_outputs_probability(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model=veto.MetaVetoNet().eval()
            a=torch.randn(2,len(veto.FEATURES),veto.WINDOW)
            b=torch.randn(2,len(veto.FEATURES),veto.WINDOW)
            context=torch.randn(2,3)
            with torch.no_grad():
                p=torch.sigmoid(model(a,b,context))
                q=torch.sigmoid(model(b,a,-context))
            self.assertEqual(tuple(p.shape),(2,))
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(((p>=0)&(p<=1)).all())
            self.assertFalse(torch.equal(p,q))

    def test_pair_rows_never_use_labels_maturing_at_or_after_test(self):
        dates=pd.bdate_range("2023-01-02",periods=220,tz="UTC")
        frames={}
        symbols=["AAA","BBB"]
        for offset,symbol in enumerate(symbols):
            frame=pd.DataFrame(index=dates)
            for j,name in enumerate(veto.FEATURES):
                frame[name]=np.linspace(.01+j*.001,.2+j*.001,len(dates))+offset*.01
            frame[veto.TARGET]=np.linspace(-.2,.3,len(dates))+offset*.02
            frames[symbol]=frame
        calibration=dates[80:180]
        cache={
            pd.Timestamp(d):np.array([0.,.2,.1],dtype=float)
            for d in calibration
        }
        rows=veto._build_pair_rows(
            frames,symbols,calibration,cache,test_start=dates[180],
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertLess(
                frames[row.candidate].index[row.candidate_loc+60],
                dates[180],
            )
            self.assertLess(
                frames[row.incumbent].index[row.incumbent_loc+60],
                dates[180],
            )

    def test_swagger_input_cannot_tune_model_or_thresholds(self):
        body={
            "confirm":"RESEARCH_CONTROL_META_VETO_NO_ORDERS",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "source_tcn_job_id":TCN,
            "source_ranking_job_id":RANK,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlMetaVetoRequest.model_validate(body)
        self.assertEqual(parsed.source_ranking_job_id,RANK)
        for extra in (
            {"threshold":.5},
            {"epochs":50},
            {"loss":"focal"},
            {"confirm":"PROMOTE_WINNER"},
        ):
            with self.assertRaises(ValidationError):
                StartControlMetaVetoRequest.model_validate({**body,**extra})

    def test_service_rejects_mismatched_rank_chain_before_thread(self):
        db=MagicMock()
        common={"status":"completed","source_job_id":SNAP,"snapshot_sha256":SHA}
        db[jobs.VALIDATION_COLLECTION].find_one.return_value={
            **common,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "source_unchanged":True,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "original_shadow":{"reproduced":True},
            },
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value={
            **common,"source_validation_job_id":VAL,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "source_unchanged":True,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_execution_feasibility_scenario",
            },
        }
        db[jobs.LIQUIDITY_COLLECTION].find_one.return_value={
            **common,"source_execution_job_id":EXEC,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "source_unchanged":True,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_liquidity_aware_fixed_hypothesis",
            },
        }
        db[jobs.TCN_COLLECTION].find_one.return_value={
            **common,"source_liquidity_job_id":LIQ,
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "source_unchanged":True,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_tcn_fixed_temporal_baseline",
            },
        }
        db[jobs.RANK_COLLECTION].find_one.return_value={
            **common,"source_tcn_job_id":"control-tcn-aaaaaaaaaaaaaaaa",
            "result":{
                "source_snapshot_sha256":SHA,"source_job_id":SNAP,
                "source_unchanged":True,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "research_kind":"control_deep_pairwise_rank_hybrid",
            },
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.MetaVetoInvalid):
                jobs.start_meta_veto_research(
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
