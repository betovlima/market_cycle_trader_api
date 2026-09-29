"""Offline reproducibility and temporal isolation tests for TinyTCN research."""
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

from market_cycle_trader_api.api.routers.control_shadow import StartControlTCNResearchRequest
from market_cycle_trader_api.engine import control_deep_learning_tcn as tcn
from market_cycle_trader_api.engine import control_deep_learning_research as research
from market_cycle_trader_api.services import control_shadow_tcn_jobs as jobs

VAL="control-validation-7821002400424ccc"
EXEC="control-execution-c38169f6c6fc4ea0"
LIQ="control-liquidity-dbe4ec6f52c74694"
SHA="6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3"
SNAP="control-shadow-870fb66e1bdc4fd0"


class FixedTinyTCNTests(TestCase):
    def setUp(self):
        self.dates=pd.bdate_range("2025-01-01",periods=160,tz="UTC")
        x=np.linspace(.001,.05,len(self.dates))
        self.frame=pd.DataFrame(
            {feature:x + i*.02 for i,feature in enumerate(tcn.FEATURES)},
            index=self.dates,
        )
        self.frame[tcn.TARGET]=np.sin(np.arange(len(self.dates))*.1)
        self.frames={"AAA":self.frame}

    def test_training_rows_do_not_use_unmatured_future_labels(self):
        rows=tcn._dataset_rows(
            self.frames,["AAA"],self.dates[:110],
            maturity_before=self.dates[110],horizon=60,
        )
        self.assertTrue(rows)
        self.assertTrue(all(
            self.frame.index[pos+60]<self.dates[110]
            for symbol,pos in rows
        ))
        self.assertLess(max(pos for _,pos in rows),50)
        # Mutating labels of rows not mature as of cutoff never changes selection.
        future=self.frame.copy()
        future.loc[self.dates[50]:,tcn.TARGET]=999.
        same=tcn._dataset_rows(
            {"AAA":future},["AAA"],self.dates[:110],
            maturity_before=self.dates[110],horizon=60,
        )
        self.assertEqual(rows,same)
        for _,pos in same:
            self.assertNotEqual(future[tcn.TARGET].iloc[pos],999.)

    def test_scaler_is_fitted_from_training_only_not_later_calendar(self):
        rows=tcn._dataset_rows(
            self.frames,["AAA"],self.dates[:100],
            maturity_before=self.dates[100],horizon=60,
        )
        old=tcn._scaling(self.frames,["AAA"],self.dates[:100],rows)
        newer=self.frame.copy()
        newer.loc[self.dates[100]:,list(tcn.FEATURES)]=100000.
        newer.loc[self.dates[100]:,tcn.TARGET]=-999.
        current=tcn._scaling({"AAA":newer},["AAA"],self.dates[:100],rows)
        np.testing.assert_array_equal(old.mean,current.mean)
        np.testing.assert_array_equal(old.std,current.std)
        self.assertEqual(old.y_mean,current.y_mean)
        self.assertEqual(old.y_std,current.y_std)

    def test_causal_tcn_encoder_has_no_future_to_past_information(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model=tcn.TinyTCN().eval()
            input_=torch.randn(1,len(tcn.FEATURES),tcn.WINDOW)
            other=input_.clone()
            other[:,:,25:]+=500.
            with torch.no_grad():
                before=model.encoder(input_)
                after=model.encoder(other)
            torch.testing.assert_close(before[:,:,:25],after[:,:,:25],
                                       rtol=0,atol=0)
            self.assertEqual(
                tuple(model(input_).shape),(1,),
            )
        self.assertEqual(tcn.WINDOW,40)

    def test_tiny_fixed_one_epoch_trains_on_cpu_without_nan(self):
        rows=[("AAA",i) for i in range(40,50)]
        scale=tcn._scaling(self.frames,["AAA"],self.dates[:100],rows)
        features,targets=tcn._arrays(self.frames,["AAA"],scale)
        model,ep,loss=tcn._fit(
            features,targets,rows,seed=43,epochs=1,
            valid=[("AAA",i) for i in range(51,55)],
        )
        self.assertEqual(ep,1)
        self.assertTrue(np.isfinite(loss))
        out=tcn._predict(model,features,[("AAA",55)],scale)
        self.assertTrue(np.isfinite(out[("AAA",55)]))

    def test_predictive_metric_maturity_is_explicit(self):
        observations=pd.DataFrame({
            "predicted_utility":[.2,.1,-.2],
            "realized_utility_if_mature":[.1,None,-.3],
        })
        report=research._predictive_metrics(observations,"test")
        self.assertEqual(report["mature_target_rows"],2)
        self.assertEqual(report["unmatured_target_rows"],1)
        self.assertAlmostEqual(report["mae"],.1)

    def test_request_rejects_model_user_tuning_and_wrong_source_ids(self):
        expected={
            "confirm":"RESEARCH_FIXED_CAUSAL_TCN_NO_ORDERS",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,
            "source_liquidity_job_id":LIQ,
            "expected_snapshot_sha256":SHA,
        }
        parsed=StartControlTCNResearchRequest.model_validate(expected)
        self.assertEqual(parsed.source_liquidity_job_id,LIQ)
        for change in ({"epochs":500},{"model_family":"transformer"},
                       {"confirm":"EXECUTE_LIVE_ORDER"}):
            with self.assertRaises(ValidationError):
                StartControlTCNResearchRequest.model_validate({**expected,**change})

    def test_server_rejects_mismatched_research_without_creating_thread(self):
        db=MagicMock()
        db[jobs.VALIDATION_COLLECTION].find_one.return_value={
            "status":"completed","source_job_id":SNAP,
            "snapshot_sha256":SHA,
            "result":{
                "source_snapshot_sha256":SHA,
                "source_job_id":SNAP,"order_submission":"never",
                "numeric_input_integrity":{"status":"verified"},
                "original_shadow":{"reproduced":True},
            },
        }
        db[jobs.EXECUTION_COLLECTION].find_one.return_value={
            "status":"completed","source_job_id":SNAP,
            "source_validation_job_id":VAL,"snapshot_sha256":SHA,
            "result":{"source_snapshot_sha256":SHA,
                      "source_job_id":SNAP,
                      "order_submission":"never",
                      "numeric_input_integrity":{"status":"verified"},
                      "research_kind":"control_execution_feasibility_scenario"},
        }
        db[jobs.LIQUIDITY_COLLECTION].find_one.return_value={
            "status":"completed","source_job_id":"control-shadow-aaaaaaaaaaaaaaaa",
            "source_validation_job_id":VAL,
            "source_execution_job_id":EXEC,"snapshot_sha256":SHA,
            "result":{"source_snapshot_sha256":SHA,
                      "source_job_id":"control-shadow-aaaaaaaaaaaaaaaa"},
        }
        with patch.object(jobs,"_require_enabled"):
            with self.assertRaises(jobs.TCNInvalid):
                jobs.start_tcn_research(
                    db,source_validation_job_id=VAL,
                    source_execution_job_id=EXEC,
                    source_liquidity_job_id=LIQ,
                    expected_snapshot_sha256=SHA,
                )
        db[jobs.COLLECTION].insert_one.assert_not_called()

    def test_reference_is_read_only_and_training_mode_is_separate(self):
        self.assertEqual(tcn.MODE,"MCT_RESEARCH_TCN_LIQUIDITY_AWARE_V1")
        self.assertEqual(tcn.FIXED_SWITCH_MARGIN,.0025)
        self.assertEqual(tcn.PATIENCE,2)
        self.assertEqual(tcn.MAX_PREDICT_ABS,3.)
        self.assertIn("market_exposure",research.KEYS)


if __name__=="__main__":
    import unittest
    unittest.main()
