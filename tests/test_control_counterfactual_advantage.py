"""Unit tests for v10.8.47 counterfactual advantage meta-veto."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from market_cycle_trader_api.engine import control_counterfactual_advantage as adv


class CounterfactualAdvantageTests(TestCase):
    def _frames(self):
        dates=pd.bdate_range("2024-01-02",periods=180,tz="UTC")
        frames={}
        for n,symbol in enumerate(("AAA","BBB")):
            frame=pd.DataFrame(index=dates)
            price=np.linspace(10+n,20+2*n,len(dates))
            frame["close"]=price
            for j,name in enumerate(adv.FEATURES):
                frame[name]=np.linspace(.01+j*.001,.2+j*.001,len(dates)) + n*.005
            frames[symbol]=frame
        return dates,frames

    def test_labels_require_full_60_session_maturity(self):
        dates,frames=self._frames()
        decisions=pd.DataFrame({
            "decision_timestamp":[dates[50],dates[100],dates[130]],
            "held_asset_at_decision":["AAA"]*3,
            "liquidity_policy_target":["BBB"]*3,
        })
        samples=adv.build_pair_samples(
            frames,decisions,maturity_before=dates[150],stride=1,
        )
        self.assertEqual([s.decision_date for s in samples],[dates[50]])
        self.assertLess(
            frames["BBB"].index[samples[0].candidate_loc+adv.MAX_HORIZON],
            dates[150],
        )

    def test_future_feature_mutation_does_not_change_pair_window(self):
        dates,frames=self._frames()
        decision=pd.DataFrame({
            "decision_timestamp":[dates[70]],
            "held_asset_at_decision":["AAA"],
            "liquidity_policy_target":["BBB"],
        })
        samples=adv.build_pair_samples(
            frames,decision,maturity_before=dates[150],stride=1,
        )
        scale=adv.fit_scale(frames,samples)
        x,_=adv._tensor(frames,samples,scale)
        mutated={k:v.copy() for k,v in frames.items()}
        for frame in mutated.values():
            frame.loc[dates[71]:,list(adv.FEATURES)]=999999.
        x2,_=adv._tensor(mutated,samples,scale)
        np.testing.assert_array_equal(x.numpy(),x2.numpy())

    def test_meta_veto_can_only_hold_incumbent_or_keep_control(self):
        target,reason=adv.apply_meta_veto(
            current_position=3,control_target=7,
            probability_positive_advantage=.20,model_enabled=True,
        )
        self.assertEqual(target,3)
        self.assertIn("VETO",reason)
        target,_=adv.apply_meta_veto(
            current_position=3,control_target=7,
            probability_positive_advantage=.70,model_enabled=True,
        )
        self.assertEqual(target,7)

    def test_disabled_or_uncertain_model_is_exact_control_fallback(self):
        for enabled,prob in ((False,.01),(True,None),(True,.50)):
            target,_=adv.apply_meta_veto(
                current_position=2,control_target=8,
                probability_positive_advantage=prob,model_enabled=enabled,
            )
            self.assertEqual(target,8)

    def test_cash_entries_and_exits_are_never_overridden(self):
        for current,target in ((0,4),(4,0),(0,0),(4,4)):
            actual,_=adv.apply_meta_veto(
                current_position=current,control_target=target,
                probability_positive_advantage=0.0,model_enabled=True,
            )
            self.assertEqual(actual,target)


if __name__=="__main__":
    import unittest
    unittest.main()
