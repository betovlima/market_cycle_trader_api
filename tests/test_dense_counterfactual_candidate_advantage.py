from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.services.dense_counterfactual_candidate_advantage import (
    replay_forced_candidate_intervention,
)
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (
    extract_candidate_divergence_episodes,
)


class _Panel:
    def __init__(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
        self.calendar = pd.DataFrame(
            {
                "fold": [1] * 6,
                "decision_session_id": list(range(6)),
                "execution_session_id": list(range(1, 7)),
                "execution_timestamp": timestamps + pd.Timedelta(days=1),
            },
            index=timestamps,
        )
        self.baseline = ["A", "B"]
        self.scores = pd.DataFrame(
            {
                "A": [0.20] * 6,
                "B": [0.10] * 6,
                "X": [0.90, 0.90, 0.90, 0.05, 0.05, 0.05],
            },
            index=timestamps,
        )
        self.opens = pd.DataFrame(
            {"A": [100.0] * 6, "B": [100.0] * 6, "X": [100.0] * 6},
            index=timestamps,
        )
        self.closes = pd.DataFrame(
            {
                "A": [101.0] * 6,
                "B": [100.0] * 6,
                "X": [100.0, 100.0, 102.0, 102.0, 100.0, 100.0],
            },
            index=timestamps,
        )

    def complete(self, symbol: str) -> bool:
        return symbol in self.scores.columns


def _settings() -> dict:
    return {
        "strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST",
        "initial_capital": 10_000.0,
        "rotation_cash_threshold": 0.0,
        "rotation_min_expected_edge": 0.001,
        "rotation_min_holding_days": 2,
        "rotation_switch_margin": 0.0005,
        "fractional_shares": True,
        "slippage_bps": 0.0,
        "commission_rate": 0.0,
        "cat_fee_per_share": 0.0,
        "sec_fee_rate": 0.0,
        "taf_fee_per_share": 0.0,
        "taf_fee_cap": 0.0,
    }


def _baseline_daily(panel: _Panel) -> pd.DataFrame:
    timestamps = panel.calendar.index
    equity = 10_000.0 * np.cumprod(np.repeat(1.01, len(timestamps)))
    daily = panel.calendar.reset_index().copy()
    if "timestamp" not in daily.columns:
        daily.rename(columns={"index": "timestamp"}, inplace=True)
    daily["selected_asset"] = "A"
    daily["decision_reason"] = ["ENTER_BEST_ASSET"] + ["HOLD_CURRENT_BEST"] * 5
    daily["strategy_equity"] = equity
    daily["net_log_return"] = np.log(1.01)
    daily.attrs["rotation_min_holding_days"] = 2
    return daily


class DenseCounterfactualCandidateAdvantageTests(unittest.TestCase):
    def test_candidate_cannot_change_prefix_before_forced_date(self) -> None:
        panel = _Panel()
        activation = panel.calendar.index[2]
        result = replay_forced_candidate_intervention(
            panel=panel,
            candidate="X",
            settings=_settings(),
            intervention_timestamp=activation,
        )
        self.assertTrue(result.intervention_applied)
        selected = result.replay.daily["selected_asset"].tolist()
        self.assertEqual(selected[:2], ["A", "A"])
        self.assertEqual(selected[2], "X")
        self.assertEqual(
            result.replay.daily.iloc[2]["decision_reason"],
            "FORCED_CANDIDATE_INTERVENTION",
        )

    def test_minimum_holding_guard_blocks_forced_intervention(self) -> None:
        panel = _Panel()
        activation = panel.calendar.index[1]
        result = replay_forced_candidate_intervention(
            panel=panel,
            candidate="X",
            settings=_settings(),
            intervention_timestamp=activation,
        )
        self.assertFalse(result.intervention_applied)
        self.assertEqual(result.blocked_reason, "MIN_HOLD_GUARD")
        self.assertNotIn(
            "X", result.replay.daily["selected_asset"].astype(str).tolist()
        )

    def test_forced_episode_reconverges_on_policy_sufficient_holding_state(self) -> None:
        panel = _Panel()
        activation = panel.calendar.index[2]
        forced = replay_forced_candidate_intervention(
            panel=panel,
            candidate="X",
            settings=_settings(),
            intervention_timestamp=activation,
        )
        baseline = _baseline_daily(panel)
        episodes = extract_candidate_divergence_episodes(
            baseline_daily=baseline,
            expanded_daily=forced.replay.daily,
            candidate="X",
            fold_id=1,
        )
        starts = pd.to_datetime(episodes["episode_start"], utc=True)
        exact = episodes.loc[starts.eq(activation)]
        self.assertEqual(len(exact), 1)
        row = exact.iloc[0]
        self.assertFalse(bool(row["right_censored"]))
        self.assertEqual(pd.Timestamp(row["episode_start"]), activation)
        self.assertEqual(pd.Timestamp(row["episode_end"]), panel.calendar.index[5])


if __name__ == "__main__":
    unittest.main()
