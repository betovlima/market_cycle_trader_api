from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v107 as research  # noqa: E402


class ContextualMarginalSignatureV107Tests(unittest.TestCase):
    def _row(
        self,
        universe: str,
        date: str,
        horizon: str,
        candidate: str,
        y: float,
        context_shift: float,
        corr: float,
    ) -> dict:
        values = {feature: 0.0 for feature in research.MODEL_FEATURES}
        values["relative__return_20"] = context_shift
        values["relative__trend_efficiency_20"] = context_shift * 0.5
        values["corr_20_to_universe"] = corr
        return {
            "universe_name": universe,
            "universe_size": 20,
            "decision_date": date,
            "horizon_end": horizon,
            "candidate": candidate,
            "evaluation_status": "completed",
            "delta_log_capital": y,
            **values,
        }

    def _frame(self) -> pd.DataFrame:
        rows = []
        specifications = [
            ("2026-01-02", "2026-03-02"),
            ("2026-02-02", "2026-03-30"),
            ("2026-03-02", "2026-04-28"),
        ]
        for date, horizon in specifications:
            for universe_index, universe in enumerate(["U0", "U1", "U2"]):
                shift = float(universe_index) * 0.1
                rows.append(self._row(universe, date, horizon, "A", 0.02 - 0.02 * universe_index, shift, 0.1 + shift))
                rows.append(self._row(universe, date, horizon, "B", 0.00, shift, 0.4 - shift))
        frame = pd.DataFrame(rows)
        frame["decision_ts"] = pd.to_datetime(frame["decision_date"], utc=True)
        frame["horizon_ts"] = pd.to_datetime(frame["horizon_end"], utc=True)
        frame["effect_sign"] = frame["delta_log_capital"].map(research._effect_sign)
        return frame

    def test_pairwise_report_detects_context_constant_feature(self) -> None:
        pairwise = research.build_pairwise(self._frame())
        report = research.pseudoreplication_report(pairwise).set_index("feature")
        self.assertTrue(bool(report.loc["relative__return_20", "context_constant_across_candidates"]))
        self.assertFalse(bool(report.loc["corr_20_to_universe", "context_constant_across_candidates"]))

    def test_sign_change_episodes_are_counted_once_per_candidate_date(self) -> None:
        episodes = research.sign_episode_summary(self._frame())
        a = episodes.loc[episodes["candidate"] == "A"]
        self.assertEqual(len(a), 3)
        self.assertTrue(bool(a["activation_change"].all()))
        self.assertFalse(bool(a["strict_positive_negative_flip"].any()))

    def test_context_state_collapses_candidates_before_inference(self) -> None:
        states = research.build_context_states(self._frame())
        self.assertEqual(len(states), 9)
        self.assertEqual(int(states["candidate_count"].min()), 2)
        report = research.context_feature_report(states, permutations=50, random_state=7)
        row = report.loc[report["feature"] == "relative__return_20"].iloc[0]
        self.assertEqual(int(row["universe_date_states"]), 9)
        self.assertEqual(int(row["decision_dates"]), 3)

    def test_non_overlapping_anchor_dates_follow_horizon_boundaries(self) -> None:
        anchors = research.non_overlapping_anchor_dates(self._frame())
        self.assertEqual(anchors, ["2026-01-02", "2026-03-02"])


if __name__ == "__main__":
    unittest.main()
