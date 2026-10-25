from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock

from market_cycle_trader_api.services.analytics import _decision_score_rows
from market_cycle_trader_api.engine.tcc_u67_operational_runtime import _observe_five_cached_utilities


def ts(day: int) -> datetime:
    return datetime(2025, 4, day, 20, 0, tzinfo=timezone.utc)


class ReplayDecisionScoresTests(unittest.TestCase):
    def test_replays_real_top_three_from_old_backtest_without_inventing_five(self):
        db = {"backtest_predictions": MagicMock()}
        db["backtest_predictions"].find.return_value = [{
            "timestamp": ts(2),
            "decision_date": ts(1),
            "current_asset": "AVGO",
            "current_score": 0.374561,
            "best_asset": "LLY",
            "best_score": 0.40745,
            "finite_score_count": 75,
            "decision_reason": "ROTATE_TO_BEST_ASSET",
            "top_1_asset": "LLY", "top_1_score": 0.40745,
            "top_2_asset": "AVGO", "top_2_score": 0.374561,
            "top_3_asset": "TSLA", "top_3_score": 0.3315,
        }]
        result = _decision_score_rows(db, "job-1", "model-1")
        self.assertEqual(len(result), 1)
        self.assertEqual([row["asset"] for row in result[0]["top_scores"]], ["LLY", "AVGO", "TSLA"])
        self.assertEqual(result[0]["finite_score_count"], 75)
        self.assertEqual(result[0]["decision_reason"], "ROTATE_TO_BEST_ASSET")
        self.assertEqual(result[0]["execution_date"][:10], "2025-04-02")
        args, kwargs = db["backtest_predictions"].find.call_args
        self.assertEqual(args[0], {"job_id": "job-1", "symbol": "PORTFOLIO", "backend": "model-1"})
        self.assertEqual(args[1]["top_5_score"], 1)

    def test_complete_top_five_is_returned_without_synthetic_scores(self):
        db = {"backtest_predictions": MagicMock()}
        db["backtest_predictions"].find.return_value = [{
            "timestamp": ts(3),
            "decision_date": ts(2),
            "top_1_asset": "AAPL", "top_1_score": 0.5,
            "top_2_asset": "NVDA", "top_2_score": 0.4,
            "top_3_asset": "MSFT", "top_3_score": 0.3,
            "top_4_asset": "AMZN", "top_4_score": 0.2,
            "top_5_asset": "LLY", "top_5_score": 0.1,
            "finite_score_count": 5,
        }]
        output = _decision_score_rows(db, "job-2", "model-2")
        self.assertEqual(len(output[0]["top_scores"]), 5)
        self.assertEqual(output[0]["top_scores"][-1], {"asset": "LLY", "score": 0.1})

    def test_rejects_empty_and_nonfinite_utility_entries(self):
        db = {"backtest_predictions": MagicMock()}
        db["backtest_predictions"].find.return_value = [
            {"timestamp": ts(2), "top_1_asset": "AVGO", "top_1_score": float("nan")},
            {"timestamp": ts(3), "top_1_asset": "LLY", "top_1_score": float("inf"),
             "top_2_asset": "AAPL", "top_2_score": 0.0},
        ]
        output = _decision_score_rows(db, "job-3", "model-3")
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["top_scores"], [{"asset": "AAPL", "score": 0.0}])

    def test_protected_u67_instrumentation_does_not_change_policy_output(self):
        decision_at = pd.Timestamp(ts(2))
        diagnostics = {}

        def scientific_policy(timestamp, holding, days):
            diagnostics[pd.Timestamp(timestamp)] = {
                "decision_reason": "HOLD_CURRENT_BEST",
                "top_1_asset": "AVGO",
                "top_1_score": 0.5,
            }
            return 1, 0.5

        symbols = ["AVGO", "LLY", "TSLA", "NVDA", "AAPL", "MSFT"]
        cache = {decision_at: np.asarray([0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])}
        observed = _observe_five_cached_utilities(scientific_policy, cache, symbols, diagnostics)
        self.assertEqual(observed(decision_at, 1, 3), (1, 0.5))
        self.assertEqual(diagnostics[decision_at]["top_4_asset"], "NVDA")
        self.assertEqual(diagnostics[decision_at]["top_5_score"], 0.1)



if __name__ == "__main__":
    unittest.main()
