from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES,
    _analysis_decision_dates,
    _cagr,
    _model_utilities,
    _precompute_model_utilities,
    _select_calendar_source_symbol,
    build_rotation_frame,
)
from market_cycle_trader_api.engine.research_challengers import (
    _soft_horizon_consensus_settings,
)


class _FakeModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.sum(axis=1).to_numpy(dtype=np.float64)


def _feature_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        feature: np.linspace(0.01, 0.02, len(index))
        for feature in ROTATION_FEATURES
    }
    data["open"] = np.linspace(100.0, 110.0, len(index))
    data["close"] = np.linspace(100.5, 110.5, len(index))
    data["high"] = np.linspace(101.0, 111.0, len(index))
    data["low"] = np.linspace(99.0, 109.0, len(index))
    data["volume"] = np.full(len(index), 1_000_000.0)
    return pd.DataFrame(data, index=index)


class TccEngineParityTests(unittest.TestCase):
    def test_calendar_source_uses_longest_valid_history(self) -> None:
        short = pd.DataFrame(
            index=pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        )
        long = pd.DataFrame(
            index=pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        )
        self.assertEqual(
            _select_calendar_source_symbol({"SHORT": short, "LONG": long}),
            "LONG",
        )

    def test_calendar_source_tie_break_is_deterministic(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        frames = {
            "BBB": pd.DataFrame(index=index),
            "AAA": pd.DataFrame(index=index),
        }
        self.assertEqual(_select_calendar_source_symbol(frames), "AAA")

    def test_build_rotation_frame_exposes_per_horizon_targets(self) -> None:
        index = pd.date_range(
            "2019-01-01",
            periods=320,
            freq="B",
            tz="America/New_York",
        )
        base = np.linspace(50.0, 90.0, len(index))
        bars = pd.DataFrame(
            {
                "open": base,
                "high": base * 1.01,
                "low": base * 0.99,
                "close": base * 1.002,
                "volume": np.linspace(1_000_000.0, 1_500_000.0, len(index)),
            },
            index=index,
        )
        config = SimpleNamespace(
            rotation_target_horizons=(5, 10),
            rotation_target_horizon_weights=(0.4, 0.6),
            slippage_bps=0.0,
            commission_rate=0.0,
            rotation_downside_penalty=0.2,
            rotation_drawdown_penalty=0.35,
            rotation_movement_capture_weight=0.35,
            rotation_trend_persistence_weight=0.20,
        )

        frame = build_rotation_frame(bars, config)

        self.assertIn("forward_horizon_utility_5", frame.columns)
        self.assertIn("forward_horizon_utility_10", frame.columns)
        self.assertIn("forward_horizon_net_log_return_5", frame.columns)
        self.assertIn("forward_horizon_net_log_return_10", frame.columns)

    def test_cagr_uses_true_initial_capital_when_supplied(self) -> None:
        index = pd.to_datetime(
            ["2020-01-01", "2021-01-01"],
            utc=True,
        )
        curve = pd.Series([150.0, 200.0], index=index, dtype=float)

        result = _cagr(curve, initial_capital=100.0)

        self.assertAlmostEqual(result, 1.0, places=2)

    def test_analysis_end_is_inclusive_for_nyse_daily_timestamp(self) -> None:
        common_dates = pd.to_datetime(
            [
                "2026-09-21 04:00:00+00:00",
                "2026-09-22 04:00:00+00:00",
                "2026-09-23 04:00:00+00:00",
            ],
            utc=True,
        )
        folds = [
            {
                "test_start_index": 1,
                "test_end_index": len(common_dates),
            }
        ]
        config = SimpleNamespace(
            start_date="2026-09-21",
            analysis_start_date="2026-09-21",
            analysis_end_date="2026-09-22",
        )

        dates = _analysis_decision_dates(common_dates, folds, config)

        self.assertEqual(
            dates[-1],
            pd.Timestamp("2026-09-22 04:00:00+00:00"),
        )
        self.assertNotIn(
            pd.Timestamp("2026-09-23 04:00:00+00:00"),
            dates,
        )

    def test_batched_utility_cache_matches_rowwise_inference(self) -> None:
        index = pd.date_range(
            "2026-01-01",
            periods=8,
            freq="B",
            tz="America/New_York",
        )
        frame = _feature_frame(index)
        frames = {"AAA": frame}
        models = {"AAA": _FakeModel()}
        timestamps = index[:6]
        config = SimpleNamespace()

        cache, profile = _precompute_model_utilities(
            models,
            frames,
            ["AAA"],
            timestamps,
            config,
        )

        self.assertEqual(profile["cache_predict_calls"], 1)
        self.assertEqual(profile["cache_session_count"], len(timestamps))
        for timestamp in timestamps:
            direct = _model_utilities(
                models,
                frames,
                ["AAA"],
                timestamp,
                config,
            )
            cached = _model_utilities(
                models,
                frames,
                ["AAA"],
                timestamp,
                config,
                utility_cache=cache,
            )
            np.testing.assert_allclose(cached, direct)

    def test_lightgbm_uses_snapshot_n_jobs_without_environment_override(self) -> None:
        import inspect

        from market_cycle_trader_api.engine import research_challengers
        from market_cycle_trader_api.services import asset_discovery_replay_cache

        target = (
            asset_discovery_replay_cache._ORIGINAL_FIT_MODELS
            or research_challengers._lightgbm_fit_models
        )
        source = inspect.getsource(target)

        self.assertIn(
            'n_jobs=int(settings["n_jobs"])',
            source,
        )
        self.assertNotIn(
            '_effective_n_jobs(int(settings["n_jobs"]))',
            source,
        )

    def test_soft_horizon_consensus_is_opt_in(self) -> None:
        base = SimpleNamespace(
            rotation_target_horizons=(5, 10, 20),
            rotation_target_horizon_weights=(0.2, 0.3, 0.5),
            research_model_settings={},
        )
        disabled = _soft_horizon_consensus_settings(base)
        self.assertFalse(disabled["enabled"])

        enabled_config = SimpleNamespace(
            rotation_target_horizons=(5, 10, 20),
            rotation_target_horizon_weights=(0.2, 0.3, 0.5),
            research_model_settings={
                "soft_horizon_consensus": {
                    "enabled": True,
                    "penalty_strength": 1.25,
                }
            },
        )
        enabled = _soft_horizon_consensus_settings(enabled_config)
        self.assertTrue(enabled["enabled"])
        self.assertEqual(enabled["penalty_strength"], 1.25)
        self.assertEqual(enabled["mode"], "weighted_rank_margin_modifier")


if __name__ == "__main__":
    unittest.main()
