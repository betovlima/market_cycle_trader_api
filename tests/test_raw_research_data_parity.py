from __future__ import annotations

from dataclasses import dataclass, replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.market_data import _download_alpaca_bars
from market_cycle_trader_api.engine.research_market_data import (
    RAW_TOTAL_CAUSAL_PROTOCOL,
    effective_research_config,
    split_normalize,
    structural_identity_issue,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
from market_cycle_trader_api.services.reproducibility import (
    market_data_manifest,
)


@dataclass(frozen=True)
class _Config:
    alpaca_adjustment: str = "all"
    research_market_data_protocol: str = RAW_TOTAL_CAUSAL_PROTOCOL
    deterministic_execution: bool = False
    numeric_thread_limit: int = 8
    research_model_settings: dict[str, object] | None = None
    research_market_data_refresh_mode: str = "reuse"

    def model_copy(self, *, update: dict[str, object]):
        return replace(self, **update)


class RawResearchDataParityTests(unittest.TestCase):
    def test_research_protocol_defaults_to_raw_total_causal(self) -> None:
        field = BacktestExecutionRequest.model_fields[
            "research_market_data_protocol"
        ]
        self.assertEqual(field.default, RAW_TOTAL_CAUSAL_PROTOCOL)

    def test_effective_research_config_forces_raw_without_mutating_source(self) -> None:
        original = _Config(
            alpaca_adjustment="all",
            deterministic_execution=False,
            numeric_thread_limit=8,
            research_model_settings={
                "lightgbm": {
                    "n_jobs": -1,
                    "early_stopping_enabled": True,
                }
            },
        )

        effective = effective_research_config(original)

        self.assertEqual(original.alpaca_adjustment, "all")
        self.assertEqual(effective.alpaca_adjustment, "raw")
        self.assertEqual(
            effective.research_market_data_protocol,
            RAW_TOTAL_CAUSAL_PROTOCOL,
        )
        self.assertEqual(
            effective.research_market_data_refresh_mode,
            "full",
        )
        self.assertFalse(effective.deterministic_execution)
        self.assertEqual(
            effective.research_model_settings["lightgbm"]["n_jobs"],
            -1,
        )
        self.assertFalse(
            effective.research_model_settings["lightgbm"][
                "early_stopping_enabled"
            ]
        )

    def test_full_research_download_matches_tcc_single_request(self) -> None:
        request = BacktestExecutionRequest(
            assets=["AAPL", "MSFT"],
            start_date="2016-01-01",
            end_date="2026-09-17",
            analysis_start_date="2016-01-01",
            analysis_end_date="2026-09-17",
            calendar_anchor_assets=["AAPL", "MSFT"],
            research_reference_assets=["AAPL"],
            research_candidate_assets=["MSFT"],
            research_market_data_mode="backtest_bootstrap_missing",
            research_market_data_protocol="raw_total_causal_v1",
            research_market_data_refresh_mode="full",
            market_data_provider="alpaca",
            alpaca_historical_feed="sip",
            alpaca_live_feed="iex",
            alpaca_adjustment="raw",
            timeframe="1Day",
            rotation_models=["xgboost_utility"],
            rotation_horizon_days=40,
            rotation_target_horizons=[5, 10, 20, 40, 60],
            rotation_target_horizon_weights=[0.10, 0.15, 0.20, 0.30, 0.25],
            rotation_minimum_training_rows=700,
            rotation_walk_forward_enabled=True,
            rotation_walk_forward_calibration_days=126,
            rotation_walk_forward_test_days=504,
            rotation_walk_forward_min_test_days=126,
            rotation_purge_days=60,
            rotation_downside_penalty=0.20,
            rotation_drawdown_penalty=0.35,
            rotation_min_holding_days=2,
            rotation_min_expected_edge=0.001,
            rotation_cash_threshold=0.0,
            rotation_switch_margin=0.0005,
            rotation_switch_margin_candidates=[0.0, 0.0025, 0.005, 0.01],
            rotation_xgb_n_estimators=300,
            rotation_xgb_learning_rate=0.035,
            rotation_xgb_max_depth=3,
            rotation_accelerator="cpu",
            rotation_allow_cpu_fallback=False,
            rotation_xgb_repetitions=1,
            rotation_seed_step=1000,
            initial_capital=10000.0,
            whole_shares=False,
            slippage_bps=0.0,
            commission_rate=0.0,
            sec_fee_rate=0.0000206,
            taf_fee_per_share=0.000195,
            taf_fee_cap=9.79,
            cat_fee_per_share=0.000003,
            xgb_min_child_weight=5.0,
            xgb_subsample=0.85,
            xgb_colsample_bytree=0.85,
            xgb_reg_alpha=0.10,
            xgb_reg_lambda=2.0,
            xgb_n_jobs=-1,
            mongo_cache_enabled=True,
            mongo_refresh_overlap_days=5,
            mongo_write_batch_size=1000,
            deterministic_execution=False,
            numeric_thread_limit=1,
            random_state=42,
            research_model_family="lightgbm_utility",
            research_model_settings={
                "schema_version": 3,
                "settings_revision": 2,
                "profile_id": "strategy",
                "lightgbm": {
                    "n_estimators": 329,
                    "learning_rate": 0.020731,
                    "max_depth": 3,
                    "num_leaves": 6,
                    "min_child_samples": 18,
                    "min_child_weight": 5.0,
                    "subsample": 0.85,
                    "subsample_freq": 0,
                    "colsample_bytree": 0.88067,
                    "reg_alpha": 0.050837,
                    "reg_lambda": 3.596305,
                    "max_bin": 255,
                    "n_jobs": -1,
                    "repetitions": 1,
                    "seed_step": 1000,
                    "random_state": 42,
                },
            },
        )
        captured: dict[str, object] = {}

        def fake_download_stock_bars(**kwargs):
            captured.update(kwargs)
            index = pd.to_datetime(
                [
                    "2016-01-04 05:00:00+00:00",
                    "2026-09-17 04:00:00+00:00",
                ],
                utc=True,
            )
            return pd.DataFrame(
                {
                    "open": [100.0, 110.0],
                    "high": [101.0, 111.0],
                    "low": [99.0, 109.0],
                    "close": [100.5, 110.5],
                    "volume": [1000.0, 1100.0],
                },
                index=index,
            )

        with patch(
            "market_cycle_trader_api.engine.market_data.get_alpaca_credentials",
            return_value={
                "api_key_id": "k",
                "secret_key": "s",
            },
        ), patch(
            "market_cycle_trader_api.engine.market_data.download_stock_bars",
            side_effect=fake_download_stock_bars,
        ):
            frame = _download_alpaca_bars(
                "AAPL",
                request,
                "2016-01-01",
                "2026-09-17",
            )

        self.assertEqual(captured["limit"], 10_000)
        self.assertEqual(captured["feed"], "sip")
        self.assertEqual(captured["adjustment"], "raw")
        self.assertEqual(
            pd.Timestamp(captured["start"]),
            pd.Timestamp("2016-01-01 00:00:00+00:00"),
        )
        self.assertEqual(
            pd.Timestamp(captured["end"]),
            pd.Timestamp("2026-09-18 00:00:00+00:00"),
        )
        self.assertEqual(
            frame.attrs["research_bar_loader"],
            "tcc_single_request_v1",
        )
        self.assertFalse(frame.attrs["research_bar_chunking"])

    def test_split_normalization_matches_tcc_rule(self) -> None:
        index = pd.to_datetime(
            [
                "2020-08-28 04:00:00+00:00",
                "2020-08-31 04:00:00+00:00",
                "2020-09-01 04:00:00+00:00",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "open": [400.0, 100.0, 105.0],
                "high": [404.0, 104.0, 109.0],
                "low": [396.0, 96.0, 101.0],
                "close": [402.0, 102.0, 107.0],
                "volume": [1_000.0, 4_000.0, 5_000.0],
            },
            index=index,
        )
        actions = [
            {
                "action_type": "forward_split",
                "ex_date": "2020-08-31",
                "old_rate": 1.0,
                "new_rate": 4.0,
                "process_date": "2020-08-31",
            }
        ]

        normalized, applied = split_normalize(frame, actions)

        self.assertEqual(len(applied), 1)
        self.assertAlmostEqual(normalized.iloc[0]["open"], 100.0)
        self.assertAlmostEqual(normalized.iloc[0]["close"], 100.5)
        self.assertAlmostEqual(normalized.iloc[0]["volume"], 4_000.0)
        self.assertAlmostEqual(normalized.iloc[1]["open"], 100.0)
        self.assertAlmostEqual(normalized.iloc[1]["close"], 102.0)
        self.assertAlmostEqual(normalized.iloc[1]["volume"], 4_000.0)
        self.assertAlmostEqual(normalized.iloc[2]["open"], 105.0)
        self.assertAlmostEqual(normalized.iloc[2]["volume"], 5_000.0)
        self.assertEqual(
            applied[0]["normalization_direction"],
            "pre_ex_date_history",
        )

    def test_dividend_does_not_adjust_model_ohlcv(self) -> None:
        index = pd.to_datetime(
            [
                "2024-01-02 05:00:00+00:00",
                "2024-01-03 05:00:00+00:00",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "open": [100.0, 99.0],
                "high": [101.0, 100.0],
                "low": [98.0, 97.0],
                "close": [100.0, 99.0],
                "volume": [1_000.0, 1_100.0],
            },
            index=index,
        )
        actions = [
            {
                "action_type": "cash_dividend",
                "ex_date": "2024-01-03",
                "cash": 1.0,
            }
        ]

        normalized, applied = split_normalize(frame, actions)

        pd.testing.assert_frame_equal(normalized, frame)
        self.assertEqual(applied, [])

    def test_structural_merger_excludes_acquiree_symbol(self) -> None:
        issue = structural_identity_issue(
            "DOC",
            [
                {
                    "action_type": "stock_merger",
                    "effective_date": "2024-03-01",
                    "process_date": "2024-03-01",
                    "acquiree_symbol": "DOC",
                    "acquirer_symbol": "PEAK",
                }
            ],
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "structural_identity_change")
        self.assertEqual(issue["symbol"], "DOC")
        self.assertEqual(issue["acquirer_symbol"], "PEAK")

    def test_structural_merger_does_not_exclude_acquirer(self) -> None:
        issue = structural_identity_issue(
            "PEAK",
            [
                {
                    "action_type": "stock_merger",
                    "effective_date": "2024-03-01",
                    "acquiree_symbol": "DOC",
                    "acquirer_symbol": "PEAK",
                }
            ],
        )
        self.assertIsNone(issue)

    def test_reproducibility_signature_includes_raw_protocol_and_splits(self) -> None:
        index = pd.date_range(
            "2020-01-02",
            periods=4,
            freq="B",
            tz="UTC",
        )
        frame = pd.DataFrame(
            {
                "open": np.arange(4, dtype=float) + 10.0,
                "high": np.arange(4, dtype=float) + 11.0,
                "low": np.arange(4, dtype=float) + 9.0,
                "close": np.arange(4, dtype=float) + 10.5,
                "volume": np.arange(4, dtype=float) + 1_000.0,
            },
            index=index,
        )
        frame.attrs["market_data_provenance"] = {
            "history_complete": True,
            "provider": "alpaca",
            "effective_provider": "alpaca",
            "historical_feed": "sip",
            "live_feed": "iex",
            "adjustment": "raw",
            "research_market_data_protocol": RAW_TOTAL_CAUSAL_PROTOCOL,
            "source_adjustment": "raw",
            "effective_adjustment": "raw_plus_causal_split_normalization",
            "corporate_action_count": 3,
            "splits_applied": 1,
            "split_normalization_direction": "pre_ex_date_history",
            "split_normalization_uses_future_events": True,
            "dividend_event_count": 2,
            "dividend_adjustment_applied": False,
            "dividend_events_used_by_model": False,
            "structural_identity_verified": True,
        }

        _, manifests = market_data_manifest({"AAA": frame})
        manifest = manifests["AAA"]

        self.assertEqual(
            manifest["research_market_data_protocol"],
            RAW_TOTAL_CAUSAL_PROTOCOL,
        )
        self.assertEqual(
            manifest["effective_adjustment"],
            "raw_plus_causal_split_normalization",
        )
        self.assertEqual(manifest["splits_applied"], 1)
        self.assertEqual(manifest["corporate_action_count"], 3)
        self.assertEqual(
            manifest["split_normalization_direction"],
            "pre_ex_date_history",
        )
        self.assertTrue(
            manifest["split_normalization_uses_future_events"]
        )
        self.assertEqual(manifest["dividend_event_count"], 2)
        self.assertFalse(manifest["dividend_adjustment_applied"])
        self.assertFalse(manifest["dividend_events_used_by_model"])


if __name__ == "__main__":
    unittest.main()
