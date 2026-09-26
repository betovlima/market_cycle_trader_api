"""Regression tests for the Strategy #12 latest safely completed XNYS cutoff."""
from __future__ import annotations

from dataclasses import dataclass, replace
import unittest
from unittest.mock import patch

import pandas as pd

from market_cycle_trader_api.api.routers import jobs
from market_cycle_trader_api.engine import market_data
from market_cycle_trader_api.engine.tcc_frozen_reference_source import (
    DATA_SOURCE_ENV, DEFAULT_SOURCE, FROZEN_SOURCE,
    selected_tcc_reference_input_source,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
from market_cycle_trader_api.tcc_v106_reference.config import CONFIG as SCIENTIFIC_TCC


@dataclass(frozen=True)
class _Config:
    end_date: str | None = "2026-09-17"

    def model_copy(self, *, update: dict):
        return replace(self, **update)


class CutoffAndSourceTests(unittest.TestCase):
    def _dynamic(self, date: str) -> str:
        with patch.object(
            market_data, "create_client",
            side_effect=AssertionError("Do not inspect stale Mongo cache for a new full-refresh job"),
        ):
            return market_data.resolve_backtest_analysis_end_date(
                _Config(end_date=None),
                now=pd.Timestamp(date),
                require_cached_common_session=False,
            )

    def test_saturday_and_sunday_use_friday_session(self):
        self.assertEqual(self._dynamic("2026-09-26T15:00:00Z"), "2026-09-25")
        self.assertEqual(self._dynamic("2026-09-27T22:00:00Z"), "2026-09-25")

    def test_friday_incomplete_bar_and_close_buffer(self):
        self.assertEqual(self._dynamic("2026-09-25T19:00:00Z"), "2026-09-24")
        self.assertEqual(self._dynamic("2026-09-25T20:10:00Z"), "2026-09-24")
        self.assertEqual(self._dynamic("2026-09-25T20:25:00Z"), "2026-09-25")

    def test_holiday_and_next_session(self):
        self.assertEqual(self._dynamic("2026-09-07T22:00:00Z"), "2026-09-04")
        self.assertEqual(self._dynamic("2026-09-28T19:00:00Z"), "2026-09-25")
        self.assertEqual(self._dynamic("2026-09-28T20:25:00Z"), "2026-09-28")

    def test_scientific_end_date_remains_locked(self):
        self.assertEqual(SCIENTIFIC_TCC.analysis_end_date, "2026-09-17")
        config = _Config(end_date=SCIENTIFIC_TCC.analysis_end_date)
        actual = market_data.resolve_backtest_analysis_end_date(
            config,
            now=pd.Timestamp("2026-09-26T15:00:00Z"),
            require_cached_common_session=False,
        )
        self.assertEqual(actual, "2026-09-17")

    def test_refreshing_job_does_not_use_stale_common_cache(self):
        with (
            patch.object(market_data, "create_client") as connect,
            patch.object(market_data, "get_database"),
            patch.object(
                market_data, "_latest_common_cached_session",
                return_value=pd.Timestamp("2026-09-17"),
            ) as stale,
        ):
            connect.return_value.close.return_value = None
            old_mode = market_data.resolve_backtest_analysis_end_date(
                _Config(end_date=None),
                now=pd.Timestamp("2026-09-26T15:00:00Z"),
            )
            new_mode = market_data.resolve_backtest_analysis_end_date(
                _Config(end_date=None),
                now=pd.Timestamp("2026-09-26T15:00:00Z"),
                require_cached_common_session=False,
            )
        self.assertEqual(old_mode, "2026-09-17")
        self.assertEqual(new_mode, "2026-09-25")
        stale.assert_called_once()

    def test_existing_strategy12_profile_is_overridden_only_for_new_job(self):
        old_profile = _Config(end_date="2026-09-17")
        selected = {"backtest_engine_binding": "tcc_v106_reference"}
        with patch.object(jobs, "selected_tcc_reference_input_source", return_value=DEFAULT_SOURCE):
            new_config, source = jobs._reference_execution_window(old_profile, selected)
        self.assertEqual(source, DEFAULT_SOURCE)
        self.assertIsNone(new_config.end_date)
        self.assertEqual(old_profile.end_date, "2026-09-17")

        with patch.object(jobs, "selected_tcc_reference_input_source", return_value=FROZEN_SOURCE):
            frozen_config, source = jobs._reference_execution_window(old_profile, selected)
        self.assertEqual(source, FROZEN_SOURCE)
        self.assertEqual(frozen_config.end_date, "2026-09-17")
        unchanged, source = jobs._reference_execution_window(
            old_profile, {"backtest_engine_binding": "ordinary"}
        )
        self.assertIs(unchanged, old_profile)
        self.assertIsNone(source)

    def test_reference_source_is_pinned_in_request(self):
        from types import SimpleNamespace
        with patch.dict(
            "os.environ",
            {DATA_SOURCE_ENV: FROZEN_SOURCE},
        ):
            self.assertEqual(
                selected_tcc_reference_input_source(
                    SimpleNamespace(tcc_reference_input_source=DEFAULT_SOURCE)
                ),
                DEFAULT_SOURCE,
            )
            self.assertEqual(
                selected_tcc_reference_input_source(
                    SimpleNamespace(tcc_reference_input_source=None)
                ),
                FROZEN_SOURCE,
            )
        self.assertIn("tcc_reference_input_source", BacktestExecutionRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
