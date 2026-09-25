from __future__ import annotations

import unittest

import pandas as pd

from scripts.audit_tcc_v106_fold1_data_parity import (
    TCC_FROZEN_MANIFEST_SHA,
    _compare_numeric_frames,
    _split_safe_float_frame,
    _clean_actions,
)
from market_cycle_trader_api.tcc_v106_reference.config import CONFIG as TCC_CONFIG
from market_cycle_trader_api.engine.research_market_data import split_normalize


class FrozenTccDataAuditTests(unittest.TestCase):
    def test_split_normalization_accepts_integer_csv_volume(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-06"], utc=True)
        raw = pd.DataFrame(
            {"open": [100, 50], "high": [101, 51], "low": [99, 49],
             "close": [100, 50], "volume": [10001, 20003]},
            index=dates,
        )
        splits = [{
            "action_type": "forward_split", "ex_date": "2020-01-06",
            "old_rate": "2", "new_rate": "3",
        }]
        frame, events = split_normalize(_split_safe_float_frame(raw), splits)
        self.assertEqual(len(events), 1)
        self.assertEqual(frame["volume"].dtype.kind, "f")
        self.assertAlmostEqual(float(frame["volume"].iloc[0]), 15001.5)
        self.assertEqual(int(raw["volume"].iloc[0]), 10001)

    def test_ca_rate_formatting_is_not_a_difference(self) -> None:
        fields = ["id", "action_type", "old_rate", "new_rate", "cusip"]
        frozen = [{
            "id": "one", "action_type": "forward_split",
            "old_rate": "1.0", "new_rate": "4.0", "cusip": "000123",
        }]
        mongo = [{
            "id": "one", "action_type": "forward_split",
            "old_rate": 1, "new_rate": 4, "cusip": "000123",
        }]
        self.assertEqual(
            _clean_actions(frozen, fields), _clean_actions(mongo, fields)
        )

    def test_environment_is_loaded_before_mongo_import(self) -> None:
        """Mongo connection settings are cached by mongo_repository on import."""
        from pathlib import Path

        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "audit_tcc_v106_fold1_data_parity.py"
        ).read_text(encoding="utf-8")
        setup = script.index("load_project_environment()")
        dependent_import = script.index(
            "from market_cycle_trader_api.engine.market_data import"
        )
        self.assertLess(setup, dependent_import)

    def test_reference_hash_is_locked(self) -> None:
        self.assertEqual(
            TCC_FROZEN_MANIFEST_SHA,
            "4e2fd225cc0ea05da56dad8f0628ca989ad332812a5fa3a7dc796b2b8a6d5128",
        )

    def test_identical_data_is_match(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True)
        frame = pd.DataFrame({"open": [10.0, 12.0]}, index=dates)
        differences = []
        result = _compare_numeric_frames(
            symbol="AAA", stage="raw_bars", frozen=frame, mct=frame.copy(),
            columns=["open"], details=differences,
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(differences, [])

    def test_first_substantial_difference_is_reported(self) -> None:
        dates = pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06"], utc=True
        )
        tcc = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=dates)
        mct = pd.DataFrame({"close": [10.0, 13.0, 14.0]}, index=dates)
        result = _compare_numeric_frames(
            symbol="AAA", stage="raw_bars", frozen=tcc, mct=mct,
            columns=["close"], details=[],
        )
        self.assertEqual(result["status"], "DIFFERENT")
        self.assertEqual(result["substantial_different_values"], 2)
        self.assertEqual(
            result["first_substantial_difference"]["timestamp"],
            "2020-01-03T00:00:00+00:00",
        )
        self.assertEqual(
            result["first_substantial_difference"]["column"], "close"
        )

    def test_rounding_only_is_not_material_divergence(self) -> None:
        date = pd.to_datetime(["2020-01-02"], utc=True)
        tcc = pd.DataFrame({"close": [10.0]}, index=date)
        mct = pd.DataFrame({"close": [10.0 + 1e-13]}, index=date)
        result = _compare_numeric_frames(
            symbol="AAA", stage="raw_bars", frozen=tcc, mct=mct,
            columns=["close"], details=[],
        )
        self.assertEqual(result["status"], "PRECISION_ONLY")
        self.assertEqual(result["substantial_different_values"], 0)

    def test_missing_date_is_not_match(self) -> None:
        tcc = pd.DataFrame(
            {"open": [1., 2.]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
        )
        mct = tcc.iloc[:1].copy()
        result = _compare_numeric_frames(
            symbol="AAA", stage="raw_bars", frozen=tcc, mct=mct,
            columns=["open"], details=[],
        )
        self.assertEqual(result["status"], "DIFFERENT")
        self.assertEqual(result["missing_in_mct"], 1)


if __name__ == "__main__":
    unittest.main()
