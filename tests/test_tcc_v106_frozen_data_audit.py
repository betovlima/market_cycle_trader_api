from __future__ import annotations

import unittest

import pandas as pd

from scripts.audit_tcc_v106_fold1_data_parity import (
    TCC_FROZEN_MANIFEST_SHA,
    _compare_numeric_frames,
)


class FrozenTccDataAuditTests(unittest.TestCase):
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
