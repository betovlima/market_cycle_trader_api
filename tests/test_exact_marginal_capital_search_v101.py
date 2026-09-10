from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_exact_marginal_capital_search_v101 as hotfix  # noqa: E402


class ExactMarginalCapitalSearchV101Tests(unittest.TestCase):
    def test_missing_local_candidate_falls_back_to_transient_full_history(self) -> None:
        index = pd.date_range("2016-01-04", periods=3, freq="B", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "high": [10.6, 11.0, 11.5],
                "low": [9.8, 10.3, 10.8],
                "close": [10.4, 10.9, 11.3],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=index,
        )
        coverage = {"history_window_complete": True}

        with patch.object(
            hotfix,
            "_ORIGINAL_LOAD_CANDIDATE_FRAME",
            side_effect=RuntimeError("Local MongoDB returned no market bars for the selected Strategy."),
        ), patch.object(
            hotfix.discovery,
            "_candidate_history_coverage",
            return_value=(frame.copy(), coverage),
        ) as fallback:
            loaded, loaded_coverage = hotfix._load_candidate_frame_with_transient_fallback(
                object(),
                "LB",
                {"interval": "1Day", "feed": "sip", "adjustment": "all"},
                pd.Timestamp("2016-01-01"),
                pd.Timestamp("2026-09-04"),
                object(),
                pd.DatetimeIndex([]),
            )

        self.assertEqual(loaded_coverage, coverage)
        self.assertEqual(
            loaded.attrs.get("exact_candidate_history_source"),
            "alpaca_transient_full_history",
        )
        fallback.assert_called_once()

    def test_local_candidate_does_not_call_transient_fallback(self) -> None:
        frame = pd.DataFrame({"close": [1.0]})
        coverage = {"history_window_complete": True}
        with patch.object(
            hotfix,
            "_ORIGINAL_LOAD_CANDIDATE_FRAME",
            return_value=(frame.copy(), coverage),
        ), patch.object(
            hotfix.discovery,
            "_candidate_history_coverage",
        ) as fallback:
            loaded, _ = hotfix._load_candidate_frame_with_transient_fallback(
                object(),
                "CCS",
                {},
                pd.Timestamp("2016-01-01"),
                pd.Timestamp("2026-09-04"),
                object(),
                pd.DatetimeIndex([]),
            )

        self.assertEqual(
            loaded.attrs.get("exact_candidate_history_source"),
            "local_mongodb_cache",
        )
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
