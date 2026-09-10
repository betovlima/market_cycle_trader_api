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

import research_exact_marginal_capital_search_v102 as hotfix  # noqa: E402


class DummyConfig:
    timeframe = "1Day"
    alpaca_historical_feed = "sip"
    alpaca_adjustment = "all"
    start_date = "2016-01-01"
    end_date = "2016-01-08"
    mongo_cache_enabled = True
    market_data_history_backfill_enabled = False

    def model_copy(self, update=None):
        clone = DummyConfig()
        for key, value in (update or {}).items():
            setattr(clone, key, value)
        return clone


class ExactMarginalCapitalSearchV102Tests(unittest.TestCase):
    def test_missing_local_candidate_uses_explicit_credentials_and_download(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [10.0, 10.2, 10.4, 10.6, 10.8],
                "high": [10.4, 10.6, 10.8, 11.0, 11.2],
                "low": [9.8, 10.0, 10.2, 10.4, 10.6],
                "close": [10.2, 10.4, 10.6, 10.8, 11.0],
                "volume": [1000.0] * 5,
            },
            index=pd.date_range("2016-01-04", periods=5, freq="B", tz="UTC"),
        )
        coverage = {"history_window_complete": True}
        hotfix._STATE.db = object()
        try:
            with patch.object(
                hotfix,
                "_ORIGINAL_LOAD_CANDIDATE_FRAME",
                side_effect=RuntimeError("Local MongoDB returned no market bars for the selected Strategy."),
            ), patch.object(
                hotfix.discovery,
                "get_alpaca_credentials",
                return_value={"api_key_id": "key", "secret_key": "secret"},
            ), patch.object(
                hotfix.discovery,
                "download_stock_bars",
                return_value=frame.copy(),
            ) as downloader, patch.object(
                hotfix.discovery,
                "validate_and_clean_bars",
                return_value=frame.copy(),
            ), patch.object(
                hotfix.discovery,
                "_history_coverage_against_baseline",
                return_value=coverage,
            ):
                loaded, loaded_coverage = hotfix._load_candidate_frame_v102(
                    object(),
                    "LB",
                    {"interval": "1Day", "feed": "sip", "adjustment": "all"},
                    pd.Timestamp("2016-01-01"),
                    pd.Timestamp("2016-01-08"),
                    DummyConfig(),
                    pd.DatetimeIndex([]),
                )
        finally:
            hotfix._STATE.db = None

        self.assertEqual(loaded_coverage, coverage)
        self.assertEqual(
            loaded.attrs.get("exact_candidate_history_source"),
            "alpaca_transient_full_history_v102",
        )
        self.assertGreaterEqual(downloader.call_count, 1)

    def test_invalid_credential_contract_is_explicit(self) -> None:
        with patch.object(hotfix.discovery, "get_alpaca_credentials", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CredentialContractError"):
                hotfix._transient_full_history(
                    object(),
                    "LB",
                    pd.Timestamp("2016-01-01"),
                    pd.Timestamp("2016-01-08"),
                    DummyConfig(),
                    pd.DatetimeIndex([]),
                )

    def test_candidate_evaluation_exposes_real_db_to_fallback_thread(self) -> None:
        sentinel = object()

        def fake_evaluation(**kwargs):
            self.assertIs(getattr(hotfix._STATE, "db", None), sentinel)
            return {"evaluation_status": "completed"}

        with patch.object(hotfix, "_ORIGINAL_CANDIDATE_EVALUATION", side_effect=fake_evaluation):
            result = hotfix._candidate_evaluation_v102(db=sentinel)

        self.assertEqual(result["evaluation_status"], "completed")
        self.assertIsNone(getattr(hotfix._STATE, "db", None))


if __name__ == "__main__":
    unittest.main()
