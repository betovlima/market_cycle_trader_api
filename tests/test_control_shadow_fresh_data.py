"""Fresh RAW/SIP MCT dados snapshot regressions; no external Alpaca calls."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

API_SRC = Path(__file__).resolve().parents[1] / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_cycle_trader_api.engine import control_shadow_market_data as market
from market_cycle_trader_api.tcc_v106_reference.config import CONFIG


def bars():
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2021-01-04", "2026-09-25")[:1050]
    dates = pd.DatetimeIndex(sessions) + pd.Timedelta(hours=4)
    x = np.arange(len(dates), dtype=float)
    prices = 100 + 0.02 * x + 2 * np.sin(x / 11)
    frame = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": 100000 + x,
    }, index=dates)
    frame.index.name = "timestamp"
    return frame


class FreshControlMarketSnapshotTests(TestCase):
    def test_creates_own_mct_dados_snapshot_with_manifest_and_structural_exclusion(self):
        raw = bars()
        cutoff = raw.index[-1].date().isoformat()
        source_config = CONFIG.model_copy(update={"start_date": "2021-01-01"})
        issue = {
            "action_type": "cash_merger",
            "acquiree_symbol": "DOC",
            "acquirer_symbol": "PEAK",
            "effective_date": cutoff,
        }
        read_actions = lambda symbol, _config: (
            ([issue], "2020-01-01", cutoff)
            if symbol == "DOC" else ([], "2020-01-01", cutoff)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dados" / "control_shadow" / "snapshots"
            with (
                patch.object(market, "ASSETS", ("AAPL", "DOC")),
                patch.object(market, "REFERENCE_ASSETS", ("AAPL",)),
                patch.object(market, "CONFIG", source_config),
                patch.object(market, "latest_safe_completed_xnys_session",
                             return_value=pd.Timestamp(cutoff)),
                patch.object(market, "_download_alpaca_bars",
                             side_effect=lambda *a, **k: raw.copy()) as download,
                patch.object(market, "_download_corporate_actions",
                             side_effect=read_actions),
            ):
                snapshot = market.download_current_control_snapshot(
                    job_id="control-shadow-a123",
                    completed_session=cutoff,
                    data_directory=root,
                    per_asset_pause_seconds=0,
                )
            self.assertEqual(download.call_count, 2)
            for call in download.call_args_list:
                self.assertEqual(call.args[1].alpaca_historical_feed, "sip")
                self.assertEqual(call.args[1].alpaca_adjustment, "raw")
                self.assertTrue(call.kwargs["single_request_daily"])
            self.assertEqual(tuple(snapshot.frames), ("AAPL",))
            self.assertEqual(snapshot.manifest["source"], "alpaca")
            self.assertEqual(snapshot.manifest["completed_session"], cutoff)
            self.assertEqual(snapshot.manifest["requested_assets"], ["AAPL", "DOC"])
            self.assertEqual(len(snapshot.manifest["structural_exclusions"]), 1)
            self.assertEqual(snapshot.manifest["structural_exclusions"][0]["symbol"], "DOC")
            self.assertTrue(snapshot.directory.joinpath("raw_bars", "DOC.csv").is_file())
            self.assertTrue(snapshot.directory.joinpath("normalized_bars", "AAPL.csv").is_file())
            self.assertFalse(snapshot.directory.joinpath("normalized_bars", "DOC.csv").exists())
            stored = json.loads((snapshot.directory / "manifest.json").read_text("utf-8"))
            self.assertEqual(stored["snapshot_sha256"], snapshot.manifest["snapshot_sha256"])
            for relative, expected in stored["file_hashes"].items():
                self.assertEqual(
                    hashlib.sha256((snapshot.directory / relative).read_bytes()).hexdigest(),
                    expected,
                )
            self.assertFalse(any(path.name.startswith(".partial-") for path in root.iterdir()))

    def test_provider_failure_fails_closed_and_cleans_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dados" / "control_shadow" / "snapshots"
            with (
                patch.object(market, "latest_safe_completed_xnys_session",
                             return_value=pd.Timestamp("2026-09-25")),
                patch.object(market, "_download_alpaca_bars",
                             side_effect=RuntimeError("Alpaca SIP 403")),
            ):
                with self.assertRaisesRegex(RuntimeError, "SIP 403"):
                    market.download_current_control_snapshot(
                        job_id="control-shadow-b456",
                        completed_session="2026-09-25",
                        data_directory=root,
                    )
            self.assertEqual(list(root.iterdir()), [])

    def test_cutoff_must_be_safely_completed(self):
        with patch.object(market, "latest_safe_completed_xnys_session",
                          return_value=pd.Timestamp("2026-09-25")):
            with self.assertRaisesRegex(ValueError, "safely completed"):
                market.download_current_control_snapshot(
                    job_id="control-shadow-future",
                    completed_session="2026-09-28",
                )


if __name__ == "__main__":
    import unittest
    unittest.main()
