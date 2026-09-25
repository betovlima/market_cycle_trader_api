from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.research_market_data import (
    StructuralResearchAssetExclusion,
    split_normalize,
)
from market_cycle_trader_api.engine.tcc_frozen_reference_source import (
    DATA_SOURCE_ENV,
    DEFAULT_SOURCE,
    FROZEN_ROOT_ENV,
    FROZEN_SOURCE,
    load_frozen_tcc_main_symbol,
    selected_tcc_reference_input_source,
    validate_frozen_tcc_main,
)

ASSETS = ("AAA", "DOC")


def _fixture(root: Path) -> dict:
    (root / "raw_bars").mkdir()
    (root / "corporate_actions").mkdir()
    bars = (
        "timestamp,open,high,low,close,volume,vwap,trade_count\n"
        "2020-01-02T00:00:00+00:00,100,102,99,101,10001,101,5\n"
        "2020-01-06T00:00:00+00:00,50,52,49,51,20003,51,6\n"
    )
    files = {}
    for symbol in ASSETS:
        raw = root / "raw_bars" / f"{symbol}.csv"
        raw.write_text(bars, encoding="utf-8")
        actions = root / "corporate_actions" / f"{symbol}.csv"
        action_csv = (
            "action_type,ex_date,old_rate,new_rate,acquiree_symbol,acquirer_symbol\n"
        )
        if symbol == "AAA":
            action_csv += "forward_split,2020-01-06,2,3,,\n"
        else:
            action_csv += "stock_merger,2020-01-06,,,DOC,NEXT\n"
        actions.write_text(action_csv, encoding="utf-8")
        files[f"raw_bars/{symbol}.csv"] = hashlib.sha256(
            raw.read_bytes()
        ).hexdigest()
        files[f"corporate_actions/{symbol}.csv"] = hashlib.sha256(
            actions.read_bytes()
        ).hexdigest()
    identity = {
        "schema_version": 2,
        "snapshot_name": "tcc-research-v1",
        "source": "alpaca",
        "bars": {
            "feed": "sip", "timeframe": "1Day", "adjustment": "raw",
            "start": "2016-01-01",
            "bar_snapshot_as_of_end": "2026-09-17",
        },
        "corporate_actions": {
            "query_start": "2014-12-31", "query_end": "2026-09-17",
            "types": ["forward_split", "stock_merger"],
        },
        "assets": list(ASSETS),
        "row_counts": {"AAA": 2, "DOC": 2},
        "corporate_action_counts": {"AAA": 1, "DOC": 1},
        "file_hashes": files,
    }
    identity_sha = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    manifest = {
        **identity,
        "snapshot_sha256": identity_sha,
        "created_for_experiment_version": "1.2.0-dev.1",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return manifest


class TCCFrozenInputTests(unittest.TestCase):
    def test_opt_in_is_explicit(self):
        with patch.dict(os.environ, {DATA_SOURCE_ENV: ""}):
            self.assertEqual(selected_tcc_reference_input_source(), DEFAULT_SOURCE)
        with patch.dict(os.environ, {DATA_SOURCE_ENV: FROZEN_SOURCE}):
            self.assertEqual(selected_tcc_reference_input_source(), FROZEN_SOURCE)
        with patch.dict(os.environ, {DATA_SOURCE_ENV: "silently_round"}):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                selected_tcc_reference_input_source()

    def test_validates_manifest_identity_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = _fixture(root)
            valid = validate_frozen_tcc_main(
                root, assets=ASSETS, expected_sha=manifest["snapshot_sha256"],
            )
            self.assertEqual(valid["snapshot_sha256"], manifest["snapshot_sha256"])
            (root / "raw_bars" / "AAA.csv").write_text(
                "changed", encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                validate_frozen_tcc_main(
                    root, assets=ASSETS,
                    expected_sha=manifest["snapshot_sha256"],
                )

    def test_manifest_identity_cannot_be_edited_without_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = _fixture(root)
            tampered = dict(manifest)
            tampered["bars"] = dict(manifest["bars"], adjustment="all")
            (root / "manifest.json").write_text(
                json.dumps(tampered), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pinned SHA"):
                validate_frozen_tcc_main(
                    root, assets=ASSETS,
                    expected_sha=manifest["snapshot_sha256"],
                )

    def test_split_fractional_volume_and_structural_doc_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = _fixture(root)
            frame = load_frozen_tcc_main_symbol(root, "AAA", manifest)
            self.assertEqual(frame["volume"].dtype, np.dtype("float64"))
            self.assertAlmostEqual(float(frame["volume"].iloc[0]), 15001.5)
            self.assertEqual(
                frame.attrs["market_data_provenance"]["research_access_path"],
                "verified_frozen_tcc_main_csv",
            )
            with self.assertRaises(StructuralResearchAssetExclusion):
                load_frozen_tcc_main_symbol(root, "DOC", manifest)

    def test_reference_runner_uses_frozen_loader_only_when_opted_in(self):
        from market_cycle_trader_api.engine import tcc_v106_reference_backtest as job
        from market_cycle_trader_api.tcc_v106_reference.config import (
            ASSETS as REFERENCE_ASSETS,
        )
        config = SimpleNamespace(
            assets=list(REFERENCE_ASSETS),
            start_date="2016-01-01",
            analysis_end_date="2026-09-17",
            end_date="2026-09-17",
            calendar_anchor_assets=(),
        )

        def frozen_loader(root, symbol, manifest):
            if symbol == "DOC":
                raise StructuralResearchAssetExclusion({
                    "symbol": symbol, "action_type": "stock_merger",
                    "acquirer_symbol": "NEXT",
                })
            return pd.DataFrame({
                "open": [1.0], "high": [1.0], "low": [1.0],
                "close": [1.0], "volume": [1.0],
            }, index=pd.to_datetime(["2020-01-02"], utc=True))

        with (
            patch.object(job, "effective_research_config", return_value=config),
            patch.object(job, "selected_tcc_reference_input_source", return_value=FROZEN_SOURCE),
            patch.object(job, "frozen_tcc_root_from_environment", return_value=Path("/unused")),
            patch.object(job, "validate_frozen_tcc_main", return_value={"assets": list(REFERENCE_ASSETS)}) as verify,
            patch.object(job, "load_frozen_tcc_main_symbol", side_effect=frozen_loader) as frozen,
            patch.object(job, "load_research_market_bars") as operational,
            patch.object(job, "validate_and_clean_bars", side_effect=lambda bars, _: bars),
            patch.object(job, "emit_progress"),
        ):
            frames, exclusions, _ = job._load_mct_market_frames(config)
            self.assertEqual(len(frames), 55)
            self.assertEqual([row["symbol"] for row in exclusions], ["DOC"])
            self.assertEqual(frozen.call_count, 56)
            verify.assert_called_once()
            operational.assert_not_called()

    def test_default_split_normalizer_handles_inferred_integer_volume(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-06"], utc=True)
        frame = pd.DataFrame({
            "open": [100, 50], "high": [101, 51],
            "low": [99, 49], "close": [100, 50],
            "volume": [10001, 20003],
        }, index=index)
        adjusted, _ = split_normalize(frame, [{
            "action_type": "forward_split",
            "ex_date": "2020-01-06",
            "old_rate": "2", "new_rate": "3",
        }])
        self.assertEqual(adjusted["volume"].dtype, np.dtype("float64"))
        self.assertAlmostEqual(float(adjusted["volume"].iloc[0]), 15001.5)
        self.assertEqual(int(frame["volume"].iloc[0]), 10001)


if __name__ == "__main__":
    unittest.main()
