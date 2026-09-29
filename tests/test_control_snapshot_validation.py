"""v10.8.40 snapshot provenance, calibration and admin endpoint tests.

All tests are offline. They never call Alpaca or create trading orders.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from market_cycle_trader_api.api.routers.control_shadow import StartControlValidationRequest
from market_cycle_trader_api.engine import control_snapshot_validation as engine
from market_cycle_trader_api.services import control_shadow_validation_jobs as service


ID = "control-shadow-aaaaaaaaaaaaaaaa"


def _snapshot(root: Path) -> tuple[Path, dict]:
    directory = root / ID
    directory.mkdir(parents=True)
    hashes = {}
    records = {}
    for symbol in ("AAPL", "NVDA"):
        for category, extension, payload in (
            ("raw_bars", "csv", b"timestamp,open,high,low,close,volume\n2026-09-28T04:00:00+00:00,10,11,9,10,100\n"),
            ("corporate_actions", "json", b"[]\n"),
            ("normalized_bars", "csv", b"timestamp,open,high,low,close,volume\n2026-09-28T04:00:00+00:00,10,11,9,10,100\n"),
        ):
            name = f"{category}/{symbol}.{extension}"
            path = directory / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        records[symbol] = {
            "status": "eligible", "normalized_rows": 1,
            "first_session": "2026-09-28", "last_session": "2026-09-28",
        }
    manifest = {
        "schema_version": 1,
        "source_contract": engine.SOURCE_CONTRACT,
        "source": "alpaca", "feed": "sip", "download_adjustment": "raw",
        "effective_adjustment": "raw_plus_split_normalization",
        "dividend_adjustment_applied": False,
        "completed_session": "2026-09-28",
        "requested_assets": ["AAPL", "NVDA"],
        "eligible_assets": ["AAPL", "NVDA"],
        "per_asset": records,
        "file_hashes": hashes,
    }
    manifest["snapshot_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest


class ControlSnapshotValidationTests(TestCase):
    def test_reopens_exact_source_files_without_download_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest = _snapshot(root)
            with (
                patch.object(engine, "ASSETS", ("AAPL", "NVDA")),
                patch.object(engine, "REFERENCE_ASSETS", ("AAPL", "NVDA")),
            ):
                frames, verified, path = engine.read_verified_control_snapshot(
                    ID, snapshot_root=root, expected_sha256=manifest["snapshot_sha256"],
                )
                self.assertEqual(path, directory)
                self.assertEqual(tuple(frames), ("AAPL", "NVDA"))
                self.assertEqual(verified["snapshot_sha256"], manifest["snapshot_sha256"])
                self.assertIsNotNone(frames["AAPL"].index.tz)
                (directory / "normalized_bars" / "AAPL.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "file hash changed"):
                    engine.read_verified_control_snapshot(
                        ID, snapshot_root=root, expected_sha256=manifest["snapshot_sha256"],
                    )
        with self.assertRaisesRegex(ValueError, "server-generated"):
            engine.read_verified_control_snapshot("../other")

    def test_calibration_curve_is_risk_adjusted_not_percent_return(self):
        dates = pd.date_range("2026-01-01", periods=3, tz="UTC")
        with patch.object(engine, "_training_transition_log_return", return_value=-0.01):
            curve, decisions, total = engine._calibration_curve(
                lambda timestamp, position, holding: (0, 0.0),
                {}, [], dates, MagicMock(initial_capital=10000.0,
                                         rotation_downside_penalty=0.20,
                                         rotation_drawdown_penalty=0.35),
            )
        self.assertLess(total, 0.0)
        self.assertEqual(len(curve), 2)
        self.assertEqual(len(decisions), 2)
        self.assertLess(float(curve["equity"].iloc[-1]), 10000.0)
        self.assertAlmostEqual(
            total, float(curve["risk_adjusted_reward"].sum()), places=12,
        )

    def test_validation_api_schema_requires_exact_existing_job_id(self):
        StartControlValidationRequest.model_validate({
            "confirm": "VALIDATE_EXISTING_CONTROL_SNAPSHOT_NO_ORDERS",
            "source_job_id": ID,
        })
        with self.assertRaises(ValidationError):
            StartControlValidationRequest.model_validate({
                "confirm": "VALIDATE_EXISTING_CONTROL_SNAPSHOT_NO_ORDERS",
                "source_job_id": ID,
                "data_directory": "C:/other",
            })
        with self.assertRaises(ValidationError):
            StartControlValidationRequest.model_validate({
                "confirm": "REFRESH_ALPACA_CONTROL_SHADOW_NO_ORDERS",
                "source_job_id": ID,
            })

    def test_source_job_must_be_completed_fresh_and_sha_pinned(self):
        db = MagicMock()
        db[service.SOURCE_COLLECTION].find_one.return_value = {
            "status": "completed",
            "completed_session": "2026-09-28",
            "snapshot_sha256": "a" * 64,
            "result": {
                "status": "shadow_only", "order_submission": "never",
                "decision_date": "2026-09-28",
                "calibration_score": -0.68,
                "calibrated_candidate_margin": 0.0025,
                "input_audit": {
                    "source_kind": "fresh_alpaca_raw_sip_local_mct_snapshot",
                    "snapshot_sha256": "a" * 64,
                },
            },
        }
        with (
            patch.object(service, "_require_enabled"),
            patch.object(service.threading, "Thread") as thread,
        ):
            created = service.start_snapshot_validation(db, source_job_id=ID)
        self.assertEqual(created["status"], "queued")
        self.assertEqual(created["source_job_id"], ID)
        self.assertEqual(created["source_download"], "never")
        self.assertFalse(created["order_eligible"])
        self.assertEqual(created["order_submission"], "never")
        thread.return_value.start.assert_called_once()
        inserted = db[service.COLLECTION].insert_one.call_args.args[0]
        self.assertEqual(inserted["active_key"], service.ACTIVE_KEY)
        service._THREADS.pop(created["job_id"], None)

    def test_ineligible_source_and_duplicate_run_fail_closed(self):
        db = MagicMock()
        db[service.SOURCE_COLLECTION].find_one.return_value = {
            "status": "failed", "error": "provider fail",
        }
        with patch.object(service, "_require_enabled"):
            with self.assertRaises(service.SnapshotValidationInvalid):
                service.start_snapshot_validation(db, source_job_id=ID)
        db[service.SOURCE_COLLECTION].find_one.return_value = {
            "status": "completed", "completed_session": "2026-09-28",
            "snapshot_sha256": "a" * 64,
            "result": {
                "status": "shadow_only", "decision_date": "2026-09-28",
                "order_submission": "never",
                "input_audit": {
                    "source_kind": "fresh_alpaca_raw_sip_local_mct_snapshot",
                    "snapshot_sha256": "a" * 64,
                },
            },
        }
        db[service.COLLECTION].insert_one.side_effect = DuplicateKeyError("duplicate")
        with patch.object(service, "_require_enabled"):
            with self.assertRaises(service.SnapshotValidationConflict):
                service.start_snapshot_validation(db, source_job_id=ID)


if __name__ == "__main__":
    import unittest
    unittest.main()
