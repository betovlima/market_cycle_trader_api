from __future__ import annotations

import unittest
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.audit_tcc_v106_fold1_calibration_parity import (
    _load_completed_audit_report,
    _status_pair,
)


class Fold1CalibrationParityTests(unittest.TestCase):
    def test_generated_root_level_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generated_audit.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "report.json", json.dumps({
                        "audit_status": "COMPLETED",
                        "fully_audited_assets": 55,
                    })
                )
            report = _load_completed_audit_report(path)
        self.assertEqual(report["audit_status"], "COMPLETED")
        self.assertEqual(report["fully_audited_assets"], 55)

    def test_folder_prefixed_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "folder_archive.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "output/tcc_v106_fold1_parity_audit/report.json",
                    '{"audit_status": "COMPLETED"}',
                )
            report = _load_completed_audit_report(path)
        self.assertEqual(report["audit_status"], "COMPLETED")

    def test_missing_or_duplicate_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad_audit.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("summary.csv", "symbol,status\\n")
            with self.assertRaisesRegex(ValueError, "exactly one report.json"):
                _load_completed_audit_report(path)
            with ZipFile(path, "w") as archive:
                archive.writestr("report.json", "{}")
                archive.writestr("nested/report.json", "{}")
            with self.assertRaisesRegex(ValueError, "exactly one report.json"):
                _load_completed_audit_report(path)

    def test_exact_match(self):
        x = np.array([[0.0, 0.4, -np.inf], [0.0, np.nan, 0.3]])
        result = _status_pair(x, x.copy())
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["substantial_different_values"], 0)

    def test_small_float_drift(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0 + 1e-13, 2.0]])
        result = _status_pair(a, b)
        self.assertEqual(result["status"], "PRECISION_ONLY")
        self.assertEqual(result["substantial_different_values"], 0)

    def test_significant_prediction_difference(self):
        result = _status_pair(
            np.array([[0.01, 0.02]]),
            np.array([[0.01, 0.04]]),
        )
        self.assertEqual(result["status"], "DIFFERENT")
        self.assertEqual(result["first_substantial_flat_index"], 1)

    def test_mismatched_dimension(self):
        result = _status_pair(np.ones((3, 2)), np.ones((3, 3)))
        self.assertEqual(result["status"], "SHAPE_MISMATCH")

    def test_direct_script_help_without_repository_on_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "audit_tcc_v106_fold1_calibration_parity.py"),
                "--help",
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--input-audit-zip", completed.stdout)

    def test_mongo_env_initialized_early(self):
        from pathlib import Path
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "audit_tcc_v106_fold1_calibration_parity.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            script.index("load_project_environment()"),
            script.index(
                "from scripts.audit_tcc_v106_fold1_data_parity import"
            ),
        )


if __name__ == "__main__":
    unittest.main()
