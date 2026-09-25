from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from scripts.audit_tcc_v106_fold1_model_crossover import (
    _first_trace_difference,
    _read_reference_report,
    _read_unique_zip_member,
    _swap_columns,
)


class Fold1FourModelCrossoverTests(unittest.TestCase):
    def test_swap_only_requested_symbol_and_not_original(self):
        base = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 6.0],
        ])
        donor = np.array([
            [0.0, 9.0, 8.0, 7.0],
            [0.0, 6.0, 5.0, 4.0],
        ])
        changed = _swap_columns(base, donor, ["A", "B", "C"], ["B"])
        np.testing.assert_array_equal(changed[:, 0], base[:, 0])
        np.testing.assert_array_equal(changed[:, 1], base[:, 1])
        np.testing.assert_array_equal(changed[:, 2], donor[:, 2])
        np.testing.assert_array_equal(changed[:, 3], base[:, 3])
        np.testing.assert_array_equal(base[:, 2], [2.0, 5.0])

    def test_first_transition_difference(self):
        base = [
            {"date": "1", "position_before": 0, "action": 1},
            {"date": "2", "position_before": 1, "action": 1},
        ]
        altered = [
            {"date": "1", "position_before": 0, "action": 1},
            {"date": "2", "position_before": 1, "action": 2},
        ]
        self.assertIsNone(_first_trace_difference(base, base))
        self.assertEqual(
            _first_trace_difference(base, altered)["date"], "2"
        )

    def test_zip_reader_accepts_root_and_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "audit.zip"
            with ZipFile(file, "w") as zipped:
                zipped.writestr("report.json", '{"ok": true}')
            self.assertEqual(
                json.loads(_read_unique_zip_member(file, "report.json")),
                {"ok": True},
            )
            with ZipFile(file, "w") as zipped:
                zipped.writestr("folder/report.json", '{"ok": true}')
            self.assertEqual(
                json.loads(_read_unique_zip_member(file, "report.json")),
                {"ok": True},
            )

    def test_duplicate_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "audit.zip"
            with ZipFile(file, "w") as zipped:
                zipped.writestr("report.json", "{}")
                zipped.writestr("nested/report.json", "{}")
            with self.assertRaisesRegex(ValueError, "Expected one"):
                _read_unique_zip_member(file, "report.json")

    def test_direct_script_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        process = subprocess.run(
            [sys.executable,
             str(root / "scripts" / "audit_tcc_v106_fold1_model_crossover.py"),
             "--help"],
            cwd=root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--calibration-zip", process.stdout)


if __name__ == "__main__":
    unittest.main()
