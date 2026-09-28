from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_tcc_v106_clmt_input_isolation import (
    _numeric_precision_by_column,
    _predict_single,
)


class CLMTInputIsolationTests(unittest.TestCase):
    def test_ulp_difference_not_material(self):
        dates = pd.to_datetime(["2017-01-03", "2017-01-04"], utc=True)
        a = pd.DataFrame({"return_1": [0.1, 0.2]}, index=dates)
        b = pd.DataFrame(
            {"return_1": [np.nextafter(0.1, 1.0), 0.2]},
            index=dates,
        )
        rows = _numeric_precision_by_column(a, b, ["return_1"])
        self.assertEqual(rows[0]["exact_different_values"], 1)
        self.assertEqual(rows[0]["material_different_values"], 0)
        self.assertEqual(rows[0]["first_exact_difference_date"], dates[0].isoformat())
        self.assertNotEqual(
            rows[0]["first_tcc_float_hex"],
            rows[0]["first_mct_float_hex"],
        )

    def test_material_target_delta(self):
        date = pd.to_datetime(["2017-01-03"], utc=True)
        a = pd.DataFrame({"forward_risk_adjusted_utility": [0.1]}, index=date)
        b = pd.DataFrame({"forward_risk_adjusted_utility": [0.2]}, index=date)
        row = _numeric_precision_by_column(
            a, b, ["forward_risk_adjusted_utility"]
        )[0]
        self.assertEqual(row["stage"], "target")
        self.assertEqual(row["material_different_values"], 1)

    def test_mismatched_dates_are_rejected(self):
        a = pd.DataFrame(
            {"return_1": [0.1]},
            index=pd.to_datetime(["2017-01-03"], utc=True),
        )
        b = pd.DataFrame(
            {"return_1": [0.1]},
            index=pd.to_datetime(["2017-01-04"], utc=True),
        )
        with self.assertRaisesRegex(ValueError, "row-date"):
            _numeric_precision_by_column(a, b, ["return_1"])

    def test_direct_powershell_style_invocation(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "audit_tcc_v106_clmt_input_isolation.py"),
                "--help",
            ],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--calibration-zip", result.stdout)


if __name__ == "__main__":
    unittest.main()
