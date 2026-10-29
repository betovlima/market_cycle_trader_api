"""Check that replay downloads contain only stored data for the selected job."""
from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import unittest
from zipfile import ZipFile

from market_cycle_trader_api.services.backtest_replay_files import (
    PUBLIC_FILES, available_backtest_files, backtest_file,
)


def sample(job_id: str = "job-123"):
    return {
        "job_id": job_id,
        "created_at": "2026-10-01T00:00:00Z",
        "finished_at": "2026-10-02T00:00:00Z",
        "metrics": {"ending_capital": 1250.75, "sharpe": 1.2},
        "rotation_summary": {"count": 1},
        "equity": [
            {"timestamp": "2026-01-02T21:00:00Z", "simulation_equity": 1000.0, "reference_equity": 1001.0, "drawdown": 0.0},
            {"timestamp": "2026-01-05T21:00:00Z", "simulation_equity": 1250.75, "reference_equity": 1010.0, "drawdown": -0.02},
        ],
        "rotations": [
            {"executed_at": "2026-01-05T14:30:00Z", "from_asset": "CASH", "to_asset": "NVDA",
             "realized_pnl": 7.5, "transaction_fees": 0.5, "buy_reason": None, "sell_reason": None}
        ],
        "decision_scores": [
            {"execution_date": "2026-01-05", "decision_date": "2026-01-02", "current_asset": "CASH",
             "current_score": 0.25, "best_asset": "NVDA", "best_score": 0.45,
             "top_scores": [{"asset": "NVDA", "score": 0.45}, {"asset": "MSFT", "score": 0.4}]}
        ],
        "monthly_returns": [{"month": "2026-01", "return": 0.25}],
    }


class ReplayFilesTests(unittest.TestCase):
    def test_list_and_download_are_bound_to_single_job(self):
        first = sample("one")
        second = sample("two")
        listing = available_backtest_files(first)
        self.assertEqual(listing["job_id"], "one")
        self.assertEqual({f["name"] for f in listing["files"]}, set(PUBLIC_FILES))
        left, _, left_name = backtest_file(first, "analytics.json")
        right, _, right_name = backtest_file(second, "analytics.json")
        self.assertEqual(json.loads(left)["job_id"], "one")
        self.assertEqual(json.loads(right)["job_id"], "two")
        self.assertNotEqual(left, right)
        self.assertIn("backtest_one_", left_name)
        self.assertIn("backtest_two_", right_name)

    def test_zip_checksums_and_source_limitations(self):
        archive_bytes, media_type, _ = backtest_file(sample(), "package.zip")
        self.assertEqual(media_type, "application/zip")
        with ZipFile(BytesIO(archive_bytes)) as archive:
            self.assertEqual(set(archive.namelist()), set(PUBLIC_FILES) - {"package.zip"})
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["source"], "persisted_backtest_analytics")
            self.assertIn("not original", manifest["limitations"])
            self.assertEqual(manifest["job_id"], "job-123")
            for entry in manifest["files"]:
                contents = archive.read(entry["name"])
                self.assertEqual(sha256(contents).hexdigest(), entry["sha256"])
                self.assertEqual(len(contents), entry["bytes"])
            rows = list(csv.DictReader(StringIO(archive.read("decision_scores.csv").decode("utf-8-sig"))))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["top_2_asset"], "MSFT")
            self.assertEqual(rows[0]["top_2_score"], "0.4")
            self.assertEqual(json.loads(archive.read("metrics.json"))["metrics"]["ending_capital"], 1250.75)

    def test_absent_decision_scores_are_not_fabricated(self):
        data = sample()
        data["decision_scores"] = []
        data["rotations"] = []
        files = {item["name"] for item in available_backtest_files(data)["files"]}
        self.assertNotIn("decision_scores.csv", files)
        self.assertNotIn("rotations.csv", files)
        for filename in ("decision_scores.csv", "rotations.csv"):
            with self.assertRaises(KeyError):
                backtest_file(data, filename)
        with ZipFile(BytesIO(backtest_file(data, "package.zip")[0])) as archive:
            self.assertNotIn("decision_scores.csv", archive.namelist())
            self.assertNotIn("rotations.csv", archive.namelist())

    def test_whitelist_and_formula_safety(self):
        data = sample("../../bad")
        data["rotations"][0]["buy_reason"] = "=HYPERLINK(\"https://evil.example\")"
        for filename in ("../secrets", "unknown.csv", ".env", "package.zip/../../other"):
            with self.assertRaises(KeyError):
                backtest_file(data, filename)
        body, _, download_name = backtest_file(data, "rotations.csv")
        text = body.decode("utf-8-sig")
        self.assertIn("'=HYPERLINK", text)
        self.assertNotIn("../", download_name)


if __name__ == "__main__":
    unittest.main()
