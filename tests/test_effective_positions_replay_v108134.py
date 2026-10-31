"""Regression: a completed U75 backtest MUST expose persisted holdings.

The reported U75 ZIP 20261031T204408-402222ac contains 1578 selected_asset
rows and exactly two CASH days.  These fixtures represent its schema.
Do not infer missing historical positions from transactions or utility scores.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
from io import BytesIO, StringIO
import json
from unittest import TestCase
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    PREDICTIONS_COLLECTION,
)
from market_cycle_trader_api.services.analytics import (
    _equity_rows, _equity_position_summary, backtest_analytics,
)
from market_cycle_trader_api.services.backtest_replay_files import (
    available_backtest_files, backtest_file,
)


def dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 4, tzinfo=timezone.utc)


class PositionPersistenceReplayTests(TestCase):
    def test_analytics_reads_actual_stored_positions_with_equity(self):
        collection = MagicMock()
        collection.find.return_value = [
            {"timestamp": dt(2022, 6, 16),
             "strategy_equity": 46700., "buy_hold_equity": 10100.,
             "selected_asset": "WDAY", "previous_asset": "WDAY",
             "decision_date": dt(2022, 6, 15)},
            {"timestamp": dt(2022, 6, 17),
             "strategy_equity": 46365.65, "buy_hold_equity": 10200.,
             "selected_asset": "CASH", "previous_asset": "WDAY",
             "trade_action": "SELL", "decision_date": dt(2022, 6, 16),
             "u75_market_stress": True, "u75_cash_override": True,
             "u75_technical_candidates": 0, "u75_effective_coverage": 75},
            {"timestamp": dt(2022, 6, 21),
             "strategy_equity": 46365.65, "buy_hold_equity": 10300.,
             "selected_asset": "META", "previous_asset": "CASH",
             "trade_action": "BUY", "decision_date": dt(2022, 6, 17),
             "u75_market_stress": False, "u75_cash_override": False,
             "u75_technical_candidates": 5, "u75_effective_coverage": 75},
        ]
        db = {PREDICTIONS_COLLECTION: collection}
        equity = _equity_rows(db, "20261031T204408-402222ac",
                             "tcc_u77_fixed_challenger")
        self.assertEqual(len(equity), 3)
        self.assertEqual([r["selected_asset"] for r in equity],
                         ["WDAY", "CASH", "META"])
        self.assertEqual([r["is_cash"] for r in equity], [False, True, False])
        self.assertEqual(equity[1]["u75_effective_coverage"], 75)
        self.assertEqual(equity[1]["u75_cash_override"], True)
        summary = _equity_position_summary(equity)
        self.assertEqual(summary["recorded_sessions"], 3)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["cash_sessions"], 1)
        self.assertEqual(summary["cash_episodes"][0]["sessions"], 1)
        self.assertEqual(summary["cash_episodes"][0]["start_at"][:10], "2022-06-17")
        query, projection = collection.find.call_args.args
        self.assertEqual(query["symbol"], "PORTFOLIO")
        self.assertEqual(query["backend"], "tcc_u77_fixed_challenger")
        self.assertEqual(projection["selected_asset"], 1)
        self.assertEqual(projection["u75_cash_override"], 1)

    def test_export_contains_recorded_position_csv_and_nominal_usd_equity(self):
        records = [
            {"timestamp": "2022-06-17T04:00:00+00:00",
             "simulation_equity": 46365.65, "reference_equity": 10000.,
             "drawdown": -.007, "selected_asset": "CASH", "is_cash": True,
             "previous_asset": "WDAY", "trade_action": "SELL",
             "decision_date": "2022-06-16T04:00:00+00:00",
             "u75_cash_override": True, "u75_market_stress": True,
             "u75_technical_candidates": 0, "u75_effective_coverage": 75},
            {"timestamp": "2022-06-21T04:00:00+00:00",
             "simulation_equity": 46365.65, "reference_equity": 10100.,
             "drawdown": -.007, "selected_asset": "META", "is_cash": False,
             "previous_asset": "CASH", "trade_action": "BUY"},
        ]
        data = {"job_id": "20261031T204408-402222ac", "equity": records}
        names = {x["name"] for x in available_backtest_files(data)["files"]}
        self.assertIn("positions_history.csv", names)
        body, media, _ = backtest_file(data, "positions_history.csv")
        self.assertIn("csv", media)
        positions = list(csv.DictReader(StringIO(body.decode("utf-8-sig"))))
        self.assertEqual(positions[0]["selected_asset"], "CASH")
        self.assertEqual(positions[0]["is_cash"], "True")
        self.assertEqual(positions[1]["selected_asset"], "META")
        equity_csv = backtest_file(data, "equity_history.csv")[0].decode("utf-8-sig")
        self.assertIn("selected_asset", equity_csv.splitlines()[0])
        with ZipFile(BytesIO(backtest_file(data, "package.zip")[0])) as archive:
            self.assertIn("positions_history.csv", archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertTrue(any(p["name"] == "positions_history.csv" and p["rows"] == 2
                                for p in manifest["files"]))

    def test_legacy_equity_is_not_labelled_as_cash_or_reconstructed(self):
        collection = MagicMock()
        collection.find.return_value = [
            {"timestamp": dt(2025, 1, 2), "strategy_equity": 1000.0},
            {"timestamp": dt(2025, 1, 3), "strategy_equity": 1100.0},
        ]
        rows = _equity_rows({PREDICTIONS_COLLECTION: collection}, "old", "old-engine")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("selected_asset" not in r and "is_cash" not in r for r in rows))
        coverage = _equity_position_summary(rows)
        self.assertFalse(coverage["complete"])
        self.assertIsNone(coverage["cash_sessions"])
        self.assertEqual(coverage["missing_sessions"], 2)
        with self.assertRaises(KeyError):
            backtest_file({"job_id": "old", "equity": rows}, "positions_history.csv")

    def test_partial_coverage_cannot_be_mistaken_for_no_cash_days(self):
        rows = [
            {"timestamp": "2025-01-02", "selected_asset": "MSFT"},
            {"timestamp": "2025-01-03"},  # Unknown, not automatically MSFT
            {"timestamp": "2025-01-06", "selected_asset": "CASH"},
        ]
        summary = _equity_position_summary(rows)
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["cash_sessions"])
        self.assertEqual(summary["recorded_sessions"], 2)
        self.assertEqual(summary["cash_episodes"][0]["sessions"], 1)

    def test_backtest_api_exposes_position_coverage_without_new_simulation(self):
        fake_job = {"id": "job-u75", "status": "completed"}
        fake_rows = [
            {"timestamp": "2022-06-17T04:00:00+00:00",
             "simulation_equity": 46365.65, "selected_asset": "CASH", "is_cash": True}
        ]
        db = MagicMock()
        from market_cycle_trader_api.services import analytics as service
        with (
            patch.object(service, "_selected_backend", return_value="tcc_u77_fixed_challenger"),
            patch.object(service, "_equity_rows", return_value=fake_rows),
            patch.object(service, "_decision_score_rows", return_value=[]),
            patch.object(service, "_trade_rows", return_value=[]),
            patch.object(service, "admin_job_rotations",
                         return_value={"rotations": [], "summary": {}}),
            patch.object(service, "_public_metrics", return_value={}),
        ):
            db.__getitem__.return_value.find_one.side_effect = [
                fake_job, {"results": []},
            ]
            result = backtest_analytics(db, "job-u75")
        self.assertEqual(result["equity"][0]["selected_asset"], "CASH")
        self.assertTrue(result["position_coverage"]["complete"])
        self.assertEqual(result["position_coverage"]["cash_sessions"], 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
