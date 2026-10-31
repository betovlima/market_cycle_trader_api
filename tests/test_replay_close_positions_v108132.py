"""A rotation decision at close(t) must not be reported as CASH at close(t)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import unittest
from unittest.mock import MagicMock

from market_cycle_trader_api.services.analytics import _equity_rows
from market_cycle_trader_api.services.backtest_replay_files import backtest_file


def day(day_number: int):
    return datetime(2025, 1, day_number, 21, tzinfo=timezone.utc)


class ActualClosePositionTests(unittest.TestCase):
    def test_replay_uses_persisted_end_of_session_position_not_decision(self):
        db = {"backtest_predictions": MagicMock()}
        # Jan 2 closes invested even though a decision to CASH has been made;
        # Jan 3 executes at next open and closes fully in CASH.
        db["backtest_predictions"].find.return_value = [
            {"timestamp": day(2), "strategy_equity": 1000.0, "selected_asset": "NVDA"},
            {"timestamp": day(3), "strategy_equity": 1002.0, "selected_asset": "CASH"},
            {"timestamp": day(6), "strategy_equity": 1002.0, "selected_asset": "MSFT"},
            {"timestamp": day(7), "strategy_equity": 1003.0},
        ]
        records = _equity_rows(db, "job", "backend")
        self.assertEqual([row["selected_asset"] for row in records], ["NVDA", "CASH", "MSFT", None])
        self.assertEqual([row["simulation_equity"] for row in records], [1000.0, 1002.0, 1002.0, 1003.0])
        projection = db["backtest_predictions"].find.call_args.args[1]
        self.assertEqual(projection["selected_asset"], 1)
        blob, mime, _ = backtest_file({"job_id": "job", "equity": records}, "equity_history.csv")
        csv_rows = list(csv.DictReader(StringIO(blob.decode("utf-8-sig"))))
        self.assertEqual([row["selected_asset"] for row in csv_rows], ["NVDA", "CASH", "MSFT", ""])
        self.assertIn("selected_asset", csv_rows[0])
        self.assertEqual(mime.split(";")[0], "text/csv")

    def test_missing_position_is_unknown_even_if_capital_is_recorded(self):
        db = {"backtest_predictions": MagicMock()}
        db["backtest_predictions"].find.return_value = [
            {"timestamp": day(2), "strategy_equity": 1.0},
            {"timestamp": day(3), "strategy_equity": 2.0, "selected_asset": None},
        ]
        self.assertEqual([x["selected_asset"] for x in _equity_rows(db, "job", "backend")], [None, None])


if __name__ == "__main__":
    unittest.main()
