"""Read-only Alpaca cumulative Live performance, separate from position P/L."""
from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_cycle_trader_api.services.live_winner_performance import (
    first_live_winner_date,
    live_winner_performance,
)
from market_cycle_trader_api.services.live_winner_session import LIVE_RUNS
from market_cycle_trader_api.services.live_order_execution import ORDER_LEDGER
from market_cycle_trader_api.services.live_trader_state import LIVE_STATE


def epoch(day: str) -> int:
    return int(datetime.fromisoformat(day + "T12:00:00+00:00").timestamp())


class LiveWinnerPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.collections = {
            ORDER_LEDGER: MagicMock(),
            LIVE_RUNS: MagicMock(),
            LIVE_STATE: MagicMock(),
        }
        self.db.__getitem__.side_effect = self.collections.__getitem__
        for collection in self.collections.values():
            collection.find_one.return_value = None

    def test_canonical_account_uses_earliest_verified_winner_or_reconciled_entry(self):
        self.db[LIVE_RUNS].find_one.return_value = {
            "execution_session": "2026-10-29",
        }
        self.db[LIVE_STATE].find_one.return_value = {
            "entry_session": "2026-10-20",
        }
        since = first_live_winner_date(
            self.db, account_id="alpaca-main", canonical_account=True,
        )
        self.assertEqual(since, date(2026, 10, 20))
        self.db[LIVE_RUNS].find_one.assert_called_once()
        self.assertEqual(
            self.db[LIVE_RUNS].find_one.call_args.args[0]["live_account_id"],
            "alpaca-main",
        )
        self.assertEqual(
            self.db[LIVE_STATE].find_one.call_args.args[0],
            {"_id": "alpaca-main"},
        )

    def test_trader_starts_at_own_grant_and_older_verified_fills_survive_reconnect(self):
        initial = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
        self.assertEqual(
            first_live_winner_date(self.db, account_id="alpaca-user", joined_at=initial),
            date(2026, 11, 4),
        )
        self.db[ORDER_LEDGER].find_one.return_value = {
            "session": "2026-10-27",
        }
        self.assertEqual(
            first_live_winner_date(self.db, account_id="alpaca-user", joined_at=initial),
            date(2026, 10, 27),
        )
        self.db[LIVE_RUNS].find_one.assert_not_called()
        self.db[LIVE_STATE].find_one.assert_not_called()
        self.db[ORDER_LEDGER].find_one.assert_called_with(
            {"broker_account_id": "alpaca-user", "status": "filled"},
            {"created_at": 1, "filled_at": 1, "session": 1},
            sort=[("created_at", 1)],
        )

    def test_broker_cashflow_adjusted_cumulative_return_is_not_equity_delta(self):
        broker = MagicMock()
        broker.get_portfolio_history.return_value = SimpleNamespace(
            timestamp=[epoch("2026-11-03"), epoch("2026-11-04")],
            equity=[100.0, 610.0],
            profit_loss=[0.0, 10.0],
            profit_loss_pct=[0.0, 0.10],
            base_value=100.0,
            base_value_asof="2026-11-02",
            cashflow={"CSD": [0, 500], "CSW": [0, 0]},
        )
        result = live_winner_performance(broker, start_date=date(2026, 11, 3))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["return_pct"], 0.10)
        self.assertEqual(result["profit_loss"], 10.0)
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(result["start_date"], "2026-11-02")
        self.assertEqual(result["winner_enrolled_date"], "2026-11-03")
        request = broker.get_portfolio_history.call_args.args[0]
        self.assertEqual(request.timeframe, "1D")
        self.assertEqual(request.cashflow_types, "ALL")
        self.assertIsNotNone(request.start)
        self.assertIsNotNone(request.end)
        broker.submit_order.assert_not_called()

    def test_missing_base_asof_is_reported_as_partial_not_full_history(self):
        broker = MagicMock()
        broker.get_portfolio_history.return_value = SimpleNamespace(
            timestamp=[epoch("2026-11-03"), epoch("2026-11-04")],
            equity=[100, 120], profit_loss=[0, 20],
            profit_loss_pct=[0, 0.20], base_value=100,
        )
        result = live_winner_performance(broker, start_date=date(2026, 11, 3))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["coverage"], "from_first_daily_close")
        self.assertEqual(result["return_pct"], 0.20)
        self.assertEqual(result["start_date"], "2026-11-03")

    def test_unfunded_or_missing_history_is_not_turned_into_zero_return(self):
        broker = MagicMock()
        self.assertIsNone(live_winner_performance(broker, start_date=None)["return_pct"])
        broker.get_portfolio_history.assert_not_called()
        broker.get_portfolio_history.return_value = SimpleNamespace(
            timestamp=[epoch("2026-11-03")], equity=[0],
            profit_loss=[0], profit_loss_pct=[0], base_value=0,
        )
        result = live_winner_performance(broker, start_date=date(2026, 11, 3))
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["profit_loss"])

    def test_stale_or_incomplete_data_does_not_show_spurious_profit(self):
        broker = MagicMock()
        broker.get_portfolio_history.return_value = SimpleNamespace(
            timestamp=[epoch("2026-10-20")], equity=[100],
            profit_loss=[500], profit_loss_pct=[5], base_value=100,
            base_value_asof="2026-10-19",
        )
        result = live_winner_performance(broker, start_date=date(2026, 11, 3))
        self.assertEqual(result["status"], "unavailable")
        broker.get_portfolio_history.side_effect = RuntimeError("Alpaca temporarily down")
        result = live_winner_performance(broker, start_date=date(2026, 11, 3))
        self.assertEqual(result["reason"], "alpaca_history_unavailable")

    def test_computed_performance_does_not_write_or_use_other_account(self):
        broker = MagicMock()
        broker.get_portfolio_history.return_value = SimpleNamespace(
            timestamp=[epoch("2026-11-04")], equity=[150],
            profit_loss=[0], profit_loss_pct=[0], base_value=150,
        )
        first_live_winner_date(
            self.db, account_id="alpaca-A",
            joined_at=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            self.db[ORDER_LEDGER].find_one.call_args.args[0]["broker_account_id"],
            "alpaca-A",
        )
        self.db[ORDER_LEDGER].insert_one.assert_not_called()
        self.db[LIVE_RUNS].update_one.assert_not_called()
        broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
