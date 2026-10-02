from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from market_cycle_trader_api.engine import tcc_control_operational_runtime as runtime


class TccControlPeakExitDiagnosticsTests(unittest.TestCase):
    def test_operational_wrapper_adds_peak_exit_diagnostics_after_scientific_run(self) -> None:
        symbols = list(runtime.TCC_CONTROL_REQUESTED_ASSETS[:2])
        index = pd.to_datetime(
            [
                "2026-01-05T05:00:00Z",
                "2026-01-06T05:00:00Z",
                "2026-01-07T05:00:00Z",
                "2026-01-08T05:00:00Z",
                "2026-01-09T05:00:00Z",
                "2026-01-12T05:00:00Z",
                "2026-01-13T05:00:00Z",
                "2026-01-14T05:00:00Z",
            ],
            utc=True,
        )
        frames = {
            symbol: pd.DataFrame(
                {
                    "open": [10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5],
                    "high": [12.0, 15.0, 18.0, 14.0, 15.0, 16.0, 15.0, 15.5],
                    "low": [9.5, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
                    "close": [11.0, 12.0, 14.0, 13.0, 14.0, 15.0, 14.5, 15.0],
                },
                index=index,
            )
            for symbol in symbols
        }
        result = SimpleNamespace(
            metrics={"strategy_ending_capital": 12345.67},
            trades=pd.DataFrame(
                [
                    {
                        "timestamp": index[0],
                        "action": "BUY",
                        "asset": symbols[0],
                        "execution_price": 10.0,
                    },
                    {
                        "timestamp": index[2],
                        "action": "SELL",
                        "asset": symbols[0],
                        "entry_timestamp": index[0],
                        "entry_price": 10.0,
                        "execution_price": 12.0,
                        "position_return": 0.2,
                        "holding_bars": 2,
                    },
                ]
            ),
        )

        with (
            patch.object(runtime, "build_tcc_control_replay_config", return_value=object()),
            patch.object(runtime, "run_research_challenger", return_value=[result]),
        ):
            output = runtime.run_tcc_control_operational_backtest(frames, object())

        self.assertIs(output[0], result)
        self.assertEqual(result.metrics["strategy_ending_capital"], 12345.67)
        sell = result.trades.loc[result.trades["action"] == "SELL"].iloc[0]
        self.assertEqual(int(sell["peak_exit_diagnostics_schema_version"]), 1)
        self.assertAlmostEqual(float(sell["peak_price_while_held"]), 15.0)
        self.assertAlmostEqual(float(sell["exit_distance_from_peak_pct"]), 20.0)
        self.assertAlmostEqual(float(sell["peak_capture_pct"]), 40.0)
        self.assertAlmostEqual(float(sell["max_runup_pct"]), 50.0)
        self.assertEqual(int(sell["days_from_peak_to_exit"]), 1)
        self.assertAlmostEqual(float(sell["post_exit_peak_5d_pct"]), 50.0)


if __name__ == "__main__":
    unittest.main()
