from __future__ import annotations

import pandas as pd

from market_cycle_trader_api.engine.rotation_diagnostics import enrich_trade_diagnostics


def _frame(opens: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=len(opens), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": opens,
            "volume": [1000] * len(opens),
        },
        index=index,
    )


def test_rotation_diagnostics_measure_value_added_excursions_and_opportunity_cost() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B", tz="UTC")
    frames = {
        "AAA": _frame([100, 102, 104, 106], [103, 105, 107, 108], [98, 101, 103, 105]),
        "BBB": _frame([50, 51, 55, 60], [51, 53, 60, 62], [49, 50, 54, 59]),
        "CCC": _frame([20, 22, 25, 30], [22, 25, 30, 31], [19, 21, 24, 29]),
    }
    records = [
        {
            "timestamp": dates[0],
            "action": "SELL",
            "asset": "AAA",
            "rotation_id": "r1",
            "rotation_from_asset": "AAA",
            "rotation_to_asset": "BBB",
            "entry_timestamp": dates[0],
            "entry_price": 100.0,
            "position_return": 0.0,
        },
        {
            "timestamp": dates[0],
            "action": "BUY",
            "asset": "BBB",
            "rotation_id": "r1",
            "rotation_from_asset": "AAA",
            "rotation_to_asset": "BBB",
            "entry_timestamp": dates[0],
            "entry_price": 50.0,
            "position_return": 0.0,
        },
        {
            "timestamp": dates[3],
            "action": "FINAL_SELL",
            "asset": "BBB",
            "entry_timestamp": dates[0],
            "entry_price": 50.0,
            "position_return": 0.20,
            "holding_bars": 3,
        },
    ]

    enriched = enrich_trade_diagnostics(records, frames, ["AAA", "BBB", "CCC"])
    buy = enriched[1]
    exit_row = enriched[2]

    assert round(float(buy["chosen_market_return"]), 6) == 0.20
    assert round(float(buy["counterfactual_previous_asset_return"]), 6) == 0.06
    assert round(float(buy["rotation_value_added"]), 6) == 0.14
    assert buy["rotation_regret"] == 0.0
    assert buy["best_alternative_asset"] == "CCC"
    assert round(float(buy["best_alternative_return"]), 6) == 0.50
    assert round(float(buy["opportunity_cost"]), 6) == 0.30
    assert round(float(exit_row["maximum_favorable_excursion"]), 6) == 0.20
    assert round(float(exit_row["maximum_adverse_excursion"]), 6) == -0.02
    assert round(float(exit_row["profit_capture_ratio"]), 6) == 1.0
