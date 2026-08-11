from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_cycle_trader_api.engine.capital_rotation import (
    _precompute_market_regime_diagnostics,
    _simulate_exact,
)


def _frame(index: pd.DatetimeIndex, base: float, daily_step: float = 1.0) -> pd.DataFrame:
    opens = np.asarray([base + daily_step * i for i in range(len(index))], dtype=float)
    closes = opens + 1.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": closes + 3.0,
            "low": opens - 2.0,
            "close": closes,
            "volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )


def _config(symbols: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        initial_capital=10_000.0,
        fractional_shares=True,
        strategy_mode="COMPOUND_ROTATION_SWING_XGBOOST",
        rotation_horizon_days=40,
        rotation_target_horizons=[5, 10, 20, 40, 60],
        rotation_walk_forward_enabled=True,
        rotation_purge_days=60,
        rotation_walk_forward_calibration_days=126,
        rotation_walk_forward_test_days=504,
        rotation_downside_penalty=0.20,
        rotation_drawdown_penalty=0.35,
        calendar_anchor_assets=list(symbols),
        research_reference_assets=list(symbols),
        research_candidate_assets=[],
    )


def _fees(_side, _quantity, _price, _config):
    return {"total_fee": 0.0}


def _slippage(price, _side, _config):
    return float(price)


def test_market_regime_diagnostics_are_point_in_time() -> None:
    dates = pd.date_range("2026-01-02", periods=30, freq="B", tz="UTC")
    frames = {
        "AAPL": _frame(dates, 100.0, 1.0),
        "MSFT": _frame(dates, 200.0, 2.0),
        "SPY": _frame(dates, 400.0, 1.5),
    }

    rows = _precompute_market_regime_diagnostics(frames, list(frames), dates)
    last = rows[dates[-1]]

    expected_spy_5 = frames["SPY"].loc[dates[-1], "close"] / frames["SPY"].loc[dates[-6], "close"] - 1.0
    expected_spy_20 = frames["SPY"].loc[dates[-1], "close"] / frames["SPY"].loc[dates[-21], "close"] - 1.0
    assert last["spy_return_5"] == pytest.approx(expected_spy_5)
    assert last["spy_return_20"] == pytest.approx(expected_spy_20)
    assert last["universe_breadth_5"] == pytest.approx(1.0)
    assert last["universe_breadth_20"] == pytest.approx(1.0)
    assert last["universe_breadth_5_valid_assets"] == 3
    assert last["universe_breadth_20_valid_assets"] == 3


def test_position_risk_diagnostics_observe_path_without_changing_trades() -> None:
    dates = pd.date_range("2026-01-02", periods=6, freq="B", tz="UTC")
    aapl = _frame(dates, 99.0, 1.0)
    msft = _frame(dates, 199.0, 1.0)
    spy = _frame(dates, 399.0, 1.0)

    # The first execution buys AAPL at 100. On the same session AAPL reaches 104,
    # trades down to 98 and closes at 102. Those values are all known by that
    # day's close and therefore are valid diagnostics for the next policy check.
    aapl.loc[dates[1], ["open", "high", "low", "close"]] = [100.0, 104.0, 98.0, 102.0]
    frames = {"AAPL": aapl, "MSFT": msft, "SPY": spy}
    symbols = list(frames)

    diagnostics = {
        dates[0]: {
            "decision_diagnostics_schema_version": 2,
            "current_asset": "CASH",
            "current_score": 0.0,
            "best_asset": "AAPL",
            "best_score": 0.40,
            "final_action_asset": "AAPL",
            "final_action_score": 0.40,
            "decision_reason": "ENTER_BEST_ASSET",
        },
        dates[1]: {
            "decision_diagnostics_schema_version": 2,
            "current_asset": "AAPL",
            "current_score": 0.35,
            "best_asset": "MSFT",
            "best_score": 0.45,
            "final_action_asset": "AAPL",
            "final_action_score": 0.35,
            "decision_reason": "SWITCH_MARGIN_GUARD",
            "current_asset_rank": 2,
        },
        dates[2]: {
            "decision_diagnostics_schema_version": 2,
            "current_asset": "AAPL",
            "current_score": 0.33,
            "best_asset": "AAPL",
            "best_score": 0.33,
            "final_action_asset": "AAPL",
            "final_action_score": 0.33,
            "decision_reason": "HOLD_CURRENT_BEST",
            "current_asset_rank": 1,
        },
        dates[3]: {
            "decision_diagnostics_schema_version": 2,
            "current_asset": "AAPL",
            "current_score": 0.20,
            "best_asset": "MSFT",
            "best_score": 0.50,
            "final_action_asset": "MSFT",
            "final_action_score": 0.50,
            "decision_reason": "ROTATE_TO_BEST_ASSET",
            "current_asset_rank": 2,
        },
        dates[4]: {
            "decision_diagnostics_schema_version": 2,
            "current_asset": "MSFT",
            "current_score": 0.48,
            "best_asset": "MSFT",
            "best_score": 0.48,
            "final_action_asset": "MSFT",
            "final_action_score": 0.48,
            "decision_reason": "HOLD_CURRENT_BEST",
            "current_asset_rank": 1,
        },
    }

    actions = {
        dates[0]: (1, 0.40),
        dates[1]: (1, 0.35),
        dates[2]: (1, 0.33),
        dates[3]: (2, 0.50),
        dates[4]: (2, 0.48),
    }

    def policy(timestamp, _current_position, _holding_days):
        return actions[pd.Timestamp(timestamp)]

    result = _simulate_exact(
        "xgboost_utility",
        policy,
        frames,
        symbols,
        dates,
        _config(symbols),
        _fees,
        _slippage,
        policy_decision_diagnostics=diagnostics,
    )

    observed = result.predictions.loc[result.predictions["decision_date"] == dates[1]].iloc[0]
    assert observed["position_entry_price"] == pytest.approx(100.0)
    assert observed["position_entry_score"] == pytest.approx(0.40)
    assert observed["position_return_since_entry"] == pytest.approx(0.02)
    assert observed["position_peak_return"] == pytest.approx(0.04)
    assert observed["position_mfe_so_far"] == pytest.approx(0.04)
    assert observed["position_mae_so_far"] == pytest.approx(-0.02)
    assert observed["position_drawdown_from_peak"] == pytest.approx(102.0 / 104.0 - 1.0)
    assert observed["score_change_from_entry"] == pytest.approx(-0.05)
    assert observed["days_current_not_top1"] == 1
    assert observed["consecutive_days_current_not_top1"] == 1

    # The diagnostic path must not create additional trades or alter the planned
    # AAPL -> MSFT sequence.
    assert result.trades["action"].tolist() == ["BUY", "SELL", "BUY", "FINAL_SELL"]
    assert result.trades["asset"].tolist() == ["AAPL", "AAPL", "MSFT", "MSFT"]
    exit_row = result.trades.loc[result.trades["action"] == "SELL"].iloc[0]
    assert exit_row["position_risk_diagnostics_schema_version"] == 1
    assert exit_row["days_current_not_top1"] == 2
