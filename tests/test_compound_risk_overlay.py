from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.compound_risk_overlay import optimize_compound_risk_overlay
from market_cycle_trader_api.engine.capital_rotation import _execute_buy
from market_cycle_trader_api.engine.compound_rotation_backtest import calculate_reference_fees, apply_slippage


def _config(**overrides):
    values = dict(
        allocation_lookback_days=126,
        allocation_cvar_confidence=0.95,
        allocation_cvar_penalty=1.0,
        allocation_turnover_penalty=0.0025,
        allocation_max_asset_weight=1.0,
        allocation_signal_scale=1.0,
        rotation_horizon_days=5,
        rotation_target_horizons=[5],
        rotation_target_horizon_weights=[1.0],
        slippage_bps=0.0,
        commission_rate=0.0,
        sec_fee_rate=0.0000278,
        taf_fee_per_share=0.000166,
        taf_fee_cap=8.30,
        cat_fee_per_share=0.000003,
        whole_shares=False,
    )
    values.update(overrides)
    config = SimpleNamespace(**values)
    config.fractional_shares = not bool(config.whole_shares)
    return config


def _frame(vol=0.01):
    dates = pd.date_range("2020-01-01", periods=1100, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, vol, len(dates))
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "open": close}, index=dates)


def test_compound_risk_overlay_uses_only_selected_asset_and_can_reach_full_exposure() -> None:
    frames = {"AAA": _frame(0.008), "BBB": _frame(0.20)}
    timestamp = frames["AAA"].index[-1]
    decision = optimize_compound_risk_overlay(
        1,
        0.3,
        frames,
        ["AAA", "BBB"],
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0},
        _config(allocation_cvar_penalty=0.0, allocation_turnover_penalty=0.0),
    )
    assert decision.weights["AAA"] > 0.999999
    assert decision.weights["BBB"] == 0.0
    assert decision.cash_weight < 1e-9
    assert decision.eligible_assets == ("AAA",)


def test_compound_risk_overlay_does_not_use_opportunity_confidence_or_relative_alpha() -> None:
    frames = {"AAA": _frame(0.01), "BBB": _frame(0.015)}
    timestamp = frames["AAA"].index[-1]
    first = optimize_compound_risk_overlay(1, 0.3, frames, ["AAA", "BBB"], timestamp, {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0}, _config())
    second = optimize_compound_risk_overlay(1, -10.0, frames, ["AAA", "BBB"], timestamp, {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0}, _config())
    assert abs(first.weights["AAA"] - second.weights["AAA"]) < 1e-12
    assert first.expected_relative_alpha == second.expected_relative_alpha == 0.0
    assert first.confidence_adjusted_allocation_reward == first.allocation_reward


def test_compound_risk_overlay_turnover_does_not_double_count_cash_leg() -> None:
    frames = {"AAA": _frame(0.01), "BBB": _frame(0.015)}
    timestamp = frames["AAA"].index[-1]
    decision = optimize_compound_risk_overlay(1, 0.3, frames, ["AAA", "BBB"], timestamp, {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0}, _config(allocation_cvar_penalty=0.0, allocation_turnover_penalty=0.0))
    assert abs(decision.turnover - decision.weights["AAA"]) < 1e-9


def test_fee_aware_execute_buy_can_fully_deploy_capital_with_cat_fee() -> None:
    config = _config(cat_fee_per_share=0.000003)
    quantity, price, fees = _execute_buy(10000.0, 100.0, config, calculate_reference_fees, apply_slippage)
    total_cost = quantity * price + float(fees["total_fee"])
    assert quantity > 99.99
    assert total_cost <= 10000.0 + 1e-9
    assert 10000.0 - total_cost < 0.02


def test_risk_overlay_technical_history_fallback_is_explicit_base_policy_not_cash() -> None:
    short = _frame(0.01).iloc[-80:].copy()
    decision = optimize_compound_risk_overlay(1, 0.3, {"AAA": short}, ["AAA"], short.index[-1], {"AAA": 0.0, "CASH": 1.0}, _config())
    assert decision.optimizer_status.startswith("technical_fallback_base_policy:")
    assert decision.weights["AAA"] == 1.0
    assert decision.cash_weight == 0.0


def _full_backtest_config(**updates):
    import json
    from pathlib import Path
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    parameterization = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json"
    payload = json.loads(parameterization.read_text(encoding="utf-8"))
    payload.update(
        {
            "strategy_mode": "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY",
            "allocation_max_asset_weight": 1.0,
            "allocation_cvar_penalty": 0.0,
            "allocation_turnover_penalty": 0.0,
            "slippage_bps": 0.0,
            **updates,
        }
    )
    return BacktestRequest.model_validate(payload)


def test_fee_aware_allocation_simulator_executes_100_percent_target_instead_of_staying_cash() -> None:
    from market_cycle_trader_api.engine.capital_rotation import _simulate_optimized_allocation
    from market_cycle_trader_api.engine.optimized_allocation import AllocationDecision

    dates = pd.date_range("2025-01-01", periods=6, freq="B", tz="UTC")
    close = np.asarray([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    frames = {"AAA": pd.DataFrame({"open": close, "close": close}, index=dates)}
    config = _full_backtest_config(assets=["AAA", "BBB"])

    def policy(_timestamp, _current_weights):
        return AllocationDecision(
            weights={"AAA": 1.0},
            cash_weight=0.0,
            expected_utility=1.0,
            expected_relative_alpha=0.0,
            confidence_adjusted_relative_alpha=0.0,
            allocation_reward=1.0,
            confidence_adjusted_allocation_reward=1.0,
            normalized_cvar=0.0,
            risk_reference=0.01,
            estimated_cvar=0.0,
            turnover=1.0,
            objective_value=1.0,
            eligible_assets=("AAA",),
            optimizer_status="test_full_target",
        )

    result = _simulate_optimized_allocation(
        "test",
        policy,
        frames,
        ["AAA"],
        dates,
        config,
        calculate_reference_fees,
        apply_slippage,
    )
    non_final = result.predictions.iloc[:-1]
    assert float(non_final["market_exposure_weight"].max()) > 0.999
    assert float(non_final["cash_weight"].min()) < 0.001
    assert result.metrics["strategy_ending_capital"] > 10000.0
    assert result.metrics["simulated_buys"] >= 1


class _TimestampModel:
    def __init__(self, values):
        self.values = values

    def predict(self, row):
        return np.asarray([self.values[pd.Timestamp(row.index[0])]], dtype=float)


def test_compound_risk_overlay_preserves_base_policy_min_hold_across_fold_policy_objects() -> None:
    from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES, _compound_risk_overlay_policy

    dates = pd.date_range("2020-01-01", periods=1100, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.linspace(0.0, 0.5, len(dates)))
    frames = {}
    for symbol in ("AAA", "BBB"):
        frame = pd.DataFrame({"open": close, "close": close}, index=dates)
        for feature in ROTATION_FEATURES:
            frame[feature] = 0.0
        frames[symbol] = frame

    first, second, third = dates[-5], dates[-4], dates[-3]
    aaa_values = {date: 0.50 for date in dates}
    bbb_values = {date: 0.40 for date in dates}
    bbb_values[second] = 0.80
    bbb_values[third] = 0.80
    models = {"AAA": _TimestampModel(aaa_values), "BBB": _TimestampModel(bbb_values)}
    config = _full_backtest_config(
        assets=["AAA", "BBB"],
        rotation_min_holding_days=2,
        rotation_cash_threshold=-0.5,
        rotation_min_expected_edge=0.0,
        rotation_switch_margin=0.0,
        allocation_cvar_penalty=0.0,
        allocation_turnover_penalty=0.0,
    )
    state = {"position": 0, "holding_days": 0}
    diagnostics = {}
    fold_one = _compound_risk_overlay_policy(models, frames, ["AAA", "BBB"], config, 0.0, state=state, decision_diagnostics=diagnostics, fold_id=1)
    fold_two = _compound_risk_overlay_policy(models, frames, ["AAA", "BBB"], config, 0.0, state=state, decision_diagnostics=diagnostics, fold_id=2)

    d1 = fold_one(first, {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0})
    d2 = fold_two(second, {"AAA": 1.0, "BBB": 0.0, "CASH": 0.0})
    d3 = fold_two(third, {"AAA": 1.0, "BBB": 0.0, "CASH": 0.0})

    assert d1.weights["AAA"] > 0.999999
    assert d2.weights["AAA"] > 0.999999
    assert diagnostics[second]["decision_reason"] == "MIN_HOLD_GUARD"
    assert d3.weights["BBB"] > 0.999999
    assert diagnostics[third]["decision_reason"] == "ROTATE_TO_BEST_ASSET"
