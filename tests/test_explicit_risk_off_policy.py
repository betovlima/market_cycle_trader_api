from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES, _xgb_policy
from market_cycle_trader_api.engine.live_policy import build_live_rotation_policy


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, _row):
        return [self.value]


def _frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC")
    output: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        values = {feature: [0.0, 0.0] for feature in ROTATION_FEATURES}
        values.update({"open": [100.0, 101.0], "close": [100.5, 101.5]})
        output[symbol] = pd.DataFrame(values, index=index)
    return output


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        strategy_mode="COMPOUND_ROTATION_SWING_RISK_OFF",
        rotation_min_holding_days=2,
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0075,
    )


def test_positive_ranking_scores_do_not_force_market_exposure_when_cash_edges_are_negative() -> None:
    symbols = ["AAPL", "MSFT", "NVDA"]
    frames = _frames(symbols)
    ranking_models = {
        "AAPL": ConstantModel(0.40),
        "MSFT": ConstantModel(0.35),
        "NVDA": ConstantModel(0.30),
    }
    cash_edge_models = {
        "AAPL": ConstantModel(-0.01),
        "MSFT": ConstantModel(-0.02),
        "NVDA": ConstantModel(-0.03),
    }
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}

    policy = _xgb_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 0, 0) == (0, 0.0)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "RISK_OFF_ENTRY_GUARD"
    assert row["strategy_risk_off_enabled"] is True
    assert row["best_asset"] == "AAPL"
    assert row["best_score"] == 0.40
    assert row["best_cash_edge"] == -0.01
    assert row["decision_is_entry"] is False


def test_risk_off_exit_overrides_minimum_holding_when_current_asset_loses_absolute_edge() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    ranking_models = {"AAPL": ConstantModel(0.50), "MSFT": ConstantModel(0.40)}
    cash_edge_models = {"AAPL": ConstantModel(-0.005), "MSFT": ConstantModel(0.0005)}
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}

    policy = _xgb_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 1, 1) == (0, 0.0)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "RISK_OFF_EXIT_TO_CASH"
    assert row["decision_is_exit_to_cash"] is True
    assert row["cash_threshold_guard_applied"] is True
    assert row["min_hold_guard_applied"] is False


def test_risk_off_can_rotate_directly_to_a_strong_eligible_asset() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    ranking_models = {"AAPL": ConstantModel(0.50), "MSFT": ConstantModel(0.45)}
    cash_edge_models = {"AAPL": ConstantModel(-0.01), "MSFT": ConstantModel(0.02)}
    timestamp = frames["AAPL"].index[0]

    policy = _xgb_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
    )

    assert policy(timestamp, 1, 1) == (2, 0.45)


def test_risk_off_hysteresis_requires_stronger_edge_for_new_entry_than_for_holding() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    ranking_models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    cash_edge_models = {"AAPL": ConstantModel(0.0005), "MSFT": ConstantModel(-0.01)}
    timestamp = frames["AAPL"].index[0]

    policy = _xgb_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
    )

    assert policy(timestamp, 0, 0) == (0, 0.0)
    assert policy(timestamp, 1, 5) == (1, 0.40)


def test_live_policy_mirrors_backtest_risk_off_semantics() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    ranking_models = {"AAPL": ConstantModel(0.50), "MSFT": ConstantModel(0.40)}
    cash_edge_models = {"AAPL": ConstantModel(-0.01), "MSFT": ConstantModel(-0.02)}
    timestamp = frames["AAPL"].index[0]

    backtest_policy = _xgb_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
    )
    live_policy = build_live_rotation_policy(
        ranking_models,
        frames,
        symbols,
        _config(),
        0.0075,
        cash_edge_models=cash_edge_models,
    )

    assert backtest_policy(timestamp, 1, 1) == (0, 0.0)
    assert live_policy(timestamp, 1, 1) == (0, 0.0)
