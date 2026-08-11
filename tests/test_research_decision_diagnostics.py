from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES, _xgb_policy


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
        rotation_min_holding_days=2,
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0075,
    )


def test_diagnostics_observe_the_same_rotation_without_changing_policy_output() -> None:
    symbols = ["AAPL", "MSFT", "NVDA"]
    frames = _frames(symbols)
    models = {
        "AAPL": ConstantModel(0.10),
        "MSFT": ConstantModel(0.12),
        "NVDA": ConstantModel(0.11),
    }
    timestamp = frames["AAPL"].index[0]

    plain = _xgb_policy(models, frames, symbols, _config(), 0.0075)
    diagnostics: dict[pd.Timestamp, dict] = {}
    observed = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        decision_diagnostics=diagnostics,
        fold_id=2,
        calibrated_switch_margin=0.005,
    )

    assert observed(timestamp, 1, 3) == plain(timestamp, 1, 3) == (2, 0.12)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "ROTATE_TO_BEST_ASSET"
    assert row["current_asset"] == "AAPL"
    assert row["best_asset"] == "MSFT"
    assert row["second_asset"] == "NVDA"
    assert row["top_1_asset"] == "MSFT"
    assert row["top_2_asset"] == "NVDA"
    assert row["top_3_asset"] == "AAPL"
    assert row["best_vs_second_gap"] == pytest.approx(0.01)
    assert row["best_vs_current_gap"] == pytest.approx(0.02)
    assert row["calibrated_switch_margin"] == pytest.approx(0.005)
    assert row["effective_switch_margin"] == pytest.approx(0.0075)
    assert row["q_current_position"] == pytest.approx(0.10)
    assert row["q_raw_best"] == pytest.approx(0.12)
    assert row["q_final_action"] == pytest.approx(0.12)


def test_switch_margin_guard_is_recorded_without_forcing_a_rotation() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.115), "MSFT": ConstantModel(0.12)}
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}
    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 1, 3) == (1, 0.115)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "SWITCH_MARGIN_GUARD"
    assert row["switch_margin_guard_applied"] is True
    assert row["best_vs_current_gap"] == pytest.approx(0.005)
    assert row["final_action_asset"] == "AAPL"


def test_minimum_holding_guard_is_visible_in_decision_diagnostics() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.10), "MSFT": ConstantModel(0.20)}
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}
    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 1, 1) == (1, 0.10)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "MIN_HOLD_GUARD"
    assert row["min_hold_guard_applied"] is True
    assert row["raw_best_asset"] == "MSFT"
    assert row["final_action_asset"] == "AAPL"


def test_diagnostic_policy_preserves_legacy_decision_semantics_across_guards() -> None:
    import numpy as np
    from market_cycle_trader_api.engine.capital_rotation import _xgb_utilities

    symbols = ["AAPL", "MSFT", "NVDA"]
    frames = _frames(symbols)
    timestamp = frames["AAPL"].index[0]
    config = _config()

    def legacy(models, switch_margin, current_position, holding_days):
        utilities = _xgb_utilities(models, frames, symbols, timestamp, config)
        best = int(np.nanargmax(utilities))
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])
        if (
            current_position > 0
            and np.isfinite(current_value)
            and holding_days < int(config.rotation_min_holding_days)
        ):
            return current_position, current_value
        minimum = float(config.rotation_cash_threshold)
        if best == 0 or best_value <= minimum:
            return 0, 0.0
        if current_position == 0:
            if best_value >= minimum + float(config.rotation_min_expected_edge):
                return best, best_value
            return 0, 0.0
        if best == current_position:
            return current_position, current_value
        required = max(float(config.rotation_switch_margin), float(switch_margin))
        if best_value >= current_value + required:
            return best, best_value
        return current_position, current_value

    scenarios = [
        {"AAPL": -0.02, "MSFT": -0.01, "NVDA": -0.03},
        {"AAPL": 0.10, "MSFT": 0.12, "NVDA": 0.11},
        {"AAPL": 0.115, "MSFT": 0.12, "NVDA": 0.10},
        {"AAPL": 0.0002, "MSFT": 0.0005, "NVDA": -0.01},
    ]
    for scores in scenarios:
        models = {symbol: ConstantModel(value) for symbol, value in scores.items()}
        for switch_margin in (0.0, 0.0025, 0.01):
            for current_position in range(0, len(symbols) + 1):
                for holding_days in range(0, 4):
                    diagnostics: dict[pd.Timestamp, dict] = {}
                    policy = _xgb_policy(
                        models,
                        frames,
                        symbols,
                        config,
                        switch_margin,
                        decision_diagnostics=diagnostics,
                    )
                    assert policy(timestamp, current_position, holding_days) == legacy(
                        models,
                        switch_margin,
                        current_position,
                        holding_days,
                    )
