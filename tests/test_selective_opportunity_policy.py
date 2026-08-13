from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES, _xgb_policy
from market_cycle_trader_api.engine.live_policy import build_live_rotation_policy
from market_cycle_trader_api.engine.selective_opportunity import SelectiveOpportunityGate, _relative_confidence


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, _row):
        return [self.value]


def _frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC")
    output: dict[str, pd.DataFrame] = {}
    for offset, symbol in enumerate(symbols):
        values = {feature: [0.01 + offset * 0.001, 0.01] for feature in ROTATION_FEATURES}
        values.update({"open": [100.0, 101.0], "close": [100.5, 101.5]})
        output[symbol] = pd.DataFrame(values, index=index)
    return output


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        strategy_mode="COMPOUND_ROTATION_SWING_SELECTIVE",
        rotation_min_holding_days=2,
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0075,
    )


def _gate(probability: float) -> SelectiveOpportunityGate:
    return SelectiveOpportunityGate(
        model=None,
        threshold=0.5,
        constant_probability=float(probability),
        training_rows=100,
        positive_rate=0.5,
        threshold_validation_rows=20,
        threshold_validation_score=0.1,
    )


def test_selective_gate_rejects_market_exposure_and_records_probability() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}

    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        opportunity_gate=_gate(0.25),
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 1, 1) == (0, 0.0)
    row = diagnostics[timestamp]
    assert row["decision_reason"] == "SELECTIVE_OPPORTUNITY_REJECT"
    assert row["strategy_selective_opportunity_enabled"] is True
    assert row["opportunity_probability"] == 0.25
    assert row["opportunity_confidence"] == 0.25
    assert row["opportunity_threshold"] == 0.5
    assert row["opportunity_accepted"] is False
    assert row["decision_is_exit_to_cash"] is True


def test_selective_gate_accepts_and_preserves_rotation_policy() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]

    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        opportunity_gate=_gate(0.80),
    )

    assert policy(timestamp, 0, 0) == (1, 0.40)


def test_live_policy_matches_selective_backtest_rejection() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]
    gate = _gate(0.20)

    backtest = _xgb_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        opportunity_gate=gate,
    )
    live = build_live_rotation_policy(
        models,
        frames,
        symbols,
        _config(),
        0.0075,
        opportunity_gate=gate,
    )

    assert backtest(timestamp, 1, 1) == (0, 0.0)
    assert live(timestamp, 1, 1) == (0, 0.0)


def test_selective_gate_fits_probability_model_and_calibrates_threshold() -> None:
    import numpy as np
    from market_cycle_trader_api.engine.selective_opportunity import fit_selective_opportunity_gate

    symbols = ["AAPL", "MSFT"]
    index = pd.date_range("2025-01-02", periods=140, freq="B", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for symbol_offset, symbol in enumerate(symbols):
        values: dict[str, list[float]] = {}
        for feature_index, feature in enumerate(ROTATION_FEATURES):
            values[feature] = [
                0.01 + 0.0002 * day + 0.0001 * feature_index + 0.001 * symbol_offset
                for day in range(len(index))
            ]
        values["open"] = [100.0 + day * 0.1 for day in range(len(index))]
        values["close"] = [100.05 + day * 0.1 for day in range(len(index))]
        values["forward_net_log_return"] = [
            (0.02 if day % 3 else -0.015) if symbol == "AAPL" else 0.005
            for day in range(len(index))
        ]
        frames[symbol] = pd.DataFrame(values, index=index)

    def utilities(_models, _frames, _symbols, timestamp):
        location = index.get_loc(timestamp)
        return np.asarray([0.0, 0.30 + location * 0.001, 0.20], dtype=float)

    gate = fit_selective_opportunity_gate(
        {},
        frames,
        symbols,
        index[:126],
        utilities,
        random_state=42,
        label_horizon=20,
    )

    assert gate.training_rows == 126
    assert 0.0 < gate.positive_rate < 1.0
    assert 0.0 <= gate.threshold <= 0.90
    assert gate.calibration_method == "prequential_relative_confidence_v2"
    assert gate.threshold_validation_rows > 0
    assert gate.threshold_validation_accepted >= 8
    assert len(gate.reference_probabilities) == 126
    probability = gate.probability({
        "best_score": 0.40,
        "second_score": 0.20,
        "best_vs_second_gap": 0.20,
        "universe_score_mean": 0.30,
        "universe_score_std": 0.10,
        "best_score_zscore": 1.0,
        "positive_score_fraction": 1.0,
        "best_return_5": 0.02,
        "best_return_20": 0.04,
        "best_return_60": 0.08,
        "best_vol_20": 0.02,
        "best_vol_60": 0.03,
        "best_trend_efficiency_20": 0.5,
        "best_trend_efficiency_60": 0.6,
        "universe_breadth_5": 0.8,
        "universe_breadth_20": 0.7,
    })
    assert 0.0 <= probability <= 1.0


def test_relative_confidence_normalizes_probability_scale() -> None:
    import numpy as np

    low_scale = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
    high_scale = np.asarray([0.70, 0.75, 0.80, 0.85, 0.90], dtype=float)

    assert _relative_confidence(0.04, low_scale) == 0.8
    assert _relative_confidence(0.85, high_scale) == 0.8


def test_gate_acceptance_uses_relative_confidence_not_raw_probability() -> None:
    import numpy as np

    class ProbabilityModel:
        def predict_proba(self, _frame):
            return np.asarray([[0.8, 0.2]], dtype=float)

    gate = SelectiveOpportunityGate(
        model=ProbabilityModel(),
        threshold=0.75,
        constant_probability=None,
        training_rows=100,
        positive_rate=0.4,
        threshold_validation_rows=40,
        threshold_validation_score=0.1,
        reference_probabilities=(0.01, 0.02, 0.03, 0.04, 0.05),
        threshold_validation_accepted=12,
    )
    values = {name: 0.1 for name in (
        "best_score", "second_score", "best_vs_second_gap", "universe_score_mean",
        "universe_score_std", "best_score_zscore", "positive_score_fraction", "best_return_5",
        "best_return_20", "best_return_60", "best_vol_20", "best_vol_60",
        "best_trend_efficiency_20", "best_trend_efficiency_60", "universe_breadth_5",
        "universe_breadth_20",
    )}

    assert gate.probability(values) == 0.2
    assert gate.confidence(values) == 1.0
