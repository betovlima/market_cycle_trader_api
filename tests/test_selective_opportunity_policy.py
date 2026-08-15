from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES, _xgb_policy
from market_cycle_trader_api.engine.live_policy import build_live_rotation_policy
from market_cycle_trader_api.engine.selective_opportunity import OPPORTUNITY_FEATURES, SelectiveOpportunityGate, _relative_confidence


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



def _cash_gate_config() -> SimpleNamespace:
    return SimpleNamespace(
        strategy_mode="COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE",
        rotation_min_holding_days=2,
        rotation_cash_threshold=0.0,
        rotation_min_expected_edge=0.001,
        rotation_switch_margin=0.0005,
    )


def _hysteresis_gate(probability: float, *, entry: float = 0.75, exit: float = 0.45) -> SelectiveOpportunityGate:
    return SelectiveOpportunityGate(
        model=None,
        threshold=float(entry),
        constant_probability=float(probability),
        training_rows=100,
        positive_rate=0.5,
        threshold_validation_rows=20,
        threshold_validation_score=0.1,
        entry_threshold=float(entry),
        exit_threshold=float(exit),
        calibration_method="prequential_absolute_probability_hysteresis_v1",
        threshold_basis="absolute_probability",
    )


def test_opportunity_cash_gate_uses_stateful_hysteresis_and_preserves_base_policy() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]

    gate = _hysteresis_gate(0.60, entry=0.75, exit=0.45)
    diagnostics: dict[pd.Timestamp, dict] = {}
    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _cash_gate_config(),
        0.0005,
        opportunity_gate=gate,
        decision_diagnostics=diagnostics,
    )

    # In CASH, confidence 0.60 is below the 0.75 entry threshold: stay in dollars.
    assert policy(timestamp, 0, 0) == (0, 0.0)
    cash_row = diagnostics[timestamp]
    assert cash_row["decision_reason"] == "OPPORTUNITY_CASH_GATE_REJECT"
    assert cash_row["opportunity_active_threshold"] == 0.75
    assert cash_row["opportunity_hysteresis_cash_block"] is True
    assert cash_row["cash_gate_base_action_asset"] == "AAPL"
    assert cash_row["cash_gate_base_decision_reason"] == "ENTER_BEST_ASSET"

    # Already invested, the same 0.60 probability is above the 0.45 exit threshold.
    # The v2 gate keeps B0 as an independent counterfactual state; the risky
    # target is still AAPL, regardless of the gated portfolio state.
    assert policy(timestamp, 1, 2) == (1, 0.40)
    market_row = diagnostics[timestamp]
    assert market_row["opportunity_active_threshold"] == 0.45
    assert market_row["opportunity_hysteresis_market_hold"] is True
    assert market_row["cash_gate_base_action_asset"] == "AAPL"


def test_opportunity_cash_gate_rejects_even_when_base_policy_would_enter() -> None:
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]
    diagnostics: dict[pd.Timestamp, dict] = {}

    policy = _xgb_policy(
        models,
        frames,
        symbols,
        _cash_gate_config(),
        0.0005,
        opportunity_gate=_hysteresis_gate(0.20),
        decision_diagnostics=diagnostics,
    )

    assert policy(timestamp, 0, 0) == (0, 0.0)
    row = diagnostics[timestamp]
    assert row["opportunity_accepted"] is False
    assert row["cash_gate_base_action_asset"] == "AAPL"
    assert row["cash_gate_base_action_score"] == 0.40
    assert row["final_action_asset"] == "CASH"


def test_hysteresis_calibration_learns_entry_and_exit_thresholds_without_test_data() -> None:
    import numpy as np
    from market_cycle_trader_api.engine.selective_opportunity import _calibrate_hysteresis_thresholds

    validation = pd.DataFrame(
        {
            "probability": [0.10, 0.20, 0.80, 0.70, 0.60, 0.35, 0.30, 0.85, 0.75, 0.25] * 4,
            "confidence": [0.10, 0.20, 0.80, 0.70, 0.60, 0.35, 0.30, 0.85, 0.75, 0.25] * 4,
            "realized_net_log_return": [-0.03, -0.02, 0.04, 0.03, 0.02, -0.02, -0.01, 0.05, 0.03, -0.02] * 4,
        }
    )
    entry, exit, score, exposed, transitions = _calibrate_hysteresis_thresholds(validation)

    assert 0.0 <= exit <= entry <= 0.90
    assert np.isfinite(score)
    assert exposed >= 8
    assert transitions >= 0


def test_fit_selective_gate_supports_stateful_hysteresis_mode() -> None:
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
            (0.025 if day % 4 else -0.02) if symbol == "AAPL" else 0.004
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
        hysteresis=True,
    )

    assert gate.calibration_method == "prequential_absolute_probability_hysteresis_v1"
    assert gate.threshold_basis == "absolute_probability"
    assert gate.entry_threshold is not None
    assert gate.exit_threshold is not None
    assert 0.0 <= gate.exit_threshold <= gate.entry_threshold <= 0.90
    assert gate.threshold == gate.entry_threshold


def test_opportunity_cash_gate_is_neutral_when_gate_always_accepts() -> None:
    """The new research overlay must not alter the protected B0 rotation policy when open."""
    symbols = ["AAPL", "MSFT"]
    frames = _frames(symbols)
    models = {"AAPL": ConstantModel(0.40), "MSFT": ConstantModel(0.30)}
    timestamp = frames["AAPL"].index[0]

    gate_config = _cash_gate_config()
    legacy_config = SimpleNamespace(**vars(gate_config))
    legacy_config.strategy_mode = "COMPOUND_ROTATION_SWING_XGBOOST"

    legacy = _xgb_policy(
        models,
        frames,
        symbols,
        legacy_config,
        0.0005,
    )
    gated = _xgb_policy(
        models,
        frames,
        symbols,
        gate_config,
        0.0005,
        opportunity_gate=_hysteresis_gate(1.0, entry=0.50, exit=0.50),
    )

    assert gated(timestamp, 0, 0) == legacy(timestamp, 0, 0)
    assert gated(timestamp, 1, 2) == legacy(timestamp, 1, 2)


def test_opportunity_cash_gate_uses_absolute_growth_probability_not_relative_rank() -> None:
    gate = SelectiveOpportunityGate(
        model=None,
        threshold=0.60,
        constant_probability=0.40,
        training_rows=100,
        positive_rate=0.5,
        threshold_validation_rows=20,
        threshold_validation_score=0.1,
        reference_probabilities=(0.10, 0.20, 0.30),
        entry_threshold=0.60,
        exit_threshold=0.45,
        calibration_method="prequential_absolute_probability_hysteresis_v1",
        threshold_basis="absolute_probability",
    )

    # 40% can be excellent relative to a weak historical distribution, but it is
    # still only 40% absolute P(positive net return). CASH must therefore win on entry.
    probability = 0.40
    relative_confidence = 1.0
    assert gate.decision_value(probability, relative_confidence) == 0.40
    assert gate.accepts(probability, relative_confidence, current_position=0) is False



def test_cash_gate_v2_regularizes_to_b0_when_cash_has_no_validation_alpha() -> None:
    from market_cycle_trader_api.engine.selective_opportunity import _fit_cash_gate_v2_from_samples

    rows = []
    for index in range(140):
        probability_driver = 0.1 + (index % 10) * 0.01
        row = {name: probability_driver + feature_index * 0.0001 for feature_index, name in enumerate(OPPORTUNITY_FEATURES)}
        rows.append({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index),
            **row,
            "realized_net_log_return": 0.01,
            "label": 1,
        })
    samples = pd.DataFrame(rows)
    gate = _fit_cash_gate_v2_from_samples(samples, 42)

    assert gate.calibration_method == "adaptive_base_policy_one_step_hysteresis_v2"
    assert gate.target_basis == "protected_base_policy_next_session_open_to_close_net_log_return"
    assert gate.target_horizon_sessions == 1
    assert gate.entry_threshold == 0.0
    assert gate.exit_threshold == 0.0
    assert gate.regularized_to_base_policy is True


def test_cash_gate_v2_requires_majority_market_exposure_when_it_intervenes() -> None:
    from market_cycle_trader_api.engine.selective_opportunity import _calibrate_cash_gate_v2_thresholds

    validation = pd.DataFrame({
        "probability": ([0.2] * 30) + ([0.8] * 70),
        "confidence": ([0.2] * 30) + ([0.8] * 70),
        "realized_net_log_return": ([-0.03] * 30) + ([0.02] * 70),
        "label": ([0] * 30) + ([1] * 70),
    })
    entry, exit, score, exposed, _transitions, alpha, exposure_ratio, regularized = (
        _calibrate_cash_gate_v2_thresholds(validation)
    )

    assert 0.0 <= exit <= entry <= 0.75
    assert score > 0.0
    assert alpha > 0.0
    assert exposed >= 60
    assert exposure_ratio >= 0.60
    assert regularized is False


def test_adaptive_cash_gate_refreshes_only_after_matured_interval() -> None:
    from market_cycle_trader_api.engine.selective_opportunity import fit_adaptive_opportunity_cash_gate

    rows = []
    for index in range(100):
        driver = (index % 20) / 20.0
        features = {name: driver + feature_index * 0.0001 for feature_index, name in enumerate(OPPORTUNITY_FEATURES)}
        realized = 0.02 if driver >= 0.5 else -0.02
        rows.append({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index),
            **features,
            "realized_net_log_return": realized,
            "label": int(realized > 0.0),
        })
    initial = pd.DataFrame(rows)
    history: list[dict] = []
    gate = fit_adaptive_opportunity_cash_gate(initial, random_state=42, shared_history=history)

    for index in range(20):
        gate.record_matured_sample(dict(rows[index], timestamp=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index)))
    assert gate.refresh_if_needed() is False
    assert gate.refresh_count == 0

    gate.record_matured_sample(dict(rows[20], timestamp=pd.Timestamp("2025-01-21", tz="UTC")))
    assert gate.refresh_if_needed() is True
    assert gate.refresh_count == 1
    assert gate.training_rows <= 252
