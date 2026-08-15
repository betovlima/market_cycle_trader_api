from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_cycle_trader_api.engine.optimized_allocation import (
    build_expected_return_samples,
    build_relative_alpha_samples,
    cross_sectional_ordinal_strength,
    cross_sectional_relative_signal,
    cross_sectional_separation_strength,
    fit_expected_return_calibrator,
    fit_relative_alpha_calibrator,
    optimize_allocation,
)
from market_cycle_trader_api.engine.concentrated_allocation import concentrated_candidate_strength, optimize_concentrated_allocation
from market_cycle_trader_api.engine.selective_opportunity import OpportunityEvaluation


class _LinearReturnCalibrator:
    method = "test_linear_net_log_return"
    sample_count = 500
    realized_return_mean = 0.01
    realized_return_std = 0.02

    def predict(self, utility: np.ndarray) -> np.ndarray:
        return np.asarray(utility, dtype=float) * 0.10


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2025-01-01", periods=180, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    output = {}
    for symbol, drift, vol in (("AAA", 0.0012, 0.010), ("BBB", 0.0008, 0.008), ("CCC", 0.0002, 0.015)):
        returns = rng.normal(drift, vol, len(dates))
        close = 100.0 * np.cumprod(1.0 + returns)
        output[symbol] = pd.DataFrame({"close": close}, index=dates)
    return output


def _config(**overrides):
    values = dict(
        allocation_minimum_utility=0.0,
        allocation_lookback_days=126,
        allocation_cvar_confidence=0.95,
        allocation_cvar_penalty=1.0,
        allocation_turnover_penalty=0.0025,
        allocation_max_asset_weight=1.0,
        allocation_signal_scale=1.0,
        slippage_bps=1.0,
        commission_rate=0.0,
        rotation_horizon_days=5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _opportunity(accepted=True, confidence=0.9):
    return OpportunityEvaluation(
        probability=0.8,
        confidence=confidence,
        accepted=accepted,
        features={},
        best_position=1,
    )


def test_optimizer_allocates_across_multiple_assets_and_can_hold_cash() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    decision = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(True),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal"
    assert abs(sum(decision.weights.values()) + decision.cash_weight - 1.0) < 1e-8
    assert all(0.0 <= weight <= 1.0000001 for weight in decision.weights.values())
    assert sum(weight > 1e-6 for weight in decision.weights.values()) >= 1
    assert 0.0 < decision.cash_weight < 1.0
    assert decision.allocation_reward > 0.0
    assert decision.confidence_adjusted_allocation_reward > 0.0
    assert decision.normalized_cvar is not None
    assert decision.risk_reference is not None
    assert decision.opportunity_threshold == 0.70


def test_rejected_opportunity_is_evidence_not_a_hard_cash_gate() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    accepted = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    rejected = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(False, confidence=0.4),
        opportunity_threshold=0.70,
    )
    assert accepted.optimizer_status == "optimal"
    assert rejected.optimizer_status == "optimal"
    assert rejected.opportunity_accepted is False
    assert 0.0 < rejected.cash_weight < 1.0
    assert rejected.cash_weight > accepted.cash_weight
    assert sum(rejected.weights.values()) > 0.0


def test_lower_opportunity_confidence_reduces_scale_free_risky_exposure() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    high = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    low = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(False, confidence=0.4),
        opportunity_threshold=0.70,
    )
    assert high.optimizer_status == "optimal"
    assert low.optimizer_status == "optimal"
    assert low.cash_weight > high.cash_weight
    assert low.confidence_adjusted_allocation_reward < high.confidence_adjusted_allocation_reward


def test_optimizer_does_not_allocate_to_nonpositive_utility_asset() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    decision = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, -0.50]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_LinearReturnCalibrator(),
        opportunity=_opportunity(True),
        opportunity_threshold=0.70,
    )
    assert decision.weights["CCC"] == 0.0


def test_expected_return_samples_trim_dates_without_complete_forward_label_horizon() -> None:
    dates = pd.date_range("2025-01-01", periods=20, freq="B", tz="UTC")
    frames = {}
    for index, symbol in enumerate(("AAA", "BBB")):
        frames[symbol] = pd.DataFrame(
            {
                "close": np.linspace(100.0 + index, 120.0 + index, len(dates)),
                "forward_net_log_return": np.linspace(-0.02, 0.04, len(dates)) + index * 0.001,
            },
            index=dates,
        )

    def utilities(_models, _frames, symbols, timestamp):
        day = dates.get_loc(timestamp)
        return np.asarray([0.0, 0.1 + day * 0.01, 0.2 + day * 0.01][: len(symbols) + 1])

    samples = build_expected_return_samples({}, frames, ["AAA", "BBB"], dates, utilities, label_horizon=5)
    assert len(samples) == (len(dates) - 5) * 2
    assert samples["timestamp"].max() == dates[-6]


def test_relative_alpha_calibrator_maps_cross_sectional_rank_into_relative_return_units() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="B", tz="UTC")
    symbols = ["AAA", "BBB"]
    frames = {}
    base = np.linspace(-0.08, 0.08, len(dates))
    for index, symbol in enumerate(symbols):
        alpha = -0.01 if index == 0 else 0.01
        frames[symbol] = pd.DataFrame(
            {
                "close": np.linspace(100.0, 150.0, len(dates)),
                "forward_net_log_return": base + alpha,
            },
            index=dates,
        )

    def utilities(_models, _frames, _symbols, timestamp):
        day = dates.get_loc(timestamp)
        common_shift = -10.0 if day % 2 == 0 else 10.0
        return np.asarray([0.0, common_shift - 0.2, common_shift + 0.2])

    calibrator = fit_relative_alpha_calibrator({}, frames, symbols, dates, utilities, label_horizon=5)
    low_scale = calibrator.predict(np.asarray([-10.0, -9.0, -8.0]))
    high_scale = calibrator.predict(np.asarray([10.0, 11.0, 12.0]))
    assert calibrator.sample_count == (len(dates) - 5) * len(symbols)
    assert calibrator.method == "out_of_sample_isotonic_cross_sectional_relative_alpha_v2"
    assert np.allclose(low_scale, high_scale)
    assert low_scale[0] < 0.0
    assert abs(low_scale[1]) < 1e-12
    assert low_scale[2] > 0.0


def test_relative_alpha_samples_are_centered_cross_sectionally() -> None:
    dates = pd.date_range("2025-01-01", periods=20, freq="B", tz="UTC")
    symbols = ["AAA", "BBB", "CCC"]
    frames = {}
    for index, symbol in enumerate(symbols):
        frames[symbol] = pd.DataFrame(
            {
                "close": np.linspace(100.0 + index, 120.0 + index, len(dates)),
                "forward_net_log_return": np.linspace(-0.03, 0.05, len(dates)) + index * 0.01,
            },
            index=dates,
        )

    def utilities(_models, _frames, _symbols, timestamp):
        day = dates.get_loc(timestamp)
        common = -5.0 + day * 0.1
        return np.asarray([0.0, common - 0.4, common, common + 0.4])

    samples = build_relative_alpha_samples({}, frames, symbols, dates, utilities, label_horizon=5)
    assert len(samples) == (len(dates) - 5) * len(symbols)
    for _, group in samples.groupby("timestamp"):
        assert abs(float(np.median(group["realized_relative_alpha"].to_numpy(dtype=float)))) < 1e-12
        assert np.allclose(sorted(group["relative_signal"].tolist()), [-1.0, 0.0, 1.0])


def test_cross_sectional_relative_signal_is_invariant_to_shift_and_positive_scale() -> None:
    first = cross_sectional_relative_signal(np.asarray([-5.0, -4.0, -3.0, -2.0]))
    second = cross_sectional_relative_signal(np.asarray([10.0, 12.0, 14.0, 16.0]))
    assert np.allclose(first, second)
    assert np.allclose(first, np.asarray([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]))


def test_optimizer_can_allocate_when_all_raw_utilities_are_negative_but_relative_alpha_is_positive() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="B", tz="UTC")
    symbols = ["AAA", "BBB", "CCC"]
    frames = _frames()
    frames = {symbol: frame.iloc[-120:].copy() for symbol, frame in frames.items()}
    for index, symbol in enumerate(symbols):
        frames[symbol]["forward_net_log_return"] = 0.001 + index * 0.002

    calibration_dates = pd.DatetimeIndex(frames["AAA"].index)

    def utilities(_models, _frames, _symbols, timestamp):
        day = calibration_dates.get_loc(timestamp)
        common = -10.0 - day * 0.001
        return np.asarray([0.0, common - 0.2, common, common + 0.2])

    calibrator = fit_relative_alpha_calibrator({}, frames, symbols, calibration_dates, utilities, label_horizon=5)
    timestamp = calibration_dates[-1]
    decision = optimize_allocation(
        np.asarray([0.0, -10.2, -10.0, -9.8]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(allocation_cvar_penalty=0.0, allocation_turnover_penalty=0.0, slippage_bps=0.0),
        expected_return_calibrator=calibrator,
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal"
    assert decision.expected_relative_alpha > 0.0
    assert decision.weights["CCC"] > 0.0
    assert decision.cash_weight < 1.0


def test_expected_return_compatibility_alias_uses_relative_alpha_calibration() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="B", tz="UTC")
    symbols = ["AAA", "BBB"]
    frames = {}
    for index, symbol in enumerate(symbols):
        frames[symbol] = pd.DataFrame(
            {
                "close": np.linspace(100.0, 150.0, len(dates)),
                "forward_net_log_return": np.linspace(-0.02, 0.04, len(dates)) + index * 0.01,
            },
            index=dates,
        )

    def utilities(_models, _frames, _symbols, timestamp):
        shift = float(dates.get_loc(timestamp))
        return np.asarray([0.0, shift - 0.2, shift + 0.2])

    calibrator = fit_expected_return_calibrator({}, frames, symbols, dates, utilities, label_horizon=5)
    predicted = calibrator.predict(np.asarray([-0.5, 0.0, 0.5]))
    assert calibrator.method == "out_of_sample_isotonic_cross_sectional_relative_alpha_v2"
    assert predicted[0] <= predicted[1] <= predicted[2]
    assert abs(predicted[1]) < 1e-12


class _NegativeAlphaDiagnosticCalibrator:
    method = "negative_alpha_diagnostic"
    sample_count = 500
    realized_return_mean = -0.01
    realized_return_std = 0.02
    realized_alpha_mean = -0.01
    realized_alpha_std = 0.02

    def predict(self, utility: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(utility, dtype=float).shape, -0.05, dtype=float)


def test_negative_calibrated_alpha_is_diagnostic_and_does_not_block_rank_eligible_assets() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    decision = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _config(),
        expected_return_calibrator=_NegativeAlphaDiagnosticCalibrator(),
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal"
    assert len(decision.eligible_assets) > 0
    assert sum(decision.weights.values()) > 0.0
    assert decision.expected_relative_alpha < 0.0
    assert decision.allocation_reward > 0.0


def test_scale_free_reward_is_invariant_to_positive_affine_utility_transform() -> None:
    first = np.asarray([-5.0, -4.0, -3.0, -2.0])
    second = 7.5 * first + 123.0
    assert np.allclose(cross_sectional_relative_signal(first), cross_sectional_relative_signal(second))
    assert np.allclose(cross_sectional_separation_strength(first), cross_sectional_separation_strength(second))
    assert np.allclose(cross_sectional_ordinal_strength(first), cross_sectional_ordinal_strength(second))


def test_scale_free_optimizer_weights_ignore_calibrated_alpha_magnitude() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    common = dict(
        frames=frames,
        symbols=symbols,
        timestamp=timestamp,
        current_weights={"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        config=_config(),
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    first = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        expected_return_calibrator=_LinearReturnCalibrator(),
        **common,
    )
    second = optimize_allocation(
        np.asarray([0.0, 20.0, 16.0, 8.0]),
        expected_return_calibrator=_NegativeAlphaDiagnosticCalibrator(),
        **common,
    )
    assert first.optimizer_status == second.optimizer_status == "optimal"
    assert abs(first.cash_weight - second.cash_weight) < 1e-9
    for symbol in symbols:
        assert abs(first.weights[symbol] - second.weights[symbol]) < 1e-9



def _concentrated_config(**overrides):
    return _config(strategy_mode="COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION", **overrides)


def test_concentrated_candidate_strength_preserves_top1_and_is_affine_invariant() -> None:
    first = np.asarray([0.30, 0.29, 0.10, -0.20])
    second = 17.0 * first - 11.0
    idx1, strength1 = concentrated_candidate_strength(first)
    idx2, strength2 = concentrated_candidate_strength(second)
    assert idx1.tolist() == idx2.tolist() == [0, 1, 2]
    assert np.allclose(strength1, strength2)
    assert strength1[0] == 1.0
    assert strength1[1] > strength1[2]


def test_concentrated_optimizer_keeps_top1_dominant_and_allows_full_concentration() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    decision = optimize_concentrated_allocation(
        np.asarray([0.0, 0.30, 0.10, -0.20]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _concentrated_config(allocation_cvar_penalty=0.0, allocation_turnover_penalty=0.0, slippage_bps=0.0),
        expected_return_calibrator=_NegativeAlphaDiagnosticCalibrator(),
        opportunity=_opportunity(True, confidence=1.0),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal_concentrated"
    assert decision.weights["AAA"] > 0.999999
    assert decision.weights["BBB"] < 1e-9
    assert decision.weights["CCC"] < 1e-9
    assert decision.cash_weight < 1e-9


def test_concentrated_optimizer_secondary_weight_is_bounded_by_top1_closeness() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    utilities = np.asarray([0.0, 0.30, 0.29, 0.28])
    _, closeness = concentrated_candidate_strength(utilities[1:])
    decision = optimize_concentrated_allocation(
        utilities,
        frames,
        symbols,
        timestamp,
        {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        _concentrated_config(allocation_cvar_penalty=1.0, allocation_turnover_penalty=0.0, slippage_bps=0.0),
        expected_return_calibrator=_NegativeAlphaDiagnosticCalibrator(),
        opportunity=_opportunity(True, confidence=0.9),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal_concentrated"
    primary = decision.weights["AAA"]
    assert primary >= decision.weights["BBB"] - 1e-9
    assert primary >= decision.weights["CCC"] - 1e-9
    assert decision.weights["BBB"] <= closeness[1] * primary + 1e-8
    assert decision.weights["CCC"] <= closeness[2] * primary + 1e-8


def test_concentrated_optimizer_lower_confidence_moves_more_compounded_capital_to_cash() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    common = dict(
        utilities=np.asarray([0.0, 0.30, 0.16, 0.08]),
        frames=frames,
        symbols=symbols,
        timestamp=timestamp,
        current_weights={"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
        config=_concentrated_config(allocation_turnover_penalty=0.0, slippage_bps=0.0),
        expected_return_calibrator=_NegativeAlphaDiagnosticCalibrator(),
        opportunity_threshold=0.70,
    )
    high = optimize_concentrated_allocation(opportunity=_opportunity(True, confidence=0.95), **common)
    low = optimize_concentrated_allocation(opportunity=_opportunity(False, confidence=0.35), **common)
    assert high.optimizer_status == low.optimizer_status == "optimal_concentrated"
    assert low.cash_weight > high.cash_weight
    assert high.weights["AAA"] >= high.weights["BBB"]
    assert low.weights["AAA"] >= low.weights["BBB"]


def test_optimized_allocation_raises_on_insufficient_risk_history_instead_of_fake_cash() -> None:
    from market_cycle_trader_api.engine.optimized_allocation import AllocationTechnicalError

    dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
    frames = {
        symbol: pd.DataFrame({"close": np.linspace(100.0, 110.0, len(dates))}, index=dates)
        for symbol in ("AAA", "BBB")
    }
    with pytest.raises(AllocationTechnicalError, match="insufficient synchronized risk history"):
        optimize_allocation(
            np.asarray([0.0, 0.20, 0.10]),
            frames,
            ["AAA", "BBB"],
            dates[-1],
            {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0},
            _config(),
            expected_return_calibrator=_LinearReturnCalibrator(),
            opportunity=_opportunity(True),
            opportunity_threshold=0.70,
        )


def test_concentrated_allocation_raises_on_insufficient_risk_history_instead_of_fake_cash() -> None:
    from market_cycle_trader_api.engine.optimized_allocation import AllocationTechnicalError

    dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
    frames = {
        symbol: pd.DataFrame({"close": np.linspace(100.0, 110.0, len(dates))}, index=dates)
        for symbol in ("AAA", "BBB")
    }
    with pytest.raises(AllocationTechnicalError, match="insufficient synchronized risk history"):
        optimize_concentrated_allocation(
            np.asarray([0.0, 0.20, 0.10]),
            frames,
            ["AAA", "BBB"],
            dates[-1],
            {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0},
            _config(),
            expected_return_calibrator=_LinearReturnCalibrator(),
            opportunity=_opportunity(True),
            opportunity_threshold=0.70,
        )
