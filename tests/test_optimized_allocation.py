from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.optimized_allocation import optimize_allocation
from market_cycle_trader_api.engine.selective_opportunity import OpportunityEvaluation


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
        allocation_max_asset_weight=0.35,
        allocation_signal_scale=1.0,
        slippage_bps=1.0,
        commission_rate=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _opportunity(accepted=True):
    return OpportunityEvaluation(
        probability=0.8,
        confidence=0.9,
        accepted=accepted,
        features={},
        best_position=1,
    )


def test_optimizer_allocates_across_multiple_assets_and_cash() -> None:
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
        opportunity=_opportunity(True),
        opportunity_threshold=0.70,
    )
    assert decision.optimizer_status == "optimal"
    assert abs(sum(decision.weights.values()) + decision.cash_weight - 1.0) < 1e-8
    assert all(0.0 <= weight <= 0.3500001 for weight in decision.weights.values())
    assert sum(weight > 1e-6 for weight in decision.weights.values()) >= 2
    assert decision.opportunity_threshold == 0.70


def test_optimizer_moves_to_cash_when_opportunity_gate_rejects() -> None:
    frames = _frames()
    symbols = sorted(frames)
    timestamp = frames[symbols[0]].index[-1]
    decision = optimize_allocation(
        np.asarray([0.0, 0.20, 0.16, 0.08]),
        frames,
        symbols,
        timestamp,
        {"AAA": 0.35, "BBB": 0.35, "CCC": 0.0, "CASH": 0.30},
        _config(),
        opportunity=_opportunity(False),
        opportunity_threshold=0.70,
    )
    assert decision.cash_weight == 1.0
    assert sum(decision.weights.values()) == 0.0
    assert decision.optimizer_status == "opportunity_rejected"


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
        opportunity=_opportunity(True),
        opportunity_threshold=0.70,
    )
    assert decision.weights["CCC"] == 0.0
