from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .optimized_allocation import AllocationDecision, _historical_return_scenarios, _safe_current_weights

COMPOUND_RISK_OVERLAY_MODE = "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY"


def compound_risk_overlay_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == COMPOUND_RISK_OVERLAY_MODE


def allocation_execution_enabled(config: Any) -> bool:
    from .concentrated_allocation import portfolio_allocation_enabled

    return portfolio_allocation_enabled(config) or compound_risk_overlay_enabled(config)


def _scenario_cvar(values: np.ndarray, confidence_level: float) -> float:
    observations = np.asarray(values, dtype=float)
    observations = observations[np.isfinite(observations)]
    if len(observations) == 0:
        return float("nan")
    losses = -observations
    level = float(np.quantile(losses, confidence_level, method="higher"))
    tail = losses[losses >= level - 1e-15]
    return max(0.0, float(np.mean(tail)) if len(tail) else level)


def _base_target(
    symbols: list[str],
    current: np.ndarray,
    target_position: int,
    target_score: float,
    *,
    status: str,
    technical_fallback: bool,
    current_cvar: float | None = None,
    reference_cvar: float | None = None,
) -> AllocationDecision:
    weights = {symbol: 0.0 for symbol in symbols}
    if target_position <= 0:
        target = np.zeros(len(symbols) + 1, dtype=float)
        target[-1] = 1.0
        return AllocationDecision(
            weights=weights,
            cash_weight=1.0,
            expected_utility=float(target_score) if np.isfinite(target_score) else 0.0,
            expected_relative_alpha=0.0,
            confidence_adjusted_relative_alpha=0.0,
            allocation_reward=0.0,
            confidence_adjusted_allocation_reward=0.0,
            normalized_cvar=0.0,
            risk_reference=reference_cvar,
            estimated_cvar=0.0,
            turnover=float(np.abs(target[:-1] - current[:-1]).sum()),
            objective_value=0.0,
            eligible_assets=(),
            optimizer_status=status,
        )
    index = int(target_position) - 1
    weights[symbols[index]] = 1.0
    target_risky = np.zeros(len(symbols), dtype=float)
    target_risky[index] = 1.0
    normalized = (
        float(current_cvar) / float(reference_cvar)
        if current_cvar is not None
        and reference_cvar is not None
        and np.isfinite(current_cvar)
        and np.isfinite(reference_cvar)
        and reference_cvar > 1e-12
        else None
    )
    return AllocationDecision(
        weights=weights,
        cash_weight=0.0,
        expected_utility=float(target_score) if np.isfinite(target_score) else 0.0,
        expected_relative_alpha=0.0,
        confidence_adjusted_relative_alpha=0.0,
        allocation_reward=1.0,
        confidence_adjusted_allocation_reward=1.0,
        normalized_cvar=normalized,
        risk_reference=reference_cvar,
        estimated_cvar=current_cvar,
        turnover=float(np.abs(target_risky - current[:-1]).sum()),
        objective_value=None if technical_fallback else 1.0,
        eligible_assets=(symbols[index],),
        optimizer_status=status,
    )


def _estimated_trade_cost_rates(frame: pd.DataFrame, timestamp: pd.Timestamp, config: Any) -> tuple[float, float]:
    if timestamp not in frame.index:
        return (0.0, 0.0)
    price = float(frame.loc[timestamp].get("close", float("nan")))
    if not np.isfinite(price) or price <= 0:
        return (0.0, 0.0)
    slippage = max(0.0, float(getattr(config, "slippage_bps", 0.0))) / 10000.0
    commission = max(0.0, float(getattr(config, "commission_rate", 0.0)))
    cat = max(0.0, float(getattr(config, "cat_fee_per_share", 0.0))) / price
    sec = max(0.0, float(getattr(config, "sec_fee_rate", 0.0)))
    taf = max(0.0, float(getattr(config, "taf_fee_per_share", 0.0))) / price
    return (slippage + commission + cat, slippage + commission + cat + sec + taf)


def optimize_compound_risk_overlay(
    target_position: int,
    target_score: float,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    current_weights: dict[str, float] | None,
    config: Any,
) -> AllocationDecision:
    current = _safe_current_weights(symbols, current_weights)
    if target_position <= 0:
        return _base_target(
            symbols,
            current,
            0,
            target_score,
            status="base_policy_cash",
            technical_fallback=False,
        )

    target_index = int(target_position) - 1
    if target_index < 0 or target_index >= len(symbols):
        return _base_target(
            symbols,
            current,
            target_position,
            target_score,
            status="technical_fallback_base_policy:invalid_target_position",
            technical_fallback=True,
        )
    target_symbol = symbols[target_index]
    frame = frames.get(target_symbol)
    if frame is None or timestamp not in frame.index:
        return _base_target(
            symbols,
            current,
            target_position,
            target_score,
            status="technical_fallback_base_policy:missing_target_market_data",
            technical_fallback=True,
        )

    configured_lookback = max(20, int(getattr(config, "allocation_lookback_days", 126)))
    current_lookback = max(252, configured_lookback)
    reference_lookback = max(756, current_lookback * 3)
    current_scenarios = _historical_return_scenarios(
        {target_symbol: frame},
        [target_symbol],
        timestamp,
        current_lookback,
        config,
    )
    reference_scenarios = _historical_return_scenarios(
        {target_symbol: frame},
        [target_symbol],
        timestamp,
        reference_lookback,
        config,
    )
    minimum_current = max(60, min(126, current_lookback // 2))
    minimum_reference = max(126, min(252, reference_lookback // 3))
    if current_scenarios.shape[0] < minimum_current or reference_scenarios.shape[0] < minimum_reference:
        return _base_target(
            symbols,
            current,
            target_position,
            target_score,
            status=(
                "technical_fallback_base_policy:insufficient_asset_risk_history"
                f":current={current_scenarios.shape[0]}:reference={reference_scenarios.shape[0]}"
            ),
            technical_fallback=True,
        )

    confidence_level = float(getattr(config, "allocation_cvar_confidence", 0.95))
    current_cvar = _scenario_cvar(current_scenarios[:, 0], confidence_level)
    reference_cvar = _scenario_cvar(reference_scenarios[:, 0], confidence_level)
    if (
        not np.isfinite(current_cvar)
        or not np.isfinite(reference_cvar)
        or reference_cvar <= 1e-12
    ):
        return _base_target(
            symbols,
            current,
            target_position,
            target_score,
            status="technical_fallback_base_policy:invalid_asset_risk_estimate",
            technical_fallback=True,
            current_cvar=current_cvar if np.isfinite(current_cvar) else None,
            reference_cvar=reference_cvar if np.isfinite(reference_cvar) else None,
        )

    normalized_risk = max(0.0, float(current_cvar) / float(reference_cvar))
    risk_aversion = max(0.0, float(getattr(config, "allocation_cvar_penalty", 1.0)))
    turnover_penalty = max(0.0, float(getattr(config, "allocation_turnover_penalty", 0.0025)))
    reward = max(1e-12, float(getattr(config, "allocation_signal_scale", 1.0)))
    max_weight = min(1.0, max(0.0, float(getattr(config, "allocation_max_asset_weight", 1.0))))
    current_target_weight = float(current[target_index])
    other_risky_weight = float(np.delete(current[:-1], target_index).sum())
    buy_cost_rate, sell_cost_rate = _estimated_trade_cost_rates(frame, timestamp, config)

    def objective(weight: float) -> float:
        w = min(max_weight, max(0.0, float(weight)))
        risk_cost = 0.5 * risk_aversion * (normalized_risk * w) ** 2
        risky_turnover = other_risky_weight + abs(w - current_target_weight)
        trade_cost = (
            buy_cost_rate * max(0.0, w - current_target_weight)
            + sell_cost_rate * max(0.0, current_target_weight - w)
        )
        return reward * w - risk_cost - turnover_penalty * risky_turnover - trade_cost

    candidates = {0.0, max_weight, min(max_weight, max(0.0, current_target_weight))}
    curvature = risk_aversion * normalized_risk * normalized_risk
    if curvature > 1e-12:
        lower_stationary = (reward + turnover_penalty + sell_cost_rate) / curvature
        upper_stationary = (reward - turnover_penalty - buy_cost_rate) / curvature
        if 0.0 <= lower_stationary <= min(max_weight, current_target_weight):
            candidates.add(float(lower_stationary))
        if max(0.0, current_target_weight) <= upper_stationary <= max_weight:
            candidates.add(float(upper_stationary))
    best_weight = max(candidates, key=lambda value: (objective(value), value))
    best_weight = min(max_weight, max(0.0, float(best_weight)))

    risky = np.zeros(len(symbols), dtype=float)
    risky[target_index] = best_weight
    cash_weight = max(0.0, 1.0 - best_weight)
    risky_turnover = float(np.abs(risky - current[:-1]).sum())
    portfolio_cvar = float(current_cvar) * best_weight
    status = "optimal_compound_risk_overlay"
    if best_weight >= max_weight - 1e-9:
        status = "optimal_compound_risk_overlay_full_exposure"
    elif best_weight <= 1e-9:
        status = "optimal_compound_risk_overlay_cash"

    return AllocationDecision(
        weights={symbol: float(risky[index]) for index, symbol in enumerate(symbols)},
        cash_weight=float(cash_weight),
        expected_utility=float(target_score) if np.isfinite(target_score) else 0.0,
        expected_relative_alpha=0.0,
        confidence_adjusted_relative_alpha=0.0,
        allocation_reward=float(best_weight),
        confidence_adjusted_allocation_reward=float(best_weight),
        normalized_cvar=float(normalized_risk),
        risk_reference=float(reference_cvar),
        estimated_cvar=float(portfolio_cvar),
        turnover=risky_turnover,
        objective_value=float(objective(best_weight)),
        eligible_assets=(target_symbol,),
        optimizer_status=status,
    )
