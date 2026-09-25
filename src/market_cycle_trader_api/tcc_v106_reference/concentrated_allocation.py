from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .optimized_allocation import (
    OPTIMIZED_ALLOCATION_MODE,
    AllocationDecision,
    AllocationTechnicalError,
    ExpectedReturnCalibrator,
    _all_cash,
    _historical_return_scenarios,
    _safe_current_weights,
    cross_sectional_relative_signal,
)

CONCENTRATED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION"


def concentrated_allocation_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == CONCENTRATED_ALLOCATION_MODE


def portfolio_allocation_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) in {OPTIMIZED_ALLOCATION_MODE, CONCENTRATED_ALLOCATION_MODE}


def concentrated_candidate_strength(utility: np.ndarray, *, candidate_limit: int = 3) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(utility, dtype=float)
    strengths = np.zeros(values.shape, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if len(finite_positions) == 0:
        return np.asarray([], dtype=int), strengths
    ranked = sorted(finite_positions.tolist(), key=lambda index: (-float(values[index]), int(index)))
    selected = np.asarray(ranked[: max(1, int(candidate_limit))], dtype=int)
    top_value = float(values[selected[0]])
    finite_values = values[finite_positions]
    center = float(np.median(finite_values))
    scale = float(np.median(np.abs(finite_values - center))) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite_values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    for position in selected:
        gap = max(0.0, (top_value - float(values[position])) / scale)
        strengths[position] = float(np.exp(-0.5 * gap * gap))
    strengths[selected[0]] = 1.0
    return selected, strengths


def optimize_concentrated_allocation(
    utilities: np.ndarray,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    current_weights: dict[str, float] | None,
    config: Any,
    *,
    expected_return_calibrator: ExpectedReturnCalibrator | None = None,
    opportunity: Any | None = None,
    opportunity_threshold: float | None = None,
) -> AllocationDecision:
    current = _safe_current_weights(symbols, current_weights)
    utility = np.asarray(utilities[1 : len(symbols) + 1], dtype=float)
    finite = np.isfinite(utility)
    if not finite.any():
        return _all_cash(
            symbols,
            current,
            status="no_finite_ranking_signal",
            opportunity=opportunity,
            opportunity_threshold=opportunity_threshold,
        )

    relative_signal = cross_sectional_relative_signal(utility)
    selected, closeness = concentrated_candidate_strength(utility, candidate_limit=3)
    if len(selected) == 0:
        return _all_cash(
            symbols,
            current,
            status="no_ranked_candidate",
            opportunity=opportunity,
            opportunity_threshold=opportunity_threshold,
        )

    minimum_relative_signal = float(getattr(config, "allocation_minimum_utility", 0.0))
    primary_index = int(selected[0])
    eligible = np.zeros(len(symbols), dtype=bool)
    eligible[primary_index] = True
    for index in selected[1:]:
        if (
            np.isfinite(relative_signal[index])
            and float(relative_signal[index]) > minimum_relative_signal
            and float(closeness[index]) > 1e-6
        ):
            eligible[index] = True

    calibrated = (
        expected_return_calibrator.predict(utility)
        if expected_return_calibrator is not None
        else np.full(utility.shape, np.nan, dtype=float)
    )
    confidence = (
        min(1.0, max(0.0, float(opportunity.confidence)))
        if opportunity is not None and getattr(opportunity, "confidence", None) is not None
        else 1.0
    )
    confidence_adjusted = calibrated * confidence
    allocation_reward_vector = np.where(eligible, closeness, 0.0)
    allocation_reward_vector[primary_index] = 1.0
    confidence_adjusted_reward = allocation_reward_vector * confidence

    lookback = int(getattr(config, "allocation_lookback_days", 126))
    scenarios = _historical_return_scenarios(frames, symbols, timestamp, lookback, config)
    minimum_scenarios = max(20, min(60, lookback // 2))
    if scenarios.shape[0] < minimum_scenarios:
        raise AllocationTechnicalError(
            f"Concentrated Allocation has insufficient synchronized risk history at {pd.Timestamp(timestamp)}: {scenarios.shape[0]} scenarios."
        )

    asset_count = len(symbols)
    scenario_count = int(scenarios.shape[0])
    cash_index = asset_count
    alpha_index = asset_count + 1
    slack_start = alpha_index + 1
    turnover_start = slack_start + scenario_count
    turnover_end = turnover_start + asset_count + 1
    cvar_index = turnover_end
    risk_penalty_index = cvar_index + 1
    variable_count = risk_penalty_index + 1

    confidence_level = float(getattr(config, "allocation_cvar_confidence", 0.95))
    risk_aversion = float(getattr(config, "allocation_cvar_penalty", 1.0))
    turnover_penalty = float(getattr(config, "allocation_turnover_penalty", 0.0025))
    max_asset_weight = float(getattr(config, "allocation_max_asset_weight", 1.0))
    signal_scale = float(getattr(config, "allocation_signal_scale", 1.0))
    estimated_cost = max(0.0, float(getattr(config, "slippage_bps", 0.0))) / 10000.0
    estimated_cost += max(0.0, float(getattr(config, "commission_rate", 0.0)))

    def scenario_cvar(values: np.ndarray) -> float:
        losses = -np.asarray(values, dtype=float)
        level = float(np.quantile(losses, confidence_level, method="higher"))
        tail = losses[losses >= level - 1e-15]
        return max(0.0, float(np.mean(tail)) if len(tail) else level)

    individual_cvars = np.asarray(
        [scenario_cvar(scenarios[:, index]) for index in range(asset_count)],
        dtype=float,
    )
    positive_cvars = individual_cvars[np.isfinite(individual_cvars) & (individual_cvars > 1e-8)]
    risk_reference = float(np.median(positive_cvars)) if len(positive_cvars) else 0.01
    risk_reference = max(risk_reference, 1e-6)
    risk_ceiling = max(
        risk_reference * 3.0,
        float(np.max(positive_cvars)) * 1.5 if len(positive_cvars) else 0.03,
    )

    c = np.zeros(variable_count, dtype=float)
    c[:asset_count] = -signal_scale * confidence_adjusted_reward
    c[turnover_start : turnover_start + asset_count] = turnover_penalty
    normalized_estimated_cost = estimated_cost / risk_reference
    c[turnover_start : turnover_start + asset_count] += normalized_estimated_cost
    c[risk_penalty_index] = risk_aversion

    a_eq = np.zeros((1, variable_count), dtype=float)
    a_eq[0, : asset_count + 1] = 1.0
    b_eq = np.asarray([1.0], dtype=float)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for scenario_index, returns in enumerate(scenarios):
        row = np.zeros(variable_count, dtype=float)
        row[:asset_count] = -returns
        row[alpha_index] = -1.0
        row[slack_start + scenario_index] = -1.0
        rows.append(row)
        rhs.append(0.0)

    cvar_row = np.zeros(variable_count, dtype=float)
    cvar_row[alpha_index] = 1.0
    cvar_row[slack_start:turnover_start] = 1.0 / max(
        1e-12,
        (1.0 - confidence_level) * scenario_count,
    )
    cvar_row[cvar_index] = -1.0
    rows.append(cvar_row)
    rhs.append(0.0)

    for risk_point in np.linspace(0.0, risk_ceiling, 13):
        slope = float(risk_point) / (risk_reference ** 2)
        intercept = -0.5 * ((float(risk_point) / risk_reference) ** 2)
        row = np.zeros(variable_count, dtype=float)
        row[cvar_index] = slope
        row[risk_penalty_index] = -1.0
        rows.append(row)
        rhs.append(float(-intercept))

    for index in selected[1:]:
        if not bool(eligible[index]):
            continue
        row = np.zeros(variable_count, dtype=float)
        row[int(index)] = 1.0
        row[primary_index] = -float(closeness[index])
        rows.append(row)
        rhs.append(0.0)

    for index in range(asset_count + 1):
        z_index = turnover_start + index
        row_positive = np.zeros(variable_count, dtype=float)
        row_positive[index] = 1.0
        row_positive[z_index] = -1.0
        rows.append(row_positive)
        rhs.append(float(current[index]))

        row_negative = np.zeros(variable_count, dtype=float)
        row_negative[index] = -1.0
        row_negative[z_index] = -1.0
        rows.append(row_negative)
        rhs.append(float(-current[index]))

    bounds: list[tuple[float | None, float | None]] = []
    for index in range(asset_count):
        bounds.append((0.0, max_asset_weight if bool(eligible[index]) else 0.0))
    bounds.append((0.0, 1.0))
    bounds.append((None, None))
    bounds.extend([(0.0, None)] * scenario_count)
    bounds.extend([(0.0, None)] * (asset_count + 1))
    bounds.append((0.0, None))
    bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=np.asarray(rows, dtype=float),
        b_ub=np.asarray(rhs, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not bool(result.success):
        raise AllocationTechnicalError(
            f"Concentrated Allocation solver failed at {pd.Timestamp(timestamp)}: {str(result.message or 'unknown')[:160]}"
        )

    solution = np.asarray(result.x, dtype=float)
    risky = np.clip(solution[:asset_count], 0.0, 1.0)
    cash_weight = float(np.clip(solution[cash_index], 0.0, 1.0))
    total = float(risky.sum() + cash_weight)
    if total <= 0 or not np.isfinite(total):
        raise AllocationTechnicalError(f"Concentrated Allocation returned an invalid solution at {pd.Timestamp(timestamp)}.")
    risky /= total
    cash_weight /= total

    portfolio_returns = scenarios @ risky
    losses = -portfolio_returns
    var_level = float(np.quantile(losses, confidence_level, method="higher"))
    tail = losses[losses >= var_level - 1e-15]
    estimated_cvar = float(np.mean(tail)) if len(tail) else var_level
    normalized_cvar = float(max(0.0, estimated_cvar) / risk_reference)
    target = np.concatenate([risky, np.asarray([cash_weight])])
    turnover = float(np.abs(target[:-1] - current[:-1]).sum())
    expected_utility = float(np.dot(np.where(np.isfinite(utility), utility, 0.0), risky))
    expected_relative_alpha = float(
        np.dot(np.where(np.isfinite(calibrated), calibrated, 0.0), risky)
    )
    confidence_adjusted_relative_alpha = float(
        np.dot(np.where(np.isfinite(confidence_adjusted), confidence_adjusted, 0.0), risky)
    )
    allocation_reward = float(np.dot(allocation_reward_vector, risky))
    confidence_adjusted_allocation_reward = float(np.dot(confidence_adjusted_reward, risky))
    eligible_assets = tuple(symbols[index] for index in range(asset_count) if eligible[index])

    return AllocationDecision(
        weights={symbol: float(risky[index]) for index, symbol in enumerate(symbols)},
        cash_weight=float(cash_weight),
        expected_utility=expected_utility,
        expected_relative_alpha=expected_relative_alpha,
        confidence_adjusted_relative_alpha=confidence_adjusted_relative_alpha,
        allocation_reward=allocation_reward,
        confidence_adjusted_allocation_reward=confidence_adjusted_allocation_reward,
        normalized_cvar=normalized_cvar,
        risk_reference=float(risk_reference),
        estimated_cvar=estimated_cvar,
        turnover=turnover,
        objective_value=float(-result.fun),
        eligible_assets=eligible_assets,
        optimizer_status="optimal_concentrated",
        opportunity_probability=float(opportunity.probability) if opportunity is not None else None,
        opportunity_confidence=float(opportunity.confidence) if opportunity is not None else None,
        opportunity_threshold=float(opportunity_threshold) if opportunity_threshold is not None else None,
        opportunity_accepted=bool(opportunity.accepted) if opportunity is not None else None,
    )
