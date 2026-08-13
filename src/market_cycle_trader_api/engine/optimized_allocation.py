from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

OPTIMIZED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"


@dataclass(frozen=True)
class AllocationDecision:
    weights: dict[str, float]
    cash_weight: float
    expected_utility: float
    estimated_cvar: float | None
    turnover: float
    objective_value: float | None
    eligible_assets: tuple[str, ...]
    optimizer_status: str
    opportunity_probability: float | None = None
    opportunity_confidence: float | None = None
    opportunity_threshold: float | None = None
    opportunity_accepted: bool | None = None


def optimized_allocation_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == OPTIMIZED_ALLOCATION_MODE


def _safe_current_weights(symbols: list[str], current_weights: dict[str, float] | None) -> np.ndarray:
    current = current_weights or {}
    risky = np.asarray([max(0.0, float(current.get(symbol, 0.0) or 0.0)) for symbol in symbols], dtype=float)
    cash = max(0.0, float(current.get("CASH", 0.0) or 0.0))
    values = np.concatenate([risky, np.asarray([cash], dtype=float)])
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        values[:] = 0.0
        values[-1] = 1.0
        return values
    return values / total


def _historical_return_scenarios(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    lookback_days: int,
) -> np.ndarray:
    series: list[pd.Series] = []
    for symbol in symbols:
        frame = frames[symbol]
        if timestamp not in frame.index:
            return np.empty((0, len(symbols)), dtype=float)
        location = frame.index.get_loc(timestamp)
        if not isinstance(location, (int, np.integer)):
            return np.empty((0, len(symbols)), dtype=float)
        start = max(0, int(location) - int(lookback_days))
        closes = frame.iloc[start : int(location) + 1]["close"].astype(float)
        returns = closes.pct_change().replace([np.inf, -np.inf], np.nan)
        returns.name = symbol
        series.append(returns)
    if not series:
        return np.empty((0, 0), dtype=float)
    joined = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if len(joined) > lookback_days:
        joined = joined.iloc[-lookback_days:]
    return joined.to_numpy(dtype=float)


def _all_cash(
    symbols: list[str],
    current: np.ndarray,
    *,
    status: str,
    opportunity: Any | None,
    opportunity_threshold: float | None = None,
) -> AllocationDecision:
    target = np.zeros(len(symbols) + 1, dtype=float)
    target[-1] = 1.0
    return AllocationDecision(
        weights={symbol: 0.0 for symbol in symbols},
        cash_weight=1.0,
        expected_utility=0.0,
        estimated_cvar=0.0,
        turnover=float(np.abs(target - current).sum()),
        objective_value=0.0,
        eligible_assets=(),
        optimizer_status=status,
        opportunity_probability=float(opportunity.probability) if opportunity is not None else None,
        opportunity_confidence=float(opportunity.confidence) if opportunity is not None else None,
        opportunity_threshold=float(opportunity_threshold) if opportunity_threshold is not None else None,
        opportunity_accepted=bool(opportunity.accepted) if opportunity is not None else None,
    )


def optimize_allocation(
    utilities: np.ndarray,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    current_weights: dict[str, float] | None,
    config: Any,
    *,
    opportunity: Any | None = None,
    opportunity_threshold: float | None = None,
) -> AllocationDecision:
    current = _safe_current_weights(symbols, current_weights)
    if opportunity is not None and not bool(opportunity.accepted):
        return _all_cash(symbols, current, status="opportunity_rejected", opportunity=opportunity, opportunity_threshold=opportunity_threshold)

    utility = np.asarray(utilities[1 : len(symbols) + 1], dtype=float)
    finite = np.isfinite(utility)
    minimum_utility = float(getattr(config, "allocation_minimum_utility", 0.0))
    eligible = finite & (utility > minimum_utility)
    if not eligible.any():
        return _all_cash(symbols, current, status="no_positive_utility", opportunity=opportunity, opportunity_threshold=opportunity_threshold)

    lookback = int(getattr(config, "allocation_lookback_days", 126))
    scenarios = _historical_return_scenarios(frames, symbols, timestamp, lookback)
    minimum_scenarios = max(20, min(60, lookback // 2))
    if scenarios.shape[0] < minimum_scenarios:
        return _all_cash(symbols, current, status="insufficient_risk_history", opportunity=opportunity, opportunity_threshold=opportunity_threshold)

    asset_count = len(symbols)
    scenario_count = int(scenarios.shape[0])
    cash_index = asset_count
    alpha_index = asset_count + 1
    slack_start = alpha_index + 1
    turnover_start = slack_start + scenario_count
    variable_count = turnover_start + asset_count + 1

    confidence = float(getattr(config, "allocation_cvar_confidence", 0.95))
    risk_aversion = float(getattr(config, "allocation_cvar_penalty", 1.0))
    turnover_penalty = float(getattr(config, "allocation_turnover_penalty", 0.0025))
    max_asset_weight = float(getattr(config, "allocation_max_asset_weight", 0.35))
    signal_scale = float(getattr(config, "allocation_signal_scale", 1.0))
    estimated_cost = max(0.0, float(getattr(config, "slippage_bps", 0.0))) / 10000.0
    estimated_cost += max(0.0, float(getattr(config, "commission_rate", 0.0)))

    c = np.zeros(variable_count, dtype=float)
    c[:asset_count] = -signal_scale * np.where(eligible, utility, 0.0)
    c[alpha_index] = risk_aversion
    c[slack_start:turnover_start] = risk_aversion / max(1e-12, (1.0 - confidence) * scenario_count)
    c[turnover_start:] = turnover_penalty
    c[turnover_start : turnover_start + asset_count] += estimated_cost

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
        return _all_cash(
            symbols,
            current,
            status=f"optimizer_failed:{str(result.message or 'unknown')[:80]}",
            opportunity=opportunity,
            opportunity_threshold=opportunity_threshold,
        )

    solution = np.asarray(result.x, dtype=float)
    risky = np.clip(solution[:asset_count], 0.0, 1.0)
    cash_weight = float(np.clip(solution[cash_index], 0.0, 1.0))
    total = float(risky.sum() + cash_weight)
    if total <= 0 or not np.isfinite(total):
        return _all_cash(symbols, current, status="invalid_solution", opportunity=opportunity, opportunity_threshold=opportunity_threshold)
    risky /= total
    cash_weight /= total

    portfolio_returns = scenarios @ risky
    losses = -portfolio_returns
    var_level = float(np.quantile(losses, confidence, method="higher"))
    tail = losses[losses >= var_level - 1e-15]
    estimated_cvar = float(np.mean(tail)) if len(tail) else var_level
    target = np.concatenate([risky, np.asarray([cash_weight])])
    turnover = float(np.abs(target - current).sum())
    expected_utility = float(np.dot(np.where(np.isfinite(utility), utility, 0.0), risky))
    eligible_assets = tuple(symbols[index] for index in range(asset_count) if eligible[index])

    return AllocationDecision(
        weights={symbol: float(risky[index]) for index, symbol in enumerate(symbols)},
        cash_weight=float(cash_weight),
        expected_utility=expected_utility,
        estimated_cvar=estimated_cvar,
        turnover=turnover,
        objective_value=float(-result.fun),
        eligible_assets=eligible_assets,
        optimizer_status="optimal",
        opportunity_probability=float(opportunity.probability) if opportunity is not None else None,
        opportunity_confidence=float(opportunity.confidence) if opportunity is not None else None,
        opportunity_threshold=float(opportunity_threshold) if opportunity_threshold is not None else None,
        opportunity_accepted=bool(opportunity.accepted) if opportunity is not None else None,
    )
