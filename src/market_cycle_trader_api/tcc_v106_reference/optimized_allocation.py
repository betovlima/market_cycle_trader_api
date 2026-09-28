from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

OPTIMIZED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"


class AllocationTechnicalError(RuntimeError):
    pass


def cross_sectional_relative_signal(utility: np.ndarray) -> np.ndarray:
    values = np.asarray(utility, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return output
    if count == 1:
        output[finite] = 1.0
        return output
    finite_values = values[finite]
    ranks = pd.Series(finite_values).rank(method="average").to_numpy(dtype=float)
    output[finite] = (2.0 * (ranks - 1.0) / float(count - 1)) - 1.0
    return output


def cross_sectional_separation_strength(utility: np.ndarray) -> np.ndarray:
    values = np.asarray(utility, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    finite_values = values[finite]
    center = float(np.median(finite_values))
    absolute_deviation = np.abs(finite_values - center)
    scale = float(np.median(absolute_deviation)) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite_values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        output[finite] = 0.0
        return output
    z_score = (finite_values - center) / scale
    output[finite] = np.tanh(np.maximum(z_score, 0.0))
    return output


def cross_sectional_ordinal_strength(utility: np.ndarray) -> np.ndarray:
    relative_signal = cross_sectional_relative_signal(utility)
    separation = cross_sectional_separation_strength(utility)
    output = np.full(relative_signal.shape, np.nan, dtype=float)
    finite = np.isfinite(relative_signal) & np.isfinite(separation)
    if not finite.any():
        return output
    rank_strength = np.clip(relative_signal[finite], 0.0, 1.0)
    output[finite] = np.clip(0.75 * rank_strength + 0.25 * separation[finite], 0.0, 1.0)
    return output


@dataclass(frozen=True)
class RelativeAlphaCalibrator:
    model: Any | None
    constant_alpha: float | None
    sample_count: int
    signal_min: float
    signal_max: float
    realized_alpha_mean: float
    realized_alpha_std: float
    center_prediction: float
    method: str

    @property
    def utility_min(self) -> float:
        return self.signal_min

    @property
    def utility_max(self) -> float:
        return self.signal_max

    @property
    def realized_return_mean(self) -> float:
        return self.realized_alpha_mean

    @property
    def realized_return_std(self) -> float:
        return self.realized_alpha_std

    def relative_signal(self, utility: np.ndarray) -> np.ndarray:
        return cross_sectional_relative_signal(utility)

    def predict(self, utility: np.ndarray) -> np.ndarray:
        signal = self.relative_signal(utility)
        output = np.full(signal.shape, np.nan, dtype=float)
        finite = np.isfinite(signal)
        if not finite.any():
            return output
        if self.model is None:
            output[finite] = float(self.constant_alpha or 0.0)
            return output
        predicted = np.asarray(self.model.predict(signal[finite]), dtype=float)
        output[finite] = predicted - float(self.center_prediction)
        return output


ExpectedReturnCalibrator = RelativeAlphaCalibrator


@dataclass(frozen=True)
class AllocationDecision:
    weights: dict[str, float]
    cash_weight: float
    expected_utility: float
    expected_relative_alpha: float
    confidence_adjusted_relative_alpha: float
    allocation_reward: float
    confidence_adjusted_allocation_reward: float
    normalized_cvar: float | None
    risk_reference: float | None
    estimated_cvar: float | None
    turnover: float
    objective_value: float | None
    eligible_assets: tuple[str, ...]
    optimizer_status: str
    opportunity_probability: float | None = None
    opportunity_confidence: float | None = None
    opportunity_threshold: float | None = None
    opportunity_accepted: bool | None = None

    @property
    def expected_net_return(self) -> float:
        return self.expected_relative_alpha

    @property
    def confidence_adjusted_expected_return(self) -> float:
        return self.confidence_adjusted_relative_alpha


def optimized_allocation_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == OPTIMIZED_ALLOCATION_MODE


def build_relative_alpha_samples(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    label_horizon: int,
) -> pd.DataFrame:
    safe_count = max(0, len(dates) - max(1, int(label_horizon)))
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for timestamp in dates[:safe_count]:
        ts = pd.Timestamp(timestamp)
        utilities = np.asarray(utilities_for_timestamp(models, frames, symbols, ts), dtype=float)
        raw_utility = np.asarray(
            [float(utilities[position]) if position < len(utilities) else float("nan") for position in range(1, len(symbols) + 1)],
            dtype=float,
        )
        realized = np.asarray(
            [
                float(frames[symbol].loc[ts].get("forward_net_log_return", float("nan")))
                if ts in frames[symbol].index
                else float("nan")
                for symbol in symbols
            ],
            dtype=float,
        )
        valid = np.isfinite(raw_utility) & np.isfinite(realized)
        if int(valid.sum()) < 2:
            continue
        signal = cross_sectional_relative_signal(np.where(valid, raw_utility, np.nan))
        benchmark = float(np.median(realized[valid]))
        relative_alpha = realized - benchmark
        for index, symbol in enumerate(symbols):
            if not bool(valid[index]) or not np.isfinite(signal[index]):
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "utility": float(raw_utility[index]),
                    "relative_signal": float(signal[index]),
                    "realized_net_log_return": float(realized[index]),
                    "realized_relative_alpha": float(relative_alpha[index]),
                }
            )
    columns = [
        "timestamp",
        "symbol",
        "utility",
        "relative_signal",
        "realized_net_log_return",
        "realized_relative_alpha",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_expected_return_samples(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    label_horizon: int,
) -> pd.DataFrame:
    return build_relative_alpha_samples(
        models,
        frames,
        symbols,
        dates,
        utilities_for_timestamp,
        label_horizon=label_horizon,
    )


def fit_relative_alpha_calibrator(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    label_horizon: int,
) -> RelativeAlphaCalibrator:
    samples = build_relative_alpha_samples(
        models,
        frames,
        symbols,
        calibration_dates,
        utilities_for_timestamp,
        label_horizon=label_horizon,
    )
    minimum_samples = max(100, len(symbols) * 4)
    if len(samples) < minimum_samples:
        raise ValueError(
            "Optimized Allocation relative-alpha calibration requires at least "
            f"{minimum_samples} valid out-of-sample asset observations; only {len(samples)} are available."
        )

    x = samples["relative_signal"].to_numpy(dtype=float)
    y = samples["realized_relative_alpha"].to_numpy(dtype=float)
    lower, upper = np.quantile(y, [0.01, 0.99]) if len(y) >= 100 else (float(np.min(y)), float(np.max(y)))
    clipped_y = np.clip(y, float(lower), float(upper))
    model = None
    constant = 0.0
    center_prediction = 0.0
    method = "zero_relative_alpha_no_rank_resolution"

    if np.unique(x).size >= 2:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=None, y_max=None)
        model.fit(x, clipped_y)
        center_prediction = float(np.asarray(model.predict(np.asarray([0.0], dtype=float)), dtype=float)[0])
        method = "out_of_sample_isotonic_cross_sectional_relative_alpha_v2"

    return RelativeAlphaCalibrator(
        model=model,
        constant_alpha=constant if model is None else None,
        sample_count=int(len(samples)),
        signal_min=float(np.min(x)),
        signal_max=float(np.max(x)),
        realized_alpha_mean=float(np.mean(clipped_y)),
        realized_alpha_std=float(np.std(clipped_y, ddof=1)) if len(clipped_y) > 1 else 0.0,
        center_prediction=float(center_prediction),
        method=method,
    )


def fit_expected_return_calibrator(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    label_horizon: int,
) -> RelativeAlphaCalibrator:
    return fit_relative_alpha_calibrator(
        models,
        frames,
        symbols,
        calibration_dates,
        utilities_for_timestamp,
        label_horizon=label_horizon,
    )


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


def _normalized_horizon_weights(config: Any) -> tuple[tuple[int, ...], np.ndarray]:
    horizons = tuple(int(value) for value in list(getattr(config, "rotation_target_horizons", []) or []))
    raw_weights = np.asarray(list(getattr(config, "rotation_target_horizon_weights", []) or []), dtype=float)
    if not horizons or len(horizons) != len(raw_weights) or not np.isfinite(raw_weights).all() or float(raw_weights.sum()) <= 0:
        horizon = max(1, int(getattr(config, "rotation_horizon_days", 5)))
        return (horizon,), np.asarray([1.0], dtype=float)
    return horizons, raw_weights / float(raw_weights.sum())


def _historical_return_scenarios(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    lookback_days: int,
    config: Any,
) -> np.ndarray:
    horizons, weights = _normalized_horizon_weights(config)
    max_horizon = max(horizons)
    series: list[pd.Series] = []
    for symbol in symbols:
        frame = frames[symbol]
        if timestamp not in frame.index:
            return np.empty((0, len(symbols)), dtype=float)
        location = frame.index.get_loc(timestamp)
        if not isinstance(location, (int, np.integer)):
            return np.empty((0, len(symbols)), dtype=float)
        start = max(0, int(location) - int(lookback_days) - max_horizon - 2)
        closes = frame.iloc[start : int(location) + 1]["close"].astype(float)
        weighted = pd.Series(0.0, index=closes.index, dtype=float)
        valid = pd.Series(True, index=closes.index, dtype=bool)
        for horizon, weight in zip(horizons, weights, strict=True):
            component = np.log(closes / closes.shift(int(horizon)))
            valid &= component.notna() & np.isfinite(component)
            weighted = weighted + float(weight) * component.fillna(0.0)
        weighted = weighted.where(valid)
        weighted.name = symbol
        series.append(weighted)
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
        expected_relative_alpha=0.0,
        confidence_adjusted_relative_alpha=0.0,
        allocation_reward=0.0,
        confidence_adjusted_allocation_reward=0.0,
        normalized_cvar=0.0,
        risk_reference=None,
        estimated_cvar=0.0,
        turnover=float(np.abs(target[:-1] - current[:-1]).sum()),
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
    expected_return_calibrator: ExpectedReturnCalibrator | None = None,
    opportunity: Any | None = None,
    opportunity_threshold: float | None = None,
) -> AllocationDecision:
    current = _safe_current_weights(symbols, current_weights)

    utility = np.asarray(utilities[1 : len(symbols) + 1], dtype=float)
    finite = np.isfinite(utility)
    relative_signal = cross_sectional_relative_signal(utility)
    ordinal_strength = cross_sectional_ordinal_strength(utility)
    minimum_relative_signal = float(getattr(config, "allocation_minimum_utility", 0.0))
    eligible = (
        finite
        & np.isfinite(relative_signal)
        & np.isfinite(ordinal_strength)
        & (relative_signal > minimum_relative_signal)
        & (ordinal_strength > 0.0)
    )
    if not eligible.any():
        return _all_cash(
            symbols,
            current,
            status="no_eligible_relative_rank",
            opportunity=opportunity,
            opportunity_threshold=opportunity_threshold,
        )

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
    allocation_reward_vector = np.where(eligible, ordinal_strength, 0.0)
    confidence_adjusted_reward = allocation_reward_vector * confidence

    lookback = int(getattr(config, "allocation_lookback_days", 126))
    scenarios = _historical_return_scenarios(frames, symbols, timestamp, lookback, config)
    minimum_scenarios = max(20, min(60, lookback // 2))
    if scenarios.shape[0] < minimum_scenarios:
        raise AllocationTechnicalError(f"Optimized Allocation has insufficient synchronized risk history at {pd.Timestamp(timestamp)}: {scenarios.shape[0]} scenarios.")

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

    individual_cvars = np.asarray([scenario_cvar(scenarios[:, index]) for index in range(asset_count)], dtype=float)
    positive_cvars = individual_cvars[np.isfinite(individual_cvars) & (individual_cvars > 1e-8)]
    risk_reference = float(np.median(positive_cvars)) if len(positive_cvars) else 0.01
    risk_reference = max(risk_reference, 1e-6)
    risk_ceiling = max(risk_reference * 3.0, float(np.max(positive_cvars)) * 1.5 if len(positive_cvars) else 0.03)

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
    cvar_row[slack_start:turnover_start] = 1.0 / max(1e-12, (1.0 - confidence_level) * scenario_count)
    cvar_row[cvar_index] = -1.0
    rows.append(cvar_row)
    rhs.append(0.0)

    for risk_point in np.linspace(0.0, risk_ceiling, 13):
        slope = 2.0 * float(risk_point) / (risk_reference ** 2)
        intercept = -((float(risk_point) / risk_reference) ** 2)
        row = np.zeros(variable_count, dtype=float)
        row[cvar_index] = slope
        row[risk_penalty_index] = -1.0
        rows.append(row)
        rhs.append(float(-intercept))

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
            f"Optimized Allocation solver failed at {pd.Timestamp(timestamp)}: {str(result.message or 'unknown')[:160]}"
        )

    solution = np.asarray(result.x, dtype=float)
    risky = np.clip(solution[:asset_count], 0.0, 1.0)
    cash_weight = float(np.clip(solution[cash_index], 0.0, 1.0))
    total = float(risky.sum() + cash_weight)
    if total <= 0 or not np.isfinite(total):
        raise AllocationTechnicalError(f"Optimized Allocation returned an invalid solution at {pd.Timestamp(timestamp)}.")
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
    expected_relative_alpha = float(np.dot(np.where(np.isfinite(calibrated), calibrated, 0.0), risky))
    confidence_adjusted_relative_alpha = float(np.dot(np.where(np.isfinite(confidence_adjusted), confidence_adjusted, 0.0), risky))
    allocation_reward = float(np.dot(np.where(np.isfinite(allocation_reward_vector), allocation_reward_vector, 0.0), risky))
    confidence_adjusted_allocation_reward = float(np.dot(np.where(np.isfinite(confidence_adjusted_reward), confidence_adjusted_reward, 0.0), risky))
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
        optimizer_status="optimal",
        opportunity_probability=float(opportunity.probability) if opportunity is not None else None,
        opportunity_confidence=float(opportunity.confidence) if opportunity is not None else None,
        opportunity_threshold=float(opportunity_threshold) if opportunity_threshold is not None else None,
        opportunity_accepted=bool(opportunity.accepted) if opportunity is not None else None,
    )
