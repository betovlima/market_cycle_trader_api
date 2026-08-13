from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import math
import subprocess
import time
from typing import Any, Callable
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .rotation_diagnostics import enrich_trade_diagnostics

LEGACY_ROTATION_MODE = 'COMPOUND_ROTATION_SWING_XGBOOST'
RISK_OFF_ROTATION_MODE = 'COMPOUND_ROTATION_SWING_RISK_OFF'
SUPPORTED_ROTATION_MODES = frozenset({LEGACY_ROTATION_MODE, RISK_OFF_ROTATION_MODE})

def _risk_off_enabled(config: Any) -> bool:
    return str(getattr(config, 'strategy_mode', LEGACY_ROTATION_MODE)) == RISK_OFF_ROTATION_MODE

ROTATION_FEATURES = [
    'return_1', 'return_2', 'return_3', 'return_5', 'return_10', 'return_20',
    'return_40', 'return_60', 'return_120',
    'vol_5', 'vol_10', 'vol_20', 'vol_40', 'vol_60',
    'vol_ratio_5_20', 'vol_ratio_10_40', 'vol_ratio_20_60',
    'ema_distance_5', 'ema_distance_10', 'ema_distance_20', 'ema_distance_50',
    'ema_distance_100', 'ema_5_vs_20', 'ema_20_vs_50', 'ema_50_vs_100',
    'ema_slope_20_5', 'ema_slope_50_10', 'ema_slope_100_20',
    'rsi_14', 'atr_pct_14',
    'distance_from_high_20', 'distance_from_low_20',
    'distance_from_high_50', 'distance_from_low_50',
    'distance_from_high_100', 'distance_from_low_100',
    'distance_from_high_200', 'distance_from_low_200',
    'channel_position_20', 'channel_position_50', 'channel_position_100', 'channel_position_200',
    'trend_efficiency_10', 'trend_efficiency_20', 'trend_efficiency_40', 'trend_efficiency_60',
    'momentum_acceleration_5_20', 'momentum_acceleration_20_60',
    'range_expansion_5_20', 'volume_zscore_20', 'volume_zscore_60', 'volume_ratio_5_20',
]


@dataclass
class RotationRunResult:
    backend: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    summary: str
    metrics: dict[str, Any]

@dataclass(frozen=True)
class RotationComputePlan:
    framework: str
    requested: str
    selected: str
    cuda_available: bool
    gpu_name: str | None
    framework_version: str | None
    cuda_runtime_version: str | None
    cuda_build: bool | None
    fallback_used: bool
    fallback_reason: str | None

def _truthy_build_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}

def _nvidia_gpu_name() -> str | None:
    try:
        completed = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=3, check=False)
        if completed.returncode == 0:
            names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if names:
                return names[0]
    except Exception:
        pass
    return None

def resolve_xgboost_compute_plan(config: Any) -> RotationComputePlan:
    requested = str(config.rotation_accelerator).strip().lower()
    allow_fallback = bool(config.rotation_allow_cpu_fallback)
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('ROTATION_ACCELERATOR must be auto, cpu or cuda.')
    version = None
    cuda_build = None
    error = None
    try:
        import xgboost as xgb
        version = str(xgb.__version__)
        build = xgb.build_info()
        if isinstance(build, dict):
            cuda_build = _truthy_build_flag(build.get('USE_CUDA'))
    except Exception as exc:
        error = str(exc)
    gpu_name = _nvidia_gpu_name()
    cuda_available = bool(cuda_build and gpu_name)
    if requested == 'cpu':
        selected = 'cpu'
    elif cuda_available:
        selected = 'cuda'
    else:
        selected = 'cpu'
    reason = None
    fallback_used = False
    if requested == 'cuda' and (not cuda_available):
        if error is not None:
            reason = f'XGBoost CUDA detection failed: {error}'
        elif not cuda_build:
            reason = 'The installed XGBoost build does not expose CUDA support'
        else:
            reason = 'No NVIDIA GPU/driver is visible to XGBoost'
        if not allow_fallback:
            raise RuntimeError(f'{reason}. Enable CPU fallback or use accelerator=auto.')
        fallback_used = True
    return RotationComputePlan(framework='xgboost', requested=requested, selected=selected, cuda_available=cuda_available, gpu_name=gpu_name, framework_version=version, cuda_runtime_version=None, cuda_build=cuda_build, fallback_used=fallback_used, fallback_reason=reason)

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean = denominator.replace(0, np.nan)
    return numerator / clean

def _rsi(close: pd.Series, period: int=14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - 100 / (1 + rs)

def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame['close'].shift(1)
    return pd.concat([frame['high'] - frame['low'], (frame['high'] - previous_close).abs(), (frame['low'] - previous_close).abs()], axis=1).max(axis=1)

def build_rotation_frame(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    horizons = [int(item) for item in config.rotation_target_horizons]
    weights = np.asarray(config.rotation_target_horizon_weights, dtype=float)
    weights = weights / weights.sum()
    max_horizon = max(horizons)

    data = bars.copy().sort_index()
    data.index = pd.to_datetime(data.index, utc=True)
    close = data['close'].astype(float)
    open_price = data['open'].astype(float)
    high = data['high'].astype(float)
    low = data['low'].astype(float)
    volume = data['volume'].astype(float)
    daily_return = close.pct_change()
    daily_range = _safe_divide(high - low, close)

    for period in [1, 2, 3, 5, 10, 20, 40, 60, 120]:
        data[f'return_{period}'] = close.pct_change(period)
    for period in [5, 10, 20, 40, 60]:
        data[f'vol_{period}'] = daily_return.rolling(period).std()
    data['vol_ratio_5_20'] = _safe_divide(data['vol_5'], data['vol_20'])
    data['vol_ratio_10_40'] = _safe_divide(data['vol_10'], data['vol_40'])
    data['vol_ratio_20_60'] = _safe_divide(data['vol_20'], data['vol_60'])

    ema: dict[int, pd.Series] = {}
    for period in [5, 10, 20, 50, 100]:
        ema[period] = close.ewm(span=period, adjust=False).mean()
        data[f'ema_distance_{period}'] = _safe_divide(close, ema[period]) - 1
    data['ema_5_vs_20'] = _safe_divide(ema[5], ema[20]) - 1
    data['ema_20_vs_50'] = _safe_divide(ema[20], ema[50]) - 1
    data['ema_50_vs_100'] = _safe_divide(ema[50], ema[100]) - 1
    data['ema_slope_20_5'] = ema[20].pct_change(5)
    data['ema_slope_50_10'] = ema[50].pct_change(10)
    data['ema_slope_100_20'] = ema[100].pct_change(20)

    data['rsi_14'] = _rsi(close) / 100.0
    atr = _true_range(data).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data['atr_pct_14'] = _safe_divide(atr, close)

    for period in [20, 50, 100, 200]:
        rolling_high = high.rolling(period).max()
        rolling_low = low.rolling(period).min()
        data[f'distance_from_high_{period}'] = _safe_divide(close, rolling_high) - 1
        data[f'distance_from_low_{period}'] = _safe_divide(close, rolling_low) - 1
        data[f'channel_position_{period}'] = _safe_divide(close - rolling_low, rolling_high - rolling_low)

    absolute_change = close.pct_change().abs()
    for period in [10, 20, 40, 60]:
        net_move = close.pct_change(period).abs()
        traveled = absolute_change.rolling(period).sum()
        data[f'trend_efficiency_{period}'] = _safe_divide(net_move, traveled)

    data['momentum_acceleration_5_20'] = data['return_5'] - data['return_20'] * (5.0 / 20.0)
    data['momentum_acceleration_20_60'] = data['return_20'] - data['return_60'] * (20.0 / 60.0)
    data['range_expansion_5_20'] = _safe_divide(daily_range.rolling(5).mean(), daily_range.rolling(20).mean())

    for period in [20, 60]:
        volume_mean = volume.rolling(period).mean()
        volume_std = volume.rolling(period).std()
        data[f'volume_zscore_{period}'] = _safe_divide(volume - volume_mean, volume_std)
    data['volume_ratio_5_20'] = _safe_divide(volume.rolling(5).mean(), volume.rolling(20).mean())

    round_trip_cost = min(
        0.25,
        2.0 * (
            max(0.0, float(config.slippage_bps)) / 10000.0
            + max(0.0, float(config.commission_rate))
        ),
    )
    net_cost_log = math.log(max(1e-12, 1.0 - round_trip_cost))

    lows = low.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_price.to_numpy(dtype=float)
    utility_components = np.full((len(data), len(horizons)), np.nan, dtype=float)
    movement_capture = np.full(len(data), np.nan, dtype=float)
    trend_persistence = np.full(len(data), np.nan, dtype=float)

    for idx in range(0, max(0, len(data) - max_horizon)):
        entry = opens[idx + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        full_closes = closes[idx + 1:idx + max_horizon + 1]
        full_highs = highs[idx + 1:idx + max_horizon + 1]
        if len(full_closes) != max_horizon:
            continue

        max_upside = max(0.0, float(np.nanmax(full_highs) / entry - 1.0))
        positive_share = float(np.mean(full_closes > entry))
        directional_changes = np.diff(full_closes)
        positive_steps = float(np.mean(directional_changes > 0)) if len(directional_changes) else 0.0
        movement_capture[idx] = math.log1p(max_upside)
        trend_persistence[idx] = 0.5 * positive_share + 0.5 * positive_steps

        for component_idx, horizon in enumerate(horizons):
            future_lows = lows[idx + 1:idx + horizon + 1]
            future_closes = closes[idx + 1:idx + horizon + 1]
            if len(future_closes) != horizon:
                continue
            gross_log_return = math.log(max(float(future_closes[-1]) / entry, 1e-12))
            minimum_low = float(np.nanmin(future_lows))
            downside = max(0.0, 1.0 - minimum_low / entry)
            path = np.concatenate(([entry], future_closes))
            running_peak = np.maximum.accumulate(path)
            drawdowns = 1.0 - np.divide(path, running_peak, out=np.ones_like(path), where=running_peak > 0)
            path_drawdown = max(0.0, float(np.nanmax(drawdowns)))
            utility_components[idx, component_idx] = (
                gross_log_return
                + net_cost_log
                - float(config.rotation_downside_penalty) * downside
                - float(config.rotation_drawdown_penalty) * path_drawdown
            )

    weighted_utility = np.nansum(utility_components * weights.reshape(1, -1), axis=1)
    invalid_rows = np.isnan(utility_components).any(axis=1)
    weighted_utility[invalid_rows] = np.nan
    
    
    
    
    data['forward_cash_edge'] = weighted_utility
    data['forward_movement_capture'] = movement_capture
    data['forward_trend_persistence'] = trend_persistence
    data['forward_risk_adjusted_utility'] = (
        weighted_utility
        + float(config.rotation_movement_capture_weight) * data['forward_movement_capture']
        + float(config.rotation_trend_persistence_weight) * data['forward_trend_persistence']
    )

    required = ROTATION_FEATURES + ['open', 'high', 'low', 'close', 'volume']
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    return data

def prepare_rotation_panel(bars_by_symbol: dict[str, pd.DataFrame], config: Any) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    








    frames = {
        symbol: build_rotation_frame(frame, config)
        for symbol, frame in bars_by_symbol.items()
        if frame is not None and not frame.empty
    }
    if len(frames) < 2:
        raise ValueError('Compound rotation needs at least two assets with valid aligned data.')

    configured_anchors = list(getattr(config, 'calendar_anchor_assets', []) or [])
    anchor_symbols = [symbol for symbol in configured_anchors if symbol in frames]
    if len(anchor_symbols) < 2:
        anchor_symbols = sorted(frames)

    common: pd.DatetimeIndex | None = None
    for symbol in anchor_symbols:
        index = pd.DatetimeIndex(frames[symbol].index)
        common = index if common is None else common.intersection(index)
    if common is None or len(common) < 700:
        raise ValueError('The anchored aligned history is too short for train/calibration/test.')

    common = common.sort_values()
    aligned = {symbol: frame.reindex(common).copy() for symbol, frame in frames.items()}
    return aligned, common

def _annualized_sharpe(curve: pd.Series, periods_per_year: float=252.0) -> float:
    returns = curve.pct_change().dropna()
    if returns.empty or float(returns.std()) <= 0:
        return float('nan')
    return float(np.sqrt(float(periods_per_year)) * returns.mean() / returns.std())

def _maximum_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float('nan')
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())

def _cagr(curve: pd.Series) -> float:
    if len(curve) < 2:
        return float('nan')
    start = pd.Timestamp(curve.index[0])
    end = pd.Timestamp(curve.index[-1])
    years = max((end - start).days / 365.25, 1 / 365.25)
    if float(curve.iloc[0]) <= 0 or float(curve.iloc[-1]) <= 0:
        return float('nan')
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)

def _geometric_trade_return(trades: pd.DataFrame) -> float:
    if trades.empty or 'position_return' not in trades:
        return float('nan')
    returns = pd.to_numeric(trades.loc[trades['action'].isin(['SELL', 'FINAL_SELL']), 'position_return'], errors='coerce').dropna()
    if returns.empty:
        return float('nan')
    gross = np.prod(1.0 + returns.clip(lower=-0.999999))
    return float(gross ** (1 / len(returns)) - 1)

def _proportional_switch_cost(config: Any, from_position: int, to_position: int) -> float:
    if from_position == to_position:
        return 0.0
    one_side = max(0.0, float(config.slippage_bps)) / 10000.0
    one_side += max(0.0, float(config.commission_rate))
    sides = int(from_position != 0) + int(to_position != 0)
    return min(0.25, one_side * sides)

def _training_transition_log_return(frames: dict[str, pd.DataFrame], symbols: list[str], date_now: pd.Timestamp, date_next: pd.Timestamp, from_position: int, to_position: int, config: Any) -> float:
    gross = 1.0
    if from_position > 0:
        symbol = symbols[from_position - 1]
        close_now = float(frames[symbol].loc[date_now, 'close'])
        open_next = float(frames[symbol].loc[date_next, 'open'])
        if close_now > 0:
            gross *= open_next / close_now
    cost = _proportional_switch_cost(config, from_position, to_position)
    gross *= max(1e-08, 1.0 - cost)
    if to_position > 0:
        symbol = symbols[to_position - 1]
        open_next = float(frames[symbol].loc[date_next, 'open'])
        close_next = float(frames[symbol].loc[date_next, 'close'])
        if open_next > 0:
            gross *= close_next / open_next
    return float(np.log(max(gross, 1e-12)))

def _risk_adjusted_reward(log_return: float, wealth_before: float, peak_before: float, config: Any) -> tuple[float, float, float]:
    wealth_after = wealth_before * math.exp(log_return)
    peak_after = max(peak_before, wealth_after)
    previous_drawdown = max(0.0, 1.0 - wealth_before / max(peak_before, 1e-12))
    current_drawdown = max(0.0, 1.0 - wealth_after / max(peak_after, 1e-12))
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown)
    downside = max(0.0, -log_return)
    reward = log_return - float(config.rotation_downside_penalty) * downside - float(config.rotation_drawdown_penalty) * drawdown_increase
    return (float(reward), float(wealth_after), float(peak_after))

def _curve_risk_adjusted_score(curve: pd.Series, config: Any) -> float:
    if curve.empty or len(curve) < 2:
        return float('nan')
    values = curve.astype(float)
    logs = np.log(values / values.shift(1)).dropna()
    peak = values.cummax()
    drawdown = 1.0 - values / peak
    drawdown_increase = drawdown.diff().clip(lower=0).fillna(0.0)
    aligned = drawdown_increase.reindex(logs.index).fillna(0.0)
    downside = (-logs).clip(lower=0)
    score = (logs - float(config.rotation_downside_penalty) * downside - float(config.rotation_drawdown_penalty) * aligned).sum()
    return float(score)

def _numeric_thread_context(config: Any):
    
    
    
    if not bool(config.deterministic_execution):
        return nullcontext()
    return threadpool_limits(limits=int(config.numeric_thread_limit))


def _fit_xgb_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    device_name: str,
    *,
    phase: str = 'training',
    progress_callback: Callable[[int, int, str], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
    target_column: str = 'forward_risk_adjusted_utility',
) -> tuple[dict[str, Any], str, str | None]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError('XGBoost Utility requires xgboost. Install requirements.txt.') from exc
    allow_fallback = bool(config.rotation_allow_cpu_fallback)
    anchor_assets = set(getattr(config, 'calendar_anchor_assets', []) or [])
    minimum_rows = int(config.rotation_minimum_training_rows)

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    def fit_on_device(effective_device: str) -> dict[str, Any]:
        fitted: dict[str, Any] = {}
        total_symbols = len(symbols)
        fit_started = time.perf_counter()
        technical(
            f'phase={phase} event=fit_start device={effective_device} '
            f'models={total_symbols} train_sessions={len(train_dates)} '
            f'estimators={int(config.rotation_xgb_n_estimators)} '
            f'workers={int(config.xgb_n_jobs)} seed={int(config.random_state)}'
        )
        with _numeric_thread_context(config):
            for position, symbol in enumerate(symbols, start=1):
                frame = frames[symbol].loc[train_dates].dropna(
                    subset=[target_column, *ROTATION_FEATURES]
                )
                if len(frame) < minimum_rows:
                    if symbol in anchor_assets:
                        raise ValueError(
                            f'{symbol}: only {len(frame)} utility rows are available; '
                            f'{minimum_rows} are required for an anchor asset.'
                        )
                    technical(
                        f'phase={phase} event=model_deferred model={position}/{total_symbols} '
                        f'asset={symbol} rows={len(frame)} required={minimum_rows}'
                    )
                    if progress_callback is not None:
                        progress_callback(position, total_symbols, effective_device)
                    continue
                model_started = time.perf_counter()
                technical(
                    f'phase={phase} event=model_start model={position}/{total_symbols} '
                    f'asset={symbol} rows={len(frame)} features={len(ROTATION_FEATURES)} '
                    f'target={target_column} device={effective_device}'
                )
                model = XGBRegressor(
                    n_estimators=int(config.rotation_xgb_n_estimators),
                    learning_rate=float(config.rotation_xgb_learning_rate),
                    max_depth=int(config.rotation_xgb_max_depth),
                    min_child_weight=float(config.xgb_min_child_weight),
                    subsample=float(config.xgb_subsample),
                    colsample_bytree=float(config.xgb_colsample_bytree),
                    reg_alpha=float(config.xgb_reg_alpha),
                    reg_lambda=float(config.xgb_reg_lambda),
                    objective='reg:squarederror',
                    tree_method='hist',
                    random_state=int(config.random_state),
                    n_jobs=int(config.xgb_n_jobs),
                    device=effective_device,
                    verbosity=0,
                )
                model.fit(
                    frame[ROTATION_FEATURES],
                    frame[target_column],
                )
                fitted[symbol] = model
                duration = time.perf_counter() - model_started
                technical(
                    f'phase={phase} event=model_complete model={position}/{total_symbols} '
                    f'asset={symbol} rows={len(frame)} device={effective_device} '
                    f'duration_seconds={duration:.3f}'
                )
                if progress_callback is not None:
                    progress_callback(position, total_symbols, effective_device)
        technical(
            f'phase={phase} event=fit_complete device={effective_device} '
            f'models={total_symbols} duration_seconds={time.perf_counter() - fit_started:.3f}'
        )
        return fitted

    try:
        return (fit_on_device(device_name), device_name, None)
    except Exception as exc:
        if device_name != 'cuda' or not allow_fallback:
            technical(
                f'phase={phase} event=fit_failed device={device_name} '
                f'error_type={type(exc).__name__} error={exc}'
            )
            raise
        fallback_reason = f'XGBoost CUDA initialization/training failed; using CPU instead: {exc}'
        technical(
            f'phase={phase} event=device_fallback from=cuda to=cpu '
            f'error_type={type(exc).__name__} error={exc}'
        )
        return (fit_on_device('cpu'), 'cpu', fallback_reason)

def _xgb_utilities(models: dict[str, Any], frames: dict[str, pd.DataFrame], symbols: list[str], timestamp: pd.Timestamp, config: Any) -> np.ndarray:
    

    values = [0.0]
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames[symbol]
        if model is None or timestamp not in frame.index:
            values.append(float('-inf'))
            continue

        row = frame.loc[[timestamp], ROTATION_FEATURES]
        if row.empty or row.isna().any(axis=None):
            values.append(float('-inf'))
            continue

        location = frame.index.get_loc(timestamp)
        if not isinstance(location, (int, np.integer)) or location + 1 >= len(frame.index):
            values.append(float('-inf'))
            continue
        next_row = frame.iloc[int(location) + 1]
        next_open = float(next_row.get('open', float('nan')))
        next_close = float(next_row.get('close', float('nan')))
        if not (np.isfinite(next_open) and next_open > 0 and np.isfinite(next_close) and next_close > 0):
            values.append(float('-inf'))
            continue

        prediction = float(model.predict(row)[0])
        values.append(prediction)
    return np.asarray(values, dtype=np.float64)

def _xgb_policy(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    switch_margin: float,
    *,
    cash_edge_models: dict[str, Any] | None = None,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    fold_id: int | None = None,
    calibrated_switch_margin: float | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    








    risk_off = _risk_off_enabled(config)
    if risk_off and cash_edge_models is None:
        raise ValueError('Explicit risk-off mode requires cash-edge models.')

    def position_asset(position: int) -> str:
        return 'CASH' if position <= 0 else symbols[position - 1]

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        utilities = _xgb_utilities(models, frames, symbols, timestamp, config)
        if not np.isfinite(utilities[1:]).any():
            return (0, 0.0)

        cash_edges = (
            _xgb_utilities(cash_edge_models or {}, frames, symbols, timestamp, config)
            if risk_off
            else utilities.copy()
        )
        ranked_positions = sorted(
            (
                position
                for position in range(1, len(utilities))
                if np.isfinite(utilities[position])
            ),
            key=lambda position: (-float(utilities[position]), symbols[position - 1]),
        )
        if not ranked_positions:
            return (0, 0.0)

        raw_best_asset_position = ranked_positions[0]
        legacy_best = int(np.nanargmax(utilities))
        best = raw_best_asset_position if risk_off else legacy_best
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])
        current_cash_edge = float(cash_edges[current_position]) if current_position < len(cash_edges) else float('-inf')
        minimum = float(config.rotation_cash_threshold)
        entry_threshold = minimum + float(config.rotation_min_expected_edge)
        required = max(float(config.rotation_switch_margin), float(switch_margin))

        ranked_assets = [
            (symbols[position - 1], float(utilities[position]))
            for position in ranked_positions
        ]
        best_asset, best_asset_score = ranked_assets[0] if ranked_assets else (None, None)
        second_asset, second_asset_score = (
            ranked_assets[1] if len(ranked_assets) > 1 else (None, None)
        )
        best_position = ranked_positions[0] if ranked_positions else 0
        best_cash_edge = (
            float(cash_edges[best_position])
            if best_position > 0 and np.isfinite(cash_edges[best_position])
            else None
        )
        second_position = ranked_positions[1] if len(ranked_positions) > 1 else 0
        second_cash_edge = (
            float(cash_edges[second_position])
            if second_position > 0 and np.isfinite(cash_edges[second_position])
            else None
        )
        best_vs_second_gap = (
            float(best_asset_score - second_asset_score)
            if best_asset_score is not None and second_asset_score is not None
            else None
        )
        best_vs_current_gap = (
            float(best_asset_score - current_value)
            if best_asset_score is not None and np.isfinite(current_value)
            else None
        )
        finite_asset_scores = np.asarray(
            [score for _, score in ranked_assets],
            dtype=np.float64,
        )
        universe_score_mean = (
            float(np.mean(finite_asset_scores)) if len(finite_asset_scores) else None
        )
        universe_score_std = (
            float(np.std(finite_asset_scores)) if len(finite_asset_scores) else None
        )
        positive_score_count = int(np.sum(finite_asset_scores > 0.0))
        current_asset_rank = None
        if current_position > 0:
            current_symbol = symbols[current_position - 1]
            current_asset_rank = next(
                (rank for rank, (asset, _) in enumerate(ranked_assets, start=1) if asset == current_symbol),
                None,
            )
        stable_std = (
            universe_score_std
            if universe_score_std is not None and universe_score_std > 1e-12
            else None
        )
        best_score_zscore = (
            float((best_asset_score - universe_score_mean) / stable_std)
            if best_asset_score is not None
            and universe_score_mean is not None
            and stable_std is not None
            else None
        )
        current_score_zscore = (
            float((current_value - universe_score_mean) / stable_std)
            if np.isfinite(current_value)
            and universe_score_mean is not None
            and stable_std is not None
            else None
        )
        best_vs_second_zscore = (
            float(best_vs_second_gap / stable_std)
            if best_vs_second_gap is not None and stable_std is not None
            else None
        )

        def finish(
            target_position: int,
            final_score: float,
            reason: str,
            *,
            min_hold_guard: bool = False,
            switch_margin_guard: bool = False,
            cash_threshold_guard: bool = False,
            expected_edge_guard: bool = False,
        ) -> tuple[int, float]:
            if decision_diagnostics is not None:
                top = ranked_positions[:3]
                final_cash_edge = (
                    float(cash_edges[target_position])
                    if target_position > 0 and np.isfinite(cash_edges[target_position])
                    else 0.0
                )
                diagnostic = {
                    'decision_diagnostics_schema_version': 3 if risk_off else 2,
                    'decision_fold_id': fold_id,
                    'strategy_risk_off_enabled': bool(risk_off),
                    'current_asset': position_asset(current_position),
                    'current_score': finite(current_value),
                    'current_cash_edge': finite(current_cash_edge),
                    'holding_days_at_decision': int(holding_days),
                    'raw_best_asset': position_asset(best),
                    'raw_best_score': finite(best_value),
                    'best_asset': best_asset,
                    'best_score': finite(best_asset_score) if best_asset_score is not None else None,
                    'best_cash_edge': best_cash_edge,
                    'second_asset': second_asset,
                    'second_score': finite(second_asset_score) if second_asset_score is not None else None,
                    'second_cash_edge': second_cash_edge,
                    'best_vs_second_gap': best_vs_second_gap,
                    'best_vs_current_gap': best_vs_current_gap,
                    'best_vs_cash_gap': best_cash_edge if risk_off else (float(best_asset_score) if best_asset_score is not None else None),
                    'cash_score': 0.0,
                    'cash_exit_threshold': minimum,
                    'cash_entry_threshold': entry_threshold,
                    'current_asset_rank': current_asset_rank,
                    'universe_score_mean': universe_score_mean,
                    'universe_score_std': universe_score_std,
                    'current_score_zscore': current_score_zscore,
                    'best_score_zscore': best_score_zscore,
                    'best_vs_second_zscore': best_vs_second_zscore,
                    'positive_score_count': positive_score_count,
                    'finite_score_count': int(len(finite_asset_scores)),
                    'rotation_cash_threshold': minimum,
                    'rotation_min_expected_edge': float(config.rotation_min_expected_edge),
                    'base_switch_margin': float(config.rotation_switch_margin),
                    'calibrated_switch_margin': (
                        float(calibrated_switch_margin)
                        if calibrated_switch_margin is not None
                        else float(switch_margin)
                    ),
                    'effective_switch_margin': required,
                    'final_action_asset': position_asset(target_position),
                    'final_action_score': finite(final_score),
                    'final_action_cash_edge': finite(final_cash_edge),
                    'decision_reason': reason,
                    'decision_is_rotation': bool(
                        current_position > 0
                        and target_position > 0
                        and target_position != current_position
                    ),
                    'decision_is_entry': bool(current_position == 0 and target_position > 0),
                    'decision_is_exit_to_cash': bool(current_position > 0 and target_position == 0),
                    'min_hold_guard_applied': bool(min_hold_guard),
                    'switch_margin_guard_applied': bool(switch_margin_guard),
                    'cash_threshold_guard_applied': bool(cash_threshold_guard),
                    'minimum_expected_edge_guard_applied': bool(expected_edge_guard),
                    'day_trade_constraint_applied': False,
                    'q_current_position': finite(current_value),
                    'q_raw_best': finite(best_value),
                    'q_final_action': finite(final_score),
                    'q_delta_final_vs_current': (
                        float(final_score - current_value)
                        if np.isfinite(final_score) and np.isfinite(current_value)
                        else None
                    ),
                    'q_gap_best_vs_second': best_vs_second_gap,
                    'raw_action_asset': position_asset(best),
                }
                for rank in range(3):
                    position = top[rank] if rank < len(top) else 0
                    asset = symbols[position - 1] if position > 0 else None
                    score = float(utilities[position]) if position > 0 else None
                    edge = float(cash_edges[position]) if position > 0 and np.isfinite(cash_edges[position]) else None
                    diagnostic[f'top_{rank + 1}_asset'] = asset
                    diagnostic[f'top_{rank + 1}_score'] = finite(score) if score is not None else None
                    diagnostic[f'top_{rank + 1}_cash_edge'] = finite(edge) if edge is not None else None
                decision_diagnostics[pd.Timestamp(timestamp)] = diagnostic
            return (target_position, final_score)

        if not risk_off:
            if (
                current_position > 0
                and np.isfinite(current_value)
                and holding_days < int(config.rotation_min_holding_days)
            ):
                return finish(
                    current_position,
                    current_value,
                    'MIN_HOLD_GUARD',
                    min_hold_guard=True,
                )
            if best == 0 or best_value <= minimum:
                return finish(
                    0,
                    0.0,
                    'CASH_THRESHOLD',
                    cash_threshold_guard=True,
                )
            if current_position == 0:
                if best_value >= entry_threshold:
                    return finish(best, best_value, 'ENTER_BEST_ASSET')
                return finish(
                    0,
                    0.0,
                    'MIN_EXPECTED_EDGE_GUARD',
                    expected_edge_guard=True,
                )
            if best == current_position:
                return finish(current_position, current_value, 'HOLD_CURRENT_BEST')
            if best_value >= current_value + required:
                return finish(best, best_value, 'ROTATE_TO_BEST_ASSET')
            return finish(
                current_position,
                current_value,
                'SWITCH_MARGIN_GUARD',
                switch_margin_guard=True,
            )

        def entry_candidates() -> list[int]:
            return [
                position
                for position in ranked_positions
                if np.isfinite(cash_edges[position])
                and float(cash_edges[position]) >= entry_threshold
            ]

        if current_position == 0:
            candidates = entry_candidates()
            if not candidates:
                return finish(
                    0,
                    0.0,
                    'RISK_OFF_ENTRY_GUARD',
                    expected_edge_guard=True,
                )
            target = candidates[0]
            return finish(target, float(utilities[target]), 'ENTER_BEST_ASSET')

        current_is_investable = (
            np.isfinite(current_cash_edge)
            and current_cash_edge > minimum
        )
        if not current_is_investable:
            candidates = entry_candidates()
            if candidates:
                target = candidates[0]
                return finish(
                    target,
                    float(utilities[target]),
                    'RISK_OFF_ROTATE_TO_ELIGIBLE',
                    cash_threshold_guard=True,
                )
            return finish(
                0,
                0.0,
                'RISK_OFF_EXIT_TO_CASH',
                cash_threshold_guard=True,
            )

        if holding_days < int(config.rotation_min_holding_days):
            return finish(
                current_position,
                current_value,
                'MIN_HOLD_GUARD',
                min_hold_guard=True,
            )

        candidates = entry_candidates()
        if not candidates:
            return finish(current_position, current_value, 'RISK_OFF_HYSTERESIS_HOLD')
        target = candidates[0]
        if target == current_position:
            return finish(current_position, current_value, 'HOLD_CURRENT_BEST')
        if float(utilities[target]) >= current_value + required:
            return finish(target, float(utilities[target]), 'ROTATE_TO_BEST_ASSET')
        return finish(
            current_position,
            current_value,
            'SWITCH_MARGIN_GUARD',
            switch_margin_guard=True,
        )

    return policy

def _simple_policy_growth(policy: Callable[[pd.Timestamp, int, int], tuple[int, float]], frames: dict[str, pd.DataFrame], symbols: list[str], decision_dates: pd.DatetimeIndex, config: Any) -> float:
    if len(decision_dates) < 2:
        return float('-inf')
    wealth = 1.0
    peak = 1.0
    position = 0
    holding = 0
    utility = 0.0
    for idx in range(len(decision_dates) - 1):
        now = decision_dates[idx]
        nxt = decision_dates[idx + 1]
        action, _ = policy(now, position, holding)
        log_return = _training_transition_log_return(frames, symbols, now, nxt, position, action, config)
        reward, wealth, peak = _risk_adjusted_reward(log_return, wealth, peak, config)
        utility += reward
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
    return float(utility)

def _execute_buy(cash: float, price: float, config: Any, fee_calculator: Callable, slippage: Callable) -> tuple[float, float, dict[str, float]]:
    execution_price = float(slippage(price, 'BUY', config))
    quantity = cash / execution_price
    for _ in range(25):
        fees = fee_calculator('BUY', quantity, execution_price, config)
        next_quantity = max(0.0, (cash - float(fees['total_fee'])) / execution_price)
        if not bool(config.fractional_shares):
            next_quantity = float(math.floor(next_quantity))
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity
    fees = fee_calculator('BUY', quantity, execution_price, config)
    return (float(quantity), execution_price, fees)

def _equal_weight_benchmark(frames: dict[str, pd.DataFrame], symbols: list[str], execution_dates: pd.DatetimeIndex, initial_capital: float, config: Any, fee_calculator: Callable, slippage: Callable) -> pd.Series:
    if len(execution_dates) < 2:
        return pd.Series(dtype=float)
    first = execution_dates[0]
    last = execution_dates[-1]
    benchmark_symbols: list[str] = []
    for symbol in symbols:
        frame = frames[symbol]
        window = frame.reindex(execution_dates)
        first_open = float(window.iloc[0].get('open', float('nan')))
        closes = pd.to_numeric(window['close'], errors='coerce')
        if (
            np.isfinite(first_open)
            and first_open > 0
            and closes.notna().all()
            and (closes > 0).all()
        ):
            benchmark_symbols.append(symbol)
    if not benchmark_symbols:
        raise ValueError('No asset has complete prices for the benchmark execution window.')

    capital_per_asset = float(initial_capital) / len(benchmark_symbols)
    quantities: dict[str, float] = {}
    residual = 0.0
    for symbol in benchmark_symbols:
        buy_price = float(slippage(float(frames[symbol].loc[first, 'open']), 'BUY', config))
        quantity = capital_per_asset / buy_price
        for _ in range(20):
            fees = fee_calculator('BUY', quantity, buy_price, config)
            next_quantity = max(0.0, (capital_per_asset - float(fees['total_fee'])) / buy_price)
            if not bool(config.fractional_shares):
                next_quantity = float(math.floor(next_quantity))
            if abs(next_quantity - quantity) < 1e-10:
                quantity = next_quantity
                break
            quantity = next_quantity
        fees = fee_calculator('BUY', quantity, buy_price, config)
        quantities[symbol] = quantity
        residual += capital_per_asset - (quantity * buy_price + float(fees['total_fee']))
    values = []
    for timestamp in execution_dates:
        equity = residual
        for symbol, quantity in quantities.items():
            equity += quantity * float(frames[symbol].loc[timestamp, 'close'])
        values.append(equity)
    series = pd.Series(values, index=execution_dates, dtype=float)
    final_cash = residual
    for symbol, quantity in quantities.items():
        sell_price = float(slippage(float(frames[symbol].loc[last, 'close']), 'SELL', config))
        fees = fee_calculator('SELL', quantity, sell_price, config)
        final_cash += quantity * sell_price - float(fees['total_fee'])
    series.iloc[-1] = final_cash
    return series

def _precompute_market_regime_diagnostics(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, dict[str, Any]]:
    




    if len(decision_dates) == 0:
        return {}

    close_columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or 'close' not in frame:
            continue
        close_columns[symbol] = pd.to_numeric(frame['close'], errors='coerce').reindex(decision_dates)
    if not close_columns:
        return {}

    closes = pd.DataFrame(close_columns, index=decision_dates, dtype=float)
    return_5 = closes.pct_change(periods=5, fill_method=None)
    return_20 = closes.pct_change(periods=20, fill_method=None)

    valid_5 = return_5.notna().sum(axis=1).replace(0, np.nan)
    valid_20 = return_20.notna().sum(axis=1).replace(0, np.nan)
    breadth_5 = (return_5 > 0.0).sum(axis=1) / valid_5
    breadth_20 = (return_20 > 0.0).sum(axis=1) / valid_20

    spy_return_5 = pd.Series(np.nan, index=decision_dates, dtype=float)
    spy_return_20 = pd.Series(np.nan, index=decision_dates, dtype=float)
    spy_realized_volatility_20 = pd.Series(np.nan, index=decision_dates, dtype=float)
    if 'SPY' in closes.columns:
        spy = closes['SPY']
        spy_return_5 = spy.pct_change(periods=5, fill_method=None)
        spy_return_20 = spy.pct_change(periods=20, fill_method=None)
        spy_realized_volatility_20 = (
            spy.pct_change(fill_method=None).rolling(20, min_periods=10).std() * math.sqrt(252.0)
        )

    output: dict[pd.Timestamp, dict[str, Any]] = {}
    for timestamp in decision_dates:
        ts = pd.Timestamp(timestamp)

        def finite(series: pd.Series) -> float | None:
            value = series.get(timestamp, np.nan)
            return float(value) if pd.notna(value) and np.isfinite(float(value)) else None

        output[ts] = {
            'market_regime_diagnostics_schema_version': 1,
            'spy_return_5': finite(spy_return_5),
            'spy_return_20': finite(spy_return_20),
            'spy_realized_volatility_20': finite(spy_realized_volatility_20),
            'universe_breadth_5': finite(breadth_5),
            'universe_breadth_20': finite(breadth_20),
            'universe_breadth_5_valid_assets': int(valid_5.get(timestamp)) if pd.notna(valid_5.get(timestamp)) else 0,
            'universe_breadth_20_valid_assets': int(valid_20.get(timestamp)) if pd.notna(valid_20.get(timestamp)) else 0,
        }
    return output


def _simulate_exact(backend: str, policy: Callable[[pd.Timestamp, int, int], tuple[int, float]], frames: dict[str, pd.DataFrame], symbols: list[str], decision_dates: pd.DatetimeIndex, config: Any, fee_calculator: Callable, slippage: Callable, decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None=None, policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None=None, trade_callback: Callable[[dict[str, Any]], None] | None=None, *, model_label: str='XGBoost Utility', method_line: str | None=None) -> RotationRunResult:
    if len(decision_dates) < 2:
        raise ValueError('The final-test interval is too short.')
    execution_dates = decision_dates[1:]
    benchmark = _equal_weight_benchmark(frames, symbols, execution_dates, float(config.initial_capital), config, fee_calculator, slippage)
    market_regime_by_date = _precompute_market_regime_diagnostics(
        frames, symbols, decision_dates
    )
    cash = float(config.initial_capital)
    position = 0
    quantity = 0.0
    entry_price = float('nan')
    entry_time = None
    position_entry_score: float | None = None
    position_peak_price = float('nan')
    position_low_price = float('nan')
    days_current_not_top1 = 0
    consecutive_days_current_not_top1 = 0
    holding_days = 0
    total_fees = 0.0
    turnover = 0.0
    rotation_count = 0
    records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    equity_values: list[float] = []
    for idx in range(len(decision_dates) - 1):
        decision_date = decision_dates[idx]
        execution_date = decision_dates[idx + 1]
        previous_position = position
        metadata = (decision_metadata or {}).get(pd.Timestamp(decision_date), {})
        fold_id = metadata.get('fold_id')
        target_position, score = policy(decision_date, position, holding_days)
        decision_diag = dict((policy_decision_diagnostics or {}).get(pd.Timestamp(decision_date), {}))
        decision_diag.update(market_regime_by_date.get(pd.Timestamp(decision_date), {}))

        if position > 0 and np.isfinite(entry_price) and entry_price > 0:
            current_symbol = symbols[position - 1]
            current_row = frames[current_symbol].loc[decision_date]
            current_close = float(current_row.get('close', float('nan')))
            current_high = float(current_row.get('high', current_close))
            current_low = float(current_row.get('low', current_close))
            if np.isfinite(current_high):
                position_peak_price = (
                    max(position_peak_price, current_high)
                    if np.isfinite(position_peak_price)
                    else current_high
                )
            if np.isfinite(current_low):
                position_low_price = (
                    min(position_low_price, current_low)
                    if np.isfinite(position_low_price)
                    else current_low
                )

            best_asset_now = decision_diag.get('best_asset')
            if best_asset_now and best_asset_now != current_symbol:
                days_current_not_top1 += 1
                consecutive_days_current_not_top1 += 1
            elif best_asset_now == current_symbol:
                consecutive_days_current_not_top1 = 0

            current_score = decision_diag.get('current_score')
            decision_diag.update({
                'position_risk_diagnostics_schema_version': 1,
                'position_entry_timestamp': entry_time,
                'position_entry_price': float(entry_price),
                'position_entry_score': position_entry_score,
                'position_return_since_entry': (
                    float(current_close / entry_price - 1.0)
                    if np.isfinite(current_close) else None
                ),
                'position_peak_return': (
                    float(position_peak_price / entry_price - 1.0)
                    if np.isfinite(position_peak_price) else None
                ),
                'position_drawdown_from_peak': (
                    float(current_close / position_peak_price - 1.0)
                    if np.isfinite(current_close)
                    and np.isfinite(position_peak_price)
                    and position_peak_price > 0
                    else None
                ),
                'position_mfe_so_far': (
                    float(position_peak_price / entry_price - 1.0)
                    if np.isfinite(position_peak_price) else None
                ),
                'position_mae_so_far': (
                    float(position_low_price / entry_price - 1.0)
                    if np.isfinite(position_low_price) else None
                ),
                'score_change_from_entry': (
                    float(current_score - position_entry_score)
                    if current_score is not None
                    and position_entry_score is not None
                    and np.isfinite(float(current_score))
                    and np.isfinite(float(position_entry_score))
                    else None
                ),
                'days_current_not_top1': int(days_current_not_top1),
                'consecutive_days_current_not_top1': int(consecutive_days_current_not_top1),
            })
        else:
            decision_diag.update({
                'position_risk_diagnostics_schema_version': 1,
                'position_entry_timestamp': None,
                'position_entry_price': None,
                'position_entry_score': None,
                'position_return_since_entry': None,
                'position_peak_return': None,
                'position_drawdown_from_peak': None,
                'position_mfe_so_far': None,
                'position_mae_so_far': None,
                'score_change_from_entry': None,
                'days_current_not_top1': 0,
                'consecutive_days_current_not_top1': 0,
            })
        day_trades: list[dict[str, Any]] = []
        if target_position != position:
            old_symbol = symbols[position - 1] if position > 0 else None
            new_symbol = symbols[target_position - 1] if target_position > 0 else None
            is_rotation = previous_position > 0 and target_position > 0
            rotation_id = f'{pd.Timestamp(execution_date).isoformat()}::{old_symbol}->{new_symbol}' if is_rotation else None
            decision_trade_fields = {
                'decision_timestamp': pd.Timestamp(decision_date),
                'rotation_id': rotation_id,
                'rotation_from_asset': old_symbol if is_rotation else None,
                'rotation_to_asset': new_symbol if is_rotation else None,
            }
            for diagnostic_key in (
                'decision_diagnostics_schema_version',
                'current_asset',
                'current_score',
                'holding_days_at_decision',
                'raw_best_asset',
                'raw_best_score',
                'best_asset',
                'best_score',
                'second_asset',
                'second_score',
                'best_vs_second_gap',
                'best_vs_current_gap',
                'best_vs_cash_gap',
                'cash_score',
                'base_switch_margin',
                'calibrated_switch_margin',
                'effective_switch_margin',
                'final_action_asset',
                'final_action_score',
                'decision_reason',
                'switch_margin_guard_applied',
                'cash_threshold_guard_applied',
                'minimum_expected_edge_guard_applied',
                'q_current_position',
                'q_raw_best',
                'q_final_action',
                'q_delta_final_vs_current',
                'q_gap_best_vs_second',
                'raw_action_asset',
                'min_hold_guard_applied',
                'day_trade_constraint_applied',
                'top_1_asset',
                'top_1_score',
                'top_2_asset',
                'top_2_score',
                'top_3_asset',
                'top_3_score',
                'current_asset_rank',
                'universe_score_mean',
                'universe_score_std',
                'current_score_zscore',
                'best_score_zscore',
                'best_vs_second_zscore',
                'positive_score_count',
                'finite_score_count',
                'position_risk_diagnostics_schema_version',
                'position_entry_timestamp',
                'position_entry_price',
                'position_entry_score',
                'position_return_since_entry',
                'position_peak_return',
                'position_drawdown_from_peak',
                'position_mfe_so_far',
                'position_mae_so_far',
                'score_change_from_entry',
                'days_current_not_top1',
                'consecutive_days_current_not_top1',
                'market_regime_diagnostics_schema_version',
                'spy_return_5',
                'spy_return_20',
                'spy_realized_volatility_20',
                'universe_breadth_5',
                'universe_breadth_20',
                'universe_breadth_5_valid_assets',
                'universe_breadth_20_valid_assets',
            ):
                decision_trade_fields[diagnostic_key] = decision_diag.get(diagnostic_key)
            if position > 0:
                symbol = symbols[position - 1]
                price = float(slippage(float(frames[symbol].loc[execution_date, 'open']), 'SELL', config))
                fees = fee_calculator('SELL', quantity, price, config)
                gross = quantity * price
                realized = quantity * (price - entry_price) - float(fees['total_fee'])
                cash += gross - float(fees['total_fee'])
                total_fees += float(fees['total_fee'])
                turnover += gross
                position_return = price / entry_price - 1 if np.isfinite(entry_price) and entry_price > 0 else 0.0
                day_trades.append({'timestamp': execution_date, 'action': 'SELL', 'asset': symbol, 'reason': f'ROTATE_TO_{new_symbol}' if new_symbol else 'MOVE_TO_CASH', 'execution_price': price, 'quantity': quantity, 'gross_trade_value': gross, **fees, 'realized_pnl': realized, 'position_return': position_return, 'holding_bars': holding_days, 'entry_timestamp': entry_time, 'entry_price': entry_price if np.isfinite(entry_price) else None, 'cash_after_trade': cash, 'shares_after_trade': 0.0, 'walk_forward_fold': fold_id, **decision_trade_fields})
                quantity = 0.0
                entry_price = float('nan')
                entry_time = None
                position_entry_score = None
                position_peak_price = float('nan')
                position_low_price = float('nan')
                days_current_not_top1 = 0
                consecutive_days_current_not_top1 = 0
                holding_days = 0
            position = target_position
            if position > 0:
                symbol = symbols[position - 1]
                raw_price = float(frames[symbol].loc[execution_date, 'open'])
                quantity, price, fees = _execute_buy(cash, raw_price, config, fee_calculator, slippage)
                gross = quantity * price
                cash -= gross + float(fees['total_fee'])
                total_fees += float(fees['total_fee'])
                turnover += gross
                entry_price = price
                entry_time = execution_date
                entry_score_value = decision_diag.get('final_action_score')
                position_entry_score = (
                    float(entry_score_value)
                    if entry_score_value is not None and np.isfinite(float(entry_score_value))
                    else None
                )
                position_peak_price = float(price)
                position_low_price = float(price)
                days_current_not_top1 = 0
                consecutive_days_current_not_top1 = 0
                holding_days = 1
                day_trades.append({'timestamp': execution_date, 'action': 'BUY', 'asset': symbol, 'reason': f'ROTATE_FROM_{old_symbol}' if old_symbol else 'BEST_CAPITAL_UTILITY', 'execution_price': price, 'quantity': quantity, 'gross_trade_value': gross, **fees, 'realized_pnl': 0.0, 'position_return': 0.0, 'holding_bars': 0, 'entry_timestamp': execution_date, 'entry_price': price, 'cash_after_trade': cash, 'shares_after_trade': quantity, 'walk_forward_fold': fold_id, **decision_trade_fields})
            if previous_position > 0 and target_position > 0:
                rotation_count += 1
        elif position > 0:
            holding_days += 1
        records.extend(day_trades)
        if trade_callback is not None:
            for trade in day_trades:
                trade_callback({**trade, 'backend': backend, 'model': model_label})
        if position > 0:
            symbol = symbols[position - 1]
            close_price = float(frames[symbol].loc[execution_date, 'close'])
            equity = cash + quantity * close_price
            selected_asset = symbol
        else:
            equity = cash
            selected_asset = 'CASH'
        equity_values.append(equity)
        actions = [trade['action'] for trade in day_trades]
        trade_action = 'ROTATE' if 'SELL' in actions and 'BUY' in actions else actions[-1] if actions else ''
        prediction_rows.append({'timestamp': execution_date, 'close': float('nan'), 'strategy_equity': equity, 'buy_hold_equity': float(benchmark.loc[execution_date]), 'trade_action': trade_action, 'trade_reason': 'COMPOUND_CAPITAL_ROTATION' if trade_action else '', 'execution_price': float(day_trades[-1]['execution_price']) if day_trades else None, 'selected_asset': selected_asset, 'previous_asset': symbols[previous_position - 1] if previous_position > 0 else 'CASH', 'decision_score': float(score), 'decision_date': decision_date, 'walk_forward_fold': fold_id, 'fold_test_start': metadata.get('test_start'), 'fold_test_end': metadata.get('test_end'), **decision_diag})
    if position > 0 and prediction_rows:
        final_date = execution_dates[-1]
        symbol = symbols[position - 1]
        price = float(slippage(float(frames[symbol].loc[final_date, 'close']), 'SELL', config))
        fees = fee_calculator('SELL', quantity, price, config)
        gross = quantity * price
        realized = quantity * (price - entry_price) - float(fees['total_fee'])
        cash += gross - float(fees['total_fee'])
        total_fees += float(fees['total_fee'])
        turnover += gross
        position_return = price / entry_price - 1 if np.isfinite(entry_price) and entry_price > 0 else 0.0
        final_trade = {'timestamp': final_date, 'action': 'FINAL_SELL', 'asset': symbol, 'reason': 'FINAL_LIQUIDATION', 'execution_price': price, 'quantity': quantity, 'gross_trade_value': gross, **fees, 'realized_pnl': realized, 'position_return': position_return, 'holding_bars': holding_days, 'entry_timestamp': entry_time, 'entry_price': entry_price, 'cash_after_trade': cash, 'shares_after_trade': 0.0, 'walk_forward_fold': prediction_rows[-1].get('walk_forward_fold')}
        records.append(final_trade)
        if trade_callback is not None:
            trade_callback({**final_trade, 'backend': backend, 'model': model_label})
        equity_values[-1] = cash
        prediction_rows[-1]['strategy_equity'] = cash
        prediction_rows[-1]['trade_action'] = prediction_rows[-1]['trade_action'] or 'FINAL_SELL'
        prediction_rows[-1]['trade_reason'] = prediction_rows[-1]['trade_reason'] or 'FINAL_LIQUIDATION'
    records = enrich_trade_diagnostics(records, frames, symbols)
    predictions = pd.DataFrame(prediction_rows).set_index('timestamp')
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    predictions.index.name = 'timestamp'
    trades = pd.DataFrame(records)
    if not trades.empty:
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
        trades = trades.sort_values('timestamp').reset_index(drop=True)
    strategy_curve = pd.Series([float(row['strategy_equity']) for row in prediction_rows], index=execution_dates, dtype=float)
    benchmark_curve = benchmark.reindex(execution_dates).astype(float)
    initial = float(config.initial_capital)
    ending = float(strategy_curve.iloc[-1])
    benchmark_ending = float(benchmark_curve.iloc[-1])
    buys = int((trades['action'] == 'BUY').sum()) if not trades.empty else 0
    sells = int(trades['action'].isin(['SELL', 'FINAL_SELL']).sum()) if not trades.empty else 0
    cash_days = int(sum((row['selected_asset'] == 'CASH' for row in prediction_rows)))
    exposure = 1.0 - cash_days / max(1, len(prediction_rows))
    completed_sells = trades.loc[trades['action'].isin(['SELL', 'FINAL_SELL'])] if not trades.empty else pd.DataFrame()
    avg_holding = float(pd.to_numeric(completed_sells['holding_bars']).mean()) if not completed_sells.empty else float('nan')
    days = max(1, (pd.Timestamp(execution_dates[-1]) - pd.Timestamp(execution_dates[0])).days)
    years = max(days / 365.25, 1 / 365.25)
    periods_per_year = 252.0
    anchor_assets = [symbol for symbol in getattr(config, 'calendar_anchor_assets', []) if symbol in symbols]
    reference_assets = [symbol for symbol in getattr(config, 'research_reference_assets', []) if symbol in symbols]
    if len(reference_assets) < 2:
        reference_assets = list(anchor_assets)
    reference_set = set(reference_assets)
    candidate_assets = [symbol for symbol in getattr(config, 'research_candidate_assets', []) if symbol in symbols and symbol not in reference_set]
    if not getattr(config, 'research_candidate_assets', None):
        candidate_assets = [symbol for symbol in symbols if symbol not in reference_set]
    metrics = {'portfolio_rotation': True, 'strategy_mode': config.strategy_mode, 'strategy_label': model_label, 'symbol': 'PORTFOLIO', 'backend': backend, 'assets': symbols, 'calendar_anchor_assets': anchor_assets, 'research_reference_assets': reference_assets, 'research_candidate_assets': candidate_assets, 'timeframe': '1Day', 'decision_horizon_days': int(config.rotation_horizon_days), 'decision_horizon_bars': None, 'decision_horizon_label': f'{int(config.rotation_horizon_days)} trading sessions', 'overnight_positions_allowed': True, 'benchmark_name': 'Equal-weight buy-and-hold across continuously available assets', 'walk_forward_enabled': bool(config.rotation_walk_forward_enabled), 'walk_forward_purge_days': int(config.rotation_purge_days), 'walk_forward_calibration_days': int(config.rotation_walk_forward_calibration_days), 'walk_forward_test_days': int(config.rotation_walk_forward_test_days), 'downside_penalty': float(config.rotation_downside_penalty), 'drawdown_penalty': float(config.rotation_drawdown_penalty), 'initial_capital': initial, 'strategy_ending_capital': ending, 'strategy_return': ending / initial - 1, 'buy_hold_ending_capital': benchmark_ending, 'buy_hold_return': benchmark_ending / initial - 1, 'excess_return': ending / initial - benchmark_ending / initial, 'strategy_maximum_drawdown': _maximum_drawdown(strategy_curve), 'buy_hold_maximum_drawdown': _maximum_drawdown(benchmark_curve), 'strategy_sharpe': _annualized_sharpe(strategy_curve, periods_per_year), 'buy_hold_sharpe': _annualized_sharpe(benchmark_curve, periods_per_year), 'strategy_cagr': _cagr(strategy_curve), 'buy_hold_cagr': _cagr(benchmark_curve), 'compound_log_growth': float(math.log(max(ending / initial, 1e-12))), 'risk_adjusted_compound_score': _curve_risk_adjusted_score(strategy_curve, config), 'market_exposure': float(exposure), 'cash_days': cash_days, 'simulated_buys': buys, 'simulated_sells': sells, 'capital_rotations': int(rotation_count), 'cycles_per_year': float(buys / years), 'average_holding_days': avg_holding, 'average_holding_bars': avg_holding, 'average_holding_minutes': None, 'geometric_trade_return': _geometric_trade_return(trades), 'total_transaction_fees': float(total_fees), 'turnover_ratio': float(turnover / max(initial, 1e-09)), 'test_start': execution_dates[0], 'test_end': execution_dates[-1], 'test_calendar_years': years}
    summary = '\n'.join(['COMPOUND CAPITAL ROTATION — SWING', '', f"Model: {metrics['strategy_label']}", f"Assets: {', '.join(symbols)}", 'Decision data: daily candles', f"Utility horizons: {', '.join(str(item) for item in config.rotation_target_horizons)} trading sessions", 'Capital pool: one shared account, reinvested after every exit/rotation', 'Decision objective: maximize smoother net compounded wealth, not predict exact tops.', f'Risk penalties: downside={config.rotation_downside_penalty:.3f}, drawdown={config.rotation_drawdown_penalty:.3f}', f'Validation: expanding walk-forward, purge={config.rotation_purge_days} sessions, fold test={config.rotation_walk_forward_test_days} sessions', '', 'OUT-OF-SAMPLE WALK-FORWARD', f'Initial capital: ${initial:,.2f}', f'Ending capital: ${ending:,.2f}', f"Total return: {metrics['strategy_return']:.2%}", f"CAGR: {metrics['strategy_cagr']:.2%}", f"Compound log growth: {metrics['compound_log_growth']:.6f}", f"Maximum drawdown: {metrics['strategy_maximum_drawdown']:.2%}", f"Sharpe estimate: {metrics['strategy_sharpe']:.3f}", f'Capital rotations: {rotation_count}', f'Buys: {buys}', f'Sells including final liquidation: {sells}', f"Cycles/year: {metrics['cycles_per_year']:.2f}", f'Average holding days: {avg_holding:.2f}', f'Time in market: {exposure:.2%}', f'Transaction fees: ${total_fees:,.2f}', '', 'BENCHMARK', 'Equal-weight buy-and-hold across assets with complete prices for the execution window.', f'Benchmark ending capital: ${benchmark_ending:,.2f}', f"Benchmark return: {metrics['buy_hold_return']:.2%}", f"Benchmark CAGR: {metrics['buy_hold_cagr']:.2%}", '', 'METHOD', '- Signals use information available at the current daily close.', '- Position changes execute at the next daily open.', (method_line or f"- XGBoost Utility predicts a weighted multi-horizon risk-adjusted utility across {config.rotation_target_horizons}."), '- Every fold is trained only on information available before that fold.', f'- A {config.rotation_purge_days}-session purge prevents forward labels from touching the next validation/test segment.', '- FINAL_LIQUIDATION is bookkeeping only and is not a model decision.'])
    return RotationRunResult(backend=backend, predictions=predictions, trades=trades, summary=summary, metrics=metrics)

def _build_walk_forward_folds(common_dates: pd.DatetimeIndex, config: Any) -> list[dict[str, Any]]:
    







    purge = max(int(config.rotation_purge_days), max(int(item) for item in config.rotation_target_horizons))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    test_days = int(config.rotation_walk_forward_test_days)
    min_test_days = int(config.rotation_walk_forward_min_test_days)
    min_train = int(config.rotation_minimum_training_rows)
    first_test_start = min_train + purge + calibration_days + purge

    if first_test_start >= len(common_dates) - min_test_days:
        available_test_rows = max(0, len(common_dates) - first_test_start)
        raise ValueError(
            'Not enough history for the locked champion walk-forward protocol: '
            f'available_test_rows={available_test_rows}, minimum_test={min_test_days}, '
            f'rows={len(common_dates)}, minimum_train={min_train}, '
            f'calibration={calibration_days}, purge={purge}.'
        )

    folds: list[dict[str, Any]] = []
    test_start = first_test_start
    fold_id = 1
    while test_start < len(common_dates):
        test_end = min(len(common_dates), test_start + test_days)
        if test_end - test_start < min_test_days:
            if folds:
                folds[-1]['test_end_index'] = len(common_dates)
                folds[-1]['test_end'] = common_dates[-1]
                folds[-1]['decision_dates'] = common_dates[folds[-1]['test_start_index'] - 1:len(common_dates)]
            break
        calibration_end = test_start - purge
        calibration_start = calibration_end - calibration_days
        train_end = calibration_start - purge
        final_fit_end = test_start - purge
        if train_end < min_train:
            raise ValueError(f'Fold {fold_id}: training rows {train_end} < {min_train}.')
        folds.append({'fold_id': fold_id, 'train_end_index': train_end, 'calibration_start_index': calibration_start, 'calibration_end_index': calibration_end, 'final_fit_end_index': final_fit_end, 'test_start_index': test_start, 'test_end_index': test_end, 'train_start': common_dates[0], 'train_end': common_dates[train_end - 1], 'calibration_start': common_dates[calibration_start], 'calibration_end': common_dates[calibration_end - 1], 'purge_start': common_dates[calibration_end], 'purge_end': common_dates[test_start - 1], 'test_start': common_dates[test_start], 'test_end': common_dates[test_end - 1], 'decision_dates': common_dates[test_start - 1:test_end]})
        fold_id += 1
        test_start = test_end
    if not folds:
        raise ValueError('No valid expanding walk-forward fold was created.')
    return folds


def _analysis_decision_dates(
    common_dates: pd.DatetimeIndex,
    folds: list[dict[str, Any]],
    config: Any,
) -> pd.DatetimeIndex:
    






    if not folds:
        raise ValueError('No walk-forward fold is available for the analysis window.')

    champion_oos_start = int(folds[0]['test_start_index'])
    champion_oos_end = int(folds[-1]['test_end_index'])

    requested_start = pd.Timestamp(getattr(config, 'analysis_start_date', config.start_date))
    requested_start = (
        requested_start.tz_localize('UTC')
        if requested_start.tzinfo is None
        else requested_start.tz_convert('UTC')
    )
    requested_execution_start = int(common_dates.searchsorted(requested_start, side='left'))

    if requested_execution_start >= champion_oos_end:
        raise ValueError(
            'The requested analysis start is after the last available champion '
            f'out-of-sample session: requested={requested_start.date()}, '
            f'last={common_dates[champion_oos_end - 1].date()}.'
        )

    execution_start = max(champion_oos_start, requested_execution_start)
    if execution_start >= champion_oos_end:
        raise ValueError('The requested analysis interval contains no executable session.')

    
    
    return common_dates[execution_start - 1:champion_oos_end]

def _scheduled_policy(policies: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]], decision_to_fold: dict[pd.Timestamp, int]) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        fold_id = decision_to_fold.get(key)
        if fold_id is None:
            raise KeyError(f'No walk-forward policy is assigned to {key}.')
        return policies[int(fold_id)](timestamp, current_position, holding_days)
    return policy

def _fold_performance(predictions: pd.DataFrame, folds: list[dict[str, Any]], initial_capital: float) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    rows = predictions.reset_index().sort_values('timestamp').reset_index(drop=True)
    output: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold['fold_id'])
        subset = rows.loc[rows['walk_forward_fold'] == fold_id]
        if subset.empty:
            continue
        first_idx = int(subset.index[0])
        strategy_start = float(initial_capital) if first_idx == 0 else float(rows.loc[first_idx - 1, 'strategy_equity'])
        benchmark_start = float(initial_capital) if first_idx == 0 else float(rows.loc[first_idx - 1, 'buy_hold_equity'])
        strategy_end = float(subset.iloc[-1]['strategy_equity'])
        benchmark_end = float(subset.iloc[-1]['buy_hold_equity'])
        curve = pd.Series([strategy_start, *subset['strategy_equity'].astype(float).tolist()])
        output.append({
            'fold_id': fold_id,
            'train_end': fold['train_end'],
            'calibration_start': fold['calibration_start'],
            'calibration_end': fold['calibration_end'],
            'purge_start': fold['purge_start'],
            'purge_end': fold['purge_end'],
            'model_test_start': fold['test_start'],
            'model_test_end': fold['test_end'],
            'test_start': pd.Timestamp(subset.iloc[0]['timestamp']),
            'test_end': pd.Timestamp(subset.iloc[-1]['timestamp']),
            'strategy_starting_capital': strategy_start,
            'strategy_ending_capital': strategy_end,
            'strategy_return': strategy_end / strategy_start - 1,
            'benchmark_return': benchmark_end / benchmark_start - 1,
            'excess_return': strategy_end / strategy_start - benchmark_end / benchmark_start,
            'maximum_drawdown': _maximum_drawdown(curve),
            'sessions': int(len(subset)),
        })
    return output

def _run_xgboost_rotation_models(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[RotationRunResult]:
    
    if config.strategy_mode not in SUPPORTED_ROTATION_MODES:
        raise ValueError(f'Unsupported compound-rotation strategy mode: {config.strategy_mode}.')
    if list(config.rotation_models) != ['xgboost_utility']:
        raise ValueError("This version supports only rotation_models=['xgboost_utility'].")
    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    folds = _build_walk_forward_folds(common_dates, config)
    xgb_plan = resolve_xgboost_compute_plan(config)
    repetitions = int(config.rotation_xgb_repetitions)
    seed_step = int(config.rotation_seed_step)
    if repetitions <= 0:
        raise ValueError('At least one XGBoost repetition is required.')
    all_decision_dates = _analysis_decision_dates(common_dates, folds, config)
    decision_to_fold: dict[pd.Timestamp, int] = {}
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] = {}
    for fold in folds:
        for timestamp in fold['decision_dates'][:-1]:
            key = pd.Timestamp(timestamp)
            decision_to_fold[key] = int(fold['fold_id'])
            decision_metadata[key] = {
                'fold_id': int(fold['fold_id']),
                'test_start': fold['test_start'],
                'test_end': fold['test_end'],
            }
    device_label = f'CUDA — {xgb_plan.gpu_name}' if xgb_plan.selected == 'cuda' else 'CPU'
    if progress_callback is not None:
        progress_callback(
            18.0,
            f'Prepared {len(symbols)} assets and {len(folds)} folds — XGBoost={device_label}',
            0,
        )

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    def emit_detail(**values: Any) -> None:
        if progress_detail_callback is not None:
            progress_detail_callback(values)

    def report(fraction: float, stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(
                20.0 + 72.0 * max(0.0, min(1.0, float(fraction))),
                stage,
                completed,
            )

    def backend_id(seed: int) -> str:
        return 'xgboost_utility' if repetitions <= 1 else f'xgboost_utility_seed_{seed}'

    def trade_wrapper(seed: int, repetition_index: int) -> Callable[[dict[str, Any]], None] | None:
        if trade_callback is None:
            return None

        def emit(trade: dict[str, Any]) -> None:
            payload = dict(trade)
            payload['model_family'] = 'xgboost_utility'
            payload['random_seed'] = seed
            payload['repetition_index'] = repetition_index
            payload['model'] = 'XGBoost Utility' + (f' · seed {seed}' if repetitions > 1 else '')
            trade_callback(payload)
        return emit

    results: list[RotationRunResult] = []
    effective_device = xgb_plan.selected
    total_folds = len(folds)
    total_models = len(symbols)
    technical(
        f'event=walk_forward_start runs={repetitions} folds={total_folds} '
        f'assets={total_models} device={effective_device} '
        f'xgboost_version={xgb_plan.framework_version or "unknown"}'
    )

    for repetition in range(repetitions):
        run_index = repetition + 1
        seed = int(config.random_state) + repetition * seed_step
        rep_config = config.model_copy(update={'random_state': seed})
        policies: dict[int, Callable] = {}
        decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        margin_details: list[dict[str, Any]] = []
        fallback_reasons: list[str] = []
        technical(
            f'event=run_start run={run_index}/{repetitions} seed={seed} '
            f'device={effective_device}'
        )
        emit_detail(
            run_index=run_index,
            run_count=repetitions,
            fold_index=0,
            fold_count=total_folds,
            phase='Preparing run',
            trained_models=0,
            total_models=total_models,
            device=effective_device.upper(),
        )

        run_base = repetition / repetitions
        run_span = 1.0 / repetitions
        training_span = run_span * 0.90
        for fold_position, fold in enumerate(folds, start=1):
            fold_span = training_span / max(1, total_folds)
            fold_base = run_base + (fold_position - 1) * fold_span
            fold_id = int(fold['fold_id'])
            train_dates = common_dates[:int(fold['train_end_index'])]
            calibration_dates = common_dates[
                int(fold['calibration_start_index']):int(fold['calibration_end_index'])
            ]
            final_fit_dates = common_dates[:int(fold['final_fit_end_index'])]
            technical(
                f'event=fold_start run={run_index}/{repetitions} '
                f'fold={fold_position}/{total_folds} fold_id={fold_id} '
                f'train_sessions={len(train_dates)} calibration_sessions={len(calibration_dates)} '
                f'final_fit_sessions={len(final_fit_dates)} device={effective_device}'
            )

            def phase_progress(
                phase_label: str,
                phase_start: float,
                phase_end: float,
            ) -> Callable[[int, int, str], None]:
                def callback(position: int, total: int, model_device: str) -> None:
                    fraction_in_phase = position / max(1, total)
                    overall = fold_base + fold_span * (
                        phase_start + (phase_end - phase_start) * fraction_in_phase
                    )
                    stage = (
                        f'Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} '
                        f'— {phase_label} {position}/{total}'
                    )
                    report(overall, stage, repetition)
                    emit_detail(
                        run_index=run_index,
                        run_count=repetitions,
                        fold_index=fold_position,
                        fold_count=total_folds,
                        phase=phase_label.title(),
                        trained_models=position,
                        total_models=total,
                        device=model_device.upper(),
                    )
                return callback

            report(
                fold_base,
                f'Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} — calibration training 0/{total_models}',
                repetition,
            )
            emit_detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase='Calibration training',
                trained_models=0,
                total_models=total_models,
                device=effective_device.upper(),
            )
            calibration_models, effective_device, fallback_reason = _fit_xgb_models(
                frames,
                symbols,
                train_dates,
                rep_config,
                effective_device,
                phase=f'run_{run_index}_fold_{fold_position}_calibration',
                progress_callback=phase_progress('calibration training', 0.02, 0.38),
                technical_log_callback=technical_log_callback,
            )
            if fallback_reason:
                fallback_reasons.append(fallback_reason)
            calibration_cash_edge_models = None
            if _risk_off_enabled(rep_config):
                calibration_cash_edge_models, effective_device, fallback_reason = _fit_xgb_models(
                    frames,
                    symbols,
                    train_dates,
                    rep_config,
                    effective_device,
                    phase=f'run_{run_index}_fold_{fold_position}_calibration_cash_edge',
                    technical_log_callback=technical_log_callback,
                    target_column='forward_cash_edge',
                )
                if fallback_reason:
                    fallback_reasons.append(fallback_reason)

            report(
                fold_base + fold_span * 0.42,
                f'Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} — evaluating rotation policy candidates',
                repetition,
            )
            emit_detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase='Policy calibration',
                trained_models=total_models,
                total_models=total_models,
                device=effective_device.upper(),
            )
            candidate_margins = tuple(
                float(value) for value in rep_config.rotation_switch_margin_candidates
            )
            best_candidate = candidate_margins[0]
            best_score = float('-inf')
            for candidate in candidate_margins:
                calibration_policy = _xgb_policy(
                    calibration_models,
                    frames,
                    symbols,
                    rep_config,
                    candidate,
                    cash_edge_models=calibration_cash_edge_models,
                )
                score = _simple_policy_growth(
                    calibration_policy,
                    frames,
                    symbols,
                    calibration_dates,
                    rep_config,
                )
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            report(
                fold_base + fold_span * 0.48,
                f'Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} — final training 0/{total_models}',
                repetition,
            )
            emit_detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase='Final training',
                trained_models=0,
                total_models=total_models,
                device=effective_device.upper(),
            )
            final_models, effective_device, fallback_reason = _fit_xgb_models(
                frames,
                symbols,
                final_fit_dates,
                rep_config,
                effective_device,
                phase=f'run_{run_index}_fold_{fold_position}_final',
                progress_callback=phase_progress('final training', 0.50, 0.90),
                technical_log_callback=technical_log_callback,
            )
            if fallback_reason:
                fallback_reasons.append(fallback_reason)
            final_cash_edge_models = None
            if _risk_off_enabled(rep_config):
                final_cash_edge_models, effective_device, fallback_reason = _fit_xgb_models(
                    frames,
                    symbols,
                    final_fit_dates,
                    rep_config,
                    effective_device,
                    phase=f'run_{run_index}_fold_{fold_position}_final_cash_edge',
                    technical_log_callback=technical_log_callback,
                    target_column='forward_cash_edge',
                )
                if fallback_reason:
                    fallback_reasons.append(fallback_reason)
            effective_margin = max(
                float(rep_config.rotation_switch_margin),
                float(best_candidate),
            )
            policies[fold_id] = _xgb_policy(
                final_models,
                frames,
                symbols,
                rep_config,
                effective_margin,
                cash_edge_models=final_cash_edge_models,
                decision_diagnostics=decision_diagnostics,
                fold_id=fold_id,
                calibrated_switch_margin=float(best_candidate),
            )
            margin_details.append({
                'fold_id': fold_id,
                'calibrated_candidate_margin': float(best_candidate),
                'effective_switch_margin': float(effective_margin),
                'calibration_risk_adjusted_score': float(best_score),
            })
            report(
                fold_base + fold_span,
                f'Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} completed',
                repetition,
            )
            emit_detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase='Fold completed',
                trained_models=total_models,
                total_models=total_models,
                device=effective_device.upper(),
            )
            technical(
                f'event=fold_complete run={run_index}/{repetitions} '
                f'fold={fold_position}/{total_folds} fold_id={fold_id} '
                f'device={effective_device}'
            )

        report(
            run_base + run_span * 0.94,
            f'Run {run_index}/{repetitions} — simulating out-of-sample portfolio',
            repetition,
        )
        emit_detail(
            run_index=run_index,
            run_count=repetitions,
            fold_index=total_folds,
            fold_count=total_folds,
            phase='Out-of-sample simulation',
            trained_models=total_models,
            total_models=total_models,
            device=effective_device.upper(),
        )
        technical(
            f'event=simulation_start run={run_index}/{repetitions} '
            f'decision_sessions={max(0, len(all_decision_dates) - 1)}'
        )
        scheduled = _scheduled_policy(policies, decision_to_fold)
        result = _simulate_exact(
            'xgboost_utility',
            scheduled,
            frames,
            symbols,
            all_decision_dates,
            rep_config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=decision_diagnostics,
            trade_callback=trade_wrapper(seed, run_index),
        )
        unique_backend = backend_id(seed)
        result.backend = unique_backend
        result.metrics['backend'] = unique_backend
        result.metrics['model_family'] = 'xgboost_utility'
        result.metrics['champion_fold_schedule_locked'] = True
        result.metrics['champion_oos_start'] = folds[0]['test_start']
        result.metrics['requested_analysis_start'] = pd.Timestamp(all_decision_dates[1])
        result.metrics['requested_analysis_end'] = pd.Timestamp(all_decision_dates[-1])
        result.metrics['random_seed'] = seed
        result.metrics['repetition_index'] = run_index
        result.metrics['repetition_count'] = repetitions
        result.metrics['strategy_label'] = 'XGBoost Utility' + (
            f' · seed {seed}' if repetitions > 1 else ''
        )
        fold_metrics = _fold_performance(
            result.predictions,
            folds,
            float(rep_config.initial_capital),
        )
        margin_by_fold = {item['fold_id']: item for item in margin_details}
        for item in fold_metrics:
            item.update(margin_by_fold.get(item['fold_id'], {}))
        effective_values = [item['effective_switch_margin'] for item in margin_details]
        candidate_values = [item['calibrated_candidate_margin'] for item in margin_details]
        result.metrics.update({
            'walk_forward_fold_count': len(folds),
            'walk_forward_folds': fold_metrics,
            'calibrated_candidate_margin_mean': float(np.mean(candidate_values)),
            'effective_switch_margin_mean': float(np.mean(effective_values)),
            'effective_switch_margin_min': float(np.min(effective_values)),
            'effective_switch_margin_max': float(np.max(effective_values)),
            'calibrated_switch_margin': float(np.mean(candidate_values)),
            'effective_switch_margin': float(np.mean(effective_values)),
            'requested_accelerator': xgb_plan.requested,
            'effective_compute_device': effective_device,
            'cuda_available': xgb_plan.cuda_available,
            'gpu_name': xgb_plan.gpu_name,
            'framework_version': xgb_plan.framework_version,
            'cuda_build': xgb_plan.cuda_build,
            'cpu_fallback_used': bool(xgb_plan.fallback_used or fallback_reasons),
            'compute_fallback_reason': (
                fallback_reasons[-1] if fallback_reasons else xgb_plan.fallback_reason
            ),
            'deterministic_execution': bool(rep_config.deterministic_execution),
            'numeric_thread_limit': int(rep_config.numeric_thread_limit),
            'xgb_n_jobs': int(rep_config.xgb_n_jobs),
            'decision_diagnostics_schema_version': 2,
            'position_risk_diagnostics_schema_version': 1,
            'market_regime_diagnostics_schema_version': 1,
            'decision_diagnostics_rows': int(len(decision_diagnostics)),
            'position_risk_diagnostics_rows': int(
                result.predictions.get('position_risk_diagnostics_schema_version', pd.Series(dtype=float)).notna().sum()
            ),
            'market_regime_diagnostics_rows': int(
                result.predictions.get('market_regime_diagnostics_schema_version', pd.Series(dtype=float)).notna().sum()
            ),
            'decision_diagnostics_rotation_rows': int(sum(
                bool(item.get('decision_is_rotation')) for item in decision_diagnostics.values()
            )),
            'decision_diagnostics_hold_rows': int(sum(
                str(item.get('decision_reason') or '').startswith(('HOLD_', 'SWITCH_MARGIN_', 'MIN_HOLD_'))
                for item in decision_diagnostics.values()
            )),
        })
        result.summary += '\n\nROBUSTNESS / COMPUTE\n'
        result.summary += f'Seed: {seed}\n'
        result.summary += f'Repetition: {run_index}/{repetitions}\n'
        result.summary += f'Compute device: {effective_device.upper()}\n'
        result.summary += f'Deterministic execution: {bool(rep_config.deterministic_execution)}\n'
        result.summary += f'XGBoost workers: {int(rep_config.xgb_n_jobs)}\n'
        result.summary += f'Numeric thread limit: {int(rep_config.numeric_thread_limit)}\n'
        if fallback_reasons:
            result.summary += f'Fallback: {fallback_reasons[-1]}\n'
        results.append(result)
        technical(
            f'event=run_complete run={run_index}/{repetitions} seed={seed} '
            f'device={effective_device} rotations={result.metrics.get("capital_rotations")} '
            f'ending_capital={result.metrics.get("strategy_ending_capital")}'
        )
        report(
            run_index / repetitions,
            f'XGBoost Utility run {run_index}/{repetitions} completed',
            run_index,
        )
        emit_detail(
            run_index=run_index,
            run_count=repetitions,
            fold_index=total_folds,
            fold_count=total_folds,
            phase='Run completed',
            trained_models=total_models,
            total_models=total_models,
            device=effective_device.upper(),
        )
    results.sort(key=lambda result: int(result.metrics.get('repetition_index', 1)))
    technical(f'event=walk_forward_complete runs={repetitions} results={len(results)}')
    return results

def run_rotation_models(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[RotationRunResult]:
    




    model_family = str(getattr(config, 'research_model_family', 'xgboost_utility'))
    if model_family == 'xgboost_utility':
        return _run_xgboost_rotation_models(
            bars_by_symbol,
            config,
            fee_calculator,
            slippage,
            progress_callback=progress_callback,
            trade_callback=trade_callback,
            progress_detail_callback=progress_detail_callback,
            technical_log_callback=technical_log_callback,
        )
    from .research_challengers import run_research_challenger
    return run_research_challenger(
        model_family,
        bars_by_symbol,
        config,
        fee_calculator,
        slippage,
        progress_callback=progress_callback,
        trade_callback=trade_callback,
        progress_detail_callback=progress_detail_callback,
        technical_log_callback=technical_log_callback,
    )

