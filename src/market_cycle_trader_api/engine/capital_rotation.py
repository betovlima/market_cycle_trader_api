from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import math
from collections import Counter
import subprocess
from typing import Any, Callable
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROTATION_FEATURES = ['return_1', 'return_3', 'return_5', 'return_10', 'return_20', 'vol_5', 'vol_20', 'ema_distance_5', 'ema_distance_10', 'ema_distance_20', 'ema_distance_50', 'ema_5_vs_20', 'ema_20_vs_50', 'rsi_14', 'atr_pct_14', 'distance_from_high_20', 'distance_from_low_20', 'volume_zscore_20']

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
    horizon_days = int(config.rotation_horizon_days)
    data = bars.copy().sort_index()
    data.index = pd.to_datetime(data.index, utc=True)
    close = data['close'].astype(float)
    open_price = data['open'].astype(float)
    high = data['high'].astype(float)
    low = data['low'].astype(float)
    volume = data['volume'].astype(float)
    daily_return = close.pct_change()
    for period in [1, 3, 5, 10, 20]:
        data[f'return_{period}'] = close.pct_change(period)
    for period in [5, 20]:
        data[f'vol_{period}'] = daily_return.rolling(period).std()
    ema = {}
    for period in [5, 10, 20, 50]:
        ema[period] = close.ewm(span=period, adjust=False).mean()
        data[f'ema_distance_{period}'] = _safe_divide(close, ema[period]) - 1
    data['ema_5_vs_20'] = _safe_divide(ema[5], ema[20]) - 1
    data['ema_20_vs_50'] = _safe_divide(ema[20], ema[50]) - 1
    data['rsi_14'] = _rsi(close) / 100.0
    atr = _true_range(data).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data['atr_pct_14'] = _safe_divide(atr, close)
    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    data['distance_from_high_20'] = _safe_divide(close, rolling_high) - 1
    data['distance_from_low_20'] = _safe_divide(close, rolling_low) - 1
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    data['volume_zscore_20'] = _safe_divide(volume - volume_mean, volume_std)
    entry_open = open_price.shift(-1)
    exit_close = close.shift(-horizon_days)
    gross_log_return = np.log(_safe_divide(exit_close, entry_open))
    round_trip_cost = min(0.25, 2.0 * (max(0.0, float(config.slippage_bps)) / 10000.0 + max(0.0, float(config.commission_rate))))
    net_cost_log = math.log(max(1e-12, 1.0 - round_trip_cost))
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = open_price.to_numpy(dtype=float)
    downside = np.full(len(data), np.nan, dtype=float)
    path_drawdown = np.full(len(data), np.nan, dtype=float)
    for idx in range(0, max(0, len(data) - horizon_days)):
        entry = opens[idx + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        future_lows = lows[idx + 1:idx + horizon_days + 1]
        future_closes = closes[idx + 1:idx + horizon_days + 1]
        if len(future_closes) != horizon_days:
            continue
        minimum_low = float(np.nanmin(future_lows))
        downside[idx] = max(0.0, 1.0 - minimum_low / entry)
        path = np.concatenate(([entry], future_closes))
        running_peak = np.maximum.accumulate(path)
        drawdowns = 1.0 - np.divide(path, running_peak, out=np.ones_like(path), where=running_peak > 0)
        path_drawdown[idx] = max(0.0, float(np.nanmax(drawdowns)))
    data['forward_net_log_return'] = gross_log_return + net_cost_log
    data['forward_downside'] = downside
    data['forward_max_drawdown'] = path_drawdown
    data['forward_risk_adjusted_utility'] = data['forward_net_log_return'] - float(config.rotation_downside_penalty) * data['forward_downside'] - float(config.rotation_drawdown_penalty) * data['forward_max_drawdown']
    required = ROTATION_FEATURES + ['open', 'high', 'low', 'close', 'volume']
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    return data

def prepare_rotation_panel(bars_by_symbol: dict[str, pd.DataFrame], config: Any) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames = {symbol: build_rotation_frame(frame, config) for symbol, frame in bars_by_symbol.items() if frame is not None and (not frame.empty)}
    if len(frames) < 2:
        raise ValueError('Compound rotation needs at least two assets with valid aligned data.')
    common: pd.DatetimeIndex | None = None
    for frame in frames.values():
        index = pd.DatetimeIndex(frame.index)
        common = index if common is None else common.intersection(index)
    if common is None or len(common) < 700:
        raise ValueError('The common aligned history is too short for train/calibration/test.')
    common = common.sort_values()
    frames = {symbol: frame.loc[common].copy() for symbol, frame in frames.items()}
    return (frames, common)

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


def _fit_xgb_models(frames: dict[str, pd.DataFrame], symbols: list[str], train_dates: pd.DatetimeIndex, config: Any, device_name: str) -> tuple[dict[str, Any], str, str | None]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError('XGBoost Utility requires xgboost. Install requirements.txt.') from exc
    allow_fallback = bool(config.rotation_allow_cpu_fallback)

    def fit_on_device(effective_device: str) -> dict[str, Any]:
        fitted: dict[str, Any] = {}
        with _numeric_thread_context(config):
            for symbol in symbols:
                frame = frames[symbol].loc[train_dates].dropna(
                    subset=['forward_risk_adjusted_utility']
                )
                if len(frame) < int(config.rotation_minimum_training_rows):
                    raise ValueError(
                        f'{symbol}: only {len(frame)} utility rows are available; '
                        f'{config.rotation_minimum_training_rows} are required.'
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
                )
                model.fit(
                    frame[ROTATION_FEATURES],
                    frame['forward_risk_adjusted_utility'],
                )
                fitted[symbol] = model
        return fitted
    try:
        return (fit_on_device(device_name), device_name, None)
    except Exception as exc:
        if device_name != 'cuda' or not allow_fallback:
            raise
        fallback_reason = f'XGBoost CUDA initialization/training failed; using CPU instead: {exc}'
        return (fit_on_device('cpu'), 'cpu', fallback_reason)

def _xgb_utilities(models: dict[str, Any], frames: dict[str, pd.DataFrame], symbols: list[str], timestamp: pd.Timestamp, config: Any) -> np.ndarray:
    values = [0.0]
    for symbol in symbols:
        row = frames[symbol].loc[[timestamp], ROTATION_FEATURES]
        prediction = float(models[symbol].predict(row)[0])
        values.append(prediction)
    return np.asarray(values, dtype=np.float64)

def _xgb_policy(models: dict[str, Any], frames: dict[str, pd.DataFrame], symbols: list[str], config: Any, switch_margin: float) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        utilities = _xgb_utilities(models, frames, symbols, timestamp, config)
        best = int(np.nanargmax(utilities))
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])
        if current_position > 0 and holding_days < int(config.rotation_min_holding_days):
            return (current_position, current_value)
        minimum = float(config.rotation_cash_threshold)
        if best == 0 or best_value <= minimum:
            return (0, 0.0)
        if current_position == 0:
            if best_value >= minimum + float(config.rotation_min_expected_edge):
                return (best, best_value)
            return (0, 0.0)
        if best == current_position:
            return (current_position, current_value)
        required = max(float(config.rotation_switch_margin), float(switch_margin))
        if best_value >= current_value + required:
            return (best, best_value)
        return (current_position, current_value)
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
    capital_per_asset = float(initial_capital) / len(symbols)
    quantities: dict[str, float] = {}
    residual = 0.0
    for symbol in symbols:
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
    last = execution_dates[-1]
    final_cash = residual
    for symbol, quantity in quantities.items():
        sell_price = float(slippage(float(frames[symbol].loc[last, 'close']), 'SELL', config))
        fees = fee_calculator('SELL', quantity, sell_price, config)
        final_cash += quantity * sell_price - float(fees['total_fee'])
    series.iloc[-1] = final_cash
    return series

def _simulate_exact(backend: str, policy: Callable[[pd.Timestamp, int, int], tuple[int, float]], frames: dict[str, pd.DataFrame], symbols: list[str], decision_dates: pd.DatetimeIndex, config: Any, fee_calculator: Callable, slippage: Callable, decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None=None, policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None=None, trade_callback: Callable[[dict[str, Any]], None] | None=None) -> RotationRunResult:
    if len(decision_dates) < 2:
        raise ValueError('The final-test interval is too short.')
    execution_dates = decision_dates[1:]
    benchmark = _equal_weight_benchmark(frames, symbols, execution_dates, float(config.initial_capital), config, fee_calculator, slippage)
    cash = float(config.initial_capital)
    position = 0
    quantity = 0.0
    entry_price = float('nan')
    entry_time = None
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
        day_trades: list[dict[str, Any]] = []
        if target_position != position:
            old_symbol = symbols[position - 1] if position > 0 else None
            new_symbol = symbols[target_position - 1] if target_position > 0 else None
            is_rotation = previous_position > 0 and target_position > 0
            rotation_id = f'{pd.Timestamp(execution_date).isoformat()}::{old_symbol}->{new_symbol}' if is_rotation else None
            decision_trade_fields = {'decision_timestamp': pd.Timestamp(decision_date), 'rotation_id': rotation_id, 'rotation_from_asset': old_symbol if is_rotation else None, 'rotation_to_asset': new_symbol if is_rotation else None, 'q_current_position': decision_diag.get('q_current_position'), 'q_raw_best': decision_diag.get('q_raw_best'), 'q_final_action': decision_diag.get('q_final_action'), 'q_delta_final_vs_current': decision_diag.get('q_delta_final_vs_current'), 'q_gap_best_vs_second': decision_diag.get('q_gap_best_vs_second'), 'raw_action_asset': decision_diag.get('raw_action_asset'), 'final_action_asset': decision_diag.get('final_action_asset'), 'min_hold_guard_applied': decision_diag.get('min_hold_guard_applied'), 'day_trade_constraint_applied': decision_diag.get('day_trade_constraint_applied')}
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
                holding_days = 1
                day_trades.append({'timestamp': execution_date, 'action': 'BUY', 'asset': symbol, 'reason': f'ROTATE_FROM_{old_symbol}' if old_symbol else 'BEST_CAPITAL_UTILITY', 'execution_price': price, 'quantity': quantity, 'gross_trade_value': gross, **fees, 'realized_pnl': 0.0, 'position_return': 0.0, 'holding_bars': 0, 'entry_timestamp': execution_date, 'entry_price': price, 'cash_after_trade': cash, 'shares_after_trade': quantity, 'walk_forward_fold': fold_id, **decision_trade_fields})
            if previous_position > 0 and target_position > 0:
                rotation_count += 1
        elif position > 0:
            holding_days += 1
        records.extend(day_trades)
        if trade_callback is not None:
            for trade in day_trades:
                trade_callback({**trade, 'backend': backend, 'model': 'XGBoost Utility'})
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
            trade_callback({**final_trade, 'backend': backend, 'model': 'XGBoost Utility'})
        equity_values[-1] = cash
        prediction_rows[-1]['strategy_equity'] = cash
        prediction_rows[-1]['trade_action'] = prediction_rows[-1]['trade_action'] or 'FINAL_SELL'
        prediction_rows[-1]['trade_reason'] = prediction_rows[-1]['trade_reason'] or 'FINAL_LIQUIDATION'
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
    metrics = {'portfolio_rotation': True, 'strategy_mode': config.strategy_mode, 'strategy_label': 'XGBoost Utility', 'symbol': 'PORTFOLIO', 'backend': backend, 'assets': symbols, 'timeframe': '1Day', 'decision_horizon_days': int(config.rotation_horizon_days), 'decision_horizon_bars': None, 'decision_horizon_label': f'{int(config.rotation_horizon_days)} trading sessions', 'overnight_positions_allowed': True, 'benchmark_name': 'Equal-weight buy-and-hold', 'walk_forward_enabled': bool(config.rotation_walk_forward_enabled), 'walk_forward_purge_days': int(config.rotation_purge_days), 'walk_forward_calibration_days': int(config.rotation_walk_forward_calibration_days), 'walk_forward_test_days': int(config.rotation_walk_forward_test_days), 'downside_penalty': float(config.rotation_downside_penalty), 'drawdown_penalty': float(config.rotation_drawdown_penalty), 'initial_capital': initial, 'strategy_ending_capital': ending, 'strategy_return': ending / initial - 1, 'buy_hold_ending_capital': benchmark_ending, 'buy_hold_return': benchmark_ending / initial - 1, 'excess_return': ending / initial - benchmark_ending / initial, 'strategy_maximum_drawdown': _maximum_drawdown(strategy_curve), 'buy_hold_maximum_drawdown': _maximum_drawdown(benchmark_curve), 'strategy_sharpe': _annualized_sharpe(strategy_curve, periods_per_year), 'buy_hold_sharpe': _annualized_sharpe(benchmark_curve, periods_per_year), 'strategy_cagr': _cagr(strategy_curve), 'buy_hold_cagr': _cagr(benchmark_curve), 'compound_log_growth': float(math.log(max(ending / initial, 1e-12))), 'risk_adjusted_compound_score': _curve_risk_adjusted_score(strategy_curve, config), 'market_exposure': float(exposure), 'cash_days': cash_days, 'simulated_buys': buys, 'simulated_sells': sells, 'capital_rotations': int(rotation_count), 'cycles_per_year': float(buys / years), 'average_holding_days': avg_holding, 'average_holding_bars': avg_holding, 'average_holding_minutes': None, 'geometric_trade_return': _geometric_trade_return(trades), 'total_transaction_fees': float(total_fees), 'turnover_ratio': float(turnover / max(initial, 1e-09)), 'test_start': execution_dates[0], 'test_end': execution_dates[-1], 'test_calendar_years': years}
    summary = '\n'.join(['COMPOUND CAPITAL ROTATION — SWING', '', f"Model: {metrics['strategy_label']}", f"Assets: {', '.join(symbols)}", 'Decision data: daily candles', f'Utility horizon: {config.rotation_horizon_days} trading sessions', 'Capital pool: one shared account, reinvested after every exit/rotation', 'Decision objective: maximize smoother net compounded wealth, not predict exact tops.', f'Risk penalties: downside={config.rotation_downside_penalty:.3f}, drawdown={config.rotation_drawdown_penalty:.3f}', f'Validation: expanding walk-forward, purge={config.rotation_purge_days} sessions, fold test={config.rotation_walk_forward_test_days} sessions', '', 'OUT-OF-SAMPLE WALK-FORWARD', f'Initial capital: ${initial:,.2f}', f'Ending capital: ${ending:,.2f}', f"Total return: {metrics['strategy_return']:.2%}", f"CAGR: {metrics['strategy_cagr']:.2%}", f"Compound log growth: {metrics['compound_log_growth']:.6f}", f"Maximum drawdown: {metrics['strategy_maximum_drawdown']:.2%}", f"Sharpe estimate: {metrics['strategy_sharpe']:.3f}", f'Capital rotations: {rotation_count}', f'Buys: {buys}', f'Sells including final liquidation: {sells}', f"Cycles/year: {metrics['cycles_per_year']:.2f}", f'Average holding days: {avg_holding:.2f}', f'Time in market: {exposure:.2%}', f'Transaction fees: ${total_fees:,.2f}', '', 'BENCHMARK', 'Equal-weight buy-and-hold across the same available assets.', f'Benchmark ending capital: ${benchmark_ending:,.2f}', f"Benchmark return: {metrics['buy_hold_return']:.2%}", f"Benchmark CAGR: {metrics['buy_hold_cagr']:.2%}", '', 'METHOD', '- Signals use information available at the current daily close.', '- Position changes execute at the next daily open.', f'- XGBoost Utility predicts {config.rotation_horizon_days}-session risk-adjusted capital utility.', '- Every fold is trained only on information available before that fold.', f'- A {config.rotation_purge_days}-session purge prevents forward labels from touching the next validation/test segment.', '- FINAL_LIQUIDATION is bookkeeping only and is not a model decision.'])
    return RotationRunResult(backend=backend, predictions=predictions, trades=trades, summary=summary, metrics=metrics)

def _build_walk_forward_folds(common_dates: pd.DatetimeIndex, config: Any) -> list[dict[str, Any]]:








    purge = max(int(config.rotation_purge_days), int(config.rotation_horizon_days))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    test_days = int(config.rotation_walk_forward_test_days)
    min_test_days = int(config.rotation_walk_forward_min_test_days)
    min_train = int(config.rotation_minimum_training_rows)
    first_test_start = min_train + purge + calibration_days + purge

    if first_test_start >= len(common_dates) - min_test_days:
        available_test_rows = max(0, len(common_dates) - first_test_start)
        raise ValueError(
            'Not enough history for the configured walk-forward protocol: '
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

    protocol_oos_start = int(folds[0]['test_start_index'])
    protocol_oos_end = int(folds[-1]['test_end_index'])

    requested_start = pd.Timestamp(getattr(config, 'analysis_start_date', config.start_date))
    requested_start = (
        requested_start.tz_localize('UTC')
        if requested_start.tzinfo is None
        else requested_start.tz_convert('UTC')
    )
    requested_execution_start = int(common_dates.searchsorted(requested_start, side='left'))

    if requested_execution_start >= protocol_oos_end:
        raise ValueError(
            'The requested analysis start is after the last available evaluation '
            f'out-of-sample session: requested={requested_start.date()}, '
            f'last={common_dates[protocol_oos_end - 1].date()}.'
        )

    execution_start = max(protocol_oos_start, requested_execution_start)
    if execution_start >= protocol_oos_end:
        raise ValueError('The requested analysis interval contains no executable session.')

    
    
    return common_dates[execution_start - 1:protocol_oos_end]

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

def _fold_robustness_metrics(folds: list[dict[str, Any]]) -> dict[str, float]:
    if not folds:
        return {
            'robust_score': float('-inf'),
            'positive_fold_ratio': 0.0,
            'folds_above_benchmark_ratio': 0.0,
            'worst_fold_return': float('nan'),
            'median_fold_return': float('nan'),
            'median_fold_excess_return': float('nan'),
            'fold_return_standard_deviation': float('nan'),
        }
    returns = np.asarray([float(item['strategy_return']) for item in folds], dtype=float)
    excess = np.asarray([float(item['excess_return']) for item in folds], dtype=float)
    drawdowns = np.asarray([abs(float(item['maximum_drawdown'])) for item in folds], dtype=float)
    positive_ratio = float(np.mean(returns > 0))
    above_ratio = float(np.mean(excess > 0))
    worst_return = float(np.min(returns))
    median_return = float(np.median(returns))
    median_excess = float(np.median(excess))
    dispersion = float(np.std(returns))
    median_drawdown = float(np.median(drawdowns))
    robust_score = (
        median_excess
        + 0.35 * median_return
        + 0.20 * positive_ratio
        + 0.20 * above_ratio
        + 0.25 * min(0.0, worst_return)
        - 0.35 * dispersion
        - 0.20 * median_drawdown
    )
    return {
        'robust_score': float(robust_score),
        'positive_fold_ratio': positive_ratio,
        'folds_above_benchmark_ratio': above_ratio,
        'worst_fold_return': worst_return,
        'median_fold_return': median_return,
        'median_fold_excess_return': median_excess,
        'fold_return_standard_deviation': dispersion,
    }


def _majority_vote_policy(
    policies: list[Callable],
    *,
    minimum_agreement: float,
) -> Callable:

    if not policies:
        raise ValueError('At least one seed policy is required for the ensemble.')

    def policy(date: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        decisions = [
            candidate(date, current_position, holding_days)
            for candidate in policies
        ]
        positions = [int(item[0]) for item in decisions]
        scores = [float(item[1]) for item in decisions]
        counts = Counter(positions)
        highest = max(counts.values())
        agreement = highest / len(decisions)
        leaders = sorted(position for position, count in counts.items() if count == highest)

        if agreement < float(minimum_agreement):
            selected_position = int(current_position)
        elif int(current_position) in leaders:
            selected_position = int(current_position)
        elif len(leaders) == 1:
            selected_position = leaders[0]
        else:
            selected_position = max(
                leaders,
                key=lambda position: float(np.median([
                    score
                    for (candidate_position, score) in decisions
                    if int(candidate_position) == position
                ])),
            )

        selected_scores = [
            score
            for candidate_position, score in decisions
            if int(candidate_position) == selected_position
        ]
        selected_score = float(np.median(selected_scores or scores))
        return selected_position, selected_score

    return policy


def run_rotation_models(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[RotationRunResult]:


    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    folds = _build_walk_forward_folds(common_dates, config)
    xgb_plan = resolve_xgboost_compute_plan(config)
    repetitions = int(config.rotation_xgb_repetitions)
    seed_step = int(config.rotation_seed_step)
    ensemble_enabled = repetitions > 1
    ensemble_minimum_agreement = (repetitions // 2 + 1) / repetitions
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
    total_outputs = repetitions + (1 if ensemble_enabled else 0)
    if progress_callback is not None:
        progress_callback(
            18.0,
            f'Prepared {len(symbols)} assets and {len(folds)} folds — '
            f'XGBoost={device_label}; seeds={repetitions}; ensemble={ensemble_enabled}',
            0,
        )

    def report(fraction: float, stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(
                20.0 + 72.0 * max(0.0, min(1.0, float(fraction))),
                stage,
                completed,
            )

    def backend_id(seed: int) -> str:
        return 'xgboost_utility' if repetitions <= 1 else f'xgboost_utility_seed_{seed}'

    def trade_wrapper(seed: int | None, repetition_index: int, label: str) -> Callable[[dict[str, Any]], None] | None:
        if trade_callback is None:
            return None

        def emit(trade: dict[str, Any]) -> None:
            payload = dict(trade)
            payload['model_family'] = 'xgboost_utility'
            payload['random_seed'] = seed
            payload['repetition_index'] = repetition_index
            payload['model'] = label
            trade_callback(payload)

        return emit

    results: list[RotationRunResult] = []
    effective_device = xgb_plan.selected
    policies_by_fold: dict[int, list[Callable]] = {int(fold['fold_id']): [] for fold in folds}
    all_margin_details: list[list[dict[str, Any]]] = []
    seed_values: list[int] = []

    for repetition in range(repetitions):
        seed = int(config.random_state) + repetition * seed_step
        seed_values.append(seed)
        rep_config = config.model_copy(update={'random_state': seed})
        policies: dict[int, Callable] = {}
        margin_details: list[dict[str, Any]] = []
        fallback_reasons: list[str] = []

        for fold_position, fold in enumerate(folds, start=1):
            overall = (repetition + (fold_position - 1) / max(1, len(folds))) / total_outputs
            report(
                overall,
                f'XGBoost seed run {repetition + 1}/{repetitions} — '
                f'fold {fold_position}/{len(folds)} — {effective_device.upper()}',
                repetition,
            )
            fold_id = int(fold['fold_id'])
            train_dates = common_dates[:int(fold['train_end_index'])]
            calibration_dates = common_dates[
                int(fold['calibration_start_index']):int(fold['calibration_end_index'])
            ]
            final_fit_dates = common_dates[:int(fold['final_fit_end_index'])]

            calibration_models, effective_device, fallback_reason = _fit_xgb_models(
                frames, symbols, train_dates, rep_config, effective_device
            )
            if fallback_reason:
                fallback_reasons.append(fallback_reason)

            candidate_margins = tuple(
                float(value) for value in rep_config.rotation_switch_margin_candidates
            )
            best_candidate = candidate_margins[0]
            best_score = float('-inf')
            for candidate in candidate_margins:
                calibration_policy = _xgb_policy(
                    calibration_models, frames, symbols, rep_config, candidate
                )
                score = _simple_policy_growth(
                    calibration_policy, frames, symbols, calibration_dates, rep_config
                )
                if score > best_score:
                    best_score = float(score)
                    best_candidate = float(candidate)

            final_models, effective_device, fallback_reason = _fit_xgb_models(
                frames, symbols, final_fit_dates, rep_config, effective_device
            )
            if fallback_reason:
                fallback_reasons.append(fallback_reason)

            effective_margin = max(
                float(rep_config.rotation_switch_margin),
                float(best_candidate),
            )
            fold_policy = _xgb_policy(
                final_models, frames, symbols, rep_config, effective_margin
            )
            policies[fold_id] = fold_policy
            policies_by_fold[fold_id].append(fold_policy)
            margin_details.append(
                {
                    'fold_id': fold_id,
                    'calibrated_candidate_margin': float(best_candidate),
                    'effective_switch_margin': float(effective_margin),
                    'calibration_risk_adjusted_score': float(best_score),
                }
            )

        all_margin_details.append(margin_details)
        scheduled = _scheduled_policy(policies, decision_to_fold)
        label = 'XGBoost Utility' + (f' · seed {seed}' if repetitions > 1 else '')
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
            trade_callback=trade_wrapper(seed, repetition + 1, label),
        )
        unique_backend = backend_id(seed)
        result.backend = unique_backend
        result.metrics['backend'] = unique_backend
        result.metrics['model_family'] = 'xgboost_utility'
        result.metrics['evaluation_schedule_locked'] = True
        result.metrics['protocol_oos_start'] = folds[0]['test_start']
        result.metrics['requested_analysis_start'] = pd.Timestamp(all_decision_dates[1])
        result.metrics['requested_analysis_end'] = pd.Timestamp(all_decision_dates[-1])
        result.metrics['random_seed'] = seed
        result.metrics['repetition_index'] = repetition + 1
        result.metrics['repetition_count'] = repetitions
        result.metrics['seed_ensemble'] = False
        result.metrics['strategy_label'] = label

        fold_metrics = _fold_performance(
            result.predictions, folds, float(rep_config.initial_capital)
        )
        margin_by_fold = {item['fold_id']: item for item in margin_details}
        for item in fold_metrics:
            item.update(margin_by_fold.get(item['fold_id'], {}))
        result.metrics.update(_fold_robustness_metrics(fold_metrics))
        effective_values = [item['effective_switch_margin'] for item in margin_details]
        candidate_values = [item['calibrated_candidate_margin'] for item in margin_details]
        result.metrics.update(
            {
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
                'compute_fallback_reason': fallback_reasons[-1] if fallback_reasons else xgb_plan.fallback_reason,
                'deterministic_execution': bool(rep_config.deterministic_execution),
                'numeric_thread_limit': int(rep_config.numeric_thread_limit),
                'xgb_n_jobs': int(rep_config.xgb_n_jobs),
            }
        )
        result.summary += '\n\nROBUSTNESS / COMPUTE\n'
        result.summary += f'Seed: {seed}\n'
        result.summary += f'Repetition: {repetition + 1}/{repetitions}\n'
        result.summary += f'Compute device: {effective_device.upper()}\n'
        result.summary += f'Deterministic execution: {bool(rep_config.deterministic_execution)}\n'
        result.summary += f'XGBoost workers: {int(rep_config.xgb_n_jobs)}\n'
        result.summary += f'Numeric thread limit: {int(rep_config.numeric_thread_limit)}\n'
        if fallback_reasons:
            result.summary += f'Fallback: {fallback_reasons[-1]}\n'
        results.append(result)
        report(
            (repetition + 1) / total_outputs,
            f'XGBoost seed run {repetition + 1}/{repetitions} completed',
            repetition + 1,
        )

    if ensemble_enabled:
        ensemble_policies: dict[int, Callable] = {}
        for fold in folds:
            fold_id = int(fold['fold_id'])
            ensemble_policies[fold_id] = _majority_vote_policy(
                policies_by_fold[fold_id],
                minimum_agreement=ensemble_minimum_agreement,
            )
        ensemble_scheduled = _scheduled_policy(ensemble_policies, decision_to_fold)
        ensemble_label = 'XGBoost Seed Ensemble'
        ensemble_result = _simulate_exact(
            'xgboost_utility_ensemble',
            ensemble_scheduled,
            frames,
            symbols,
            all_decision_dates,
            config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            trade_callback=trade_wrapper(None, repetitions + 1, ensemble_label),
        )
        ensemble_result.backend = 'xgboost_utility_ensemble'
        ensemble_result.metrics['backend'] = ensemble_result.backend
        ensemble_result.metrics['model_family'] = 'xgboost_utility'
        ensemble_result.metrics['strategy_label'] = ensemble_label
        ensemble_result.metrics['seed_ensemble'] = True
        ensemble_result.metrics['ensemble_seeds'] = seed_values
        ensemble_result.metrics['ensemble_min_agreement'] = float(ensemble_minimum_agreement)
        ensemble_result.metrics['random_seed'] = None
        ensemble_result.metrics['repetition_index'] = repetitions + 1
        ensemble_result.metrics['repetition_count'] = repetitions
        ensemble_result.metrics['evaluation_schedule_locked'] = True
        ensemble_result.metrics['protocol_oos_start'] = folds[0]['test_start']
        ensemble_result.metrics['requested_analysis_start'] = pd.Timestamp(all_decision_dates[1])
        ensemble_result.metrics['requested_analysis_end'] = pd.Timestamp(all_decision_dates[-1])
        ensemble_result.metrics['walk_forward_fold_count'] = len(folds)
        ensemble_fold_metrics = _fold_performance(
            ensemble_result.predictions, folds, float(config.initial_capital)
        )
        ensemble_result.metrics['walk_forward_folds'] = ensemble_fold_metrics
        ensemble_result.metrics.update(_fold_robustness_metrics(ensemble_fold_metrics))
        ensemble_result.metrics['requested_accelerator'] = xgb_plan.requested
        ensemble_result.metrics['effective_compute_device'] = effective_device
        ensemble_result.metrics['cuda_available'] = xgb_plan.cuda_available
        ensemble_result.metrics['gpu_name'] = xgb_plan.gpu_name
        ensemble_result.metrics['framework_version'] = xgb_plan.framework_version
        ensemble_result.metrics['cuda_build'] = xgb_plan.cuda_build
        ensemble_result.metrics['deterministic_execution'] = bool(config.deterministic_execution)
        ensemble_result.metrics['numeric_thread_limit'] = int(config.numeric_thread_limit)
        ensemble_result.metrics['xgb_n_jobs'] = int(config.xgb_n_jobs)
        ensemble_result.summary += '\n\nPRODUCTION SEED ENSEMBLE\n'
        ensemble_result.summary += f'Seeds: {", ".join(str(seed) for seed in seed_values)}\n'
        ensemble_result.summary += (
            f'Minimum agreement: {ensemble_minimum_agreement:.0%}\n'
        )
        ensemble_result.summary += (
            'The production decision combines seed policies instead of selecting the '
            'best seed after observing test returns.\n'
        )
        results.append(ensemble_result)
        report(1.0, 'XGBoost production seed ensemble completed', total_outputs)

    results.sort(key=lambda result: int(result.metrics.get('repetition_index', 1)))
    return results
