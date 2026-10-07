from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .diagnosticos import enriquecer_diagnosticos_operacoes

BASE_ROTATION_MODE = "COMPOUND_ROTATION_SWING_LIGHTGBM"
SUPPORTED_ROTATION_MODES = frozenset({BASE_ROTATION_MODE})

ROTATION_FEATURES = [
    "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
    "return_40", "return_60", "return_120",
    "vol_5", "vol_10", "vol_20", "vol_40", "vol_60",
    "vol_ratio_5_20", "vol_ratio_10_40", "vol_ratio_20_60",
    "ema_distance_5", "ema_distance_10", "ema_distance_20", "ema_distance_50",
    "ema_distance_100", "ema_5_vs_20", "ema_20_vs_50", "ema_50_vs_100",
    "ema_slope_20_5", "ema_slope_50_10", "ema_slope_100_20",
    "rsi_14", "atr_pct_14",
    "distance_from_high_20", "distance_from_low_20",
    "distance_from_high_50", "distance_from_low_50",
    "distance_from_high_100", "distance_from_low_100",
    "distance_from_high_200", "distance_from_low_200",
    "channel_position_20", "channel_position_50",
    "channel_position_100", "channel_position_200",
    "trend_efficiency_10", "trend_efficiency_20",
    "trend_efficiency_40", "trend_efficiency_60",
    "momentum_acceleration_5_20", "momentum_acceleration_20_60",
    "range_expansion_5_20", "volume_zscore_20", "volume_zscore_60",
    "volume_ratio_5_20",
]


@dataclass
class RotationRunResult:
    backend: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    summary: str
    metrics: dict[str, Any]


def _dividir_com_seguranca(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean = denominator.replace(0, np.nan)
    return numerator / clean

def _rsi(close: pd.Series, period: int=14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _dividir_com_seguranca(avg_gain, avg_loss)
    return 100 - 100 / (1 + rs)

def _amplitude_verdadeira(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame['close'].shift(1)
    return pd.concat([frame['high'] - frame['low'], (frame['high'] - previous_close).abs(), (frame['low'] - previous_close).abs()], axis=1).max(axis=1)

def construir_quadro_rotacao(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
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
    daily_range = _dividir_com_seguranca(high - low, close)

    for period in [1, 2, 3, 5, 10, 20, 40, 60, 120]:
        data[f'return_{period}'] = close.pct_change(period)
    for period in [5, 10, 20, 40, 60]:
        data[f'vol_{period}'] = daily_return.rolling(period).std()
    data['vol_ratio_5_20'] = _dividir_com_seguranca(data['vol_5'], data['vol_20'])
    data['vol_ratio_10_40'] = _dividir_com_seguranca(data['vol_10'], data['vol_40'])
    data['vol_ratio_20_60'] = _dividir_com_seguranca(data['vol_20'], data['vol_60'])

    ema: dict[int, pd.Series] = {}
    for period in [5, 10, 20, 50, 100]:
        ema[period] = close.ewm(span=period, adjust=False).mean()
        data[f'ema_distance_{period}'] = _dividir_com_seguranca(close, ema[period]) - 1
    data['ema_5_vs_20'] = _dividir_com_seguranca(ema[5], ema[20]) - 1
    data['ema_20_vs_50'] = _dividir_com_seguranca(ema[20], ema[50]) - 1
    data['ema_50_vs_100'] = _dividir_com_seguranca(ema[50], ema[100]) - 1
    data['ema_slope_20_5'] = ema[20].pct_change(5)
    data['ema_slope_50_10'] = ema[50].pct_change(10)
    data['ema_slope_100_20'] = ema[100].pct_change(20)

    data['rsi_14'] = _rsi(close) / 100.0
    atr = _amplitude_verdadeira(data).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data['atr_pct_14'] = _dividir_com_seguranca(atr, close)

    for period in [20, 50, 100, 200]:
        rolling_high = high.rolling(period).max()
        rolling_low = low.rolling(period).min()
        data[f'distance_from_high_{period}'] = _dividir_com_seguranca(close, rolling_high) - 1
        data[f'distance_from_low_{period}'] = _dividir_com_seguranca(close, rolling_low) - 1
        data[f'channel_position_{period}'] = _dividir_com_seguranca(close - rolling_low, rolling_high - rolling_low)

    absolute_change = close.pct_change().abs()
    for period in [10, 20, 40, 60]:
        net_move = close.pct_change(period).abs()
        traveled = absolute_change.rolling(period).sum()
        data[f'trend_efficiency_{period}'] = _dividir_com_seguranca(net_move, traveled)

    data['momentum_acceleration_5_20'] = data['return_5'] - data['return_20'] * (5.0 / 20.0)
    data['momentum_acceleration_20_60'] = data['return_20'] - data['return_60'] * (20.0 / 60.0)
    data['range_expansion_5_20'] = _dividir_com_seguranca(daily_range.rolling(5).mean(), daily_range.rolling(20).mean())

    for period in [20, 60]:
        volume_mean = volume.rolling(period).mean()
        volume_std = volume.rolling(period).std()
        data[f'volume_zscore_{period}'] = _dividir_com_seguranca(volume - volume_mean, volume_std)
    data['volume_ratio_5_20'] = _dividir_com_seguranca(volume.rolling(5).mean(), volume.rolling(20).mean())

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
    net_return_components = np.full((len(data), len(horizons)), np.nan, dtype=float)
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
            net_log_return = gross_log_return + net_cost_log
            net_return_components[idx, component_idx] = net_log_return
            utility_components[idx, component_idx] = (
                net_log_return
                - float(config.rotation_downside_penalty) * downside
                - float(config.rotation_drawdown_penalty) * path_drawdown
            )

    weighted_utility = np.nansum(utility_components * weights.reshape(1, -1), axis=1)
    weighted_net_log_return = np.nansum(net_return_components * weights.reshape(1, -1), axis=1)
    invalid_rows = np.isnan(utility_components).any(axis=1)
    weighted_utility[invalid_rows] = np.nan
    weighted_net_log_return[invalid_rows] = np.nan
    
    
    
    
    for component_idx, horizon in enumerate(horizons):
        data[f'forward_horizon_utility_{int(horizon)}'] = utility_components[:, component_idx]
        data[f'forward_horizon_net_log_return_{int(horizon)}'] = net_return_components[:, component_idx]

    data['forward_net_log_return'] = weighted_net_log_return
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

def _selecionar_ativo_fonte_calendario(
    frames: dict[str, pd.DataFrame],
) -> str:
    """Escolhe de forma deterministica o ativo com maior historico valido."""
    if not frames:
        raise ValueError("No valid asset frames are available for calendar selection.")

    def rank(symbol: str) -> tuple[int, int, int, str]:
        index = pd.DatetimeIndex(frames[symbol].index)
        if index.empty:
            return (0, 2**63 - 1, 0, symbol)
        first = pd.Timestamp(index.min()).value
        last = pd.Timestamp(index.max()).value
        return (-len(index), first, -last, symbol)

    return sorted(frames, key=rank)[0]


def preparar_painel_rotacao(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    calendar_override: pd.DatetimeIndex | None = None,
    calendar_source_label: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, str]:
    frames = {
        symbol: construir_quadro_rotacao(frame, config)
        for symbol, frame in bars_by_symbol.items()
        if frame is not None and not frame.empty
    }
    if len(frames) < 2:
        raise ValueError(
            "Compound rotation needs at least two assets with valid data."
        )

    if calendar_override is None:
        calendar_symbol = _selecionar_ativo_fonte_calendario(frames)
        calendar = pd.DatetimeIndex(frames[calendar_symbol].index).sort_values()
    else:
        calendar = pd.DatetimeIndex(calendar_override).sort_values().unique()
        if calendar.empty:
            raise ValueError("The fixed market calendar is empty.")
        calendar_symbol = str(
            calendar_source_label or "FIXED_REFERENCE_CALENDAR"
        )
    minimum_calendar_rows = max(
        700,
        int(getattr(config, "rotation_minimum_training_rows", 700)),
    )
    if len(calendar) < minimum_calendar_rows:
        raise ValueError(
            "The automatically selected market calendar is too short for "
            "train/calibration/test."
        )

    aligned = {
        symbol: frame.reindex(calendar).copy()
        for symbol, frame in frames.items()
    }
    return aligned, calendar, calendar_symbol

def _sharpe_anualizado(curve: pd.Series, periods_per_year: float=252.0) -> float:
    returns = curve.pct_change().dropna()
    if returns.empty or float(returns.std()) <= 0:
        return float('nan')
    return float(np.sqrt(float(periods_per_year)) * returns.mean() / returns.std())

def _drawdown_maximo(curve: pd.Series) -> float:
    if curve.empty:
        return float('nan')
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())

def _cagr(curve: pd.Series, initial_capital: float | None = None) -> float:
    if len(curve) < 2:
        return float('nan')
    start = pd.Timestamp(curve.index[0])
    end = pd.Timestamp(curve.index[-1])
    years = max((end - start).days / 365.25, 1 / 365.25)
    initial = float(curve.iloc[0]) if initial_capital is None else float(initial_capital)
    if not np.isfinite(initial) or initial <= 0 or float(curve.iloc[-1]) <= 0:
        return float('nan')
    return float((curve.iloc[-1] / initial) ** (1 / years) - 1)

def _retorno_geometrico_operacoes(trades: pd.DataFrame) -> float:
    if trades.empty or 'position_return' not in trades:
        return float('nan')
    returns = pd.to_numeric(trades.loc[trades['action'].isin(['SELL', 'FINAL_SELL']), 'position_return'], errors='coerce').dropna()
    if returns.empty:
        return float('nan')
    gross = np.prod(1.0 + returns.clip(lower=-0.999999))
    return float(gross ** (1 / len(returns)) - 1)

def _custo_troca_proporcional(config: Any, from_position: int, to_position: int) -> float:
    if from_position == to_position:
        return 0.0
    one_side = max(0.0, float(config.slippage_bps)) / 10000.0
    one_side += max(0.0, float(config.commission_rate))
    sides = int(from_position != 0) + int(to_position != 0)
    return min(0.25, one_side * sides)

def _retorno_log_transicao_treinamento(frames: dict[str, pd.DataFrame], symbols: list[str], date_now: pd.Timestamp, date_next: pd.Timestamp, from_position: int, to_position: int, config: Any) -> float:
    gross = 1.0
    if from_position > 0:
        symbol = symbols[from_position - 1]
        close_now = float(frames[symbol].loc[date_now, 'close'])
        open_next = float(frames[symbol].loc[date_next, 'open'])
        if close_now > 0:
            gross *= open_next / close_now
    cost = _custo_troca_proporcional(config, from_position, to_position)
    gross *= max(1e-08, 1.0 - cost)
    if to_position > 0:
        symbol = symbols[to_position - 1]
        open_next = float(frames[symbol].loc[date_next, 'open'])
        close_next = float(frames[symbol].loc[date_next, 'close'])
        if open_next > 0:
            gross *= close_next / open_next
    return float(np.log(max(gross, 1e-12)))

def _recompensa_ajustada_risco(log_return: float, wealth_before: float, peak_before: float, config: Any) -> tuple[float, float, float]:
    wealth_after = wealth_before * math.exp(log_return)
    peak_after = max(peak_before, wealth_after)
    previous_drawdown = max(0.0, 1.0 - wealth_before / max(peak_before, 1e-12))
    current_drawdown = max(0.0, 1.0 - wealth_after / max(peak_after, 1e-12))
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown)
    downside = max(0.0, -log_return)
    reward = log_return - float(config.rotation_downside_penalty) * downside - float(config.rotation_drawdown_penalty) * drawdown_increase
    return (float(reward), float(wealth_after), float(peak_after))

def _pontuacao_curva_ajustada_risco(curve: pd.Series, config: Any) -> float:
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

def _contexto_threads_numericas(config: Any):
    
    
    
    if not bool(config.deterministic_execution):
        return nullcontext()
    return threadpool_limits(limits=int(config.numeric_thread_limit))

def _precalcular_utilidades_modelo(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamps: pd.DatetimeIndex,
    config: Any,
) -> tuple[dict[pd.Timestamp, np.ndarray], dict[str, Any]]:
    dates = pd.DatetimeIndex(timestamps)
    matrix = np.full(
        (len(dates), len(symbols) + 1),
        float("-inf"),
        dtype=np.float64,
    )
    matrix[:, 0] = 0.0
    started = time.perf_counter()
    predict_calls = 0
    predicted_rows = 0

    for column, symbol in enumerate(symbols, start=1):
        model = models.get(symbol)
        frame = frames.get(symbol)
        if model is None or frame is None or frame.empty:
            continue

        locations = frame.index.get_indexer(dates)
        candidate_rows = np.flatnonzero(
            (locations >= 0) & (locations + 1 < len(frame.index))
        )
        if len(candidate_rows) == 0:
            continue

        positions = locations[candidate_rows]
        features = frame.iloc[positions][ROTATION_FEATURES]
        feature_ok = ~features.isna().any(axis=1).to_numpy()
        next_rows = frame.iloc[positions + 1]
        next_open = pd.to_numeric(
            next_rows["open"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        next_close = pd.to_numeric(
            next_rows["close"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        market_ok = (
            np.isfinite(next_open)
            & (next_open > 0.0)
            & np.isfinite(next_close)
            & (next_close > 0.0)
        )
        valid_mask = feature_ok & market_ok
        if not np.any(valid_mask):
            continue

        valid_rows = candidate_rows[valid_mask]
        valid_positions = locations[valid_rows]
        batch = frame.iloc[valid_positions][ROTATION_FEATURES]
        prediction = np.asarray(
            model.predict(batch),
            dtype=np.float64,
        )
        matrix[valid_rows, column] = prediction
        predict_calls += 1
        predicted_rows += int(len(prediction))

    cache = {
        pd.Timestamp(date): matrix[row].copy()
        for row, date in enumerate(dates)
    }
    return cache, {
        "cache_build_seconds": time.perf_counter() - started,
        "cache_session_count": int(len(dates)),
        "cache_symbol_count": int(len(symbols)),
        "cache_predict_calls": int(predict_calls),
        "cache_predicted_rows": int(predicted_rows),
        "cache_mode": "batched_lightgbm_prediction",
    }

def _utilidades_modelo(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
    *,
    utility_cache: dict[pd.Timestamp, np.ndarray] | None = None,
) -> np.ndarray:
    key = pd.Timestamp(timestamp)
    if utility_cache is not None:
        cached = utility_cache.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float64).copy()

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

def _politica_utilidade(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    switch_margin: float,
    *,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    fold_id: int | None = None,
    calibrated_switch_margin: float | None = None,
    utility_cache: dict[pd.Timestamp, np.ndarray] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Single-position Control policy used by both Control and Soft."""

    def position_asset(position: int) -> str:
        return "CASH" if position <= 0 else symbols[position - 1]

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        utilities = _utilidades_modelo(
            models,
            frames,
            symbols,
            timestamp,
            config,
            utility_cache=utility_cache,
        )
        if not np.isfinite(utilities[1:]).any():
            return 0, 0.0

        ranked_positions = sorted(
            (
                position
                for position in range(1, len(utilities))
                if np.isfinite(utilities[position])
            ),
            key=lambda position: (
                -float(utilities[position]),
                symbols[position - 1],
            ),
        )
        if not ranked_positions:
            return 0, 0.0

        best = int(np.nanargmax(utilities))
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])
        minimum = float(config.rotation_cash_threshold)
        entry_threshold = minimum + float(config.rotation_min_expected_edge)
        required = max(
            float(config.rotation_switch_margin),
            float(switch_margin),
        )

        ranked_assets = [
            (symbols[position - 1], float(utilities[position]))
            for position in ranked_positions
        ]
        best_asset, best_asset_score = ranked_assets[0]
        second_asset, second_asset_score = (
            ranked_assets[1] if len(ranked_assets) > 1 else (None, None)
        )
        best_vs_second_gap = (
            float(best_asset_score - second_asset_score)
            if second_asset_score is not None
            else None
        )
        best_vs_current_gap = (
            float(best_asset_score - current_value)
            if np.isfinite(current_value)
            else None
        )
        finite_asset_scores = np.asarray(
            [score for _, score in ranked_assets],
            dtype=np.float64,
        )
        universe_score_mean = float(np.mean(finite_asset_scores))
        universe_score_std = float(np.std(finite_asset_scores))
        positive_score_count = int(np.sum(finite_asset_scores > 0.0))
        current_asset_rank = None
        if current_position > 0:
            current_symbol = symbols[current_position - 1]
            current_asset_rank = next(
                (
                    rank
                    for rank, (asset, _) in enumerate(ranked_assets, start=1)
                    if asset == current_symbol
                ),
                None,
            )
        stable_std = universe_score_std if universe_score_std > 1e-12 else None
        best_score_zscore = (
            float((best_asset_score - universe_score_mean) / stable_std)
            if stable_std is not None
            else None
        )
        current_score_zscore = (
            float((current_value - universe_score_mean) / stable_std)
            if np.isfinite(current_value) and stable_std is not None
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
                diagnostic = {
                    "decision_diagnostics_schema_version": 2,
                    "decision_fold_id": fold_id,
                    "current_asset": position_asset(current_position),
                    "current_score": finite(current_value),
                    "holding_days_at_decision": int(holding_days),
                    "raw_best_asset": position_asset(best),
                    "raw_best_score": finite(best_value),
                    "best_asset": best_asset,
                    "best_score": finite(best_asset_score),
                    "second_asset": second_asset,
                    "second_score": (
                        finite(second_asset_score)
                        if second_asset_score is not None
                        else None
                    ),
                    "best_vs_second_gap": best_vs_second_gap,
                    "best_vs_current_gap": best_vs_current_gap,
                    "best_vs_cash_gap": float(best_asset_score),
                    "cash_score": 0.0,
                    "cash_exit_threshold": minimum,
                    "cash_entry_threshold": entry_threshold,
                    "current_asset_rank": current_asset_rank,
                    "universe_score_mean": universe_score_mean,
                    "universe_score_std": universe_score_std,
                    "current_score_zscore": current_score_zscore,
                    "best_score_zscore": best_score_zscore,
                    "best_vs_second_zscore": best_vs_second_zscore,
                    "positive_score_count": positive_score_count,
                    "finite_score_count": int(len(finite_asset_scores)),
                    "rotation_cash_threshold": minimum,
                    "rotation_min_expected_edge": float(
                        config.rotation_min_expected_edge
                    ),
                    "base_switch_margin": float(config.rotation_switch_margin),
                    "calibrated_switch_margin": (
                        float(calibrated_switch_margin)
                        if calibrated_switch_margin is not None
                        else float(switch_margin)
                    ),
                    "effective_switch_margin": required,
                    "final_action_asset": position_asset(target_position),
                    "final_action_score": finite(final_score),
                    "decision_reason": reason,
                    "decision_is_rotation": bool(
                        current_position > 0
                        and target_position > 0
                        and target_position != current_position
                    ),
                    "decision_is_entry": bool(
                        current_position == 0 and target_position > 0
                    ),
                    "decision_is_exit_to_cash": bool(
                        current_position > 0 and target_position == 0
                    ),
                    "min_hold_guard_applied": bool(min_hold_guard),
                    "switch_margin_guard_applied": bool(switch_margin_guard),
                    "cash_threshold_guard_applied": bool(cash_threshold_guard),
                    "minimum_expected_edge_guard_applied": bool(
                        expected_edge_guard
                    ),
                    "day_trade_constraint_applied": False,
                    "q_current_position": finite(current_value),
                    "q_raw_best": finite(best_value),
                    "q_final_action": finite(final_score),
                    "q_delta_final_vs_current": (
                        float(final_score - current_value)
                        if np.isfinite(final_score)
                        and np.isfinite(current_value)
                        else None
                    ),
                    "q_gap_best_vs_second": best_vs_second_gap,
                    "raw_action_asset": position_asset(best),
                }
                for rank in range(3):
                    position = top[rank] if rank < len(top) else 0
                    asset = symbols[position - 1] if position > 0 else None
                    score = float(utilities[position]) if position > 0 else None
                    diagnostic[f"top_{rank + 1}_asset"] = asset
                    diagnostic[f"top_{rank + 1}_score"] = (
                        finite(score) if score is not None else None
                    )
                decision_diagnostics[pd.Timestamp(timestamp)] = diagnostic
            return target_position, final_score

        if (
            current_position > 0
            and np.isfinite(current_value)
            and holding_days < int(config.rotation_min_holding_days)
        ):
            return finish(
                current_position,
                current_value,
                "MIN_HOLD_GUARD",
                min_hold_guard=True,
            )
        if best == 0 or best_value <= minimum:
            return finish(
                0,
                0.0,
                "CASH_THRESHOLD",
                cash_threshold_guard=True,
            )
        if current_position == 0:
            if best_value >= entry_threshold:
                return finish(best, best_value, "ENTER_BEST_ASSET")
            return finish(
                0,
                0.0,
                "MIN_EXPECTED_EDGE_GUARD",
                expected_edge_guard=True,
            )
        if best == current_position:
            return finish(
                current_position,
                current_value,
                "HOLD_CURRENT_BEST",
            )
        if best_value >= current_value + required:
            return finish(best, best_value, "ROTATE_TO_BEST_ASSET")
        return finish(
            current_position,
            current_value,
            "SWITCH_MARGIN_GUARD",
            switch_margin_guard=True,
        )

    return policy

def _crescimento_politica_simples(policy: Callable[[pd.Timestamp, int, int], tuple[int, float]], frames: dict[str, pd.DataFrame], symbols: list[str], decision_dates: pd.DatetimeIndex, config: Any) -> float:
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
        log_return = _retorno_log_transicao_treinamento(frames, symbols, now, nxt, position, action, config)
        reward, wealth, peak = _recompensa_ajustada_risco(log_return, wealth, peak, config)
        utility += reward
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
    return float(utility)

def _executar_compra(cash: float, price: float, config: Any, fee_calculator: Callable, slippage: Callable) -> tuple[float, float, dict[str, float]]:
    execution_price = float(slippage(price, 'BUY', config))
    quantity = cash / execution_price
    for _ in range(25):
        fees = fee_calculator('BUY', quantity, execution_price, config)
        next_quantity = max(0.0, (cash - float(fees['total_fee'])) / execution_price)
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity
    fees = fee_calculator('BUY', quantity, execution_price, config)
    return (float(quantity), execution_price, fees)

def _benchmark_pesos_iguais(frames: dict[str, pd.DataFrame], symbols: list[str], execution_dates: pd.DatetimeIndex, initial_capital: float, config: Any, fee_calculator: Callable, slippage: Callable) -> pd.Series:
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

def _precalcular_diagnosticos_regime_mercado(
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

def _simular_exato(backend: str, policy: Callable[[pd.Timestamp, int, int], tuple[int, float]], frames: dict[str, pd.DataFrame], symbols: list[str], decision_dates: pd.DatetimeIndex, config: Any, fee_calculator: Callable, slippage: Callable, decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None=None, policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None=None, trade_callback: Callable[[dict[str, Any]], None] | None=None, *, model_label: str='LightGBM Utility', method_line: str | None=None, simulation_progress_callback: Callable[[float, str], None] | None = None, benchmark_override: pd.Series | None = None, benchmark_override_name: str | None = None) -> RotationRunResult:
    if len(decision_dates) < 2:
        raise ValueError('The final-test interval is too short.')

    simulation_started = time.perf_counter()
    simulation_timing: dict[str, float] = {}
    total_sessions = max(1, len(decision_dates) - 1)

    def simulation_progress(fraction: float, stage: str) -> None:
        if simulation_progress_callback is not None:
            simulation_progress_callback(
                max(0.0, min(1.0, float(fraction))),
                str(stage),
            )

    execution_dates = decision_dates[1:]
    simulation_progress(0.01, "OOS benchmark")
    phase_started = time.perf_counter()
    if benchmark_override is None:
        benchmark = _benchmark_pesos_iguais(
            frames,
            symbols,
            execution_dates,
            float(config.initial_capital),
            config,
            fee_calculator,
            slippage,
        )
        benchmark_name = (
            "Equal-weight buy-and-hold across continuously available assets"
        )
    else:
        benchmark = pd.Series(
            benchmark_override,
            dtype=float,
        ).reindex(execution_dates)
        if (
            benchmark.isna().any()
            or not np.isfinite(benchmark.to_numpy(dtype=float)).all()
            or (benchmark <= 0.0).any()
        ):
            raise ValueError(
                "Fixed benchmark override does not cover the complete "
                "execution window with positive finite values."
            )
        benchmark_name = str(
            benchmark_override_name
            or "Fixed external buy-and-hold benchmark"
        )
    simulation_timing["benchmark_seconds"] = time.perf_counter() - phase_started

    simulation_progress(0.08, "OOS market-regime diagnostics")
    phase_started = time.perf_counter()
    market_regime_by_date = _precalcular_diagnosticos_regime_mercado(
        frames, symbols, decision_dates
    )
    simulation_timing["market_regime_seconds"] = time.perf_counter() - phase_started
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
    replay_started = time.perf_counter()
    policy_seconds = 0.0
    progress_stride = max(1, total_sessions // 20)
    simulation_progress(0.12, f"OOS portfolio replay 0/{total_sessions}")
    for idx in range(len(decision_dates) - 1):
        decision_date = decision_dates[idx]
        execution_date = decision_dates[idx + 1]
        previous_position = position
        metadata = (decision_metadata or {}).get(pd.Timestamp(decision_date), {})
        fold_id = metadata.get('fold_id')
        policy_started = time.perf_counter()
        target_position, score = policy(decision_date, position, holding_days)
        policy_seconds += time.perf_counter() - policy_started
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
            from_asset = old_symbol or 'CASH'
            to_asset = new_symbol or 'CASH'
            rotation_id = f'{pd.Timestamp(execution_date).isoformat()}::{from_asset}->{to_asset}'
            decision_trade_fields = {
                'decision_timestamp': pd.Timestamp(decision_date),
                'rotation_id': rotation_id,
                'rotation_from_asset': from_asset,
                'rotation_to_asset': to_asset,
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
                quantity, price, fees = _executar_compra(cash, raw_price, config, fee_calculator, slippage)
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
        completed_sessions = idx + 1
        if (
            completed_sessions % progress_stride == 0
            or completed_sessions == total_sessions
        ):
            replay_fraction = completed_sessions / total_sessions
            simulation_progress(
                0.12 + 0.86 * replay_fraction,
                f"OOS portfolio replay {completed_sessions}/{total_sessions}",
            )
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
    records = enriquecer_diagnosticos_operacoes(records, frames, symbols)
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
    metrics = {
        "portfolio_rotation": True,
        "strategy_mode": config.strategy_mode,
        "strategy_label": model_label,
        "symbol": "PORTFOLIO",
        "backend": backend,
        "assets": symbols,
        "timeframe": "1Day",
        "decision_horizons": list(config.rotation_target_horizons),
        "overnight_positions_allowed": True,
        "benchmark_name": benchmark_name,
        "walk_forward_enabled": True,
        "walk_forward_purge_days": int(config.rotation_purge_days),
        "walk_forward_calibration_days": int(
            config.rotation_walk_forward_calibration_days
        ),
        "walk_forward_test_days": int(config.rotation_walk_forward_test_days),
        "downside_penalty": float(config.rotation_downside_penalty),
        "drawdown_penalty": float(config.rotation_drawdown_penalty),
        "initial_capital": initial,
        "strategy_ending_capital": ending,
        "strategy_return": ending / initial - 1,
        "buy_hold_ending_capital": benchmark_ending,
        "buy_hold_return": benchmark_ending / initial - 1,
        "excess_return": ending / initial - benchmark_ending / initial,
        "strategy_maximum_drawdown": _drawdown_maximo(strategy_curve),
        "buy_hold_maximum_drawdown": _drawdown_maximo(benchmark_curve),
        "strategy_sharpe": _sharpe_anualizado(
            strategy_curve,
            periods_per_year,
        ),
        "buy_hold_sharpe": _sharpe_anualizado(
            benchmark_curve,
            periods_per_year,
        ),
        "strategy_cagr": _cagr(strategy_curve, initial),
        "buy_hold_cagr": _cagr(benchmark_curve, initial),
        "compound_log_growth": float(
            math.log(max(ending / initial, 1e-12))
        ),
        "risk_adjusted_compound_score": _pontuacao_curva_ajustada_risco(
            strategy_curve,
            config,
        ),
        "market_exposure": float(exposure),
        "cash_days": cash_days,
        "simulated_buys": buys,
        "simulated_sells": sells,
        "capital_rotations": int(rotation_count),
        "cycles_per_year": float(buys / years),
        "average_holding_days": avg_holding,
        "average_holding_bars": avg_holding,
        "average_holding_minutes": None,
        "geometric_trade_return": _retorno_geometrico_operacoes(trades),
        "total_transaction_fees": float(total_fees),
        "turnover_ratio": float(turnover / max(initial, 1e-09)),
        "test_start": execution_dates[0],
        "test_end": execution_dates[-1],
        "test_calendar_years": years,
    }

    simulation_timing["portfolio_replay_seconds"] = time.perf_counter() - replay_started
    simulation_timing["policy_seconds"] = float(policy_seconds)
    simulation_timing["accounting_seconds"] = max(
        0.0,
        simulation_timing["portfolio_replay_seconds"] - float(policy_seconds),
    )
    simulation_timing["total_seconds"] = time.perf_counter() - simulation_started
    simulation_timing["session_count"] = int(total_sessions)
    metrics["simulation_profile"] = dict(simulation_timing)
    simulation_progress(1.0, f"OOS portfolio replay {total_sessions}/{total_sessions} completed")

    summary = '\n'.join(['COMPOUND CAPITAL ROTATION — SWING', '', f"Model: {metrics['strategy_label']}", f"Assets: {', '.join(symbols)}", 'Decision data: daily candles', f"Utility horizons: {', '.join(str(item) for item in config.rotation_target_horizons)} trading sessions", 'Capital pool: one shared account, reinvested after every exit/rotation', 'Decision objective: maximize smoother net compounded wealth, not predict exact tops.', f'Risk penalties: downside={config.rotation_downside_penalty:.3f}, drawdown={config.rotation_drawdown_penalty:.3f}', f'Validation: expanding walk-forward, purge={config.rotation_purge_days} sessions, fold test={config.rotation_walk_forward_test_days} sessions', '', 'OUT-OF-SAMPLE WALK-FORWARD', f'Initial capital: ${initial:,.2f}', f'Ending capital: ${ending:,.2f}', f"Total return: {metrics['strategy_return']:.2%}", f"CAGR: {metrics['strategy_cagr']:.2%}", f"Compound log growth: {metrics['compound_log_growth']:.6f}", f"Maximum drawdown: {metrics['strategy_maximum_drawdown']:.2%}", f"Sharpe estimate: {metrics['strategy_sharpe']:.3f}", f'Capital rotations: {rotation_count}', f'Buys: {buys}', f'Sells including final liquidation: {sells}', f"Cycles/year: {metrics['cycles_per_year']:.2f}", f'Average holding days: {avg_holding:.2f}', f'Time in market: {exposure:.2%}', f'Transaction fees: ${total_fees:,.2f}', '', 'BENCHMARK', benchmark_name + '.', f'Benchmark ending capital: ${benchmark_ending:,.2f}', f"Benchmark return: {metrics['buy_hold_return']:.2%}", f"Benchmark CAGR: {metrics['buy_hold_cagr']:.2%}", '', 'METHOD', '- Signals use information available at the current daily close.', '- Position changes execute at the next daily open.', (method_line or f"- LightGBM Utility predicts a weighted multi-horizon risk-adjusted utility across {config.rotation_target_horizons}."), '- Every fold is trained only on information available before that fold.', f'- A {config.rotation_purge_days}-session purge prevents forward labels from touching the next validation/test segment.', '- FINAL_LIQUIDATION is bookkeeping only and is not a model decision.'])
    return RotationRunResult(backend=backend, predictions=predictions, trades=trades, summary=summary, metrics=metrics)

def _construir_folds_walk_forward(common_dates: pd.DatetimeIndex, config: Any) -> list[dict[str, Any]]:
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

    requested_fold_count = getattr(config, 'walk_forward_fold_count_override', None)
    explicit_test_ranges: list[tuple[int, int]] | None = None
    if requested_fold_count is not None:
        fold_count = int(requested_fold_count)
        available_test_rows = int(len(common_dates) - first_test_start)
        if available_test_rows < fold_count * min_test_days:
            raise ValueError(
                'Not enough out-of-sample history for the requested walk-forward fold count: '
                f'folds={fold_count}, available_test_rows={available_test_rows}, '
                f'minimum_test_rows_per_fold={min_test_days}.'
            )
        base_size, remainder = divmod(available_test_rows, fold_count)
        explicit_test_ranges = []
        cursor = int(first_test_start)
        for index in range(fold_count):
            size = base_size + (1 if index < remainder else 0)
            test_end = cursor + size
            explicit_test_ranges.append((cursor, test_end))
            cursor = test_end

    folds: list[dict[str, Any]] = []
    ranges = explicit_test_ranges
    if ranges is None:
        ranges = []
        test_start = first_test_start
        while test_start < len(common_dates):
            test_end = min(len(common_dates), test_start + test_days)
            if test_end - test_start < min_test_days:
                if ranges:
                    previous_start, _ = ranges[-1]
                    ranges[-1] = (previous_start, len(common_dates))
                break
            ranges.append((test_start, test_end))
            test_start = test_end

    for fold_id, (test_start, test_end) in enumerate(ranges, start=1):
        if test_end - test_start < min_test_days:
            raise ValueError(
                f'Fold {fold_id}: test rows {test_end - test_start} < {min_test_days}.'
            )
        calibration_end = test_start - purge
        calibration_start = calibration_end - calibration_days
        train_end = calibration_start - purge
        final_fit_end = test_start - purge
        if train_end < min_train:
            raise ValueError(f'Fold {fold_id}: training rows {train_end} < {min_train}.')
        folds.append({
            'fold_id': fold_id,
            'train_end_index': train_end,
            'calibration_start_index': calibration_start,
            'calibration_end_index': calibration_end,
            'final_fit_end_index': final_fit_end,
            'test_start_index': test_start,
            'test_end_index': test_end,
            'train_start': common_dates[0],
            'train_end': common_dates[train_end - 1],
            'calibration_start': common_dates[calibration_start],
            'calibration_end': common_dates[calibration_end - 1],
            'purge_start': common_dates[calibration_end],
            'purge_end': common_dates[test_start - 1],
            'test_start': common_dates[test_start],
            'test_end': common_dates[test_end - 1],
            'decision_dates': common_dates[test_start - 1:test_end],
        })
    if not folds:
        raise ValueError('No valid expanding walk-forward fold was created.')
    return folds

def _datas_decisao_analise(
    common_dates: pd.DatetimeIndex,
    folds: list[dict[str, Any]],
    config: Any,
) -> pd.DatetimeIndex:
    






    if not folds:
        raise ValueError('No walk-forward fold is available for the analysis window.')

    champion_oos_start = int(folds[0]['test_start_index'])
    champion_oos_end = int(folds[-1]['test_end_index'])

    requested_start = pd.Timestamp(config.analysis_start_date)
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

    requested_end_value = getattr(config, 'analysis_end_date', None)
    execution_end = champion_oos_end
    if requested_end_value:
        requested_end = pd.Timestamp(requested_end_value)
        requested_end = (
            requested_end.tz_localize('UTC')
            if requested_end.tzinfo is None
            else requested_end.tz_convert('UTC')
        )
        # analysis_end_date representa uma data de mercado inclusiva, nao um
        # instante UTC. Barras diarias da NYSE podem aparecer como 04:00/05:00
        # UTC; comparar com 00:00 UTC excluiria a propria sessao final.
        requested_end_exclusive = requested_end.normalize() + pd.Timedelta(days=1)
        requested_execution_end = int(
            common_dates.searchsorted(requested_end_exclusive, side='left')
        )
        execution_end = min(champion_oos_end, requested_execution_end)

    if execution_start >= execution_end:
        raise ValueError('The requested analysis interval contains no executable session.')

    return common_dates[execution_start - 1:execution_end]

def _politica_agendada(policies: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]], decision_to_fold: dict[pd.Timestamp, int]) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        fold_id = decision_to_fold.get(key)
        if fold_id is None:
            raise KeyError(f'No walk-forward policy is assigned to {key}.')
        return policies[int(fold_id)](timestamp, current_position, holding_days)
    return policy

def _desempenho_folds(predictions: pd.DataFrame, folds: list[dict[str, Any]], initial_capital: float) -> list[dict[str, Any]]:
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
            'maximum_drawdown': _drawdown_maximo(curve),
            'sessions': int(len(subset)),
        })
    return output

