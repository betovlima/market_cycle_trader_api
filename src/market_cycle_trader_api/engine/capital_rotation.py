from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from copy import deepcopy
import math
import multiprocessing
import subprocess
import threading
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd


ROTATION_FEATURES = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "vol_5",
    "vol_20",
    "ema_distance_5",
    "ema_distance_10",
    "ema_distance_20",
    "ema_distance_50",
    "ema_5_vs_20",
    "ema_20_vs_50",
    "rsi_14",
    "atr_pct_14",
    "distance_from_high_20",
    "distance_from_low_20",
    "volume_zscore_20",
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
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _nvidia_gpu_name() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0:
            names = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
            if names:
                return names[0]
    except Exception:
        pass
    return None


def resolve_qrdqn_compute_plan(config: Any) -> RotationComputePlan:
    requested = str(
        config.rotation_accelerator
    ).strip().lower()
    allow_fallback = bool(
        config.rotation_allow_cpu_fallback
    )
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "ROTATION_ACCELERATOR must be auto, cpu or cuda."
        )

    version = None
    cuda_runtime = None
    cuda_available = False
    gpu_name = None
    error = None
    try:
        import torch

        version = str(torch.__version__)
        cuda_runtime = (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else None
        )
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        error = str(exc)

    if requested == "cpu":
        selected = "cpu"
    elif cuda_available:
        selected = "cuda"
    else:
        selected = "cpu"

    reason = None
    fallback_used = False
    if requested == "cuda" and not cuda_available:
        reason = (
            "PyTorch CUDA is not available"
            if error is None
            else f"PyTorch CUDA detection failed: {error}"
        )
        if not allow_fallback:
            raise RuntimeError(
                f"{reason}. Enable CPU fallback or use accelerator=auto."
            )
        fallback_used = True

    return RotationComputePlan(
        framework="pytorch",
        requested=requested,
        selected=selected,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        framework_version=version,
        cuda_runtime_version=cuda_runtime,
        cuda_build=(cuda_runtime is not None),
        fallback_used=fallback_used,
        fallback_reason=reason,
    )


def resolve_xgboost_compute_plan(config: Any) -> RotationComputePlan:
    requested = str(
        config.rotation_accelerator
    ).strip().lower()
    allow_fallback = bool(
        config.rotation_allow_cpu_fallback
    )
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "ROTATION_ACCELERATOR must be auto, cpu or cuda."
        )

    version = None
    cuda_build = None
    error = None
    try:
        import xgboost as xgb

        version = str(xgb.__version__)
        build = xgb.build_info()
        if isinstance(build, dict):
            cuda_build = _truthy_build_flag(build.get("USE_CUDA"))
    except Exception as exc:
        error = str(exc)

    gpu_name = _nvidia_gpu_name()
    cuda_available = bool(cuda_build and gpu_name)
    if requested == "cpu":
        selected = "cpu"
    elif cuda_available:
        selected = "cuda"
    else:
        selected = "cpu"

    reason = None
    fallback_used = False
    if requested == "cuda" and not cuda_available:
        if error is not None:
            reason = f"XGBoost CUDA detection failed: {error}"
        elif not cuda_build:
            reason = "The installed XGBoost build does not expose CUDA support"
        else:
            reason = "No NVIDIA GPU/driver is visible to XGBoost"
        if not allow_fallback:
            raise RuntimeError(
                f"{reason}. Enable CPU fallback or use accelerator=auto."
            )
        fallback_used = True

    return RotationComputePlan(
        framework="xgboost",
        requested=requested,
        selected=selected,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        framework_version=version,
        cuda_runtime_version=None,
        cuda_build=cuda_build,
        fallback_used=fallback_used,
        fallback_reason=reason,
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean = denominator.replace(0, np.nan)
    return numerator / clean


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_rotation_frame(
    bars: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    horizon_days = int(config.rotation_horizon_days)
    data = bars.copy().sort_index()
    data.index = pd.to_datetime(data.index, utc=True)


    close = data["close"].astype(float)
    open_price = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)

    daily_return = close.pct_change()
    for period in [1, 3, 5, 10, 20]:
        data[f"return_{period}"] = close.pct_change(period)

    for period in [5, 20]:
        data[f"vol_{period}"] = daily_return.rolling(period).std()

    ema = {}
    for period in [5, 10, 20, 50]:
        ema[period] = close.ewm(span=period, adjust=False).mean()
        data[f"ema_distance_{period}"] = _safe_divide(close, ema[period]) - 1

    data["ema_5_vs_20"] = _safe_divide(ema[5], ema[20]) - 1
    data["ema_20_vs_50"] = _safe_divide(ema[20], ema[50]) - 1
    data["rsi_14"] = _rsi(close) / 100.0

    atr = _true_range(data).ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    data["atr_pct_14"] = _safe_divide(atr, close)

    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    data["distance_from_high_20"] = _safe_divide(close, rolling_high) - 1
    data["distance_from_low_20"] = _safe_divide(close, rolling_low) - 1

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    data["volume_zscore_20"] = _safe_divide(
        volume - volume_mean,
        volume_std,
    )






    entry_open = open_price.shift(-1)
    exit_close = close.shift(-horizon_days)
    gross_log_return = np.log(_safe_divide(exit_close, entry_open))

    round_trip_cost = min(
        0.25,
        2.0 * (
            max(0.0, float(config.slippage_bps)) / 10_000.0
            + max(0.0, float(config.commission_rate))
        ),
    )
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
        future_lows = lows[idx + 1 : idx + horizon_days + 1]
        future_closes = closes[idx + 1 : idx + horizon_days + 1]
        if len(future_closes) != horizon_days:
            continue

        minimum_low = float(np.nanmin(future_lows))
        downside[idx] = max(0.0, 1.0 - minimum_low / entry)

        path = np.concatenate(([entry], future_closes))
        running_peak = np.maximum.accumulate(path)
        drawdowns = 1.0 - np.divide(
            path,
            running_peak,
            out=np.ones_like(path),
            where=running_peak > 0,
        )
        path_drawdown[idx] = max(0.0, float(np.nanmax(drawdowns)))

    data["forward_net_log_return"] = gross_log_return + net_cost_log
    data["forward_downside"] = downside
    data["forward_max_drawdown"] = path_drawdown
    data["forward_risk_adjusted_utility"] = (
        data["forward_net_log_return"]
        - float(config.rotation_downside_penalty) * data["forward_downside"]
        - float(config.rotation_drawdown_penalty) * data["forward_max_drawdown"]
    )


    required = ROTATION_FEATURES + ["open", "high", "low", "close", "volume"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    return data


def prepare_rotation_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames = {
        symbol: build_rotation_frame(frame, config)
        for symbol, frame in bars_by_symbol.items()
        if frame is not None and not frame.empty
    }
    if len(frames) < 2:
        raise ValueError(
            "Compound rotation needs at least two assets with valid aligned data."
        )

    common: pd.DatetimeIndex | None = None
    for frame in frames.values():
        index = pd.DatetimeIndex(frame.index)
        common = index if common is None else common.intersection(index)

    if common is None or len(common) < 700:
        raise ValueError(
            "The common aligned history is too short for train/calibration/test."
        )

    common = common.sort_values()
    frames = {symbol: frame.loc[common].copy() for symbol, frame in frames.items()}
    return frames, common


def _normalization(
    frames: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    config: Any,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    result: dict[str, tuple[pd.Series, pd.Series]] = {}
    features = ROTATION_FEATURES
    for symbol, frame in frames.items():
        sample = frame.loc[train_dates, features]
        mean = sample.mean()
        std = sample.std().replace(0, 1.0).fillna(1.0)
        result[symbol] = (mean, std)
    return result


def _state_vector(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    current_position: int,
    holding_days: int,
    config: Any,
) -> np.ndarray:
    chunks = []
    features = ROTATION_FEATURES
    for symbol in symbols:
        mean, std = normalization[symbol]
        values = (
            (frames[symbol].loc[timestamp, features] - mean) / std
        ).clip(-8, 8)
        chunks.append(values.to_numpy(dtype=np.float32))

    position = np.zeros(len(symbols) + 1, dtype=np.float32)
    position[int(current_position)] = 1.0
    chunks.append(position)
    chunks.append(
        np.asarray(
            [min(float(holding_days), 60.0) / 60.0],
            dtype=np.float32,
        )
    )
    return np.concatenate(chunks).astype(np.float32)


def _build_qrdqn_feature_cache(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
) -> dict[pd.Timestamp, np.ndarray]:
    matrices: list[np.ndarray] = []
    features = ROTATION_FEATURES
    for symbol in symbols:
        mean, std = normalization[symbol]
        values = (
            (frames[symbol].loc[dates, features] - mean) / std
        ).clip(-8, 8)
        matrices.append(values.to_numpy(dtype=np.float32))
    combined = np.concatenate(matrices, axis=1)
    return {
        pd.Timestamp(timestamp): combined[index]
        for index, timestamp in enumerate(dates)
    }


def _state_vector_from_base(
    base_features: np.ndarray,
    asset_count: int,
    current_position: int,
    holding_days: int,
) -> np.ndarray:
    position = np.zeros(asset_count + 1, dtype=np.float32)
    position[int(current_position)] = 1.0
    return np.concatenate(
        [
            base_features,
            position,
            np.asarray(
                [min(float(holding_days), 60.0) / 60.0],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)


def _build_qrdqn_price_cache(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    opens = np.column_stack(
        [
            frames[symbol].loc[dates, "open"].to_numpy(dtype=np.float64)
            for symbol in symbols
        ]
    )
    closes = np.column_stack(
        [
            frames[symbol].loc[dates, "close"].to_numpy(dtype=np.float64)
            for symbol in symbols
        ]
    )
    return {
        pd.Timestamp(timestamp): (opens[index], closes[index])
        for index, timestamp in enumerate(dates)
    }


def _training_transition_log_return_cached(
    price_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
    date_now: pd.Timestamp,
    date_next: pd.Timestamp,
    from_position: int,
    to_position: int,
    config: Any,
) -> float:
    _, close_now = price_cache[pd.Timestamp(date_now)]
    open_next, close_next = price_cache[pd.Timestamp(date_next)]
    gross = 1.0
    if from_position > 0:
        old_close = float(close_now[from_position - 1])
        next_open = float(open_next[from_position - 1])
        if old_close > 0:
            gross *= next_open / old_close

    gross *= max(
        1e-8,
        1.0 - _proportional_switch_cost(
            config,
            from_position,
            to_position,
        ),
    )

    if to_position > 0:
        next_open = float(open_next[to_position - 1])
        next_close = float(close_next[to_position - 1])
        if next_open > 0:
            gross *= next_close / next_open
    return float(np.log(max(gross, 1e-12)))


def _annualized_sharpe(
    curve: pd.Series,
    periods_per_year: float = 252.0,
) -> float:
    returns = curve.pct_change().dropna()
    if returns.empty or float(returns.std()) <= 0:
        return float("nan")
    return float(
        np.sqrt(float(periods_per_year))
        * returns.mean()
        / returns.std()
    )


def _maximum_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())


def _cagr(curve: pd.Series) -> float:
    if len(curve) < 2:
        return float("nan")
    start = pd.Timestamp(curve.index[0])
    end = pd.Timestamp(curve.index[-1])
    years = max((end - start).days / 365.25, 1 / 365.25)
    if float(curve.iloc[0]) <= 0 or float(curve.iloc[-1]) <= 0:
        return float("nan")
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)


def _geometric_trade_return(trades: pd.DataFrame) -> float:
    if trades.empty or "position_return" not in trades:
        return float("nan")
    returns = pd.to_numeric(
        trades.loc[
            trades["action"].isin(["SELL", "FINAL_SELL"]),
            "position_return",
        ],
        errors="coerce",
    ).dropna()
    if returns.empty:
        return float("nan")
    gross = np.prod(1.0 + returns.clip(lower=-0.999999))
    return float(gross ** (1 / len(returns)) - 1)


def _proportional_switch_cost(config: Any, from_position: int, to_position: int) -> float:
    if from_position == to_position:
        return 0.0
    one_side = max(0.0, float(config.slippage_bps)) / 10_000.0
    one_side += max(0.0, float(config.commission_rate))
    sides = int(from_position != 0) + int(to_position != 0)
    return min(0.25, one_side * sides)


def _training_transition_log_return(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    date_now: pd.Timestamp,
    date_next: pd.Timestamp,
    from_position: int,
    to_position: int,
    config: Any,
) -> float:
    gross = 1.0
    if from_position > 0:
        symbol = symbols[from_position - 1]
        close_now = float(frames[symbol].loc[date_now, "close"])
        open_next = float(frames[symbol].loc[date_next, "open"])
        if close_now > 0:
            gross *= open_next / close_now

    cost = _proportional_switch_cost(config, from_position, to_position)
    gross *= max(1e-8, 1.0 - cost)

    if to_position > 0:
        symbol = symbols[to_position - 1]
        open_next = float(frames[symbol].loc[date_next, "open"])
        close_next = float(frames[symbol].loc[date_next, "close"])
        if open_next > 0:
            gross *= close_next / open_next

    return float(np.log(max(gross, 1e-12)))


def _risk_adjusted_reward(
    log_return: float,
    wealth_before: float,
    peak_before: float,
    config: Any,
) -> tuple[float, float, float]:
    wealth_after = wealth_before * math.exp(log_return)
    peak_after = max(peak_before, wealth_after)
    previous_drawdown = max(0.0, 1.0 - wealth_before / max(peak_before, 1e-12))
    current_drawdown = max(0.0, 1.0 - wealth_after / max(peak_after, 1e-12))
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown)
    downside = max(0.0, -log_return)
    reward = (
        log_return
        - float(config.rotation_downside_penalty) * downside
        - float(config.rotation_drawdown_penalty) * drawdown_increase
    )
    return float(reward), float(wealth_after), float(peak_after)


def _curve_risk_adjusted_score(curve: pd.Series, config: Any) -> float:
    if curve.empty or len(curve) < 2:
        return float("nan")
    values = curve.astype(float)
    logs = np.log(values / values.shift(1)).dropna()
    peak = values.cummax()
    drawdown = 1.0 - values / peak
    drawdown_increase = drawdown.diff().clip(lower=0).fillna(0.0)
    aligned = drawdown_increase.reindex(logs.index).fillna(0.0)
    downside = (-logs).clip(lower=0)
    score = (
        logs
        - float(config.rotation_downside_penalty) * downside
        - float(config.rotation_drawdown_penalty) * aligned
    ).sum()
    return float(score)

def _fit_xgb_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    device_name: str,
) -> tuple[dict[str, Any], str, str | None]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost Utility requires xgboost. Install requirements.txt."
        ) from exc

    allow_fallback = bool(
        config.rotation_allow_cpu_fallback
    )

    def fit_on_device(effective_device: str) -> dict[str, Any]:
        fitted: dict[str, Any] = {}
        for symbol in symbols:
            frame = frames[symbol].loc[train_dates].dropna(
                subset=["forward_risk_adjusted_utility"]
            )
            if len(frame) < int(config.rotation_minimum_training_rows):
                raise ValueError(
                    f"{symbol}: only {len(frame)} utility rows are available; "
                    f"{config.rotation_minimum_training_rows} are required."
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
                objective="reg:squarederror",
                tree_method="hist",
                random_state=int(config.random_state),
                n_jobs=int(config.xgb_n_jobs),
                device=effective_device,
            )
            model.fit(
                frame[ROTATION_FEATURES],
                frame["forward_risk_adjusted_utility"],
            )
            fitted[symbol] = model
        return fitted

    try:
        return fit_on_device(device_name), device_name, None
    except Exception as exc:
        if device_name != "cuda" or not allow_fallback:
            raise
        fallback_reason = (
            "XGBoost CUDA initialization/training failed; "
            f"using CPU instead: {exc}"
        )
        return fit_on_device("cpu"), "cpu", fallback_reason


def _xgb_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
) -> np.ndarray:
    values = [0.0]
    for symbol in symbols:
        row = frames[symbol].loc[[timestamp], ROTATION_FEATURES]
        prediction = float(models[symbol].predict(row)[0])
        values.append(prediction)
    return np.asarray(values, dtype=np.float64)


def _xgb_policy(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    switch_margin: float,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        utilities = _xgb_utilities(
            models,
            frames,
            symbols,
            timestamp,
            config,
        )
        best = int(np.nanargmax(utilities))
        best_value = float(utilities[best])
        current_value = float(utilities[current_position])

        if (
            current_position > 0
            and holding_days < int(config.rotation_min_holding_days)
        ):
            return current_position, current_value

        minimum = float(config.rotation_cash_threshold)
        if best == 0 or best_value <= minimum:
            return 0, 0.0

        if current_position == 0:
            if best_value >= minimum + float(config.rotation_min_expected_edge):
                return best, best_value
            return 0, 0.0

        if best == current_position:
            return current_position, current_value

        required = max(
            float(config.rotation_switch_margin),
            float(switch_margin),
        )
        if best_value >= current_value + required:
            return best, best_value
        return current_position, current_value

    return policy


def _simple_policy_growth(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
) -> float:
    if len(decision_dates) < 2:
        return float("-inf")
    wealth = 1.0
    peak = 1.0
    position = 0
    holding = 0
    utility = 0.0
    for idx in range(len(decision_dates) - 1):
        now = decision_dates[idx]
        nxt = decision_dates[idx + 1]
        action, _ = policy(now, position, holding)
        log_return = _training_transition_log_return(
            frames,
            symbols,
            now,
            nxt,
            position,
            action,
            config,
        )
        reward, wealth, peak = _risk_adjusted_reward(
            log_return,
            wealth,
            peak,
            config,
        )
        utility += reward
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
    return float(utility)


class _QRNetwork:
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        quantile_count: int,
        hidden_dim: int,
        device: str,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "QR-DQN requires PyTorch. Install requirements.txt."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.action_count = action_count
        self.quantile_count = quantile_count
        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count * quantile_count),
        ).to(self.device)

    def quantiles(self, tensor):
        output = self.model(tensor)
        return output.reshape(
            -1,
            self.action_count,
            self.quantile_count,
        )


class _ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.bootstrap_discounts = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, bootstrap_discount) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.bootstrap_discounts[i] = float(bootstrap_discount)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=int(batch_size))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.bootstrap_discounts[indices],
        )


class _NStepAccumulator:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.pending: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def _emit_one(self):
        count = min(self.n_step, len(self.pending))
        first_state, first_action, _, _, _ = self.pending[0]
        reward = 0.0
        final_next_state = self.pending[count - 1][3]
        final_done = self.pending[count - 1][4]
        for offset in range(count):
            reward += (self.gamma ** offset) * float(self.pending[offset][2])
            if self.pending[offset][4]:
                count = offset + 1
                final_next_state = self.pending[offset][3]
                final_done = True
                break
        self.pending.pop(0)
        return (
            first_state,
            first_action,
            float(reward),
            final_next_state,
            final_done,
            float(self.gamma ** count),
        )

    def append(self, state, action, reward, next_state, done):
        self.pending.append((state, int(action), float(reward), next_state, bool(done)))
        emitted = []
        if len(self.pending) >= self.n_step:
            emitted.append(self._emit_one())
        if done:
            while self.pending:
                emitted.append(self._emit_one())
        return emitted


def _qrdqn_action_snapshot(
    network: _QRNetwork,
    state: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    torch = network.torch
    with torch.no_grad():
        tensor = torch.as_tensor(
            state[None, :],
            dtype=torch.float32,
            device=network.device,
        )
        quantiles = network.quantiles(tensor)[0]
        means = quantiles.mean(dim=1)
        action = int(torch.argmax(means).item())
        score = float(means[action].item())
        values = means.detach().cpu().numpy().astype(float)
        return action, score, values


def _qrdqn_greedy_action(
    network: _QRNetwork,
    state: np.ndarray,
) -> tuple[int, float]:
    action, score, _ = _qrdqn_action_snapshot(network, state)
    return action, score


def _qrdqn_validation_growth(
    network: _QRNetwork,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    feature_cache: dict[pd.Timestamp, np.ndarray] | None = None,
    price_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    if len(dates) < 2:
        return float("-inf")
    position = 0
    holding = 0
    wealth = 1.0
    peak = 1.0
    utility = 0.0
    asset_count = len(symbols)
    for idx in range(len(dates) - 1):
        now = dates[idx]
        nxt = dates[idx + 1]
        if feature_cache is not None:
            state = _state_vector_from_base(
                feature_cache[pd.Timestamp(now)],
                asset_count,
                position,
                holding,
            )
        else:
            state = _state_vector(
                frames,
                symbols,
                now,
                normalization,
                position,
                holding,
                config,
            )
        action, _ = _qrdqn_greedy_action(network, state)
        if (
            position > 0
            and holding < int(config.rotation_min_holding_days)
            and action != position
        ):
            action = position
        action = int(action)
        if price_cache is not None:
            log_return = _training_transition_log_return_cached(
                price_cache,
                now,
                nxt,
                position,
                action,
                config,
            )
        else:
            log_return = _training_transition_log_return(
                frames,
                symbols,
                now,
                nxt,
                position,
                action,
                config,
            )
        reward, wealth, peak = _risk_adjusted_reward(
            log_return,
            wealth,
            peak,
            config,
        )
        utility += reward
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
    return float(utility)

def _train_qrdqn(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    device_name: str,
    seed: int,
    progress_callback: Callable[[float], None] | None = None,
    feature_cache: dict[pd.Timestamp, np.ndarray] | None = None,
    price_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[_QRNetwork, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "QR-DQN requires PyTorch >= 2.10. Install requirements.txt."
        ) from exc

    seed = int(seed)
    rng = np.random.default_rng(seed)

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QR-DQN was assigned CUDA, but PyTorch cannot access CUDA."
            )
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass




    cache_dates = train_dates.union(calibration_dates)
    if feature_cache is None:
        feature_cache = _build_qrdqn_feature_cache(
            frames,
            symbols,
            cache_dates,
            normalization,
            config,
        )
    if price_cache is None:
        price_cache = _build_qrdqn_price_cache(
            frames,
            symbols,
            cache_dates,
        )

    sample_state = _state_vector_from_base(
        feature_cache[pd.Timestamp(train_dates[0])],
        len(symbols),
        0,
        0,
    )
    online = _QRNetwork(
        len(sample_state),
        len(symbols) + 1,
        int(config.qrdqn_n_quantiles),
        int(config.qrdqn_hidden_dim),
        device_name,
    )
    target = _QRNetwork(
        len(sample_state),
        len(symbols) + 1,
        int(config.qrdqn_n_quantiles),
        int(config.qrdqn_hidden_dim),
        device_name,
    )



    generator = torch.Generator(device=online.device)
    generator.manual_seed(seed)
    for module in online.model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(
                module.weight,
                a=math.sqrt(5),
                generator=generator,
            )
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    module.weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(
                    module.bias,
                    -bound,
                    bound,
                    generator=generator,
                )
    target.model.load_state_dict(online.model.state_dict())

    optimizer = torch.optim.Adam(
        online.model.parameters(),
        lr=float(config.qrdqn_learning_rate),
    )
    buffer = _ReplayBuffer(
        int(config.qrdqn_replay_size),
        len(sample_state),
    )
    n_step = max(1, int(config.qrdqn_n_step))
    n_step_accumulator = _NStepAccumulator(
        n_step,
        float(config.qrdqn_gamma),
    )

    quantile_count = int(config.qrdqn_n_quantiles)
    tau = (
        (
            torch.arange(
                quantile_count,
                device=online.device,
                dtype=torch.float32,
            )
            + 0.5
        )
        / quantile_count
    ).view(1, quantile_count, 1)

    total_steps = int(config.qrdqn_training_steps)
    progress_interval = max(1, total_steps // 20)
    if progress_callback is not None:
        progress_callback(0.0)

    episode_length = min(
        int(config.qrdqn_episode_days),
        max(20, len(train_dates) - 2),
    )
    eval_every = max(250, int(config.qrdqn_eval_every_steps))
    early_stopping_enabled = bool(
        config.qrdqn_early_stopping_enabled
    )
    min_training_steps = int(
        config.qrdqn_min_training_steps
    )
    early_stopping_patience = int(
        config.qrdqn_early_stopping_patience
    )




    checkpoint_selection_start_step = (
        min_training_steps if early_stopping_enabled else 0
    )

    best_score = float("-inf")
    best_state = deepcopy(online.model.state_dict())
    best_step = 0
    no_improvement_evals = 0
    eligible_validation_evals = 0
    ignored_pre_min_validation_evals = 0
    stopped_early = False

    epsilon_start = float(config.qrdqn_epsilon_start)
    epsilon_end = float(config.qrdqn_epsilon_end)

    position = 0
    holding = 0
    episode_wealth = 1.0
    episode_peak = 1.0
    max_start = max(1, len(train_dates) - episode_length - 1)
    start_idx = int(rng.integers(0, max_start))
    date_idx = start_idx
    episode_end = min(
        len(train_dates) - 1,
        start_idx + episode_length,
    )
    asset_count = len(symbols)
    steps_used = 0

    for step in range(total_steps):
        steps_used = step + 1
        now = train_dates[date_idx]
        nxt = train_dates[date_idx + 1]
        state = _state_vector_from_base(
            feature_cache[pd.Timestamp(now)],
            asset_count,
            position,
            holding,
        )

        fraction = step / max(1, total_steps - 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        if rng.random() < epsilon:
            action = int(rng.integers(0, len(symbols) + 1))
        else:
            action, _ = _qrdqn_greedy_action(online, state)

        if (
            position > 0
            and holding < int(config.rotation_min_holding_days)
            and action != position
        ):
            action = position
        action = int(action)

        log_return = _training_transition_log_return_cached(
            price_cache,
            now,
            nxt,
            position,
            action,
            config,
        )
        reward, episode_wealth, episode_peak = _risk_adjusted_reward(
            log_return,
            episode_wealth,
            episode_peak,
            config,
        )

        next_holding = (
            holding + 1
            if action == position and action > 0
            else (1 if action > 0 else 0)
        )
        next_state = _state_vector_from_base(
            feature_cache[pd.Timestamp(nxt)],
            asset_count,
            action,
            next_holding,
        )
        done = date_idx + 1 >= episode_end

        for transition in n_step_accumulator.append(
            state,
            action,
            reward,
            next_state,
            done,
        ):
            buffer.add(*transition)
        position = action
        holding = next_holding

        if buffer.size >= int(config.qrdqn_learning_starts):
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                bootstrap_discounts,
            ) = buffer.sample(int(config.qrdqn_batch_size), rng)

            states_t = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=online.device,
            )
            actions_t = torch.as_tensor(
                actions,
                dtype=torch.long,
                device=online.device,
            )
            rewards_t = torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=online.device,
            )
            next_states_t = torch.as_tensor(
                next_states,
                dtype=torch.float32,
                device=online.device,
            )
            dones_t = torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=online.device,
            )
            bootstrap_discounts_t = torch.as_tensor(
                bootstrap_discounts,
                dtype=torch.float32,
                device=online.device,
            )

            current_all = online.quantiles(states_t)
            batch_index = torch.arange(
                current_all.shape[0],
                device=online.device,
            )
            current = current_all[batch_index, actions_t, :]

            with torch.no_grad():
                next_online = online.quantiles(next_states_t).mean(dim=2)
                next_actions = torch.argmax(next_online, dim=1)
                next_target_all = target.quantiles(next_states_t)
                next_target = next_target_all[
                    batch_index,
                    next_actions,
                    :,
                ]
                target_quantiles = rewards_t[:, None] + (
                    bootstrap_discounts_t[:, None]
                    * (1.0 - dones_t[:, None])
                    * next_target
                )

            td = target_quantiles[:, None, :] - current[:, :, None]
            abs_td = td.abs()
            huber = torch.where(
                abs_td <= 1.0,
                0.5 * td.pow(2),
                abs_td - 0.5,
            )
            quantile_weight = torch.abs(
                tau - (td.detach() < 0).float()
            )
            loss = (quantile_weight * huber).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                online.model.parameters(),
                max_norm=10.0,
            )
            optimizer.step()

        if (step + 1) % int(config.qrdqn_target_update_steps) == 0:
            target.model.load_state_dict(online.model.state_dict())

        if (step + 1) % eval_every == 0:
            score = _qrdqn_validation_growth(
                online,
                frames,
                symbols,
                calibration_dates,
                normalization,
                config,
                feature_cache=feature_cache,
                price_cache=price_cache,
            )
            checkpoint_eligible = (
                not early_stopping_enabled
                or step + 1 >= checkpoint_selection_start_step
            )

            if checkpoint_eligible:
                eligible_validation_evals += 1
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = deepcopy(online.model.state_dict())
                    best_step = step + 1
                    no_improvement_evals = 0
                else:
                    no_improvement_evals += 1

                if (
                    early_stopping_enabled
                    and no_improvement_evals >= early_stopping_patience
                ):
                    stopped_early = True
                    break
            else:



                ignored_pre_min_validation_evals += 1

        if progress_callback is not None and (
            (step + 1) % progress_interval == 0
            or step + 1 == total_steps
        ):
            progress_callback((step + 1) / max(1, total_steps))

        date_idx += 1
        if done:
            position = 0
            holding = 0
            episode_wealth = 1.0
            episode_peak = 1.0
            start_idx = int(rng.integers(0, max_start))
            date_idx = start_idx
            episode_end = min(
                len(train_dates) - 1,
                start_idx + episode_length,
            )

    if best_score == float("-inf"):
        best_state = deepcopy(online.model.state_dict())
        best_step = steps_used
    online.model.load_state_dict(best_state)
    online.model.eval()
    if progress_callback is not None:
        progress_callback(1.0)
    return online, {
        "seed": seed,
        "requested_steps": total_steps,
        "steps_used": steps_used,
        "best_step": best_step,
        "best_validation_score": float(best_score),
        "stopped_early": stopped_early,
        "early_stopping_enabled": early_stopping_enabled,
        "minimum_training_steps": min_training_steps,
        "checkpoint_selection_start_step": checkpoint_selection_start_step,
        "eligible_validation_evals": eligible_validation_evals,
        "ignored_pre_min_validation_evals": ignored_pre_min_validation_evals,
        "n_step": n_step,
    }


_QRDQN_REPETITION_PROCESS_CONTEXT: dict[str, Any] = {}


def _initialize_qrdqn_repetition_process(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    fold_contexts: dict[int, dict[str, Any]],
    config_values: dict[str, Any],
    device_name: str,
    torch_num_threads: int,
) -> None:
    """Initialize an isolated QR-DQN repetition worker.

    Repetition-level parallelism must use processes rather than Python threads.
    PyTorch CPU kernels and some backend state are process-global; running
    independent RL trainings in ThreadPoolExecutor workers changed numerical
    trajectories enough to select different checkpoints for the same seeds.
    A spawned process gives every repetition its own PyTorch runtime and RNG
    state while preserving the configured seed.
    """
    global _QRDQN_REPETITION_PROCESS_CONTEXT

    if int(torch_num_threads) > 0:
        try:
            import torch
            torch.set_num_threads(int(torch_num_threads))
        except Exception:
            pass

    _QRDQN_REPETITION_PROCESS_CONTEXT = {
        "frames": frames,
        "symbols": symbols,
        "fold_contexts": fold_contexts,
        "config_values": config_values,
        "device_name": device_name,
    }


def _qrdqn_repetition_process_task(
    repetition: int,
    seed: int,
) -> dict[str, Any]:
    """Train one repetition in a spawned process and return CPU model weights."""
    context = _QRDQN_REPETITION_PROCESS_CONTEXT
    if not context:
        raise RuntimeError("QR-DQN process worker was not initialized.")

    rep_values = dict(context["config_values"])
    rep_values["random_state"] = int(seed)
    rep_config = SimpleNamespace(**rep_values)

    trained_folds: dict[int, dict[str, Any]] = {}
    for fold_id in sorted(context["fold_contexts"]):
        fold_context = context["fold_contexts"][fold_id]
        network, diagnostics = _train_qrdqn(
            context["frames"],
            context["symbols"],
            fold_context["train_dates"],
            fold_context["calibration_dates"],
            fold_context["normalization"],
            rep_config,
            context["device_name"],
            int(seed),
            progress_callback=None,
            feature_cache=fold_context["feature_cache"],
            price_cache=fold_context["price_cache"],
        )
        trained_folds[int(fold_id)] = {
            "state_dict": {
                name: tensor.detach().cpu().numpy()
                for name, tensor in network.model.state_dict().items()
            },
            "diagnostics": diagnostics,
        }

    return {
        "repetition": int(repetition),
        "seed": int(seed),
        "trained_folds": trained_folds,
    }


def _restore_qrdqn_network_from_process(
    state_dict_values: dict[str, np.ndarray],
    symbols: list[str],
    fold_context: dict[str, Any],
    config: Any,
    device_name: str,
) -> _QRNetwork:
    """Rebuild a QR-DQN network in the API process from isolated worker weights."""
    import torch

    sample_state = _state_vector_from_base(
        fold_context["feature_cache"][pd.Timestamp(fold_context["train_dates"][0])],
        len(symbols),
        0,
        0,
    )
    network = _QRNetwork(
        len(sample_state),
        len(symbols) + 1,
        int(config.qrdqn_n_quantiles),
        int(config.qrdqn_hidden_dim),
        device_name,
    )
    restored = {
        name: torch.as_tensor(value, device=network.device)
        for name, value in state_dict_values.items()
    }
    network.model.load_state_dict(restored)
    network.model.eval()
    return network


def _qrdqn_policy(
    network: _QRNetwork,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        state = _state_vector(
            frames,
            symbols,
            timestamp,
            normalization,
            current_position,
            holding_days,
            config,
        )
        raw_action, score, q_values = _qrdqn_action_snapshot(network, state)
        proposed_action = raw_action
        min_hold_guard_applied = False
        if (
            current_position > 0
            and holding_days < int(config.rotation_min_holding_days)
            and proposed_action != current_position
        ):
            proposed_action = current_position
            min_hold_guard_applied = True
        action = int(proposed_action)

        if decision_diagnostics is not None:
            labels = ["CASH", *symbols]
            sorted_values = np.sort(q_values)
            second_best = (
                float(sorted_values[-2])
                if len(sorted_values) > 1
                else float(sorted_values[-1])
            )
            current_q = float(q_values[current_position])
            final_q = float(q_values[action])
            diagnostic = {
                "decision_timestamp": pd.Timestamp(timestamp),
                "holding_bars_at_decision": int(holding_days),
                "current_position_index": int(current_position),
                "current_asset": labels[current_position],
                "raw_action_index": int(raw_action),
                "raw_action_asset": labels[raw_action],
                "final_action_index": int(action),
                "final_action_asset": labels[action],
                "q_current_position": current_q,
                "q_raw_best": float(q_values[raw_action]),
                "q_final_action": final_q,
                "q_delta_final_vs_current": float(final_q - current_q),
                "q_gap_best_vs_second": float(q_values[raw_action] - second_best),
                "min_hold_guard_applied": bool(min_hold_guard_applied),
                "day_trade_constraint_applied": bool(action != proposed_action),
            }
            for index, label in enumerate(labels):
                safe_label = str(label).replace("-", "_").replace(".", "_")
                diagnostic[f"q_value_{safe_label}"] = float(q_values[index])
            decision_diagnostics[pd.Timestamp(timestamp)] = diagnostic

        return action, score

    return policy


def _execute_buy(
    cash: float,
    price: float,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
) -> tuple[float, float, dict[str, float]]:
    execution_price = float(slippage(price, "BUY", config))
    quantity = cash / execution_price
    for _ in range(25):
        fees = fee_calculator("BUY", quantity, execution_price, config)
        next_quantity = max(
            0.0,
            (cash - float(fees["total_fee"])) / execution_price,
        )
        if not bool(config.fractional_shares):
            next_quantity = float(math.floor(next_quantity))
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity
    fees = fee_calculator("BUY", quantity, execution_price, config)
    return float(quantity), execution_price, fees


def _equal_weight_benchmark(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    execution_dates: pd.DatetimeIndex,
    initial_capital: float,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
) -> pd.Series:
    if len(execution_dates) < 2:
        return pd.Series(dtype=float)

    first = execution_dates[0]
    capital_per_asset = float(initial_capital) / len(symbols)
    quantities: dict[str, float] = {}
    residual = 0.0
    for symbol in symbols:
        buy_price = float(
            slippage(float(frames[symbol].loc[first, "open"]), "BUY", config)
        )
        quantity = capital_per_asset / buy_price
        for _ in range(20):
            fees = fee_calculator("BUY", quantity, buy_price, config)
            next_quantity = max(
                0.0,
                (capital_per_asset - float(fees["total_fee"])) / buy_price,
            )
            if not bool(config.fractional_shares):
                next_quantity = float(math.floor(next_quantity))
            if abs(next_quantity - quantity) < 1e-10:
                quantity = next_quantity
                break
            quantity = next_quantity
        fees = fee_calculator("BUY", quantity, buy_price, config)
        quantities[symbol] = quantity
        residual += capital_per_asset - (
            quantity * buy_price + float(fees["total_fee"])
        )

    values = []
    for timestamp in execution_dates:
        equity = residual
        for symbol, quantity in quantities.items():
            equity += quantity * float(frames[symbol].loc[timestamp, "close"])
        values.append(equity)

    series = pd.Series(values, index=execution_dates, dtype=float)

    last = execution_dates[-1]
    final_cash = residual
    for symbol, quantity in quantities.items():
        sell_price = float(
            slippage(float(frames[symbol].loc[last, "close"]), "SELL", config)
        )
        fees = fee_calculator("SELL", quantity, sell_price, config)
        final_cash += quantity * sell_price - float(fees["total_fee"])
    series.iloc[-1] = final_cash
    return series


def _simulate_exact(
    backend: str,
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None = None,
    policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RotationRunResult:
    if len(decision_dates) < 2:
        raise ValueError("The final-test interval is too short.")

    execution_dates = decision_dates[1:]
    benchmark = _equal_weight_benchmark(
        frames,
        symbols,
        execution_dates,
        float(config.initial_capital),
        config,
        fee_calculator,
        slippage,
    )

    cash = float(config.initial_capital)
    position = 0
    quantity = 0.0
    entry_price = float("nan")
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
        metadata = (
            (decision_metadata or {}).get(pd.Timestamp(decision_date), {})
        )
        fold_id = metadata.get("fold_id")
        target_position, score = policy(
            decision_date,
            position,
            holding_days,
        )
        decision_diag = dict(
            (policy_decision_diagnostics or {}).get(
                pd.Timestamp(decision_date),
                {},
            )
        )

        day_trades: list[dict[str, Any]] = []
        if target_position != position:
            old_symbol = (
                symbols[position - 1]
                if position > 0
                else None
            )
            new_symbol = (
                symbols[target_position - 1]
                if target_position > 0
                else None
            )
            is_rotation = previous_position > 0 and target_position > 0
            rotation_id = (
                f"{pd.Timestamp(execution_date).isoformat()}::{old_symbol}->{new_symbol}"
                if is_rotation
                else None
            )
            decision_trade_fields = {
                "decision_timestamp": pd.Timestamp(decision_date),
                "rotation_id": rotation_id,
                "rotation_from_asset": old_symbol if is_rotation else None,
                "rotation_to_asset": new_symbol if is_rotation else None,
                "q_current_position": decision_diag.get("q_current_position"),
                "q_raw_best": decision_diag.get("q_raw_best"),
                "q_final_action": decision_diag.get("q_final_action"),
                "q_delta_final_vs_current": decision_diag.get("q_delta_final_vs_current"),
                "q_gap_best_vs_second": decision_diag.get("q_gap_best_vs_second"),
                "raw_action_asset": decision_diag.get("raw_action_asset"),
                "final_action_asset": decision_diag.get("final_action_asset"),
                "min_hold_guard_applied": decision_diag.get("min_hold_guard_applied"),
                "day_trade_constraint_applied": decision_diag.get("day_trade_constraint_applied"),
            }

            if position > 0:
                symbol = symbols[position - 1]
                price = float(
                    slippage(
                        float(frames[symbol].loc[execution_date, "open"]),
                        "SELL",
                        config,
                    )
                )
                fees = fee_calculator("SELL", quantity, price, config)
                gross = quantity * price
                realized = (
                    quantity * (price - entry_price)
                    - float(fees["total_fee"])
                )
                cash += gross - float(fees["total_fee"])
                total_fees += float(fees["total_fee"])
                turnover += gross
                position_return = (
                    price / entry_price - 1
                    if np.isfinite(entry_price) and entry_price > 0
                    else 0.0
                )
                day_trades.append(
                    {
                        "timestamp": execution_date,
                        "action": "SELL",
                        "asset": symbol,
                        "reason": (
                            f"ROTATE_TO_{new_symbol}"
                            if new_symbol
                            else "MOVE_TO_CASH"
                        ),
                        "execution_price": price,
                        "quantity": quantity,
                        "gross_trade_value": gross,
                        **fees,
                        "realized_pnl": realized,
                        "position_return": position_return,
                        "holding_bars": holding_days,
                        "entry_timestamp": entry_time,
                        "entry_price": (
                            entry_price if np.isfinite(entry_price) else None
                        ),
                        "cash_after_trade": cash,
                        "shares_after_trade": 0.0,
                        "walk_forward_fold": fold_id,
                        **decision_trade_fields,
                    }
                )
                quantity = 0.0
                entry_price = float("nan")
                entry_time = None
                holding_days = 0

            position = target_position

            if position > 0:
                symbol = symbols[position - 1]
                raw_price = float(frames[symbol].loc[execution_date, "open"])
                quantity, price, fees = _execute_buy(
                    cash,
                    raw_price,
                    config,
                    fee_calculator,
                    slippage,
                )
                gross = quantity * price
                cash -= gross + float(fees["total_fee"])
                total_fees += float(fees["total_fee"])
                turnover += gross
                entry_price = price
                entry_time = execution_date
                holding_days = 1
                day_trades.append(
                    {
                        "timestamp": execution_date,
                        "action": "BUY",
                        "asset": symbol,
                        "reason": (
                            f"ROTATE_FROM_{old_symbol}"
                            if old_symbol
                            else "BEST_CAPITAL_UTILITY"
                        ),
                        "execution_price": price,
                        "quantity": quantity,
                        "gross_trade_value": gross,
                        **fees,
                        "realized_pnl": 0.0,
                        "position_return": 0.0,
                        "holding_bars": 0,
                        "entry_timestamp": execution_date,
                        "entry_price": price,
                        "cash_after_trade": cash,
                        "shares_after_trade": quantity,
                        "walk_forward_fold": fold_id,
                        **decision_trade_fields,
                    }
                )

            if previous_position > 0 and target_position > 0:
                rotation_count += 1
        elif position > 0:
            holding_days += 1





        records.extend(day_trades)
        if trade_callback is not None:
            for trade in day_trades:
                trade_callback(
                    {
                        **trade,
                        "backend": backend,
                        "model": (
                            "XGBoost Utility"
                            if backend == "xgboost_utility"
                            else "QR-DQN"
                        ),
                    }
                )

        if position > 0:
            symbol = symbols[position - 1]
            close_price = float(frames[symbol].loc[execution_date, "close"])
            equity = cash + quantity * close_price
            selected_asset = symbol
        else:
            equity = cash
            selected_asset = "CASH"

        equity_values.append(equity)
        actions = [trade["action"] for trade in day_trades]
        trade_action = (
            "ROTATE"
            if "SELL" in actions and "BUY" in actions
            else (actions[-1] if actions else "")
        )
        prediction_rows.append(
            {
                "timestamp": execution_date,
                "close": float("nan"),
                "strategy_equity": equity,
                "buy_hold_equity": float(benchmark.loc[execution_date]),
                "trade_action": trade_action,
                "trade_reason": (
                    "COMPOUND_CAPITAL_ROTATION"
                    if trade_action
                    else ""
                ),
                "execution_price": (
                    float(day_trades[-1]["execution_price"])
                    if day_trades
                    else None
                ),
                "selected_asset": selected_asset,
                "previous_asset": (
                    symbols[previous_position - 1]
                    if previous_position > 0
                    else "CASH"
                ),
                "decision_score": float(score),
                "decision_date": decision_date,
                "walk_forward_fold": fold_id,
                "fold_test_start": metadata.get("test_start"),
                "fold_test_end": metadata.get("test_end"),
                **decision_diag,
            }
        )


    if position > 0 and prediction_rows:
        final_date = execution_dates[-1]
        symbol = symbols[position - 1]
        price = float(
            slippage(
                float(frames[symbol].loc[final_date, "close"]),
                "SELL",
                config,
            )
        )
        fees = fee_calculator("SELL", quantity, price, config)
        gross = quantity * price
        realized = (
            quantity * (price - entry_price)
            - float(fees["total_fee"])
        )
        cash += gross - float(fees["total_fee"])
        total_fees += float(fees["total_fee"])
        turnover += gross
        position_return = (
            price / entry_price - 1
            if np.isfinite(entry_price) and entry_price > 0
            else 0.0
        )
        final_trade = {
                "timestamp": final_date,
                "action": "FINAL_SELL",
                "asset": symbol,
                "reason": "FINAL_LIQUIDATION",
                "execution_price": price,
                "quantity": quantity,
                "gross_trade_value": gross,
                **fees,
                "realized_pnl": realized,
                "position_return": position_return,
                "holding_bars": holding_days,
                "entry_timestamp": entry_time,
                "entry_price": entry_price,
                "cash_after_trade": cash,
                "shares_after_trade": 0.0,
                "walk_forward_fold": prediction_rows[-1].get("walk_forward_fold"),
            }
        records.append(final_trade)
        if trade_callback is not None:
            trade_callback(
                {
                    **final_trade,
                    "backend": backend,
                    "model": (
                        "XGBoost Utility"
                        if backend == "xgboost_utility"
                        else "QR-DQN"
                    ),
                }
            )
        equity_values[-1] = cash
        prediction_rows[-1]["strategy_equity"] = cash
        prediction_rows[-1]["trade_action"] = (
            prediction_rows[-1]["trade_action"] or "FINAL_SELL"
        )
        prediction_rows[-1]["trade_reason"] = (
            prediction_rows[-1]["trade_reason"] or "FINAL_LIQUIDATION"
        )

    predictions = pd.DataFrame(prediction_rows).set_index("timestamp")
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    predictions.index.name = "timestamp"
    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        trades = trades.sort_values("timestamp").reset_index(drop=True)

    strategy_curve = pd.Series(
        [float(row["strategy_equity"]) for row in prediction_rows],
        index=execution_dates,
        dtype=float,
    )
    benchmark_curve = benchmark.reindex(execution_dates).astype(float)
    initial = float(config.initial_capital)
    ending = float(strategy_curve.iloc[-1])
    benchmark_ending = float(benchmark_curve.iloc[-1])

    buys = int((trades["action"] == "BUY").sum()) if not trades.empty else 0
    sells = int(
        trades["action"].isin(["SELL", "FINAL_SELL"]).sum()
    ) if not trades.empty else 0
    cash_days = int(
        sum(row["selected_asset"] == "CASH" for row in prediction_rows)
    )
    exposure = 1.0 - (cash_days / max(1, len(prediction_rows)))
    completed_sells = (
        trades.loc[trades["action"].isin(["SELL", "FINAL_SELL"])]
        if not trades.empty
        else pd.DataFrame()
    )
    avg_holding = (
        float(pd.to_numeric(completed_sells["holding_bars"]).mean())
        if not completed_sells.empty
        else float("nan")
    )

    days = max(
        1,
        (pd.Timestamp(execution_dates[-1]) - pd.Timestamp(execution_dates[0])).days,
    )
    years = max(days / 365.25, 1 / 365.25)

    periods_per_year = 252.0
    metrics = {
        "portfolio_rotation": True,
        "strategy_mode": config.strategy_mode,
        "strategy_label": (
            "XGBoost Utility"
            if backend == "xgboost_utility"
            else "QR-DQN"
        ),
        "symbol": "PORTFOLIO",
        "backend": backend,
        "assets": symbols,
        "timeframe": "1Day",
        "decision_horizon_days": int(config.rotation_horizon_days),
        "decision_horizon_bars": None,
        "decision_horizon_label": f"{int(config.rotation_horizon_days)} trading sessions",
        "overnight_positions_allowed": True,
        "benchmark_name": "Equal-weight buy-and-hold",
        "walk_forward_enabled": bool(config.rotation_walk_forward_enabled),
        "walk_forward_purge_days": int(config.rotation_purge_days),
        "walk_forward_calibration_days": int(config.rotation_walk_forward_calibration_days),
        "walk_forward_test_days": int(config.rotation_walk_forward_test_days),
        "downside_penalty": float(config.rotation_downside_penalty),
        "drawdown_penalty": float(config.rotation_drawdown_penalty),
        "initial_capital": initial,
        "strategy_ending_capital": ending,
        "strategy_return": ending / initial - 1,
        "buy_hold_ending_capital": benchmark_ending,
        "buy_hold_return": benchmark_ending / initial - 1,
        "excess_return": (
            ending / initial - benchmark_ending / initial
        ),
        "strategy_maximum_drawdown": _maximum_drawdown(strategy_curve),
        "buy_hold_maximum_drawdown": _maximum_drawdown(benchmark_curve),
        "strategy_sharpe": _annualized_sharpe(strategy_curve, periods_per_year),
        "buy_hold_sharpe": _annualized_sharpe(benchmark_curve, periods_per_year),
        "strategy_cagr": _cagr(strategy_curve),
        "buy_hold_cagr": _cagr(benchmark_curve),
        "compound_log_growth": float(math.log(max(ending / initial, 1e-12))),
        "risk_adjusted_compound_score": _curve_risk_adjusted_score(strategy_curve, config),
        "market_exposure": float(exposure),
        "cash_days": cash_days,
        "simulated_buys": buys,
        "simulated_sells": sells,
        "capital_rotations": int(rotation_count),
        "cycles_per_year": float(buys / years),
        "average_holding_days": avg_holding,
        "average_holding_bars": avg_holding,
        "average_holding_minutes": None,
        "geometric_trade_return": _geometric_trade_return(trades),
        "total_transaction_fees": float(total_fees),
        "turnover_ratio": float(turnover / max(initial, 1e-9)),
        "test_start": execution_dates[0],
        "test_end": execution_dates[-1],
        "test_calendar_years": years,
    }

    summary = "\n".join(
        [
            "COMPOUND CAPITAL ROTATION — SWING",
            "",
            f"Model: {metrics['strategy_label']}",
            f"Assets: {', '.join(symbols)}",
            "Decision data: daily candles",
            f"Utility horizon: {config.rotation_horizon_days} trading sessions",
            "Capital pool: one shared account, reinvested after every exit/rotation",
            "Decision objective: maximize smoother net compounded wealth, not predict exact tops.",
            f"Risk penalties: downside={config.rotation_downside_penalty:.3f}, drawdown={config.rotation_drawdown_penalty:.3f}",
            f"Validation: expanding walk-forward, purge={config.rotation_purge_days} sessions, fold test={config.rotation_walk_forward_test_days} sessions",
            "",
            "OUT-OF-SAMPLE WALK-FORWARD",
            f"Initial capital: ${initial:,.2f}",
            f"Ending capital: ${ending:,.2f}",
            f"Total return: {metrics['strategy_return']:.2%}",
            f"CAGR: {metrics['strategy_cagr']:.2%}",
            f"Compound log growth: {metrics['compound_log_growth']:.6f}",
            f"Maximum drawdown: {metrics['strategy_maximum_drawdown']:.2%}",
            f"Sharpe estimate: {metrics['strategy_sharpe']:.3f}",
            f"Capital rotations: {rotation_count}",
            f"Buys: {buys}",
            f"Sells including final liquidation: {sells}",
            f"Cycles/year: {metrics['cycles_per_year']:.2f}",
            f"Average holding days: {avg_holding:.2f}",
            f"Time in market: {exposure:.2%}",
            f"Transaction fees: ${total_fees:,.2f}",
            "",
            "BENCHMARK",
            "Equal-weight buy-and-hold across the same available assets.",
            f"Benchmark ending capital: ${benchmark_ending:,.2f}",
            f"Benchmark return: {metrics['buy_hold_return']:.2%}",
            f"Benchmark CAGR: {metrics['buy_hold_cagr']:.2%}",
            "",
            "METHOD",
            "- Signals use information available at the current daily close.",
            "- Position changes execute at the next daily open.",
            f"- XGBoost Utility predicts {config.rotation_horizon_days}-session risk-adjusted capital utility.",
            f"- QR-DQN uses {int(config.qrdqn_n_step)}-step discounted risk-adjusted returns.",
            "- Every fold is trained only on information available before that fold.",
            f"- A {config.rotation_purge_days}-session purge prevents forward labels from touching the next validation/test segment.",
            "- FINAL_LIQUIDATION is bookkeeping only and is not a model decision.",
        ]
    )

    return RotationRunResult(
        backend=backend,
        predictions=predictions,
        trades=trades,
        summary=summary,
        metrics=metrics,
    )


def _build_walk_forward_folds(
    common_dates: pd.DatetimeIndex,
    config: Any,
) -> list[dict[str, Any]]:
    purge = max(
        int(config.rotation_purge_days),
        int(config.rotation_horizon_days),
    )
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    test_days = int(config.rotation_walk_forward_test_days)
    min_test_days = int(config.rotation_walk_forward_min_test_days)
    min_train = int(config.rotation_minimum_training_rows)

    first_test_start = min_train + purge + calibration_days + purge
    if first_test_start >= len(common_dates) - min_test_days:
        raise ValueError(
            "Not enough common history for expanding walk-forward: "
            f"rows={len(common_dates)}, minimum_train={min_train}, "
            f"calibration={calibration_days}, purge={purge}, "
            f"minimum_test={min_test_days}."
        )

    folds: list[dict[str, Any]] = []
    test_start = first_test_start
    fold_id = 1
    while test_start < len(common_dates):
        test_end = min(len(common_dates), test_start + test_days)
        if test_end - test_start < min_test_days:
            if folds:


                folds[-1]["test_end_index"] = len(common_dates)
                folds[-1]["test_end"] = common_dates[-1]
                folds[-1]["decision_dates"] = common_dates[
                    folds[-1]["test_start_index"] - 1 : len(common_dates)
                ]
            break

        calibration_end = test_start - purge
        calibration_start = calibration_end - calibration_days
        train_end = calibration_start - purge
        final_fit_end = test_start - purge
        if train_end < min_train:
            raise ValueError(
                f"Fold {fold_id}: training rows {train_end} < {min_train}."
            )

        folds.append(
            {
                "fold_id": fold_id,
                "train_end_index": train_end,
                "calibration_start_index": calibration_start,
                "calibration_end_index": calibration_end,
                "final_fit_end_index": final_fit_end,
                "test_start_index": test_start,
                "test_end_index": test_end,
                "train_start": common_dates[0],
                "train_end": common_dates[train_end - 1],
                "calibration_start": common_dates[calibration_start],
                "calibration_end": common_dates[calibration_end - 1],
                "purge_start": common_dates[calibration_end],
                "purge_end": common_dates[test_start - 1],
                "test_start": common_dates[test_start],
                "test_end": common_dates[test_end - 1],
                "decision_dates": common_dates[test_start - 1 : test_end],
            }
        )
        fold_id += 1
        test_start = test_end

    if not folds:
        raise ValueError("No valid expanding walk-forward fold was created.")
    return folds


def _scheduled_policy(
    policies: dict[int, Callable[[pd.Timestamp, int, int], tuple[int, float]]],
    decision_to_fold: dict[pd.Timestamp, int],
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        fold_id = decision_to_fold.get(key)
        if fold_id is None:
            raise KeyError(f"No walk-forward policy is assigned to {key}.")
        return policies[int(fold_id)](
            timestamp,
            current_position,
            holding_days,
        )
    return policy


def _fold_performance(
    predictions: pd.DataFrame,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    rows = predictions.reset_index().sort_values("timestamp").reset_index(drop=True)
    output: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        subset = rows.loc[rows["walk_forward_fold"] == fold_id]
        if subset.empty:
            continue
        first_idx = int(subset.index[0])
        strategy_start = (
            float(initial_capital)
            if first_idx == 0
            else float(rows.loc[first_idx - 1, "strategy_equity"])
        )
        benchmark_start = (
            float(initial_capital)
            if first_idx == 0
            else float(rows.loc[first_idx - 1, "buy_hold_equity"])
        )
        strategy_end = float(subset.iloc[-1]["strategy_equity"])
        benchmark_end = float(subset.iloc[-1]["buy_hold_equity"])
        curve = pd.Series(
            [strategy_start, *subset["strategy_equity"].astype(float).tolist()]
        )
        output.append(
            {
                "fold_id": fold_id,
                "train_end": fold["train_end"],
                "calibration_start": fold["calibration_start"],
                "calibration_end": fold["calibration_end"],
                "purge_start": fold["purge_start"],
                "purge_end": fold["purge_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "strategy_starting_capital": strategy_start,
                "strategy_ending_capital": strategy_end,
                "strategy_return": strategy_end / strategy_start - 1,
                "benchmark_return": benchmark_end / benchmark_start - 1,
                "excess_return": (
                    strategy_end / strategy_start
                    - benchmark_end / benchmark_start
                ),
                "maximum_drawdown": _maximum_drawdown(curve),
                "sessions": int(len(subset)),
            }
        )
    return output


def run_rotation_models(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[RotationRunResult]:
    frames, common_dates = prepare_rotation_panel(
        bars_by_symbol,
        config,
    )
    symbols = sorted(frames)
    folds = _build_walk_forward_folds(common_dates, config)
    xgb_plan = resolve_xgboost_compute_plan(config)
    qrdqn_plan = resolve_qrdqn_compute_plan(config)

    selected_models = [
        model
        for model in ("xgboost_utility", "qrdqn")
        if model in config.rotation_models
    ]
    if not selected_models:
        raise ValueError("Select at least one compound-rotation model.")

    xgb_repetitions = (
        int(config.rotation_xgb_repetitions)
        if "xgboost_utility" in selected_models
        else 0
    )
    qrdqn_repetitions = (
        int(config.rotation_qrdqn_repetitions)
        if "qrdqn" in selected_models
        else 0
    )
    seed_step = int(config.rotation_seed_step)
    total_runs = xgb_repetitions + qrdqn_repetitions
    if total_runs <= 0:
        raise ValueError("At least one robustness repetition is required.")

    first_test_start = int(folds[0]["test_start_index"])
    final_test_end = int(folds[-1]["test_end_index"])
    all_decision_dates = common_dates[first_test_start - 1 : final_test_end]

    decision_to_fold: dict[pd.Timestamp, int] = {}
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] = {}
    for fold in folds:
        for timestamp in fold["decision_dates"][:-1]:
            key = pd.Timestamp(timestamp)
            decision_to_fold[key] = int(fold["fold_id"])
            decision_metadata[key] = {
                "fold_id": int(fold["fold_id"]),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            }

    progress_lock = threading.Lock()
    family_progress = {
        "xgboost_utility": 0.0,
        "qrdqn": 0.0,
    }
    family_weights = {
        "xgboost_utility": xgb_repetitions,
        "qrdqn": qrdqn_repetitions,
    }
    completed_runs = 0

    def report_family(
        family: str,
        fraction: float,
        stage: str,
    ) -> None:
        nonlocal completed_runs
        with progress_lock:
            family_progress[family] = max(0.0, min(1.0, float(fraction)))
            weighted = sum(
                family_progress[name] * family_weights[name]
                for name in family_progress
            ) / max(1, total_runs)
            completed = completed_runs
        if progress_callback is not None:
            progress_callback(
                20.0 + 72.0 * weighted,
                stage,
                completed,
            )

    def mark_run_completed(family: str, stage: str) -> None:
        nonlocal completed_runs
        with progress_lock:
            completed_runs += 1
            completed = completed_runs
        if progress_callback is not None:
            weighted = sum(
                family_progress[name] * family_weights[name]
                for name in family_progress
            ) / max(1, total_runs)
            progress_callback(
                20.0 + 72.0 * weighted,
                stage,
                completed,
            )

    xgb_label = (
        f"CUDA — {xgb_plan.gpu_name}"
        if xgb_plan.selected == "cuda"
        else "CPU"
    )
    qr_label = (
        f"CUDA — {qrdqn_plan.gpu_name}"
        if qrdqn_plan.selected == "cuda"
        else "CPU"
    )
    if progress_callback is not None:
        progress_callback(
            18.0,
            (
                f"Prepared {len(symbols)} assets and {len(folds)} folds — "
                f"XGBoost={xgb_label}; QR-DQN={qr_label}; "
                f"parallel_models={bool(config.rotation_parallel_models)}"
            ),
            0,
        )

    def backend_id(family: str, seed: int, repetitions: int) -> str:
        if repetitions <= 1:
            return family
        return f"{family}_seed_{seed}"

    def trade_wrapper(
        family: str,
        seed: int,
        repetition_index: int,
    ) -> Callable[[dict[str, Any]], None] | None:
        if trade_callback is None:
            return None
        def emit(trade: dict[str, Any]) -> None:
            payload = dict(trade)
            payload["model_family"] = family
            payload["random_seed"] = seed
            payload["repetition_index"] = repetition_index
            payload["model"] = (
                "XGBoost Utility"
                if family == "xgboost_utility"
                else "QR-DQN"
            ) + (
                f" · seed {seed}"
                if (
                    (family == "xgboost_utility" and xgb_repetitions > 1)
                    or (family == "qrdqn" and qrdqn_repetitions > 1)
                )
                else ""
            )
            trade_callback(payload)
        return emit

    def run_xgb_family() -> list[RotationRunResult]:
        family_results: list[RotationRunResult] = []
        if xgb_repetitions <= 0:
            return family_results
        effective_device = xgb_plan.selected
        fallback_reasons: list[str] = []
        for repetition in range(xgb_repetitions):
            seed = int(config.random_state) + repetition * seed_step
            rep_config = config.model_copy(update={"random_state": seed})
            policies: dict[int, Callable] = {}
            margin_details: list[dict[str, Any]] = []
            for fold_position, fold in enumerate(folds, start=1):
                overall = (
                    repetition
                    + (fold_position - 1) / max(1, len(folds))
                ) / xgb_repetitions
                report_family(
                    "xgboost_utility",
                    overall,
                    (
                        f"XGBoost Utility run {repetition + 1}/{xgb_repetitions} "
                        f"— fold {fold_position}/{len(folds)} — {effective_device.upper()}"
                    ),
                )
                fold_id = int(fold["fold_id"])
                train_dates = common_dates[: int(fold["train_end_index"])]
                calibration_dates = common_dates[
                    int(fold["calibration_start_index"])
                    : int(fold["calibration_end_index"])
                ]
                final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

                (
                    calibration_models,
                    effective_device,
                    fallback_reason,
                ) = _fit_xgb_models(
                    frames,
                    symbols,
                    train_dates,
                    rep_config,
                    effective_device,
                )
                if fallback_reason:
                    fallback_reasons.append(fallback_reason)
                candidate_margins = tuple(
                    float(value)
                    for value in rep_config.rotation_switch_margin_candidates
                )
                best_candidate = candidate_margins[0]
                best_score = float("-inf")
                for candidate in candidate_margins:
                    calibration_policy = _xgb_policy(
                        calibration_models,
                        frames,
                        symbols,
                        rep_config,
                        candidate,
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

                (
                    final_models,
                    effective_device,
                    fallback_reason,
                ) = _fit_xgb_models(
                    frames,
                    symbols,
                    final_fit_dates,
                    rep_config,
                    effective_device,
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
                )
                margin_details.append(
                    {
                        "fold_id": fold_id,
                        "calibrated_candidate_margin": float(best_candidate),
                        "effective_switch_margin": float(effective_margin),
                        "calibration_risk_adjusted_score": float(best_score),
                    }
                )

            scheduled = _scheduled_policy(policies, decision_to_fold)
            result = _simulate_exact(
                "xgboost_utility",
                scheduled,
                frames,
                symbols,
                all_decision_dates,
                rep_config,
                fee_calculator,
                slippage,
                decision_metadata=decision_metadata,
                trade_callback=trade_wrapper(
                    "xgboost_utility",
                    seed,
                    repetition + 1,
                ),
            )
            unique_backend = backend_id(
                "xgboost_utility",
                seed,
                xgb_repetitions,
            )
            result.backend = unique_backend
            result.metrics["backend"] = unique_backend
            result.metrics["model_family"] = "xgboost_utility"
            result.metrics["random_seed"] = seed
            result.metrics["repetition_index"] = repetition + 1
            result.metrics["repetition_count"] = xgb_repetitions
            result.metrics["strategy_label"] = (
                "XGBoost Utility"
                + (f" · seed {seed}" if xgb_repetitions > 1 else "")
            )

            fold_metrics = _fold_performance(
                result.predictions,
                folds,
                float(rep_config.initial_capital),
            )
            margin_by_fold = {
                item["fold_id"]: item
                for item in margin_details
            }
            for item in fold_metrics:
                item.update(margin_by_fold.get(item["fold_id"], {}))
            effective_values = [
                item["effective_switch_margin"]
                for item in margin_details
            ]
            candidate_values = [
                item["calibrated_candidate_margin"]
                for item in margin_details
            ]
            result.metrics.update(
                {
                    "walk_forward_fold_count": len(folds),
                    "walk_forward_folds": fold_metrics,
                    "calibrated_candidate_margin_mean": float(np.mean(candidate_values)),
                    "effective_switch_margin_mean": float(np.mean(effective_values)),
                    "effective_switch_margin_min": float(np.min(effective_values)),
                    "effective_switch_margin_max": float(np.max(effective_values)),
                    "calibrated_switch_margin": float(np.mean(candidate_values)),
                    "effective_switch_margin": float(np.mean(effective_values)),
                    "requested_accelerator": xgb_plan.requested,
                    "effective_compute_device": effective_device,
                    "cuda_available": xgb_plan.cuda_available,
                    "gpu_name": xgb_plan.gpu_name,
                    "framework_version": xgb_plan.framework_version,
                    "cuda_build": xgb_plan.cuda_build,
                    "cpu_fallback_used": bool(
                        xgb_plan.fallback_used or fallback_reasons
                    ),
                    "compute_fallback_reason": (
                        fallback_reasons[-1]
                        if fallback_reasons
                        else xgb_plan.fallback_reason
                    ),
                    "parallel_models_enabled": bool(
                        config.rotation_parallel_models
                    ),
                }
            )
            result.summary += "\n\nROBUSTNESS / COMPUTE\n"
            result.summary += f"Seed: {seed}\n"
            result.summary += f"Repetition: {repetition + 1}/{xgb_repetitions}\n"
            result.summary += f"Compute device: {effective_device.upper()}\n"
            if fallback_reasons:
                result.summary += f"Fallback: {fallback_reasons[-1]}\n"
            family_results.append(result)
            report_family(
                "xgboost_utility",
                (repetition + 1) / xgb_repetitions,
                f"XGBoost Utility run {repetition + 1}/{xgb_repetitions} completed",
            )
            mark_run_completed(
                "xgboost_utility",
                f"XGBoost Utility run {repetition + 1}/{xgb_repetitions} completed",
            )
        return family_results

    def run_qrdqn_family() -> list[RotationRunResult]:
        family_results: list[RotationRunResult] = []
        if qrdqn_repetitions <= 0:
            return family_results

        requested_fold_workers = max(
            1,
            int(config.qrdqn_parallel_folds),
        )
        requested_repetition_workers = max(
            1,
            int(config.qrdqn_parallel_repetitions),
        )
        effective_repetition_workers = min(
            requested_repetition_workers,
            qrdqn_repetitions,
        )
        if qrdqn_plan.selected != "cpu":
                                                                                 
                                                                              
                                                                      
            effective_repetition_workers = 1

        effective_fold_workers = min(requested_fold_workers, len(folds))
        if effective_repetition_workers > 1:
                                                                                    
                                                                                    
                                                                                 
                                        
            effective_fold_workers = 1

        torch_runtime_num_threads = None
        torch_effective_num_threads = None
        if qrdqn_plan.selected == "cpu":
            try:
                import torch
                torch_runtime_num_threads = int(torch.get_num_threads())
                requested_torch_threads = int(
                    config.qrdqn_torch_num_threads
                )
                if requested_torch_threads > 0:
                    torch.set_num_threads(requested_torch_threads)
                torch_effective_num_threads = int(torch.get_num_threads())
            except Exception:
                torch_runtime_num_threads = None
                torch_effective_num_threads = None

                                                                                  
                                                                                    
        fold_contexts: dict[int, dict[str, Any]] = {}
        for fold_position, fold in enumerate(folds, start=1):
            fold_id = int(fold["fold_id"])
            train_dates = common_dates[: int(fold["train_end_index"])]
            calibration_dates = common_dates[
                int(fold["calibration_start_index"])
                : int(fold["calibration_end_index"])
            ]
            normalization = _normalization(frames, train_dates, config)
            cache_dates = train_dates.union(calibration_dates)
            fold_contexts[fold_id] = {
                "fold_position": fold_position,
                "fold": fold,
                "train_dates": train_dates,
                "calibration_dates": calibration_dates,
                "normalization": normalization,
                "feature_cache": _build_qrdqn_feature_cache(
                    frames,
                    symbols,
                    cache_dates,
                    normalization,
                    config,
                ),
                "price_cache": _build_qrdqn_price_cache(
                    frames,
                    symbols,
                    cache_dates,
                ),
            }

        repetition_progress = {
            repetition: 0.0
            for repetition in range(qrdqn_repetitions)
        }
        repetition_progress_lock = threading.Lock()

        def report_repetition_training(
            repetition: int,
            rep_fraction: float,
            stage: str,
        ) -> None:
            with repetition_progress_lock:
                repetition_progress[repetition] = max(
                    0.0,
                    min(1.0, float(rep_fraction)),
                )
                aggregate = float(np.mean(list(repetition_progress.values())))
                                                                            
                                                                                
            report_family("qrdqn", aggregate * 0.90, stage)

        def train_repetition(repetition: int) -> dict[str, Any]:
            seed = int(config.random_state) + repetition * seed_step
            rep_config = config.model_copy(update={"random_state": seed})
            policies: dict[int, Callable] = {}
            training_details: dict[int, dict[str, Any]] = {}
            decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
            fold_progress: dict[int, float] = {
                int(fold["fold_id"]): 0.0
                for fold in folds
            }
            fold_progress_lock = threading.Lock()

            def run_fold(
                fold_position: int,
                fold: dict[str, Any],
            ) -> tuple[int, Callable, dict[str, Any]]:
                fold_id = int(fold["fold_id"])
                context = fold_contexts[fold_id]
                fold_seed = seed

                def update_fold(local_fraction: float) -> None:
                    with fold_progress_lock:
                        fold_progress[fold_id] = max(
                            0.0,
                            min(1.0, float(local_fraction)),
                        )
                        rep_fold_fraction = float(
                            np.mean(list(fold_progress.values()))
                        )
                    report_repetition_training(
                        repetition,
                        rep_fold_fraction,
                        (
                            f"QR-DQN run {repetition + 1}/{qrdqn_repetitions} "
                            f"— fold {fold_position}/{len(folds)} "
                            f"— training {local_fraction * 100:.0f}% "
                            f"— {qrdqn_plan.selected.upper()} "
                            f"— {effective_repetition_workers} repetition worker(s); "
                            f"{effective_fold_workers} fold worker(s)"
                        ),
                    )

                network, diagnostics = _train_qrdqn(
                    frames,
                    symbols,
                    context["train_dates"],
                    context["calibration_dates"],
                    context["normalization"],
                    rep_config,
                    qrdqn_plan.selected,
                    fold_seed,
                    progress_callback=update_fold,
                    feature_cache=context["feature_cache"],
                    price_cache=context["price_cache"],
                )
                policy = _qrdqn_policy(
                    network,
                    frames,
                    symbols,
                    context["normalization"],
                    rep_config,
                    decision_diagnostics=decision_diagnostics,
                )
                diagnostics.update(
                    {
                        "fold_id": fold_id,
                        "fold_position": fold_position,
                    }
                )
                return fold_id, policy, diagnostics

            if effective_fold_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=effective_fold_workers,
                    thread_name_prefix=f"qrdqn-fold-r{repetition + 1}",
                ) as pool:
                    futures = {
                        pool.submit(run_fold, position, fold): position
                        for position, fold in enumerate(folds, start=1)
                    }
                    for future in as_completed(futures):
                        fold_id, policy, diagnostics = future.result()
                        policies[fold_id] = policy
                        training_details[fold_id] = diagnostics
            else:
                for position, fold in enumerate(folds, start=1):
                    fold_id, policy, diagnostics = run_fold(position, fold)
                    policies[fold_id] = policy
                    training_details[fold_id] = diagnostics

            report_repetition_training(
                repetition,
                1.0,
                f"QR-DQN run {repetition + 1}/{qrdqn_repetitions} training completed",
            )
            return {
                "repetition": repetition,
                "seed": seed,
                "rep_config": rep_config,
                "policies": policies,
                "training_details": training_details,
                "decision_diagnostics": decision_diagnostics,
            }

        trained_repetitions: dict[int, dict[str, Any]] = {}
        if effective_repetition_workers > 1:
                                                                              
                                                                                     
                                                                                    
                                                                                    
                                                                                
            if hasattr(config, "model_dump"):
                process_config_values = config.model_dump(mode="python")
            else:
                process_config_values = dict(vars(config))

            requested_torch_threads = int(
                config.qrdqn_torch_num_threads
            )
            process_torch_threads = (
                requested_torch_threads
                if requested_torch_threads > 0
                else int(torch_runtime_num_threads or 0)
            )
            process_context = multiprocessing.get_context("spawn")
            completed_repetitions = 0

            with ProcessPoolExecutor(
                max_workers=effective_repetition_workers,
                mp_context=process_context,
                initializer=_initialize_qrdqn_repetition_process,
                initargs=(
                    frames,
                    symbols,
                    fold_contexts,
                    process_config_values,
                    qrdqn_plan.selected,
                    process_torch_threads,
                ),
            ) as pool:
                futures = {}
                for repetition in range(qrdqn_repetitions):
                    seed = int(config.random_state) + repetition * seed_step
                    future = pool.submit(
                        _qrdqn_repetition_process_task,
                        repetition,
                        seed,
                    )
                    futures[future] = repetition

                for future in as_completed(futures):
                    process_bundle = future.result()
                    repetition = int(process_bundle["repetition"])
                    seed = int(process_bundle["seed"])
                    rep_config = config.model_copy(update={"random_state": seed})
                    policies: dict[int, Callable] = {}
                    training_details: dict[int, dict[str, Any]] = {}
                    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

                    for fold_id, trained_fold in process_bundle[
                        "trained_folds"
                    ].items():
                        fold_id = int(fold_id)
                        fold_context = fold_contexts[fold_id]
                        network = _restore_qrdqn_network_from_process(
                            trained_fold["state_dict"],
                            symbols,
                            fold_context,
                            rep_config,
                            qrdqn_plan.selected,
                        )
                        policies[fold_id] = _qrdqn_policy(
                            network,
                            frames,
                            symbols,
                            fold_context["normalization"],
                            rep_config,
                            decision_diagnostics=decision_diagnostics,
                        )
                        diagnostics = dict(trained_fold["diagnostics"])
                        diagnostics.update(
                            {
                                "fold_id": fold_id,
                                "fold_position": int(
                                    fold_context["fold_position"]
                                ),
                            }
                        )
                        training_details[fold_id] = diagnostics

                    trained_repetitions[repetition] = {
                        "repetition": repetition,
                        "seed": seed,
                        "rep_config": rep_config,
                        "policies": policies,
                        "training_details": training_details,
                        "decision_diagnostics": decision_diagnostics,
                    }
                    completed_repetitions += 1
                    report_family(
                        "qrdqn",
                        (completed_repetitions / qrdqn_repetitions) * 0.90,
                        (
                            f"QR-DQN isolated process run "
                            f"{completed_repetitions}/{qrdqn_repetitions} completed"
                        ),
                    )
        else:
            for repetition in range(qrdqn_repetitions):
                bundle = train_repetition(repetition)
                trained_repetitions[repetition] = bundle

                                                                                  
                                                                                        
        for repetition in range(qrdqn_repetitions):
            bundle = trained_repetitions[repetition]
            seed = int(bundle["seed"])
            rep_config = bundle["rep_config"]
            policies = bundle["policies"]
            training_details = bundle["training_details"]
            decision_diagnostics = bundle["decision_diagnostics"]

            scheduled = _scheduled_policy(policies, decision_to_fold)
            result = _simulate_exact(
                "qrdqn",
                scheduled,
                frames,
                symbols,
                all_decision_dates,
                rep_config,
                fee_calculator,
                slippage,
                decision_metadata=decision_metadata,
                policy_decision_diagnostics=decision_diagnostics,
                trade_callback=trade_wrapper(
                    "qrdqn",
                    seed,
                    repetition + 1,
                ),
            )
            unique_backend = backend_id(
                "qrdqn",
                seed,
                qrdqn_repetitions,
            )
            result.backend = unique_backend
            result.metrics["backend"] = unique_backend
            result.metrics["model_family"] = "qrdqn"
            result.metrics["random_seed"] = seed
            result.metrics["repetition_index"] = repetition + 1
            result.metrics["repetition_count"] = qrdqn_repetitions
            result.metrics["strategy_label"] = (
                "QR-DQN"
                + (f" · seed {seed}" if qrdqn_repetitions > 1 else "")
            )
            fold_metrics = _fold_performance(
                result.predictions,
                folds,
                float(rep_config.initial_capital),
            )
            for item in fold_metrics:
                item.update(training_details.get(item["fold_id"], {}))
            used_steps = [
                int(item["steps_used"])
                for item in training_details.values()
            ]
            early_count = sum(
                bool(item.get("stopped_early"))
                for item in training_details.values()
            )
            best_steps = [
                int(item.get("best_step", 0))
                for item in training_details.values()
            ]
            result.metrics.update(
                {
                    "walk_forward_fold_count": len(folds),
                    "walk_forward_folds": fold_metrics,
                    "requested_accelerator": qrdqn_plan.requested,
                    "effective_compute_device": qrdqn_plan.selected,
                    "cuda_available": qrdqn_plan.cuda_available,
                    "gpu_name": qrdqn_plan.gpu_name,
                    "framework_version": qrdqn_plan.framework_version,
                    "torch_cuda_version": qrdqn_plan.cuda_runtime_version,
                    "cuda_build": qrdqn_plan.cuda_build,
                    "cpu_fallback_used": qrdqn_plan.fallback_used,
                    "compute_fallback_reason": qrdqn_plan.fallback_reason,
                    "qrdqn_parallel_folds_requested": requested_fold_workers,
                    "qrdqn_parallel_folds_effective": effective_fold_workers,
                    "qrdqn_parallel_repetitions_requested": requested_repetition_workers,
                    "qrdqn_parallel_repetitions_effective": effective_repetition_workers,
                    "qrdqn_repetition_parallelism_backend": (
                        "process_spawn"
                        if effective_repetition_workers > 1
                        else "sequential"
                    ),
                    "qrdqn_shared_fold_cache_enabled": True,
                    "qrdqn_shared_fold_cache_count": len(fold_contexts),
                    "qrdqn_torch_num_threads_runtime_default": torch_runtime_num_threads,
                    "qrdqn_torch_num_threads_requested": int(
                        config.qrdqn_torch_num_threads
                    ),
                    "qrdqn_torch_num_threads": torch_effective_num_threads,
                    "qrdqn_torch_thread_policy": (
                        "configured"
                        if int(config.qrdqn_torch_num_threads) > 0
                        else "runtime_default_unchanged"
                    ),
                    "qrdqn_fold_seed_mode": "shared_repetition_seed",
                    "qrdqn_repetition_seed": int(seed),
                    "qrdqn_training_steps_requested": int(rep_config.qrdqn_training_steps),
                    "qrdqn_training_steps_mean_used": float(np.mean(used_steps)),
                    "qrdqn_training_steps_max_used": int(max(used_steps)),
                    "qrdqn_early_stopping_enabled": bool(
                        rep_config.qrdqn_early_stopping_enabled
                    ),
                    "qrdqn_min_training_steps": int(
                        rep_config.qrdqn_min_training_steps
                    ),
                    "qrdqn_n_step": int(getattr(rep_config, "qrdqn_n_step", 1)),
                    "qrdqn_best_step_min": int(min(best_steps)),
                    "qrdqn_best_step_max": int(max(best_steps)),
                    "qrdqn_early_stopped_folds": int(early_count),
                    "diagnostics_version": "1.12.18",
                    "q_value_diagnostics_enabled": False,
                    "policy_modified_by_diagnostics": False,
                    "rotation_counterfactual_horizons_bars": [],
                    "parallel_models_enabled": bool(
                        config.rotation_parallel_models
                    ),
                }
            )
            result.summary += "\n\nROBUSTNESS / COMPUTE\n"
            result.summary += f"Seed: {seed}\n"
            result.summary += f"Repetition: {repetition + 1}/{qrdqn_repetitions}\n"
            result.summary += f"Compute device: {qrdqn_plan.selected.upper()}\n"
            result.summary += f"Parallel repetition workers: {effective_repetition_workers}\n"
            result.summary += f"Parallel fold workers: {effective_fold_workers}\n"
            result.summary += f"Shared fold cache: ON ({len(fold_contexts)} fold(s))\n"
            result.summary += (
                "PyTorch CPU threads: "
                + (
                    str(torch_effective_num_threads)
                    if torch_effective_num_threads is not None
                    else "unknown"
                )
                + (
                    f" (runtime default was {torch_runtime_num_threads})"
                    if (
                        torch_runtime_num_threads is not None
                        and torch_effective_num_threads is not None
                        and torch_runtime_num_threads != torch_effective_num_threads
                    )
                    else ""
                )
                + "\n"
            )
            result.summary += (
                f"Fold seed mode: shared repetition seed ({seed})\n"
            )
            result.summary += (
                f"N-step return: {int(getattr(rep_config, 'qrdqn_n_step', 1))}\n"
            )
            result.summary += (
                f"Mean training steps used: {np.mean(used_steps):.0f}/"
                f"{int(rep_config.qrdqn_training_steps)}\n"
            )
            result.summary += (
                "Early stopping: "
                f"{'ON' if rep_config.qrdqn_early_stopping_enabled else 'OFF'}\n"
            )
            result.summary += (
                f"Minimum eligible checkpoint step: "
                f"{int(rep_config.qrdqn_min_training_steps) if rep_config.qrdqn_early_stopping_enabled else 0}\n"
            )
            result.summary += (
                f"Best checkpoint range: {min(best_steps)}–{max(best_steps)}\n"
            )
            result.summary += f"Early-stopped folds: {early_count}/{len(folds)}\n"
            if qrdqn_plan.fallback_reason:
                result.summary += f"Fallback: {qrdqn_plan.fallback_reason}\n"
            family_results.append(result)
            report_family(
                "qrdqn",
                0.90 + 0.10 * ((repetition + 1) / qrdqn_repetitions),
                f"QR-DQN run {repetition + 1}/{qrdqn_repetitions} completed",
            )
            mark_run_completed(
                "qrdqn",
                f"QR-DQN run {repetition + 1}/{qrdqn_repetitions} completed",
            )
        return family_results

    results: list[RotationRunResult] = []
    run_parallel = bool(
        config.rotation_parallel_models
    ) and len(selected_models) > 1

    if run_parallel:
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rotation-model",
        ) as pool:
            futures = []
            if xgb_repetitions:
                futures.append(pool.submit(run_xgb_family))
            if qrdqn_repetitions:
                futures.append(pool.submit(run_qrdqn_family))
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        if xgb_repetitions:
            results.extend(run_xgb_family())
        if qrdqn_repetitions:
            results.extend(run_qrdqn_family())

    order = {
        "xgboost_utility": 0,
        "qrdqn": 1,
    }
    results.sort(
        key=lambda result: (
            order.get(
                str(result.metrics.get("model_family", result.backend)),
                99,
            ),
            int(result.metrics.get("repetition_index", 1)),
        )
    )
    return results

