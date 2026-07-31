from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import exchange_calendars as xcals


MARKET_TIMEZONE = "America/New_York"
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_LAST_BAR_MINUTE = 15 * 60 + 45
SOURCE_BAR_MINUTES = 15
NYSE_CALENDAR_NAME = "XNYS"
_NYSE_CALENDAR = xcals.get_calendar(NYSE_CALENDAR_NAME)

OPEN_CLOSE_FEATURES = [
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
    "previous_intraday_return",
    "previous_opening_gap",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]


@dataclass
class OpenCloseRunResult:
    backend: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    summary: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ComputePlan:
    requested: str
    selected: str
    cuda_available: bool
    gpu_name: str | None
    framework_version: str | None
    fallback_used: bool
    fallback_reason: str | None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


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


def _session_path_drawdown(group: pd.DataFrame) -> float:
    entry = float(group.iloc[0]["open"])
    closes = group["close"].astype(float).to_numpy()
    if not np.isfinite(entry) or entry <= 0 or closes.size == 0:
        return float("nan")
    path = np.concatenate(([entry], closes))
    running_peak = np.maximum.accumulate(path)
    drawdowns = 1.0 - np.divide(
        path,
        running_peak,
        out=np.ones_like(path),
        where=running_peak > 0,
    )
    return max(0.0, float(np.nanmax(drawdowns)))


def _expected_session_bar_starts(session_date: Any) -> pd.DatetimeIndex:
    label = pd.Timestamp(session_date).normalize()
    if not _NYSE_CALENDAR.is_session(label):
        return pd.DatetimeIndex([], tz="UTC")
    session_open = pd.Timestamp(_NYSE_CALENDAR.session_open(label)).tz_convert("UTC")
    session_close = pd.Timestamp(_NYSE_CALENDAR.session_close(label)).tz_convert("UTC")
    last_bar_start = session_close - pd.Timedelta(minutes=SOURCE_BAR_MINUTES)
    return pd.date_range(
        start=session_open,
        end=last_bar_start,
        freq=f"{SOURCE_BAR_MINUTES}min",
        tz="UTC",
    )


def _regular_session_quality(
    group: pd.DataFrame,
    session_date: Any,
    *,
    now_utc: pd.Timestamp | None = None,
) -> dict[str, Any]:
    expected = _expected_session_bar_starts(session_date)
    if expected.empty:
        return {
            "usable": False,
            "quality": "NOT_A_SESSION",
            "expected_bar_count": 0,
            "observed_bar_count": 0,
            "missing_internal_bar_count": 0,
            "official_close_timestamp": None,
        }

    label = pd.Timestamp(session_date).normalize()
    official_close = pd.Timestamp(_NYSE_CALENDAR.session_close(label)).tz_convert("UTC")
    current_utc = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.tz_localize("UTC")
    else:
        current_utc = current_utc.tz_convert("UTC")

    observed = pd.DatetimeIndex(pd.to_datetime(group.index, utc=True)).sort_values().unique()
    observed_expected = observed.intersection(expected)
    open_present = expected[0] in observed_expected
    close_bar_present = expected[-1] in observed_expected
    missing_internal = expected[1:-1].difference(observed_expected)

    if current_utc < official_close:
        quality = "IN_PROGRESS"
        usable = False
    elif not open_present and not close_bar_present:
        quality = "MISSING_OPEN_AND_CLOSE"
        usable = False
    elif not open_present:
        quality = "MISSING_OPEN"
        usable = False
    elif not close_bar_present:
        quality = "MISSING_CLOSE"
        usable = False
    elif len(missing_internal) == 0:
        quality = "COMPLETE"
        usable = True
    elif len(missing_internal) == 1:
        quality = "INTERNAL_GAP"
        usable = True
    else:
        quality = "MULTIPLE_INTERNAL_GAPS"
        usable = True

    return {
        "usable": bool(usable),
        "quality": quality,
        "expected_bar_count": int(len(expected)),
        "observed_bar_count": int(len(observed_expected)),
        "missing_internal_bar_count": int(len(missing_internal)),
        "official_close_timestamp": official_close,
    }


def _training_round_trip_cost_rate(open_price: pd.Series, config: Any) -> pd.Series:
    slip = 2.0 * max(0.0, float(getattr(config, "slippage_bps", 0.0))) / 10_000.0
    commission = 2.0 * max(0.0, float(getattr(config, "commission_rate", 0.0)))
    sec = max(0.0, float(getattr(config, "sec_fee_rate", 0.0)))
    taf_per_share = max(0.0, float(getattr(config, "taf_fee_per_share", 0.0)))
    cat_per_share = max(0.0, float(getattr(config, "cat_fee_per_share", 0.0)))
    share_based = (taf_per_share + 2.0 * cat_per_share) / open_price.clip(lower=1e-9)
    return (slip + commission + sec + share_based).clip(lower=0.0, upper=0.25)


def build_open_close_frame(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    data = bars.copy().sort_index()
    data.index = pd.to_datetime(data.index, utc=True)
    local_index = data.index.tz_convert(MARKET_TIMEZONE)
    minutes = local_index.hour * 60 + local_index.minute
    regular = (minutes >= SESSION_OPEN_MINUTE) & (minutes <= SESSION_LAST_BAR_MINUTE)
    data = data.loc[regular].copy()
    if data.empty:
        return data

    data["_session_date"] = pd.Series(
        data.index.tz_convert(MARKET_TIMEZONE).date,
        index=data.index,
        dtype="object",
    )

    rows: list[dict[str, Any]] = []
    for session_date, group in data.groupby("_session_date", sort=True):
        group = group.sort_index()
        if group.empty:
            continue

        quality = _regular_session_quality(group, session_date)
        if not bool(quality["usable"]):


            continue

        expected_starts = _expected_session_bar_starts(session_date)
        group = group.loc[group.index.intersection(expected_starts)].sort_index()
        if group.empty:
            continue
        first_ts = pd.Timestamp(expected_starts[0])
        last_ts = pd.Timestamp(expected_starts[-1])
        open_price = float(group.loc[first_ts, "open"])
        close_price = float(group.loc[last_ts, "close"])
        if not np.isfinite(open_price) or open_price <= 0 or not np.isfinite(close_price):
            continue
        rows.append(
            {
                "timestamp": first_ts,
                "session_close_timestamp": pd.Timestamp(quality["official_close_timestamp"]),
                "open": open_price,
                "high": float(group["high"].astype(float).max()),
                "low": float(group["low"].astype(float).min()),
                "close": close_price,
                "volume": float(group["volume"].astype(float).sum()),
                "source_bar_count": int(quality["observed_bar_count"]),
                "expected_source_bar_count": int(quality["expected_bar_count"]),
                "missing_internal_bar_count": int(quality["missing_internal_bar_count"]),
                "session_data_quality": str(quality["quality"]),
                "intraday_path_drawdown": _session_path_drawdown(group),
            }
        )

    if not rows:
        return pd.DataFrame()

    daily = pd.DataFrame(rows).set_index("timestamp").sort_index()
    close = daily["close"].astype(float)
    open_price = daily["open"].astype(float)
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)
    volume = daily["volume"].astype(float)

    raw_features = pd.DataFrame(index=daily.index)
    daily_return = close.pct_change()
    for period in [1, 3, 5, 10, 20]:
        raw_features[f"return_{period}"] = close.pct_change(period)
    for period in [5, 20]:
        raw_features[f"vol_{period}"] = daily_return.rolling(period).std()

    ema: dict[int, pd.Series] = {}
    for period in [5, 10, 20, 50]:
        ema[period] = close.ewm(span=period, adjust=False).mean()
        raw_features[f"ema_distance_{period}"] = _safe_divide(close, ema[period]) - 1
    raw_features["ema_5_vs_20"] = _safe_divide(ema[5], ema[20]) - 1
    raw_features["ema_20_vs_50"] = _safe_divide(ema[20], ema[50]) - 1
    raw_features["rsi_14"] = _rsi(close) / 100.0

    atr = _true_range(daily).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    raw_features["atr_pct_14"] = _safe_divide(atr, close)
    raw_features["distance_from_high_20"] = _safe_divide(close, high.rolling(20).max()) - 1
    raw_features["distance_from_low_20"] = _safe_divide(close, low.rolling(20).min()) - 1
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    raw_features["volume_zscore_20"] = _safe_divide(volume - volume_mean, volume_std)



    shifted = raw_features.shift(1)
    for column in shifted.columns:
        daily[column] = shifted[column]

    opening_gap = _safe_divide(open_price, close.shift(1)) - 1
    intraday_return = _safe_divide(close, open_price) - 1
    daily["previous_opening_gap"] = opening_gap.shift(1)
    daily["previous_intraday_return"] = intraday_return.shift(1)

    local = daily.index.tz_convert(MARKET_TIMEZONE)
    weekday = local.weekday.to_numpy(dtype=float)
    month = local.month.to_numpy(dtype=float)
    daily["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    daily["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)
    daily["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    daily["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    cost_rate = _training_round_trip_cost_rate(open_price, config)
    gross_multiplier = _safe_divide(close, open_price)
    net_multiplier = gross_multiplier * (1.0 - cost_rate)
    daily["forward_net_log_return"] = np.log(net_multiplier.clip(lower=1e-12))
    daily["forward_downside"] = (1.0 - _safe_divide(low, open_price)).clip(lower=0.0)
    daily["forward_max_drawdown"] = pd.to_numeric(
        daily["intraday_path_drawdown"], errors="coerce"
    ).clip(lower=0.0)
    daily["forward_risk_adjusted_utility"] = (
        daily["forward_net_log_return"]
        - float(config.rotation_downside_penalty) * daily["forward_downside"]
        - float(config.rotation_drawdown_penalty) * daily["forward_max_drawdown"]
    )
    daily["open_to_close_return"] = intraday_return

    required = OPEN_CLOSE_FEATURES + [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session_close_timestamp",
        "forward_risk_adjusted_utility",
    ]
    daily = daily.replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=required)
    return daily


def prepare_open_close_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames = {
        symbol: build_open_close_frame(frame, config)
        for symbol, frame in bars_by_symbol.items()
        if frame is not None and not frame.empty
    }
    if len(frames) < 2:
        raise ValueError("Open-Close Day Trade needs at least two assets with valid session data.")

    common: pd.DatetimeIndex | None = None
    for frame in frames.values():
        index = pd.DatetimeIndex(frame.index)
        common = index if common is None else common.intersection(index)
    if common is None:
        raise ValueError("No common Open-Close sessions are available.")
    common = common.sort_values()
    minimum = int(config.rotation_minimum_training_rows) + int(config.rotation_walk_forward_calibration_days) + 2
    if len(common) < minimum:
        raise ValueError(
            "The common Open-Close history is too short for walk-forward training: "
            f"sessions={len(common)}, required>{minimum}."
        )
    return {symbol: frame.loc[common].copy() for symbol, frame in frames.items()}, common


def _panel_session_quality_summary(
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    counts: dict[str, int] = {
        "COMPLETE": 0,
        "INTERNAL_GAP": 0,
        "MULTIPLE_INTERNAL_GAPS": 0,
    }
    missing_internal_total = 0
    affected_sessions: set[str] = set()
    affected_by_symbol: dict[str, int] = {}

    for symbol, frame in frames.items():
        subset = frame.loc[common_dates]
        symbol_affected = 0
        for timestamp, row in subset.iterrows():
            quality = str(row.get("session_data_quality", "COMPLETE"))
            counts[quality] = counts.get(quality, 0) + 1
            missing = int(row.get("missing_internal_bar_count", 0) or 0)
            missing_internal_total += missing
            if missing > 0:
                symbol_affected += 1
                affected_sessions.add(pd.Timestamp(timestamp).strftime("%Y-%m-%d"))
        affected_by_symbol[symbol] = symbol_affected

    return {
        "policy": "official session closed + official open bar present + official final bar present; internal gaps retained with quality metadata",
        "common_session_count": int(len(common_dates)),
        "asset_session_quality_counts": counts,
        "sessions_with_any_internal_gap": int(len(affected_sessions)),
        "missing_internal_bars_total": int(missing_internal_total),
        "sessions_with_internal_gap_by_symbol": affected_by_symbol,
    }


def _build_folds(common_dates: pd.DatetimeIndex, config: Any) -> list[dict[str, Any]]:
    purge = max(1, int(config.rotation_purge_days))
    calibration = int(config.rotation_walk_forward_calibration_days)
    test = int(config.rotation_walk_forward_test_days)
    min_test = int(config.rotation_walk_forward_min_test_days)
    min_train = int(config.rotation_minimum_training_rows)

    first_test = min_train + purge + calibration + purge
    if first_test >= len(common_dates) - min_test:
        raise ValueError(
            "Not enough Open-Close sessions for expanding walk-forward: "
            f"sessions={len(common_dates)}, minimum_train={min_train}, "
            f"calibration={calibration}, purge={purge}, minimum_test={min_test}."
        )

    folds: list[dict[str, Any]] = []
    test_start = first_test
    fold_id = 1
    while test_start < len(common_dates):
        test_end = min(len(common_dates), test_start + test)
        if test_end - test_start < min_test:
            if folds:
                folds[-1]["test_end_index"] = len(common_dates)
                folds[-1]["test_end"] = common_dates[-1]
                folds[-1]["decision_dates"] = common_dates[
                    folds[-1]["test_start_index"] : len(common_dates)
                ]
            break
        calibration_end = test_start - purge
        calibration_start = calibration_end - calibration
        train_end = calibration_start - purge
        if train_end < min_train:
            raise ValueError(f"Fold {fold_id}: training sessions {train_end} < {min_train}.")
        folds.append(
            {
                "fold_id": fold_id,
                "train_end_index": train_end,
                "calibration_start_index": calibration_start,
                "calibration_end_index": calibration_end,
                "final_fit_end_index": test_start - purge,
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
                "decision_dates": common_dates[test_start:test_end],
            }
        )
        fold_id += 1
        test_start = test_end
    if not folds:
        raise ValueError("No valid Open-Close walk-forward fold was created.")
    return folds


def _normalization(
    frames: dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    result: dict[str, tuple[pd.Series, pd.Series]] = {}
    for symbol, frame in frames.items():
        sample = frame.loc[train_dates, OPEN_CLOSE_FEATURES]
        mean = sample.mean()
        std = sample.std().replace(0, 1.0).fillna(1.0)
        result[symbol] = (mean, std)
    return result


def _feature_matrix(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for symbol in symbols:
        mean, std = normalization[symbol]
        values = ((frames[symbol].loc[dates, OPEN_CLOSE_FEATURES] - mean) / std).clip(-8, 8)
        matrices.append(values.to_numpy(dtype=np.float32))
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)


def _utility_matrix(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    cash = np.zeros((len(dates), 1), dtype=np.float32)
    assets = np.column_stack(
        [
            frames[symbol].loc[dates, "forward_risk_adjusted_utility"].to_numpy(dtype=np.float32)
            for symbol in symbols
        ]
    )
    return np.concatenate([cash, assets], axis=1)


def _resolve_qrdqn_plan(config: Any) -> ComputePlan:
    requested = str(getattr(config, "rotation_accelerator", "auto")).lower()
    allow_fallback = bool(getattr(config, "rotation_allow_cpu_fallback", True))
    version = None
    cuda_available = False
    gpu_name = None
    error = None
    try:
        import torch
        version = str(torch.__version__)
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
    fallback = requested == "cuda" and selected != "cuda"
    reason = None
    if fallback:
        reason = error or "PyTorch CUDA is not available"
        if not allow_fallback:
            raise RuntimeError(reason)
    return ComputePlan(requested, selected, cuda_available, gpu_name, version, fallback, reason)


def _resolve_xgb_plan(config: Any) -> ComputePlan:
    requested = str(getattr(config, "rotation_accelerator", "auto")).lower()
    allow_fallback = bool(getattr(config, "rotation_allow_cpu_fallback", True))
    version = None
    cuda_available = False
    gpu_name = None
    error = None
    try:
        import xgboost as xgb
        version = str(xgb.__version__)
        info = xgb.build_info()
        cuda_build = str(info.get("USE_CUDA", "false")).lower() in {"1", "true", "yes", "on"}
        if cuda_build:
            try:
                import subprocess
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if out.returncode == 0 and out.stdout.strip():
                    gpu_name = out.stdout.splitlines()[0].strip()
                    cuda_available = True
            except Exception:
                pass
    except Exception as exc:
        error = str(exc)
    if requested == "cpu":
        selected = "cpu"
    elif cuda_available:
        selected = "cuda"
    else:
        selected = "cpu"
    fallback = requested == "cuda" and selected != "cuda"
    reason = None
    if fallback:
        reason = error or "XGBoost CUDA is not available"
        if not allow_fallback:
            raise RuntimeError(reason)
    return ComputePlan(requested, selected, cuda_available, gpu_name, version, fallback, reason)


class _QRNetwork:
    def __init__(self, input_dim: int, action_count: int, quantile_count: int, hidden_dim: int, device: str) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("QR-DQN requires PyTorch.") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.action_count = action_count
        self.quantile_count = quantile_count
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count * quantile_count),
        ).to(self.device)

    def quantiles(self, x):
        return self.model(x).reshape(-1, self.action_count, self.quantile_count)


class _ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, state: np.ndarray, action: int, reward: float) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=int(batch_size))
        return self.states[idx], self.actions[idx], self.rewards[idx]


def _q_snapshot(network: _QRNetwork, state: np.ndarray) -> tuple[int, float, np.ndarray]:
    torch = network.torch
    with torch.no_grad():
        tensor = torch.as_tensor(state[None, :], dtype=torch.float32, device=network.device)
        means = network.quantiles(tensor)[0].mean(dim=1)
        action = int(torch.argmax(means).item())
        values = means.detach().cpu().numpy().astype(float)
        return action, float(values[action]), values


def _train_qrdqn_bandit(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    device: str,
    seed: int,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[_QRNetwork, dict[str, Any]]:
    import torch

    rng = np.random.default_rng(int(seed))
    train_x = _feature_matrix(frames, symbols, train_dates, normalization)
    train_u = _utility_matrix(frames, symbols, train_dates)
    cal_x = _feature_matrix(frames, symbols, calibration_dates, normalization)
    cal_u = _utility_matrix(frames, symbols, calibration_dates)

    online = _QRNetwork(
        train_x.shape[1],
        len(symbols) + 1,
        int(config.qrdqn_n_quantiles),
        int(config.qrdqn_hidden_dim),
        device,
    )
    generator = torch.Generator(device=online.device)
    generator.manual_seed(int(seed))
    for module in online.model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(module.bias, -bound, bound, generator=generator)

    optimizer = torch.optim.Adam(online.model.parameters(), lr=float(config.qrdqn_learning_rate))
    buffer = _ReplayBuffer(int(config.qrdqn_replay_size), train_x.shape[1])
    q_count = int(config.qrdqn_n_quantiles)
    tau = ((torch.arange(q_count, device=online.device, dtype=torch.float32) + 0.5) / q_count).view(1, q_count, 1)

    total_steps = int(config.qrdqn_training_steps)
    min_steps = int(config.qrdqn_min_training_steps)
    eval_every = max(250, int(config.qrdqn_eval_every_steps))
    best_score = float("-inf")
    best_state = deepcopy(online.model.state_dict())
    best_step = 0
    eligible = 0
    ignored = 0
    progress_interval = max(1, total_steps // 20)

    if progress_callback is not None:
        progress_callback(0.0)

    for step in range(total_steps):
        row = int(rng.integers(0, len(train_dates)))
        state = train_x[row]
        fraction = step / max(1, total_steps - 1)
        epsilon = float(config.qrdqn_epsilon_start) + fraction * (
            float(config.qrdqn_epsilon_end) - float(config.qrdqn_epsilon_start)
        )
        if rng.random() < epsilon:
            action = int(rng.integers(0, len(symbols) + 1))
        else:
            action, _, _ = _q_snapshot(online, state)
        reward = float(train_u[row, action])
        buffer.add(state, action, reward)

        if buffer.size >= int(config.qrdqn_learning_starts):
            states, actions, rewards = buffer.sample(int(config.qrdqn_batch_size), rng)
            states_t = torch.as_tensor(states, dtype=torch.float32, device=online.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=online.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=online.device)
            all_quantiles = online.quantiles(states_t)
            batch_index = torch.arange(all_quantiles.shape[0], device=online.device)
            current = all_quantiles[batch_index, actions_t, :]
            target = rewards_t[:, None].expand(-1, q_count)
            td = target[:, None, :] - current[:, :, None]
            abs_td = td.abs()
            huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
            quantile_weight = torch.abs(tau - (td.detach() < 0).float())
            loss = (quantile_weight * huber).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.model.parameters(), max_norm=10.0)
            optimizer.step()

        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                states_t = torch.as_tensor(cal_x, dtype=torch.float32, device=online.device)
                means = online.quantiles(states_t).mean(dim=2)
                actions = torch.argmax(means, dim=1).detach().cpu().numpy()
            score = float(cal_u[np.arange(len(cal_u)), actions].sum())
            if step + 1 >= min_steps:
                eligible += 1
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = deepcopy(online.model.state_dict())
                    best_step = step + 1
            else:
                ignored += 1

        if progress_callback is not None and ((step + 1) % progress_interval == 0 or step + 1 == total_steps):
            progress_callback((step + 1) / max(1, total_steps))

    if best_step == 0:
        best_state = deepcopy(online.model.state_dict())
        best_step = total_steps
        with torch.no_grad():
            states_t = torch.as_tensor(cal_x, dtype=torch.float32, device=online.device)
            means = online.quantiles(states_t).mean(dim=2)
            actions = torch.argmax(means, dim=1).detach().cpu().numpy()
        best_score = float(cal_u[np.arange(len(cal_u)), actions].sum())

    online.model.load_state_dict(best_state)
    online.model.eval()
    if progress_callback is not None:
        progress_callback(1.0)
    return online, {
        "seed": int(seed),
        "requested_steps": total_steps,
        "steps_used": total_steps,
        "best_step": int(best_step),
        "best_validation_score": float(best_score),
        "stopped_early": False,
        "early_stopping_enabled": False,
        "minimum_training_steps": min_steps,
        "checkpoint_selection_start_step": min_steps,
        "eligible_validation_evals": int(eligible),
        "ignored_pre_min_validation_evals": int(ignored),
        "learning_problem": "distributional_contextual_bandit",
        "effective_gamma": 0.0,
    }


def _fit_xgb_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    config: Any,
    device: str,
) -> tuple[dict[str, Any], str, str | None]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost Utility requires xgboost.") from exc

    allow_fallback = bool(getattr(config, "rotation_allow_cpu_fallback", True))

    def fit(effective: str) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for symbol in symbols:
            sample = frames[symbol].loc[dates].dropna(subset=["forward_risk_adjusted_utility"])
            if len(sample) < int(config.rotation_minimum_training_rows):
                raise ValueError(
                    f"{symbol}: only {len(sample)} Open-Close rows are available; "
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
                device=effective,
            )
            model.fit(sample[OPEN_CLOSE_FEATURES], sample["forward_risk_adjusted_utility"])
            models[symbol] = model
        return models

    try:
        return fit(device), device, None
    except Exception as exc:
        if device != "cuda" or not allow_fallback:
            raise
        reason = f"XGBoost CUDA training failed; using CPU: {exc}"
        return fit("cpu"), "cpu", reason


def _xgb_snapshot(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
) -> tuple[int, float, np.ndarray]:
    values = [0.0]
    for symbol in symbols:
        row = frames[symbol].loc[[timestamp], OPEN_CLOSE_FEATURES]
        values.append(float(models[symbol].predict(row)[0]))
    utilities = np.asarray(values, dtype=float)
    best = int(np.nanargmax(utilities))
    threshold = max(float(config.rotation_cash_threshold), float(config.rotation_min_expected_edge))
    if best == 0 or float(utilities[best]) <= threshold:
        return 0, 0.0, utilities
    return best, float(utilities[best]), utilities


def _maximum_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def _annualized_sharpe(curve: pd.Series) -> float:
    returns = curve.pct_change().dropna()
    if returns.empty or float(returns.std()) <= 0:
        return float("nan")
    return float(np.sqrt(252.0) * returns.mean() / returns.std())


def _cagr(curve: pd.Series) -> float:
    if len(curve) < 2 or float(curve.iloc[0]) <= 0 or float(curve.iloc[-1]) <= 0:
        return float("nan")
    years = max((pd.Timestamp(curve.index[-1]) - pd.Timestamp(curve.index[0])).days / 365.25, 1 / 365.25)
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)


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
        next_quantity = max(0.0, (cash - float(fees["total_fee"])) / execution_price)
        if not bool(config.fractional_shares):
            next_quantity = float(math.floor(next_quantity))
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity
    fees = fee_calculator("BUY", quantity, execution_price, config)
    return float(quantity), execution_price, fees


def _equal_weight_open_close_benchmark(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    initial_capital: float,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
) -> pd.Series:
    capital = float(initial_capital)
    values: list[float] = []
    timestamps: list[pd.Timestamp] = []
    for date in dates:
        per_asset = capital / len(symbols)
        ending = 0.0
        close_ts = pd.Timestamp(frames[symbols[0]].loc[date, "session_close_timestamp"])
        for symbol in symbols:
            buy_raw = float(frames[symbol].loc[date, "open"])
            quantity, buy_price, buy_fees = _execute_buy(per_asset, buy_raw, config, fee_calculator, slippage)
            residual = per_asset - quantity * buy_price - float(buy_fees["total_fee"])
            sell_price = float(slippage(float(frames[symbol].loc[date, "close"]), "SELL", config))
            sell_fees = fee_calculator("SELL", quantity, sell_price, config)
            ending += residual + quantity * sell_price - float(sell_fees["total_fee"])
        capital = ending
        values.append(capital)
        timestamps.append(close_ts)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)


def _equal_weight_buy_hold_reference(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    initial_capital: float,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
) -> pd.Series:
    first = dates[0]
    per_asset = float(initial_capital) / len(symbols)
    quantities: dict[str, float] = {}
    residual = 0.0
    for symbol in symbols:
        quantity, price, fees = _execute_buy(
            per_asset,
            float(frames[symbol].loc[first, "open"]),
            config,
            fee_calculator,
            slippage,
        )
        quantities[symbol] = quantity
        residual += per_asset - quantity * price - float(fees["total_fee"])

    values: list[float] = []
    timestamps: list[pd.Timestamp] = []
    for date in dates:
        equity = residual + sum(
            quantity * float(frames[symbol].loc[date, "close"])
            for symbol, quantity in quantities.items()
        )
        values.append(equity)
        timestamps.append(pd.Timestamp(frames[symbols[0]].loc[date, "session_close_timestamp"]))
    last = dates[-1]
    final_cash = residual
    for symbol, quantity in quantities.items():
        sell = float(slippage(float(frames[symbol].loc[last, "close"]), "SELL", config))
        fees = fee_calculator("SELL", quantity, sell, config)
        final_cash += quantity * sell - float(fees["total_fee"])
    values[-1] = final_cash
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)


def _fold_performance(predictions: pd.DataFrame, folds: list[dict[str, Any]], initial_capital: float) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    rows = predictions.reset_index(drop=False)
    output: list[dict[str, Any]] = []
    previous_strategy = float(initial_capital)
    previous_benchmark = float(initial_capital)
    for fold in folds:
        fold_id = int(fold["fold_id"])
        subset = rows.loc[rows["walk_forward_fold"] == fold_id]
        if subset.empty:
            continue
        strategy_end = float(subset.iloc[-1]["strategy_equity"])
        benchmark_end = float(subset.iloc[-1]["buy_hold_equity"])
        curve = pd.Series([previous_strategy, *subset["strategy_equity"].astype(float).tolist()])
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
                "strategy_starting_capital": previous_strategy,
                "strategy_ending_capital": strategy_end,
                "strategy_return": strategy_end / previous_strategy - 1,
                "benchmark_return": benchmark_end / previous_benchmark - 1,
                "excess_return": strategy_end / previous_strategy - benchmark_end / previous_benchmark,
                "maximum_drawdown": _maximum_drawdown(curve),
                "sessions": int(len(subset)),
            }
        )
        previous_strategy = strategy_end
        previous_benchmark = benchmark_end
    return output


def _simulate(
    backend: str,
    policies: dict[int, Callable[[pd.Timestamp], tuple[int, float, np.ndarray]]],
    folds: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    trade_callback: Callable[[dict[str, Any]], None] | None,
) -> OpenCloseRunResult:
    decision_to_fold: dict[pd.Timestamp, int] = {}
    dates: list[pd.Timestamp] = []
    for fold in folds:
        for date in fold["decision_dates"]:
            key = pd.Timestamp(date)
            decision_to_fold[key] = int(fold["fold_id"])
            dates.append(key)
    decision_dates = pd.DatetimeIndex(sorted(set(dates)))

    benchmark = _equal_weight_open_close_benchmark(
        frames, symbols, decision_dates, float(config.initial_capital), config, fee_calculator, slippage
    )
    buy_hold_reference = _equal_weight_buy_hold_reference(
        frames, symbols, decision_dates, float(config.initial_capital), config, fee_calculator, slippage
    )

    cash = float(config.initial_capital)
    total_fees = 0.0
    turnover = 0.0
    records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    invested_sessions = 0
    winning_sessions = 0
    realized_returns: list[float] = []

    for date in decision_dates:
        fold_id = decision_to_fold[pd.Timestamp(date)]
        action, score, values = policies[fold_id](pd.Timestamp(date))
        labels = ["CASH", *symbols]
        close_ts = pd.Timestamp(frames[symbols[0]].loc[date, "session_close_timestamp"])
        starting_cash = cash
        selected_asset = "CASH"
        trade_action = "CASH"
        session_return = 0.0
        execution_price = None

        if action > 0:
            invested_sessions += 1
            symbol = symbols[action - 1]
            selected_asset = symbol
            quantity, buy_price, buy_fees = _execute_buy(
                cash,
                float(frames[symbol].loc[date, "open"]),
                config,
                fee_calculator,
                slippage,
            )
            gross_buy = quantity * buy_price
            cash_after_buy = cash - gross_buy - float(buy_fees["total_fee"])
            total_fees += float(buy_fees["total_fee"])
            turnover += gross_buy
            buy_record = {
                "timestamp": pd.Timestamp(date),
                "action": "BUY",
                "asset": symbol,
                "reason": "OPEN_CLOSE_DAILY_SELECTION",
                "execution_price": buy_price,
                "quantity": quantity,
                "gross_trade_value": gross_buy,
                **buy_fees,
                "realized_pnl": 0.0,
                "position_return": 0.0,
                "holding_bars": 0,
                "holding_sessions": 0,
                "entry_timestamp": pd.Timestamp(date),
                "entry_price": buy_price,
                "cash_after_trade": cash_after_buy,
                "shares_after_trade": quantity,
                "walk_forward_fold": fold_id,
                "decision_timestamp": pd.Timestamp(date),
            }
            records.append(buy_record)
            if trade_callback is not None:
                trade_callback({**buy_record, "backend": backend, "model": "XGBoost Utility" if backend == "xgboost_utility" else "QR-DQN"})

            sell_price = float(slippage(float(frames[symbol].loc[date, "close"]), "SELL", config))
            sell_fees = fee_calculator("SELL", quantity, sell_price, config)
            gross_sell = quantity * sell_price
            realized = quantity * (sell_price - buy_price) - float(buy_fees["total_fee"]) - float(sell_fees["total_fee"])
            cash = cash_after_buy + gross_sell - float(sell_fees["total_fee"])
            total_fees += float(sell_fees["total_fee"])
            turnover += gross_sell
            session_return = cash / starting_cash - 1 if starting_cash > 0 else 0.0
            realized_returns.append(session_return)
            if session_return > 0:
                winning_sessions += 1
            execution_price = sell_price
            trade_action = "OPEN_TO_CLOSE"
            sell_record = {
                "timestamp": close_ts,
                "action": "SELL",
                "asset": symbol,
                "reason": "DAY_TRADE_CLOSE",
                "execution_price": sell_price,
                "quantity": quantity,
                "gross_trade_value": gross_sell,
                **sell_fees,
                "realized_pnl": realized,
                "position_return": sell_price / buy_price - 1 if buy_price > 0 else 0.0,
                "holding_bars": int(frames[symbol].loc[date, "source_bar_count"]),
                "holding_sessions": 1,
                "entry_timestamp": pd.Timestamp(date),
                "entry_price": buy_price,
                "cash_after_trade": cash,
                "shares_after_trade": 0.0,
                "walk_forward_fold": fold_id,
                "decision_timestamp": pd.Timestamp(date),
            }
            records.append(sell_record)
            if trade_callback is not None:
                trade_callback({**sell_record, "backend": backend, "model": "XGBoost Utility" if backend == "xgboost_utility" else "QR-DQN"})

        sorted_values = np.sort(values)
        second_best = float(sorted_values[-2]) if len(sorted_values) > 1 else float(sorted_values[-1])
        row: dict[str, Any] = {
            "timestamp": close_ts,
            "decision_date": pd.Timestamp(date),
            "strategy_equity": cash,
            "buy_hold_equity": float(benchmark.loc[close_ts]),
            "reference_buy_hold_equity": float(buy_hold_reference.loc[close_ts]),
            "trade_action": trade_action,
            "trade_reason": "OPEN_CLOSE_DAILY_SELECTION" if action > 0 else "MODEL_SELECTED_CASH",
            "execution_price": execution_price,
            "selected_asset": selected_asset,
            "previous_asset": "CASH",
            "decision_score": float(score),
            "walk_forward_fold": fold_id,
            "fold_test_start": next(f["test_start"] for f in folds if int(f["fold_id"]) == fold_id),
            "fold_test_end": next(f["test_end"] for f in folds if int(f["fold_id"]) == fold_id),
            "session_return": float(session_return),
            "q_gap_best_vs_second": float(values[action] - second_best) if action < len(values) else None,
        }
        for i, label in enumerate(labels):
            safe = str(label).replace("-", "_").replace(".", "_")
            row[f"decision_value_{safe}"] = float(values[i])
        prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows).set_index("timestamp")
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    predictions.index.name = "timestamp"
    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        trades = trades.sort_values("timestamp").reset_index(drop=True)

    strategy_curve = predictions["strategy_equity"].astype(float)
    benchmark_curve = predictions["buy_hold_equity"].astype(float)
    reference_curve = predictions["reference_buy_hold_equity"].astype(float)
    initial = float(config.initial_capital)
    ending = float(strategy_curve.iloc[-1])
    benchmark_ending = float(benchmark_curve.iloc[-1])
    reference_ending = float(reference_curve.iloc[-1])
    years = max((strategy_curve.index[-1] - strategy_curve.index[0]).days / 365.25, 1 / 365.25)
    cash_sessions = len(decision_dates) - invested_sessions

    metrics: dict[str, Any] = {
        "portfolio_rotation": True,
        "strategy_mode": config.strategy_mode,
        "strategy_label": "XGBoost Utility" if backend == "xgboost_utility" else "QR-DQN",
        "symbol": "PORTFOLIO",
        "backend": backend,
        "model_family": backend,
        "assets": symbols,
        "timeframe": "15Min",
        "source_timeframe": "15Min",
        "decision_frequency": "1 decision per trading session",
        "entry_rule": "Pre-open selection from completed prior-session data; execution at regular-session open",
        "exit_rule": "Regular-session close / Market-On-Close equivalent",
        "decision_horizon_days": 1,
        "decision_horizon_bars": None,
        "decision_horizon_label": "Open → Close (same trading session)",
        "overnight_positions_allowed": False,
        "intraday_rotations_allowed": False,
        "maximum_entries_per_session": 1,
        "maximum_exits_per_session": 1,
        "benchmark_name": "Equal-weight open-to-close",
        "reference_benchmark_name": "Equal-weight buy-and-hold",
        "walk_forward_enabled": True,
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
        "excess_return": ending / initial - benchmark_ending / initial,
        "reference_buy_hold_ending_capital": reference_ending,
        "reference_buy_hold_return": reference_ending / initial - 1,
        "strategy_maximum_drawdown": _maximum_drawdown(strategy_curve),
        "buy_hold_maximum_drawdown": _maximum_drawdown(benchmark_curve),
        "reference_buy_hold_maximum_drawdown": _maximum_drawdown(reference_curve),
        "strategy_sharpe": _annualized_sharpe(strategy_curve),
        "buy_hold_sharpe": _annualized_sharpe(benchmark_curve),
        "reference_buy_hold_sharpe": _annualized_sharpe(reference_curve),
        "strategy_cagr": _cagr(strategy_curve),
        "buy_hold_cagr": _cagr(benchmark_curve),
        "reference_buy_hold_cagr": _cagr(reference_curve),
        "compound_log_growth": float(math.log(max(ending / initial, 1e-12))),
        "risk_adjusted_compound_score": float(np.log(strategy_curve / strategy_curve.shift(1)).dropna().sum()),
        "market_exposure": invested_sessions / max(1, len(decision_dates)),
        "cash_days": int(cash_sessions),
        "invested_sessions": int(invested_sessions),
        "simulated_buys": int(invested_sessions),
        "simulated_sells": int(invested_sessions),
        "capital_rotations": 0,
        "cycles_per_year": float(invested_sessions / years),
        "average_holding_days": 0.0,
        "average_holding_bars": 26.0,
        "average_holding_minutes": 390.0,
        "geometric_trade_return": float(np.prod(1.0 + np.asarray(realized_returns)) ** (1 / len(realized_returns)) - 1) if realized_returns else float("nan"),
        "winning_sessions": int(winning_sessions),
        "losing_sessions": int(invested_sessions - winning_sessions),
        "session_win_rate": winning_sessions / max(1, invested_sessions),
        "average_invested_session_return": float(np.mean(realized_returns)) if realized_returns else 0.0,
        "total_transaction_fees": float(total_fees),
        "turnover_ratio": float(turnover / max(initial, 1e-9)),
        "test_start": decision_dates[0],
        "test_end": decision_dates[-1],
        "test_calendar_years": years,
    }

    summary = "\n".join(
        [
            "COMPOUND CAPITAL ROTATION — DAY TRADE OPEN→CLOSE",
            "",
            f"Model: {metrics['strategy_label']}",
            f"Assets: {', '.join(symbols)}",
            f"Market-data source: {str(getattr(config, 'market_data_provider', 'alpaca')).upper()} 15-minute bars aggregated into one session decision row",
            "Decision timing: once per session before the regular-session open",
            "Execution rule: at most one BUY at the regular-session open and one SELL at the same-session close",
            "Intraday rotations: prohibited",
            "Overnight exposure: prohibited",
            "Look-ahead guard: all model price/volume inputs come from completed prior sessions; the current opening print is reserved for execution only",
            "Session boundary guard: official close must have passed and official open/final 15-minute bars must exist; internal historical gaps are retained with quality metadata",
            "",
            "OUT-OF-SAMPLE WALK-FORWARD",
            f"Initial capital: ${initial:,.2f}",
            f"Ending capital: ${ending:,.2f}",
            f"Total return: {metrics['strategy_return']:.2%}",
            f"Equal-weight Open→Close: {metrics['buy_hold_return']:.2%}",
            f"Equal-weight Buy & Hold reference: {metrics['reference_buy_hold_return']:.2%}",
            f"Maximum drawdown: {metrics['strategy_maximum_drawdown']:.2%}",
            f"Sharpe: {metrics['strategy_sharpe']:.3f}",
            f"Invested sessions: {invested_sessions}/{len(decision_dates)}",
            f"Session win rate: {metrics['session_win_rate']:.2%}",
            f"Total simulated fees: ${total_fees:,.2f}",
        ]
    )
    return OpenCloseRunResult(backend, predictions, trades, summary, metrics)


def run_open_close_models(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[OpenCloseRunResult]:
    frames, common_dates = prepare_open_close_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    session_quality = _panel_session_quality_summary(frames, common_dates)
    folds = _build_folds(common_dates, config)
    selected = [model for model in ("xgboost_utility", "qrdqn") if model in config.rotation_models]
    if not selected:
        raise ValueError("Select at least one Open-Close model.")

    xgb_plan = _resolve_xgb_plan(config)
    qr_plan = _resolve_qrdqn_plan(config)
    if progress_callback is not None:
        progress_callback(
            18.0,
            f"Prepared {len(symbols)} assets and {len(folds)} Open-Close folds — one decision/session",
            0,
        )

    results: list[OpenCloseRunResult] = []
    total_work = max(1, len(selected) * len(folds))
    work_done = 0

    if "xgboost_utility" in selected:
        policies: dict[int, Callable[[pd.Timestamp], tuple[int, float, np.ndarray]]] = {}
        fold_details: dict[int, dict[str, Any]] = {}
        effective_device = xgb_plan.selected
        fallback_reasons: list[str] = []
        for fold in folds:
            fold_id = int(fold["fold_id"])
            fit_dates = common_dates[: int(fold["final_fit_end_index"])]
            models, effective_device, fallback = _fit_xgb_models(frames, symbols, fit_dates, config, effective_device)
            if fallback:
                fallback_reasons.append(fallback)
            policies[fold_id] = lambda timestamp, m=models: _xgb_snapshot(m, frames, symbols, timestamp, config)
            calibration_dates = common_dates[int(fold["calibration_start_index"]): int(fold["calibration_end_index"])]
            calibration_scores = []
            for date in calibration_dates:
                action, _, _ = _xgb_snapshot(models, frames, symbols, date, config)
                calibration_scores.append(0.0 if action == 0 else float(frames[symbols[action - 1]].loc[date, "forward_risk_adjusted_utility"]))
            fold_details[fold_id] = {
                "validation_score": float(np.sum(calibration_scores)),
                "effective_compute_device": effective_device,
            }
            work_done += 1
            if progress_callback is not None:
                progress_callback(20.0 + 70.0 * work_done / total_work, f"XGBoost Open-Close fold {fold_id}/{len(folds)}", 0)
        result = _simulate("xgboost_utility", policies, folds, frames, symbols, config, fee_calculator, slippage, trade_callback)
        result.metrics.update(
            {
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _fold_performance(result.predictions, folds, float(config.initial_capital)),
                "requested_accelerator": xgb_plan.requested,
                "effective_compute_device": effective_device,
                "cuda_available": xgb_plan.cuda_available,
                "gpu_name": xgb_plan.gpu_name,
                "framework_version": xgb_plan.framework_version,
                "cpu_fallback_used": bool(fallback_reasons) or xgb_plan.fallback_used,
                "compute_fallback_reason": "; ".join(dict.fromkeys(fallback_reasons)) or xgb_plan.fallback_reason,
                "open_close_feature_count": len(OPEN_CLOSE_FEATURES),
                "lookahead_guard": "all model price/volume inputs come from completed prior sessions; entry executes at current session open",
                "session_boundary_guard": session_quality,
            }
        )
        results.append(result)

    if "qrdqn" in selected:
        seed = int(config.random_state)
        policies: dict[int, Callable[[pd.Timestamp], tuple[int, float, np.ndarray]]] = {}
        training_details: dict[int, dict[str, Any]] = {}
        for fold in folds:
            fold_id = int(fold["fold_id"])
            train_dates = common_dates[: int(fold["train_end_index"])]
            calibration_dates = common_dates[int(fold["calibration_start_index"]): int(fold["calibration_end_index"])]
            normalization = _normalization(frames, train_dates)

            def update_fraction(fraction: float, fid=fold_id):
                if progress_callback is not None:
                    base = work_done / total_work
                    partial = fraction / total_work
                    progress_callback(20.0 + 70.0 * (base + partial), f"QR-DQN Open-Close fold {fid}/{len(folds)}", 0)

            network, diag = _train_qrdqn_bandit(
                frames,
                symbols,
                train_dates,
                calibration_dates,
                normalization,
                config,
                qr_plan.selected,
                seed + fold_id,
                progress_callback=update_fraction,
            )
            policies[fold_id] = lambda timestamp, n=network, norm=normalization: (
                lambda snap: snap
            )(_q_snapshot(n, _feature_matrix(frames, symbols, pd.DatetimeIndex([timestamp]), norm)[0]))
            training_details[fold_id] = diag
            work_done += 1
            if progress_callback is not None:
                progress_callback(20.0 + 70.0 * work_done / total_work, f"QR-DQN Open-Close fold {fold_id}/{len(folds)} completed", 0)

        result = _simulate("qrdqn", policies, folds, frames, symbols, config, fee_calculator, slippage, trade_callback)
        fold_metrics = _fold_performance(result.predictions, folds, float(config.initial_capital))
        for item in fold_metrics:
            item.update(training_details.get(int(item["fold_id"]), {}))
        best_steps = [int(v.get("best_step", 0)) for v in training_details.values()]
        result.metrics.update(
            {
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": fold_metrics,
                "requested_accelerator": qr_plan.requested,
                "effective_compute_device": qr_plan.selected,
                "cuda_available": qr_plan.cuda_available,
                "gpu_name": qr_plan.gpu_name,
                "framework_version": qr_plan.framework_version,
                "cpu_fallback_used": qr_plan.fallback_used,
                "compute_fallback_reason": qr_plan.fallback_reason,
                "qrdqn_training_steps_requested": int(config.qrdqn_training_steps),
                "qrdqn_training_steps_mean_used": float(config.qrdqn_training_steps),
                "qrdqn_min_training_steps": int(config.qrdqn_min_training_steps),
                "qrdqn_best_step_min": int(min(best_steps)) if best_steps else None,
                "qrdqn_best_step_max": int(max(best_steps)) if best_steps else None,
                "qrdqn_early_stopped_folds": 0,
                "qrdqn_learning_problem": "distributional_contextual_bandit",
                "qrdqn_effective_gamma": 0.0,
                "open_close_feature_count": len(OPEN_CLOSE_FEATURES),
                "lookahead_guard": "all model price/volume inputs come from completed prior sessions; entry executes at current session open",
                "session_boundary_guard": session_quality,
            }
        )
        result.summary += "\n\nQR-DQN OPEN-CLOSE DESIGN\n"
        result.summary += "Learning problem: distributional contextual bandit (one action/session)\n"
        result.summary += "Effective gamma: 0.0 because every position is flat by the session close\n"
        result.summary += f"Minimum eligible checkpoint step: {int(config.qrdqn_min_training_steps)}\n"
        results.append(result)

    order = {"xgboost_utility": 0, "qrdqn": 1}
    results.sort(key=lambda item: order.get(item.backend, 99))
    return results
