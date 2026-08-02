"""
Multi-asset local-extrema research backtest.

IMPORTANT
---------
- No real or paper orders are sent.
- Swing market data can be loaded from Alpaca or Yahoo Finance according to configuration.
- Day Trade Open→Close supports Alpaca (default) or Yahoo Finance 15-minute source bars and aggregates them to one session-level decision.
- No Alpaca trading orders are sent in Feature 9; all BUY/SELL actions remain local simulations.
- Every BUY and SELL is simulated locally.
- Each new BUY reinvests 100% of the available cash after previous profits/losses.

What changed from the first experiment
--------------------------------------
1. Multiple assets are evaluated with the same methodology.
2. BOTTOM and TOP are separate binary classification problems.
3. Thresholds are calibrated on a validation period that is separate from the final test.
4. The final test uses walk-forward retraining.
5. Strategy modes isolate Fibonacci, TOP-reversal, structural-trend, and hybrid exits.
6. Naive rolling-low/rolling-high event baselines are reported.
7. The exact-price/return regressor is intentionally removed from the decision logic.
8. Configuration, jobs, predictions, trades, metrics, comparisons, and failures are stored in MongoDB.

Install
-------
    python -m pip install -r requirements.txt

Run
---
    python multi_asset_extrema_backtest.py --job-id <job_id>

The FastAPI application creates the job in MongoDB and starts this engine.
No .env file or filesystem result directory is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol


def _configure_console_utf8() -> None:
    """Keep engine stdout/stderr aligned with the UTF-8 subprocess reader.

    Windows can default redirected console streams to cp1252.  The engine emits
    Unicode labels (for example ``Open→Close``), so explicitly reconfigure the
    text streams before the first progress/log line is written.
    """

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Some embedded/captured streams cannot be reconfigured.  The
                # API launcher also sets PYTHONIOENCODING=utf-8 as a fallback.
                pass


_configure_console_utf8()

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight



BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from alpaca_market_data import download_stock_bars as download_alpaca_stock_bars  # noqa: E402

from mongo_repository import (  # noqa: E402
    JOBS_COLLECTION,
    ALPACA_MARKET_BARS_COLLECTION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
    get_alpaca_credentials,
    replace_comparison,
    replace_run_result,
    utc_now,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLASSIC_FIBONACCI_RATIOS = (
    1.272,
    1.414,
    1.618,
    2.000,
    2.618,
    3.618,
    4.236,
)


SWING_TIMEFRAME_RULES = {
    "1Week": "W-FRI",
    "2Weeks": "2W-FRI",
    "3Weeks": "3W-FRI",
    "4Weeks": "4W-FRI",
}

MINIMUM_BARS_BY_TIMEFRAME = {
    "1Day": 800,
    "1Hour": 800,
    "30Min": 800,
    "15Min": 800,
    "5Min": 800,
    "1Week": 800,
    "2Weeks": 400,
    "3Weeks": 270,
    "4Weeks": 200,
}


def normalize_classic_fibonacci_ratio(value: Any) -> float:
    """Validate and normalize one supported Fibonacci extension ratio."""
    normalized = round(float(value), 3)
    if normalized not in CLASSIC_FIBONACCI_RATIOS:
        allowed = ", ".join(
            f"{ratio:.3f}"
            for ratio in CLASSIC_FIBONACCI_RATIOS
        )
        raise ValueError(
            "FIBONACCI_TARGET_RATIO must be one of the classic "
            f"extensions: {allowed}."
        )
    return normalized


@dataclass(frozen=True)
class BacktestConfig:
    assets: tuple[str, ...] = (
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
        "JPM",
        "SPY",
    )
    start_date: str = "2000-01-01"
    end_date: str | None = None
    timeframe: str = "1Day"
    market_data_provider: str = "alpaca"
    alpaca_feed: str = "iex"
    alpaca_adjustment: str = "all"

    # Model implementations to run: histgb and/or xgboost.
    model_backends: tuple[str, ...] = ("histgb",)
    parameter_mode: str = "general"
    asset_overrides: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    strategy_mode: str = "COMPOUND_ROTATION_SWING_XGBOOST"

    # Label definition.
    future_horizon: int = 5
    extrema_lookback: int = 10
    reversal_threshold: float = 0.03
    extrema_tolerance: float = 0.01
    event_tolerance_bars: int = 2

    # Chronological development/test split.
    calibration_fraction: float = 0.15
    test_fraction: float = 0.20
    retrain_every_bars: int = 63
    minimum_training_rows: int = 500

    # Threshold calibration.
    threshold_min: float = 0.25
    threshold_max: float = 0.85
    threshold_step: float = 0.025

    # Independent BOTTOM/TOP calibration controls.
    bottom_threshold_max: float = 0.75
    top_threshold_max: float = 0.85
    bottom_min_precision: float = 0.60
    bottom_min_recall: float = 0.30
    top_min_precision: float = 0.45
    top_min_recall: float = 0.00
    minimum_calibration_signals: int = 3
    bottom_min_calibration_signals: int = 3
    top_min_calibration_signals: int = 3

    # Entry filters.
    entry_max_rsi: float = 60.0
    entry_require_above_ema50: bool = False
    entry_cooldown_bars: int = 3

    # Trend-pullback entry supplements the calibrated BOTTOM entry.
    trend_pullback_entry_enabled: bool = True
    trend_pullback_ema: int = 20
    trend_pullback_rsi_min: float = 40.0
    trend_pullback_rsi_max: float = 65.0
    trend_pullback_touch_tolerance: float = 0.02
    trend_pullback_require_positive_return: bool = True

    # Bull-market regime used by pullback entries and adaptive exits.
    adaptive_bull_regime_enabled: bool = True
    bull_regime_ema_fast: int = 20
    bull_regime_ema_slow: int = 50
    bull_regime_require_price_above_slow: bool = True
    bull_regime_require_slow_ema_rising: bool = True
    bull_regime_entry_enabled: bool = False
    bull_regime_entry_confirmation_bars: int = 3

    # Hybrid exit.
    exit_top_probability: bool = False
    exit_trend_breakdown: bool = True
    exit_atr_trailing_stop: bool = True
    minimum_holding_bars: int = 3
    atr_trailing_multiplier: float = 3.0
    top_tighten_trailing: bool = True
    tightened_atr_multiplier: float = 1.5
    trend_exit_ema_fast: int = 5
    trend_exit_ema_slow: int = 20
    trend_breakdown_confirmation_bars: int = 2
    trend_breakdown_require_slow_ema_decline: bool = True

    # Slower exit while the bullish regime remains active.
    bull_exit_ema_fast: int = 20
    bull_exit_ema_slow: int = 50
    bull_exit_confirmation_bars: int = 3
    bull_exit_require_slow_ema_decline: bool = True

    # Fibonacci target calculated only with information known at entry.
    exit_fibonacci_target: bool = True
    fibonacci_target_ratio: float = 1.618
    fibonacci_swing_lookback: int = 50
    fibonacci_low_lookback: int = 5

    # Multi-timeframe TOP exit. The structural model keeps the current
    # BOTTOM entry; a weekly TOP model arms the exit and daily price action
    # confirms the reversal before selling at the next daily open.
    mtf_top_signal_timeframe: str = "1Week"
    mtf_top_confirmation_timeframe: str = "1Day"
    mtf_top_future_horizon: int = 4
    mtf_top_extrema_lookback: int = 10
    mtf_top_reversal_threshold: float = 0.10
    mtf_top_extrema_tolerance: float = 0.03
    mtf_top_probability_floor: float = 0.60
    mtf_top_retrain_every_bars: int = 13
    mtf_top_minimum_training_rows: int = 500
    mtf_daily_confirmation_ema: int = 20
    mtf_daily_confirmation_bars: int = 2
    mtf_daily_require_negative_return: bool = True
    mtf_daily_require_ema_decline: bool = True
    mtf_daily_require_lower_high: bool = False
    mtf_top_signal_valid_days: int = 20

    # Explicit TOP-exit guards. A confirmed TOP can sell only when the
    # current position is not below the configured return floor and price
    # remains sufficiently close to its rolling high.
    mtf_top_min_position_return: float = 0.0
    mtf_top_high_lookback_weeks: int = 26
    mtf_top_max_distance_from_high: float = 0.10

    mtf_exit_quality_horizon_days: int = 20

    # Exit Risk V1. BOTTOM remains structural; exit decisions use a separate
    # weekly XGBoost model trained on a first-touch downside/upside barrier
    # target rather than on geometric TOP labels.
    # Single-backend field is retained for backward compatibility.
    exit_risk_model_backend: str = "xgboost"
    exit_risk_compare_models: bool = True
    exit_risk_model_backends: tuple[str, ...] = (
        "xgboost",
        "histgb",
        "catboost",
    )
    exit_risk_signal_timeframe: str = "1Week"
    exit_risk_horizon_weeks: int = 8
    # We optimize for a useful neighborhood around the turning regime rather
    # than pretending the exact weekly bar is knowable in advance.
    exit_risk_event_tolerance_weeks: int = 2
    exit_risk_down_barrier: float = 0.12
    exit_risk_up_barrier: float = 0.08
    exit_risk_probability_floor: float = 0.60
    exit_risk_threshold_max: float = 0.85
    exit_risk_min_precision: float = 0.55
    exit_risk_min_recall: float = 0.20
    exit_risk_min_calibration_signals: int = 5
    exit_risk_hard_calibration_gate: bool = True
    exit_risk_retrain_every_bars: int = 26
    exit_risk_minimum_training_rows: int = 300
    exit_risk_reentry_enabled: bool = True
    exit_risk_reentry_cooldown_days: int = 5

    # Daily swing experiment. This keeps the same first-touch Exit Risk
    # concept but learns and evaluates it on 1Day bars.
    swing_exit_horizon_days: int = 10
    swing_exit_event_tolerance_days: int = 3
    swing_exit_down_barrier: float = 0.06
    swing_exit_up_barrier: float = 0.04
    swing_exit_retrain_every_bars: int = 20
    swing_exit_minimum_training_rows: int = 500

    # Compound capital rotation. Daily candles remain the information
    # frequency, while the decision utility horizon is one trading week.
    rotation_models: tuple[str, ...] = (
        "xgboost_utility",
        "qrdqn",
    )
    rotation_horizon_days: int = 5
    rotation_minimum_training_rows: int = 700
    rotation_walk_forward_enabled: bool = True
    rotation_walk_forward_calibration_days: int = 126
    rotation_walk_forward_test_days: int = 504
    rotation_walk_forward_min_test_days: int = 126
    rotation_purge_days: int = 5
    rotation_downside_penalty: float = 0.20
    rotation_drawdown_penalty: float = 0.35
    rotation_min_holding_days: int = 2
    rotation_min_expected_edge: float = 0.001
    rotation_cash_threshold: float = 0.0
    rotation_switch_margin: float = 0.005
    rotation_switch_margin_candidates: tuple[float, ...] = (
        0.0,
        0.0025,
        0.005,
        0.01,
    )
    rotation_xgb_n_estimators: int = 300
    rotation_xgb_learning_rate: float = 0.035
    rotation_xgb_max_depth: int = 3
    rotation_accelerator: str = "auto"
    rotation_allow_cpu_fallback: bool = True
    rotation_parallel_models: bool = True
    rotation_xgb_repetitions: int = 1
    rotation_qrdqn_repetitions: int = 1
    rotation_seed_step: int = 1_000

    qrdqn_training_steps: int = 15_000
    qrdqn_parallel_folds: int = 2
    qrdqn_early_stopping_enabled: bool = False
    qrdqn_early_stopping_patience: int = 4
    qrdqn_min_training_steps: int = 5_000
    qrdqn_episode_days: int = 252
    qrdqn_replay_size: int = 30_000
    qrdqn_learning_starts: int = 750
    qrdqn_batch_size: int = 128
    qrdqn_learning_rate: float = 0.0003
    qrdqn_gamma: float = 0.99
    qrdqn_n_quantiles: int = 25
    qrdqn_hidden_dim: int = 128
    qrdqn_target_update_steps: int = 250
    qrdqn_eval_every_steps: int = 1000
    qrdqn_epsilon_start: float = 1.0
    qrdqn_epsilon_end: float = 0.05
    qrdqn_device: str = "cpu"

    # Capital simulation.
    initial_capital: float = 10_000.0
    fractional_shares: bool = True
    slippage_bps: float = 0.0

    # Alpaca schedule used only as a fee reference.
    commission_rate: float = 0.0
    sec_fee_rate: float = 0.0000206
    taf_fee_per_share: float = 0.000195
    taf_fee_cap: float = 9.79
    cat_fee_per_share: float = 0.000003

    # HistGradientBoosting parameters.
    hist_max_iter: int = 300
    hist_learning_rate: float = 0.04
    hist_max_leaf_nodes: int = 15
    hist_min_samples_leaf: int = 25
    hist_l2_regularization: float = 2.0

    # XGBoost parameters.
    xgb_n_estimators: int = 350
    xgb_learning_rate: float = 0.035
    xgb_max_depth: int = 3
    xgb_min_child_weight: float = 5.0
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_gamma: float = 0.0
    xgb_reg_alpha: float = 0.10
    xgb_reg_lambda: float = 2.0
    xgb_n_jobs: int = -1
    xgb_device: str = "cpu"

    # CatBoost is used only as an Exit Risk candidate in the comparison mode.
    catboost_iterations: int = 350
    catboost_learning_rate: float = 0.035
    catboost_depth: int = 6
    catboost_l2_leaf_reg: float = 3.0
    catboost_random_strength: float = 1.0
    catboost_thread_count: int = -1

    max_parallel_workers: int = 3
    cuda_parallel_workers: int = 1

    # yfinance behavior.
    yfinance_auto_adjust: bool = True
    yfinance_repair: bool = False
    yfinance_timeout: int = 30
    yfinance_fallback_period: str = "max"

    # Local MongoDB market-data cache.
    mongo_cache_enabled: bool = True
    mongo_collection: str = "market_bars"
    mongo_refresh_overlap_days: int = 7
    mongo_server_timeout_ms: int = 2000
    mongo_write_batch_size: int = 1000

    random_state: int = 42


FEATURE_COLUMNS = [
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "gap_return",
    "body_return",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "realized_vol_5",
    "realized_vol_10",
    "realized_vol_20",
    "return_mean_5",
    "return_mean_10",
    "return_mean_20",
    "ema_distance_5",
    "ema_distance_10",
    "ema_distance_20",
    "ema_distance_50",
    "ema_5_vs_20",
    "ema_20_vs_50",
    "ema_20_slope_5",
    "ema_50_slope_10",
    "rsi_14",
    "atr_pct_14",
    "volume_change_1",
    "volume_change_5",
    "volume_zscore_20",
    "distance_from_high_10",
    "distance_from_low_10",
    "distance_from_high_20",
    "distance_from_low_20",
    "distance_from_high_50",
    "distance_from_low_50",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

EXIT_RISK_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "exit_return_4",
    "exit_return_8",
    "exit_return_13",
    "exit_return_26",
    "exit_distance_from_high_26",
    "exit_distance_from_high_52",
    "exit_ema10_slope_4",
    "exit_ema20_slope_4",
    "exit_momentum_deceleration",
    "exit_rsi_change_4",
    "exit_atr_expansion_13",
    "exit_upper_wick_atr",
]


class ProbabilityClassifier(Protocol):
    classes_: np.ndarray

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass
class BinaryModels:
    bottom_model: ProbabilityClassifier
    top_model: ProbabilityClassifier


@dataclass
class SplitData:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    train_end_position: int
    calibration_end_position: int


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_events: int
    actual_events: int


@dataclass
class AssetRunResult:
    symbol: str
    backend: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    summary_text: str
    metrics: dict[str, Any]


@dataclass
class ExitRiskBottomContext:
    structural_dataset: pd.DataFrame
    structural_split: SplitData
    bottom_calibration: ThresholdResult
    structural_calibration: pd.DataFrame
    structural_predictions: pd.DataFrame


# -----------------------------------------------------------------------------
# MongoDB configuration and arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-asset local-extrema walk-forward backtest."
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="MongoDB backtest job identifier created by app.py.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use deterministic synthetic data for a technical validation.",
    )
    return parser.parse_args()


def config_from_mapping(values: dict[str, Any]) -> BacktestConfig:
    defaults = BacktestConfig()
    allowed = {field.name for field in fields(BacktestConfig)}
    payload = {
        key: value
        for key, value in values.items()
        if key in allowed
    }

    if "assets" in payload:
        payload["assets"] = tuple(
            str(item).strip().upper()
            for item in payload["assets"]
            if str(item).strip()
        )
    if "model_backends" in payload:
        payload["model_backends"] = tuple(
            str(item).strip().lower()
            for item in payload["model_backends"]
            if str(item).strip()
        )
    if "exit_risk_model_backends" in payload:
        payload["exit_risk_model_backends"] = tuple(
            str(item).strip().lower()
            for item in payload["exit_risk_model_backends"]
            if str(item).strip()
        )
    if "rotation_models" in payload:
        payload["rotation_models"] = tuple(
            str(item).strip().lower()
            for item in payload["rotation_models"]
            if str(item).strip()
        )
    if "rotation_switch_margin_candidates" in payload:
        payload["rotation_switch_margin_candidates"] = tuple(
            float(item)
            for item in payload["rotation_switch_margin_candidates"]
        )

    if "fibonacci_target_ratio" in payload:
        payload["fibonacci_target_ratio"] = (
            normalize_classic_fibonacci_ratio(
                payload["fibonacci_target_ratio"]
            )
        )

    if "asset_overrides" in payload:
        payload["asset_overrides"] = {
            str(symbol).upper(): dict(parameters)
            for symbol, parameters in (
                payload["asset_overrides"] or {}
            ).items()
        }

    # The API exposes whole_shares while the engine stores the inverse.
    if "whole_shares" in values:
        payload["fractional_shares"] = not bool(values["whole_shares"])

    merged = {
        field.name: getattr(defaults, field.name)
        for field in fields(BacktestConfig)
    }
    merged.update(payload)
    return BacktestConfig(**merged)


def load_config(job_id: str) -> BacktestConfig:
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)

        job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"_id": 0})
        if job is None:
            raise ValueError(f"Backtest job not found in MongoDB: {job_id}")

        request_snapshot = job.get("request")
        if not request_snapshot:
            raise ValueError(
                f"Backtest job has no execution snapshot: {job_id}"
            )

        # The job request is immutable and is the only execution source.
        config = config_from_mapping(dict(request_snapshot))

        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {
                "$set": {
                    "effective_config": bson_value(serializable_config(config)),
                    "updated_at": utc_now(),
                }
            },
        )
        return config
    finally:
        client.close()


def is_swing_timeframe(timeframe: str) -> bool:
    return timeframe in SWING_TIMEFRAME_RULES


def swing_timeframe_weeks(timeframe: str) -> int:
    return {
        "1Week": 1,
        "2Weeks": 2,
        "3Weeks": 3,
        "4Weeks": 4,
    }[timeframe]


def source_timeframe(timeframe: str) -> str:
    return "1Day" if is_swing_timeframe(timeframe) else timeframe


def expanded_source_start(
    start_date: str,
    timeframe: str,
) -> str:
    if not is_swing_timeframe(timeframe):
        return start_date
    weeks = swing_timeframe_weeks(timeframe)
    start = pd.Timestamp(start_date)
    return (
        start - pd.Timedelta((weeks + 1) * 7, unit="D")
    ).strftime("%Y-%m-%d")


def aggregate_swing_bars(
    daily_bars: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    if not is_swing_timeframe(config.timeframe):
        return trim_downloaded_range(
            daily_bars,
            config.start_date,
            config.end_date,
        )
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame()

    source = daily_bars.copy().sort_index()
    source.index = pd.to_datetime(source.index, utc=True)
    source = source[~source.index.duplicated(keep="last")]

    result = source.resample(
        SWING_TIMEFRAME_RULES[config.timeframe],
        label="right",
        closed="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )
    result.index.name = "timestamp"

    normalized_end = normalize_end_date(config.end_date)
    if normalized_end:
        cutoff = pd.Timestamp(normalized_end, tz="UTC")
        result = result[result.index.normalize() <= cutoff]
    else:
        # Conservatively omit the current period until the next calendar day.
        cutoff = pd.Timestamp(date.today(), tz="UTC")
        result = result[result.index.normalize() < cutoff]

    requested_start = pd.Timestamp(config.start_date, tz="UTC")
    result = result[result.index >= requested_start]

    if normalized_end:
        requested_end = pd.Timestamp(normalized_end, tz="UTC")
        result = result[result.index < requested_end]

    return result.sort_index()


def validate_config(config: BacktestConfig) -> None:
    if config.max_parallel_workers < 1:
        raise ValueError("MAX_PARALLEL_WORKERS must be >= 1.")
    if config.cuda_parallel_workers < 1:
        raise ValueError("CUDA_PARALLEL_WORKERS must be >= 1.")
    if not config.assets:
        raise ValueError("ASSETS must contain at least one ticker.")
    valid_backends = {"histgb", "xgboost"}
    invalid = set(config.model_backends) - valid_backends
    if invalid:
        raise ValueError(f"Unsupported MODEL_BACKENDS: {sorted(invalid)}")
    supported_timeframes = {
        "1Day",
        "1Hour",
        "30Min",
        "15Min",
        "5Min",
        "1Week",
        "2Weeks",
        "3Weeks",
        "4Weeks",
    }
    if config.timeframe not in supported_timeframes:
        raise ValueError(f"Unsupported TIMEFRAME: {config.timeframe}")
    if config.strategy_mode not in {
        "ADAPTIVE_HYBRID",
        "BOTTOM_REVERSAL_FIBONACCI",
        "BOTTOM_REVERSAL_TOP_EXIT",
        "BOTTOM_ENTRY_MTF_TOP_EXIT",
        "BOTTOM_ENTRY_EXIT_RISK_V1",
        "BOTTOM_ENTRY_EXIT_RISK_SWING_1D",
        "COMPOUND_ROTATION_SWING_XGBOOST",
        "COMPOUND_ROTATION_SWING_QRDQN",
        "COMPOUND_ROTATION_SWING_1W",
        "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
        "STRUCTURAL_TREND",
    }:
        raise ValueError("Unsupported STRATEGY_MODE.")
    if config.future_horizon < 1:
        raise ValueError("FUTURE_HORIZON must be >= 1.")
    if config.extrema_lookback < 2:
        raise ValueError("EXTREMA_LOOKBACK must be >= 2.")
    if config.calibration_fraction <= 0 or config.test_fraction <= 0:
        raise ValueError("CALIBRATION_FRACTION and TEST_FRACTION must be positive.")
    if config.calibration_fraction + config.test_fraction >= 0.60:
        raise ValueError(
            "CALIBRATION_FRACTION + TEST_FRACTION must leave at least 40% for training."
        )
    if not 0 < config.threshold_min < config.threshold_max < 1:
        raise ValueError("Threshold range must satisfy 0 < min < max < 1.")
    if not config.threshold_min < config.bottom_threshold_max < 1:
        raise ValueError(
            "BOTTOM_THRESHOLD_MAX must be greater than THRESHOLD_MIN "
            "and lower than 1."
        )
    if not config.threshold_min < config.top_threshold_max < 1:
        raise ValueError(
            "TOP_THRESHOLD_MAX must be greater than THRESHOLD_MIN "
            "and lower than 1."
        )
    if not 0 <= config.bottom_min_recall < 1:
        raise ValueError("BOTTOM_MIN_RECALL must be in [0, 1).")
    if not 0 <= config.top_min_recall < 1:
        raise ValueError("TOP_MIN_RECALL must be in [0, 1).")
    if config.bottom_min_calibration_signals < 1:
        raise ValueError("BOTTOM_MIN_CALIBRATION_SIGNALS must be >= 1.")
    if config.top_min_calibration_signals < 1:
        raise ValueError("TOP_MIN_CALIBRATION_SIGNALS must be >= 1.")
    if config.threshold_step <= 0:
        raise ValueError("THRESHOLD_STEP must be positive.")
    valid_exit_backends = {"xgboost", "histgb", "catboost"}
    if config.exit_risk_model_backend not in valid_exit_backends:
        raise ValueError(
            "EXIT_RISK_MODEL_BACKEND must be xgboost, histgb or catboost."
        )
    if not config.exit_risk_model_backends:
        raise ValueError(
            "EXIT_RISK_MODEL_BACKENDS must contain at least one model."
        )
    invalid_exit_backends = (
        set(config.exit_risk_model_backends) - valid_exit_backends
    )
    if invalid_exit_backends:
        raise ValueError(
            "Unsupported EXIT_RISK_MODEL_BACKENDS: "
            f"{sorted(invalid_exit_backends)}"
        )
    if config.exit_risk_event_tolerance_weeks < 0:
        raise ValueError(
            "EXIT_RISK_EVENT_TOLERANCE_WEEKS must be >= 0."
        )
    if config.exit_risk_signal_timeframe != "1Week":
        raise ValueError("EXIT_RISK_SIGNAL_TIMEFRAME must be 1Week.")
    if config.exit_risk_horizon_weeks < 2:
        raise ValueError("EXIT_RISK_HORIZON_WEEKS must be >= 2.")
    if not 0 < config.exit_risk_down_barrier < 1:
        raise ValueError("EXIT_RISK_DOWN_BARRIER must be in (0, 1).")
    if not 0 < config.exit_risk_up_barrier < 1:
        raise ValueError("EXIT_RISK_UP_BARRIER must be in (0, 1).")
    if not 0 < config.exit_risk_probability_floor < 1:
        raise ValueError("EXIT_RISK_PROBABILITY_FLOOR must be in (0, 1).")
    if config.exit_risk_probability_floor > config.exit_risk_threshold_max:
        raise ValueError(
            "EXIT_RISK_PROBABILITY_FLOOR cannot exceed "
            "EXIT_RISK_THRESHOLD_MAX."
        )
    if not config.threshold_min < config.exit_risk_threshold_max < 1:
        raise ValueError(
            "EXIT_RISK_THRESHOLD_MAX must be greater than THRESHOLD_MIN and lower than 1."
        )
    if not 0 <= config.exit_risk_min_recall < 1:
        raise ValueError("EXIT_RISK_MIN_RECALL must be in [0, 1).")
    if config.exit_risk_min_calibration_signals < 1:
        raise ValueError("EXIT_RISK_MIN_CALIBRATION_SIGNALS must be >= 1.")
    if config.exit_risk_retrain_every_bars < 1:
        raise ValueError("EXIT_RISK_RETRAIN_EVERY_BARS must be >= 1.")
    if config.exit_risk_minimum_training_rows < 100:
        raise ValueError("EXIT_RISK_MINIMUM_TRAINING_ROWS must be >= 100.")
    if config.exit_risk_reentry_cooldown_days < 0:
        raise ValueError("EXIT_RISK_REENTRY_COOLDOWN_DAYS must be >= 0.")
    if config.strategy_mode in {"COMPOUND_ROTATION_SWING_XGBOOST", "COMPOUND_ROTATION_SWING_QRDQN", "COMPOUND_ROTATION_SWING_1W"}:
        expected_rotation_models = (
            ("qrdqn",)
            if config.strategy_mode == "COMPOUND_ROTATION_SWING_QRDQN"
            else ("xgboost_utility",)
        )
        if tuple(config.rotation_models) != expected_rotation_models:
            raise ValueError(
                f"{config.strategy_mode} requires ROTATION_MODELS={list(expected_rotation_models)}."
            )
        if config.market_data_provider not in {"alpaca", "yahoo"}:
            raise ValueError("Swing Compound Capital Rotation supports MARKET_DATA_PROVIDER=alpaca or yahoo.")
        if config.market_data_provider == "alpaca":
            if config.alpaca_feed not in {"iex", "sip"}:
                raise ValueError("ALPACA_FEED must be iex or sip.")
            if config.alpaca_adjustment not in {"raw", "split", "dividend", "all"}:
                raise ValueError("ALPACA_ADJUSTMENT must be raw, split, dividend or all.")
        if config.timeframe != "1Day":
            raise ValueError(
                "Swing Compound Capital Rotation requires TIMEFRAME=1Day."
            )
        valid_rotation_models = {"xgboost_utility", "qrdqn"}
        if not config.rotation_models:
            raise ValueError("ROTATION_MODELS must contain at least one model.")
        invalid_rotation = set(config.rotation_models) - valid_rotation_models
        if invalid_rotation:
            raise ValueError(
                f"Unsupported ROTATION_MODELS: {sorted(invalid_rotation)}"
            )
        allowed_swing_horizons = {5, 10, 20, 40, 60}
        if config.rotation_horizon_days not in allowed_swing_horizons:
            raise ValueError(
                "ROTATION_HORIZON_DAYS for Swing must be one of "
                "5, 10, 20, 40, or 60 trading sessions."
            )
        if not config.rotation_walk_forward_enabled:
            raise ValueError(
                "V8.1.0 requires true expanding walk-forward validation."
            )
        if config.rotation_purge_days < config.rotation_horizon_days:
            raise ValueError(
                "ROTATION_PURGE_DAYS must be >= ROTATION_HORIZON_DAYS "
                "to prevent forward-label leakage."
            )
        if config.rotation_walk_forward_calibration_days < 40:
            raise ValueError(
                "ROTATION_WALK_FORWARD_CALIBRATION_DAYS must be >= 40."
            )
        if config.rotation_walk_forward_test_days < 63:
            raise ValueError(
                "ROTATION_WALK_FORWARD_TEST_DAYS must be >= 63."
            )
        if config.rotation_walk_forward_min_test_days < 20:
            raise ValueError(
                "ROTATION_WALK_FORWARD_MIN_TEST_DAYS must be >= 20."
            )
        if config.rotation_downside_penalty < 0:
            raise ValueError("ROTATION_DOWNSIDE_PENALTY must be >= 0.")
        if config.rotation_drawdown_penalty < 0:
            raise ValueError("ROTATION_DRAWDOWN_PENALTY must be >= 0.")
        if config.rotation_minimum_training_rows < 300:
            raise ValueError(
                "ROTATION_MINIMUM_TRAINING_ROWS must be >= 300."
            )
        if config.rotation_min_holding_days < 0:
            raise ValueError("ROTATION_MIN_HOLDING_DAYS must be >= 0.")
        if config.qrdqn_training_steps < 500:
            raise ValueError("QRDQN_TRAINING_STEPS must be >= 500.")
        if config.rotation_xgb_repetitions < 1:
            raise ValueError("ROTATION_XGB_REPETITIONS must be >= 1.")
        if config.rotation_qrdqn_repetitions < 1:
            raise ValueError("ROTATION_QRDQN_REPETITIONS must be >= 1.")
        if config.rotation_seed_step < 1:
            raise ValueError("ROTATION_SEED_STEP must be >= 1.")
        if config.qrdqn_parallel_folds < 1:
            raise ValueError("QRDQN_PARALLEL_FOLDS must be >= 1.")
        if config.qrdqn_early_stopping_patience < 1:
            raise ValueError("QRDQN_EARLY_STOPPING_PATIENCE must be >= 1.")
        if config.qrdqn_min_training_steps < 500:
            raise ValueError("QRDQN_MIN_TRAINING_STEPS must be >= 500.")
        if config.qrdqn_n_quantiles < 5:
            raise ValueError("QRDQN_N_QUANTILES must be >= 5.")
        if config.rotation_accelerator not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "ROTATION_ACCELERATOR must be auto, cpu or cuda."
            )
        if config.qrdqn_device not in {"cpu", "cuda"}:
            raise ValueError("QRDQN_DEVICE must be cpu or cuda.")
    if config.strategy_mode == "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE":
        if config.market_data_provider not in {"alpaca", "yahoo"}:
            raise ValueError("COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE supports MARKET_DATA_PROVIDER=alpaca or yahoo.")
        if config.market_data_provider == "alpaca":
            if config.alpaca_feed not in {"iex", "sip"}:
                raise ValueError("ALPACA_FEED must be iex or sip.")
            if config.alpaca_adjustment not in {"raw", "split", "dividend", "all"}:
                raise ValueError("ALPACA_ADJUSTMENT must be raw, split, dividend or all.")
        if config.market_data_provider == "yahoo":
            start = pd.Timestamp(config.start_date, tz="UTC")
            yahoo_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)
            if start < yahoo_cutoff:
                raise ValueError(
                    "Yahoo Finance 15-minute history is limited to roughly the last 60 days. "
                    "Use Alpaca for long Day Trade training windows or choose a recent Yahoo start date."
                )
        if config.timeframe != "15Min":
            raise ValueError(
                "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE requires TIMEFRAME=15Min as the source data."
            )
        valid_rotation_models = {"xgboost_utility", "qrdqn"}
        if not config.rotation_models:
            raise ValueError("Select at least one Open-Close model.")
        invalid_rotation = set(config.rotation_models) - valid_rotation_models
        if invalid_rotation:
            raise ValueError(f"Unsupported Open-Close models: {sorted(invalid_rotation)}")
        if config.rotation_horizon_days != 1:
            raise ValueError(
                "Day Trade Open-Close fixes ROTATION_HORIZON_DAYS=1 session."
            )
        if not config.rotation_walk_forward_enabled:
            raise ValueError(
                "Day Trade Open-Close requires expanding walk-forward validation."
            )
        if config.rotation_purge_days < 1:
            raise ValueError("Day Trade Open-Close requires at least one purge session.")
        if config.rotation_walk_forward_calibration_days < 60:
            raise ValueError("Day Trade Open-Close requires at least 60 calibration sessions.")
        if config.rotation_walk_forward_test_days < 40:
            raise ValueError("Day Trade Open-Close requires at least 40 test sessions.")
        if config.rotation_walk_forward_min_test_days < 20:
            raise ValueError("Day Trade Open-Close requires at least 20 sessions in a short final test fold.")
        if len(config.assets) < 2:
            raise ValueError("Compound rotation needs at least two assets.")

    if config.strategy_mode == "BOTTOM_ENTRY_EXIT_RISK_SWING_1D":
        if config.timeframe != "1Day":
            raise ValueError(
                "BOTTOM_ENTRY_EXIT_RISK_SWING_1D requires TIMEFRAME=1Day."
            )
        if config.swing_exit_horizon_days < 2:
            raise ValueError("SWING_EXIT_HORIZON_DAYS must be >= 2.")
        if config.swing_exit_event_tolerance_days < 0:
            raise ValueError(
                "SWING_EXIT_EVENT_TOLERANCE_DAYS must be >= 0."
            )
        if not 0 < config.swing_exit_down_barrier < 1:
            raise ValueError(
                "SWING_EXIT_DOWN_BARRIER must be in (0, 1)."
            )
        if not 0 < config.swing_exit_up_barrier < 1:
            raise ValueError(
                "SWING_EXIT_UP_BARRIER must be in (0, 1)."
            )
        if config.swing_exit_retrain_every_bars < 1:
            raise ValueError(
                "SWING_EXIT_RETRAIN_EVERY_BARS must be >= 1."
            )
        if config.swing_exit_minimum_training_rows < 100:
            raise ValueError(
                "SWING_EXIT_MINIMUM_TRAINING_ROWS must be >= 100."
            )
    if config.mtf_top_min_position_return < -1:
        raise ValueError(
            "MTF_TOP_MIN_POSITION_RETURN must be greater than or equal to -1."
        )
    if config.mtf_top_high_lookback_weeks < 1:
        raise ValueError(
            "MTF_TOP_HIGH_LOOKBACK_WEEKS must be greater than or equal to 1."
        )
    if not 0 <= config.mtf_top_max_distance_from_high < 1:
        raise ValueError(
            "MTF_TOP_MAX_DISTANCE_FROM_HIGH must be in [0, 1)."
        )
    if config.initial_capital <= 0:
        raise ValueError("INITIAL_CAPITAL must be positive.")
    normalize_classic_fibonacci_ratio(
        config.fibonacci_target_ratio
    )
    if config.fibonacci_swing_lookback < 2:
        raise ValueError("FIBONACCI_SWING_LOOKBACK must be >= 2.")
    if config.fibonacci_low_lookback < 1:
        raise ValueError("FIBONACCI_LOW_LOOKBACK must be >= 1.")
    if config.tightened_atr_multiplier <= 0:
        raise ValueError("TIGHTENED_ATR_MULTIPLIER must be positive.")
    if config.trend_breakdown_confirmation_bars < 1:
        raise ValueError("TREND_BREAKDOWN_CONFIRMATION_BARS must be >= 1.")
    allowed_signal_emas = {5, 10, 20, 50}
    ema_parameters = {
        "TREND_PULLBACK_EMA": config.trend_pullback_ema,
        "BULL_REGIME_EMA_FAST": config.bull_regime_ema_fast,
        "BULL_REGIME_EMA_SLOW": config.bull_regime_ema_slow,
        "BULL_EXIT_EMA_FAST": config.bull_exit_ema_fast,
        "BULL_EXIT_EMA_SLOW": config.bull_exit_ema_slow,
    }
    for name, value in ema_parameters.items():
        if value not in allowed_signal_emas:
            raise ValueError(
                f"{name} must be one of 5, 10, 20, 50."
            )
    if config.bull_regime_ema_fast >= config.bull_regime_ema_slow:
        raise ValueError(
            "BULL_REGIME_EMA_FAST must be lower than "
            "BULL_REGIME_EMA_SLOW."
        )
    if config.bull_regime_entry_confirmation_bars < 1:
        raise ValueError(
            "BULL_REGIME_ENTRY_CONFIRMATION_BARS must be >= 1."
        )
    if config.bull_exit_ema_fast >= config.bull_exit_ema_slow:
        raise ValueError(
            "BULL_EXIT_EMA_FAST must be lower than "
            "BULL_EXIT_EMA_SLOW."
        )
    if config.bull_exit_confirmation_bars < 1:
        raise ValueError(
            "BULL_EXIT_CONFIRMATION_BARS must be >= 1."
        )
    if not 0 <= config.trend_pullback_rsi_min <= 100:
        raise ValueError(
            "TREND_PULLBACK_RSI_MIN must be between 0 and 100."
        )
    if not 0 <= config.trend_pullback_rsi_max <= 100:
        raise ValueError(
            "TREND_PULLBACK_RSI_MAX must be between 0 and 100."
        )
    if config.trend_pullback_rsi_min > config.trend_pullback_rsi_max:
        raise ValueError(
            "TREND_PULLBACK_RSI_MIN cannot exceed "
            "TREND_PULLBACK_RSI_MAX."
        )
    if not 0 <= config.trend_pullback_touch_tolerance < 1:
        raise ValueError(
            "TREND_PULLBACK_TOUCH_TOLERANCE must be in [0, 1)."
        )
    if config.strategy_mode in {
        "BOTTOM_ENTRY_MTF_TOP_EXIT",
        "BOTTOM_ENTRY_EXIT_RISK_V1",
    }:
        if config.timeframe not in {"2Weeks", "3Weeks", "4Weeks"}:
            raise ValueError(
                "Multi-timeframe exit strategies require a structural "
                "timeframe of 2Weeks, 3Weeks, or 4Weeks."
            )
        if config.mtf_top_confirmation_timeframe != "1Day":
            raise ValueError("MTF_TOP_CONFIRMATION_TIMEFRAME must be 1Day.")
        if config.mtf_daily_confirmation_ema not in allowed_signal_emas:
            raise ValueError(
                "MTF_DAILY_CONFIRMATION_EMA must be one of 5, 10, 20, 50."
            )
    if config.strategy_mode == "BOTTOM_ENTRY_MTF_TOP_EXIT":
        if config.mtf_top_signal_timeframe != "1Week":
            raise ValueError("MTF_TOP_SIGNAL_TIMEFRAME must be 1Week.")
        if not 0 < config.mtf_top_probability_floor < 1:
            raise ValueError("MTF_TOP_PROBABILITY_FLOOR must be in (0, 1).")


# -----------------------------------------------------------------------------
# Data loading and cleaning
# -----------------------------------------------------------------------------


def normalize_end_date(value: str | None) -> str | None:
    if not value:
        return None

    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid end date: {value}")

    # Internal Alpaca cache/backfill calls may provide ISO-8601 timestamps
    # carrying a UTC offset (for example ``2022-01-03T14:30:00Z``), while
    # ``date.today()`` is timezone-naive.  Normalize to a timezone-naive UTC
    # calendar value before comparing dates so pandas never mixes tz-aware and
    # tz-naive timestamps.
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)

    today = pd.Timestamp(date.today())
    if parsed.normalize() >= today:
        return None
    return parsed.strftime("%Y-%m-%d")


def filter_non_trading_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only actual market observations.

    yfinance normally omits weekends and exchange holidays already. This
    defensive filter removes any accidental daily weekend rows while leaving
    intraday data untouched.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result = result[result.index.dayofweek < 5]
    return result


def buffered_yfinance_end(
    requested_end: str | None,
    timeframe: str,
) -> str | None:
    """
    yfinance treats `end` as exclusive. For daily data, add a seven-calendar-day
    retrieval buffer so a one-day range that falls on a weekend or holiday does
    not produce a false delisted warning.
    """
    normalized = normalize_end_date(requested_end)
    if normalized is None:
        return None

    end_ts = pd.Timestamp(normalized)
    if source_timeframe(timeframe) == "1Day":
        end_ts += pd.Timedelta(7, unit="D")
    else:
        end_ts += pd.Timedelta(1, unit="D")

    return end_ts.strftime("%Y-%m-%d")


def trim_downloaded_range(
    frame: pd.DataFrame,
    requested_start: str,
    requested_end: str | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    start_ts = pd.Timestamp(requested_start)
    start_ts = (
        start_ts.tz_localize("UTC")
        if start_ts.tzinfo is None
        else start_ts.tz_convert("UTC")
    )
    result = result.loc[result.index >= start_ts]

    normalized_end = normalize_end_date(requested_end)
    if normalized_end is not None:
        end_ts = pd.Timestamp(normalized_end)
        end_ts = (
            end_ts.tz_localize("UTC")
            if end_ts.tzinfo is None
            else end_ts.tz_convert("UTC")
        )
        result = result.loc[result.index < end_ts]

    return filter_non_trading_rows(result)



def yfinance_interval(timeframe: str) -> str:
    mapping = {
        "1Day": "1d",
        "1Hour": "1h",
        "30Min": "30m",
        "15Min": "15m",
        "5Min": "5m",
        "1Week": "1d",
        "2Weeks": "1d",
        "3Weeks": "1d",
        "4Weeks": "1d",
    }
    return mapping[timeframe]


def download_yfinance_bars(
    symbol: str,
    config: BacktestConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    interval = yfinance_interval(config.timeframe)
    requested_start = start_date or config.start_date
    requested_end = end_date if end_date is not None else config.end_date
    end = buffered_yfinance_end(requested_end, config.timeframe)
    errors: list[str] = []

    history_kwargs: dict[str, Any] = {
        "start": requested_start,
        "interval": interval,
        "auto_adjust": config.yfinance_auto_adjust,
        "repair": config.yfinance_repair,
        "actions": False,
        "timeout": config.yfinance_timeout,
        "raise_errors": False,
    }
    if end:
        history_kwargs["end"] = end

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.Ticker(symbol).history(**history_kwargs)
        frame = normalize_yfinance_frame(frame, symbol)
        frame = trim_downloaded_range(frame, requested_start, requested_end)
        if not frame.empty:
            return frame
        errors.append("Ticker.history returned no rows in the requested trading range")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Ticker.history failed: {exc}")

    download_kwargs: dict[str, Any] = {
        "tickers": symbol,
        "start": requested_start,
        "interval": interval,
        "auto_adjust": config.yfinance_auto_adjust,
        "repair": config.yfinance_repair,
        "progress": False,
        "threads": False,
        "timeout": config.yfinance_timeout,
        "group_by": "column",
    }
    if end:
        download_kwargs["end"] = end

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.download(**download_kwargs)
        frame = normalize_yfinance_frame(frame, symbol)
        frame = trim_downloaded_range(frame, requested_start, requested_end)
        if not frame.empty:
            return frame
        errors.append("yf.download returned no rows in the requested trading range")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"yf.download failed: {exc}")

    if source_timeframe(config.timeframe) == "1Day":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = yf.Ticker(symbol).history(
                    period=config.yfinance_fallback_period,
                    interval="1d",
                    auto_adjust=config.yfinance_auto_adjust,
                    repair=config.yfinance_repair,
                    actions=False,
                    timeout=config.yfinance_timeout,
                    raise_errors=False,
                )
            frame = normalize_yfinance_frame(frame, symbol)
            frame = trim_downloaded_range(frame, requested_start, requested_end)
            if not frame.empty:
                return frame
            errors.append("period=max fallback returned no rows in the requested trading range")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"period=max fallback failed: {exc}")

    if allow_empty:
        return pd.DataFrame()

    details = " | ".join(errors)
    raise RuntimeError(
        f"yfinance returned no trading bars for {symbol}. Attempts: {details}"
    )



def _mongo(config: BacktestConfig):
    from pymongo import ASCENDING

    client = create_client()
    db = get_database(client)
    col = db[config.mongo_collection]
    col.create_index(
        [("symbol", ASCENDING), ("interval", ASCENDING), ("timestamp", ASCENDING)],
        unique=True,
        name="uq_market_bar",
    )
    return client, col


def _normalize_timestamp(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _row_document(row: Any, symbol: str, interval: str, updated_at: datetime) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "timestamp": _normalize_timestamp(row.Index),
        "updated_at": updated_at,
    }
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        value = getattr(row, column, None)
        if value is not None and pd.notna(value):
            doc[column] = float(value)
    return doc


def _chunked(items: list[Any], batch_size: int):
    size = max(1, batch_size)
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _insert_initial_bars(
    col: Any,
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    batch_size: int,
) -> int:
    if frame.empty:
        return 0

    updated_at = datetime.now(timezone.utc)
    documents = [
        {**_row_document(row, symbol, interval, updated_at), "created_at": updated_at}
        for row in frame.itertuples()
    ]

    inserted = 0
    for batch in _chunked(documents, batch_size):
        result = col.insert_many(batch, ordered=False)
        inserted += len(result.inserted_ids)
    return inserted


def _upsert_bars(
    col: Any,
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    batch_size: int,
) -> dict[str, int]:
    from pymongo import UpdateOne

    totals = {"processed": 0, "inserted": 0, "updated": 0, "matched": 0}
    if frame.empty:
        return totals

    updated_at = datetime.now(timezone.utc)
    operations: list[Any] = []

    for row in frame.itertuples():
        doc = _row_document(row, symbol, interval, updated_at)
        operations.append(
            UpdateOne(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "timestamp": doc["timestamp"],
                },
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": updated_at},
                },
                upsert=True,
            )
        )

    for batch in _chunked(operations, batch_size):
        result = col.bulk_write(batch, ordered=False)
        totals["processed"] += len(batch)
        totals["inserted"] += result.upserted_count
        totals["updated"] += result.modified_count
        totals["matched"] += result.matched_count

    return totals


def _read_bars(col: Any, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    q = {"symbol": symbol, "interval": interval, "timestamp": {"$gte": start.to_pydatetime()}}
    if end is not None:
        q["timestamp"]["$lt"] = end.to_pydatetime()

    projection = {
        "_id": 0,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "vwap": 1,
        "trade_count": 1,
    }
    rows = list(col.find(q, projection).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count")
        if column in frame.columns
    ]
    return frame[columns].sort_index()


def load_yfinance_bars(
    symbol: str,
    config: BacktestConfig,
) -> pd.DataFrame:
    source_start_value = expanded_source_start(
        config.start_date,
        config.timeframe,
    )

    if not config.mongo_cache_enabled:
        downloaded = download_yfinance_bars(
            symbol,
            config,
            source_start_value,
            config.end_date,
        )
        return aggregate_swing_bars(downloaded, config)

    interval = yfinance_interval(config.timeframe)
    start = pd.Timestamp(source_start_value, tz="UTC")
    normalized_end = normalize_end_date(config.end_date)
    end = (
        pd.Timestamp(normalized_end, tz="UTC")
        if normalized_end
        else None
    )

    try:
        client, col = _mongo(config)
    except Exception as exc:
        print(
            f"MongoDB unavailable: {exc}. "
            "Downloading from yfinance."
        )
        downloaded = download_yfinance_bars(
            symbol,
            config,
            source_start_value,
            config.end_date,
        )
        return aggregate_swing_bars(downloaded, config)

    try:
        first = col.find_one(
            {"symbol": symbol, "interval": interval},
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", 1)],
        )
        last = col.find_one(
            {"symbol": symbol, "interval": interval},
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", -1)],
        )

        if first is None:
            print(
                f"Cache miss for {symbol}; "
                "downloading full source history..."
            )
            downloaded = download_yfinance_bars(
                symbol,
                config,
                source_start_value,
                config.end_date,
            )
            inserted = _insert_initial_bars(
                col,
                downloaded,
                symbol,
                interval,
                config.mongo_write_batch_size,
            )
            print(
                f"Inserted {inserted:,} {symbol} source bars "
                "into MongoDB."
            )
        else:
            first_ts = pd.Timestamp(first["timestamp"])
            last_ts = pd.Timestamp(last["timestamp"])
            first_ts = (
                first_ts.tz_localize("UTC")
                if first_ts.tzinfo is None
                else first_ts.tz_convert("UTC")
            )
            last_ts = (
                last_ts.tz_localize("UTC")
                if last_ts.tzinfo is None
                else last_ts.tz_convert("UTC")
            )

            if start < first_ts:
                historical = download_yfinance_bars(
                    symbol,
                    config,
                    source_start_value,
                    first_ts.strftime("%Y-%m-%d"),
                    allow_empty=True,
                )
                stats = _upsert_bars(
                    col,
                    historical,
                    symbol,
                    interval,
                    config.mongo_write_batch_size,
                )
                print(
                    f"Backfilled {stats['processed']:,} {symbol} "
                    f"source bars ({stats['inserted']:,} inserted, "
                    f"{stats['updated']:,} updated)."
                )

            overlap = pd.Timedelta(
                int(config.mongo_refresh_overlap_days),
                unit="D",
            )
            refresh = max(start, last_ts - overlap)
            if end is None or refresh < end:
                print(
                    f"Refreshing recent {symbol} source bars "
                    f"from {refresh.date()}..."
                )
                recent = download_yfinance_bars(
                    symbol,
                    config,
                    refresh.strftime("%Y-%m-%d"),
                    config.end_date,
                )
                stats = _upsert_bars(
                    col,
                    recent,
                    symbol,
                    interval,
                    config.mongo_write_batch_size,
                )
                print(
                    f"Processed {stats['processed']:,} recent "
                    f"{symbol} source bars "
                    f"({stats['inserted']:,} inserted, "
                    f"{stats['updated']:,} updated)."
                )

        cached = _read_bars(
            col,
            symbol,
            interval,
            start,
            end,
        )
        if cached.empty:
            raise RuntimeError("MongoDB cache returned no bars")

        print(
            f"Loaded {len(cached):,} {symbol} source bars "
            "from MongoDB cache."
        )
        prepared = aggregate_swing_bars(cached, config)
        if is_swing_timeframe(config.timeframe):
            print(
                f"Aggregated {len(prepared):,} {config.timeframe} "
                f"{symbol} swing bars from daily OHLCV."
            )
        return prepared
    finally:
        client.close()



def _alpaca_mongo(config: BacktestConfig):
    from pymongo import ASCENDING

    client = create_client()
    db = get_database(client)
    col = db[ALPACA_MARKET_BARS_COLLECTION]
    col.create_index(
        [
            ("symbol", ASCENDING),
            ("interval", ASCENDING),
            ("feed", ASCENDING),
            ("adjustment", ASCENDING),
            ("timestamp", ASCENDING),
        ],
        unique=True,
        name="uq_alpaca_market_bar",
    )
    return client, db, col


def _alpaca_identity(symbol: str, config: BacktestConfig) -> dict[str, str]:
    return {
        "symbol": symbol,
        "interval": config.timeframe,
        "feed": config.alpaca_feed,
        "adjustment": config.alpaca_adjustment,
    }


def _alpaca_row_document(
    row: Any,
    identity: dict[str, str],
    updated_at: datetime,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        **identity,
        "timestamp": _normalize_timestamp(row.Index),
        "updated_at": updated_at,
    }
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        value = getattr(row, column, None)
        if value is not None and pd.notna(value):
            doc[column] = float(value)
    return doc


def _upsert_alpaca_bars(
    col: Any,
    frame: pd.DataFrame,
    identity: dict[str, str],
    batch_size: int,
) -> dict[str, int]:
    from pymongo import UpdateOne

    totals = {"processed": 0, "inserted": 0, "updated": 0, "matched": 0}
    if frame.empty:
        return totals
    updated_at = datetime.now(timezone.utc)
    operations: list[Any] = []
    for row in frame.itertuples():
        doc = _alpaca_row_document(row, identity, updated_at)
        query = {**identity, "timestamp": doc["timestamp"]}
        operations.append(
            UpdateOne(
                query,
                {"$set": doc, "$setOnInsert": {"created_at": updated_at}},
                upsert=True,
            )
        )
    for batch in _chunked(operations, batch_size):
        result = col.bulk_write(batch, ordered=False)
        totals["processed"] += len(batch)
        totals["inserted"] += result.upserted_count
        totals["updated"] += result.modified_count
        totals["matched"] += result.matched_count
    return totals


def _read_alpaca_bars(
    col: Any,
    identity: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    q: dict[str, Any] = {
        **identity,
        "timestamp": {"$gte": start.to_pydatetime()},
    }
    if end is not None:
        q["timestamp"]["$lt"] = end.to_pydatetime()
    projection = {
        "_id": 0,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "vwap": 1,
        "trade_count": 1,
    }
    rows = list(col.find(q, projection).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count")
        if column in frame.columns
    ]
    return frame[columns].sort_index()


def _download_alpaca_bars(
    symbol: str,
    config: BacktestConfig,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    client = create_client()
    try:
        db = get_database(client)
        credentials = get_alpaca_credentials(db)
    finally:
        client.close()

    normalized_end = normalize_end_date(end_date)
    end_value: str | None = normalized_end
    frame = download_alpaca_stock_bars(
        api_key_id=credentials["api_key_id"],
        secret_key=credentials["secret_key"],
        symbol=symbol,
        timeframe=config.timeframe,
        start=start_date,
        end=end_value,
        feed=config.alpaca_feed,
        adjustment=config.alpaca_adjustment,
    )
    return trim_downloaded_range(frame, start_date, end_date)


def load_alpaca_bars(symbol: str, config: BacktestConfig) -> pd.DataFrame:
    source_start_value = expanded_source_start(config.start_date, config.timeframe)
    normalized_end = normalize_end_date(config.end_date)
    start = pd.Timestamp(source_start_value, tz="UTC")
    end = pd.Timestamp(normalized_end, tz="UTC") if normalized_end else None

    if not config.mongo_cache_enabled:
        downloaded = _download_alpaca_bars(
            symbol,
            config,
            source_start_value,
            config.end_date,
        )
        return aggregate_swing_bars(downloaded, config)

    client, db, col = _alpaca_mongo(config)
    identity = _alpaca_identity(symbol, config)
    try:
        first = col.find_one(
            identity,
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", 1)],
        )
        last = col.find_one(
            identity,
            {"timestamp": 1, "_id": 0},
            sort=[("timestamp", -1)],
        )

        if first is None:
            print(
                f"Alpaca cache miss for {symbol} ({config.alpaca_feed}); downloading full {config.timeframe} history..."
            )
            downloaded = _download_alpaca_bars(
                symbol,
                config,
                source_start_value,
                config.end_date,
            )
            stats = _upsert_alpaca_bars(
                col,
                downloaded,
                identity,
                config.mongo_write_batch_size,
            )
            print(
                f"Stored {stats['processed']:,} Alpaca {symbol} bars "
                f"({stats['inserted']:,} inserted, {stats['updated']:,} updated)."
            )
        else:
            first_ts = pd.Timestamp(first["timestamp"])
            first_ts = first_ts.tz_localize("UTC") if first_ts.tzinfo is None else first_ts.tz_convert("UTC")
            last_ts = pd.Timestamp(last["timestamp"])
            last_ts = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")

            # ``start_date`` is a calendar-date boundary (midnight UTC),
            # whereas the first regular-session Alpaca bar naturally starts
            # later on that same day.  Treat the cache as covered when both
            # timestamps fall on the same UTC calendar date; otherwise every
            # run would incorrectly attempt a same-day backfill before the
            # market open.
            if start.normalize() < first_ts.normalize():
                historical_end = first_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                historical = _download_alpaca_bars(
                    symbol,
                    config,
                    source_start_value,
                    historical_end,
                )
                stats = _upsert_alpaca_bars(
                    col,
                    historical,
                    identity,
                    config.mongo_write_batch_size,
                )
                print(
                    f"Backfilled {stats['processed']:,} Alpaca {symbol} bars "
                    f"({stats['inserted']:,} inserted, {stats['updated']:,} updated)."
                )

            overlap = pd.Timedelta(int(config.mongo_refresh_overlap_days), unit="D")
            refresh = max(start, last_ts - overlap)
            if end is None or refresh < end:
                recent = _download_alpaca_bars(
                    symbol,
                    config,
                    refresh.isoformat(),
                    config.end_date,
                )
                stats = _upsert_alpaca_bars(
                    col,
                    recent,
                    identity,
                    config.mongo_write_batch_size,
                )
                print(
                    f"Refreshed {stats['processed']:,} Alpaca {symbol} bars "
                    f"({stats['inserted']:,} inserted, {stats['updated']:,} updated)."
                )

        cached = _read_alpaca_bars(col, identity, start, end)
        if cached.empty:
            raise RuntimeError("Alpaca MongoDB cache returned no bars")
        print(
            f"Loaded {len(cached):,} {symbol} {config.timeframe} bars from Alpaca/MongoDB "
            f"(feed={config.alpaca_feed}, adjustment={config.alpaca_adjustment})."
        )
        return aggregate_swing_bars(cached, config)
    finally:
        client.close()


def load_market_bars(symbol: str, config: BacktestConfig) -> pd.DataFrame:
    provider = str(config.market_data_provider or "alpaca").strip().lower()
    if provider == "alpaca":
        return load_alpaca_bars(symbol, config)
    if provider == "yahoo":
        return load_yfinance_bars(symbol, config)
    raise ValueError(f"Unsupported market data provider: {provider}")

def normalize_yfinance_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):
        level_values = [set(map(str, result.columns.get_level_values(i))) for i in range(result.columns.nlevels)]
        selected = None
        for level_index, values in enumerate(level_values):
            if symbol in values:
                try:
                    selected = result.xs(symbol, axis=1, level=level_index, drop_level=True)
                    break
                except Exception:  # noqa: BLE001
                    pass
        if selected is not None:
            result = selected
        else:
            result.columns = result.columns.get_level_values(0)

    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    rename_map = {
        "adj_close": "close",
    }
    result = result.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        return pd.DataFrame()

    result = result[required].copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    index = pd.to_datetime(result.index, errors="coerce", utc=True)
    result.index = index
    result.index.name = "timestamp"
    result = result[~result.index.isna()]
    return result.dropna().sort_index()


def validate_and_clean_bars(
    bars: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    bars = filter_non_trading_rows(bars)
    if bars.empty:
        raise ValueError("The OHLCV dataset is empty.")
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    result = bars.copy()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result[(result[required[:4]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    minimum_bars = MINIMUM_BARS_BY_TIMEFRAME[config.timeframe]
    if len(result) < minimum_bars:
        hint = (
            " Use an earlier start date, preferably 2000-01-01 "
            "or the earliest available date for swing timeframes."
            if is_swing_timeframe(config.timeframe)
            else ""
        )
        raise ValueError(
            f"Only {len(result)} valid {config.timeframe} bars were "
            f"loaded; at least {minimum_bars} are required."
            f"{hint}"
        )
    return result


def generate_demo_data(symbol: str, config: BacktestConfig) -> pd.DataFrame:
    seed = config.random_state + sum(ord(character) for character in symbol)
    rng = np.random.default_rng(seed)
    periods = 7000 if is_swing_timeframe(config.timeframe) else 2300
    index = pd.bdate_range(config.start_date, periods=periods, tz="UTC")

    regime = np.zeros(periods)
    regime[:500] = 0.0004
    regime[500:850] = -0.00015
    regime[850:1450] = 0.00075
    regime[1450:1750] = -0.00025
    regime[1750:] = 0.00055
    cycle = 0.0015 * np.sin(np.arange(periods) / 24.0)
    returns = regime + cycle + rng.normal(0, 0.018, periods)
    close = 30 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.004, periods - 1))]
    spread = np.abs(rng.normal(0.012, 0.006, periods))
    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)
    volume = rng.integers(5_000_000, 80_000_000, periods)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


# -----------------------------------------------------------------------------
# Feature and label engineering
# -----------------------------------------------------------------------------


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    difference = close.diff()
    gains = difference.clip(lower=0)
    losses = -difference.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = safe_divide(average_gain, average_loss)
    return 100 - (100 / (1 + relative_strength))


def true_range(data: pd.DataFrame) -> pd.Series:
    previous_close = data["close"].shift(1)
    return pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def future_rolling_max(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def future_rolling_min(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def build_dataset(bars: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    data = bars.copy()
    close = data["close"]
    open_price = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    returns_1 = close.pct_change()
    for period in [1, 2, 3, 5, 10, 20]:
        data[f"return_{period}"] = close.pct_change(period)

    data["gap_return"] = safe_divide(open_price, close.shift(1)) - 1
    data["body_return"] = safe_divide(close, open_price) - 1
    data["range_pct"] = safe_divide(high - low, close)
    data["upper_wick_pct"] = safe_divide(high - pd.concat([open_price, close], axis=1).max(axis=1), close)
    data["lower_wick_pct"] = safe_divide(pd.concat([open_price, close], axis=1).min(axis=1) - low, close)

    for window in [5, 10, 20]:
        data[f"realized_vol_{window}"] = returns_1.rolling(window).std()
        data[f"return_mean_{window}"] = returns_1.rolling(window).mean()

    ema_values: dict[int, pd.Series] = {}
    for window in [5, 10, 20, 50]:
        ema_values[window] = close.ewm(span=window, adjust=False).mean()
        data[f"ema_{window}"] = ema_values[window]
        data[f"ema_distance_{window}"] = safe_divide(close, ema_values[window]) - 1

    data["ema_5_vs_20"] = safe_divide(ema_values[5], ema_values[20]) - 1
    data["ema_20_vs_50"] = safe_divide(ema_values[20], ema_values[50]) - 1
    data["ema_20_slope_5"] = ema_values[20].pct_change(5)
    data["ema_50_slope_10"] = ema_values[50].pct_change(10)
    data["rsi_14"] = rsi(close, 14)

    atr = true_range(data).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_14"] = atr
    data["atr_pct_14"] = safe_divide(atr, close)

    data["volume_change_1"] = volume.pct_change()
    data["volume_change_5"] = volume.pct_change(5)
    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std()
    data["volume_zscore_20"] = safe_divide(volume - volume_mean_20, volume_std_20)

    for window in [10, 20, 50]:
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        data[f"distance_from_high_{window}"] = safe_divide(close, rolling_high) - 1
        data[f"distance_from_low_{window}"] = safe_divide(close, rolling_low) - 1

    weekday = data.index.dayofweek
    month = data.index.month
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    if is_swing_timeframe(config.timeframe):
        # Multi-week candles can have wide intraperiod wicks. For swing
        # labels, use completed closing prices for both local extrema and
        # future reversal measurement.
        future_high = future_rolling_max(
            close,
            config.future_horizon,
        )
        future_low = future_rolling_min(
            close,
            config.future_horizon,
        )
        recent_low = close.rolling(
            config.extrema_lookback,
        ).min()
        recent_high = close.rolling(
            config.extrema_lookback,
        ).max()
    else:
        future_high = future_rolling_max(
            high,
            config.future_horizon,
        )
        future_low = future_rolling_min(
            low,
            config.future_horizon,
        )
        recent_low = low.rolling(
            config.extrema_lookback,
        ).min()
        recent_high = high.rolling(
            config.extrema_lookback,
        ).max()

    data["future_max_return"] = (
        safe_divide(future_high, close) - 1
    )
    data["future_min_return"] = (
        safe_divide(future_low, close) - 1
    )

    near_recent_low = (
        close
        <= recent_low * (1 + config.extrema_tolerance)
    )
    near_recent_high = (
        close
        >= recent_high * (1 - config.extrema_tolerance)
    )

    data["actual_bottom"] = (
        near_recent_low
        & (data["future_max_return"] >= config.reversal_threshold)
        & (data["future_max_return"] > data["future_min_return"].abs())
    )
    data["actual_top"] = (
        near_recent_high
        & (data["future_min_return"] <= -config.reversal_threshold)
        & (data["future_min_return"].abs() > data["future_max_return"])
    )

    # Naive baselines use only information known at the current close.
    data["baseline_bottom_signal"] = near_recent_low
    data["baseline_top_signal"] = near_recent_high

    required = FEATURE_COLUMNS + [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_5",
        "ema_20",
        "ema_50",
        "atr_14",
        "actual_bottom",
        "actual_top",
        "future_max_return",
        "future_min_return",
    ]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    data["actual_bottom"] = data["actual_bottom"].astype(bool)
    data["actual_top"] = data["actual_top"].astype(bool)
    return data


# -----------------------------------------------------------------------------
# Splits, models, calibration, walk-forward
# -----------------------------------------------------------------------------


def split_dataset(data: pd.DataFrame, config: BacktestConfig) -> SplitData:
    total_rows = len(data)
    calibration_start = int(total_rows * (1 - config.test_fraction - config.calibration_fraction))
    test_start = int(total_rows * (1 - config.test_fraction))
    purge = config.future_horizon

    train_end = calibration_start - purge
    calibration_end = test_start - purge

    if train_end < config.minimum_training_rows:
        raise ValueError(
            f"Training contains only {train_end} rows; MINIMUM_TRAINING_ROWS is "
            f"{config.minimum_training_rows}."
        )
    if calibration_end <= calibration_start + 30:
        raise ValueError("Calibration period is too short.")
    if total_rows <= test_start + 30:
        raise ValueError("Test period is too short.")

    return SplitData(
        train=data.iloc[:train_end].copy(),
        calibration=data.iloc[calibration_start:calibration_end].copy(),
        test=data.iloc[test_start:].copy(),
        train_end_position=train_end,
        calibration_end_position=test_start,
    )


def positive_scale(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives <= 0:
        raise ValueError("The training set contains no positive events.")
    return max(1.0, negatives / positives)


def build_binary_classifier(
    backend: str,
    y: pd.Series,
    config: BacktestConfig,
) -> ProbabilityClassifier:
    if backend == "histgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=config.hist_max_iter,
            learning_rate=config.hist_learning_rate,
            max_leaf_nodes=config.hist_max_leaf_nodes,
            min_samples_leaf=config.hist_min_samples_leaf,
            l2_regularization=config.hist_l2_regularization,
            early_stopping=False,
            random_state=config.random_state,
        )

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=config.xgb_device,
            n_estimators=config.xgb_n_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            min_child_weight=config.xgb_min_child_weight,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            gamma=config.xgb_gamma,
            reg_alpha=config.xgb_reg_alpha,
            reg_lambda=config.xgb_reg_lambda,
            scale_pos_weight=positive_scale(y),
            n_jobs=config.xgb_n_jobs,
            random_state=config.random_state,
        )

    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError(
                "catboost is not installed. On Python 3.14 use CatBoost "
                "1.2.9 or newer: python -m pip install 'catboost>=1.2.9,<2'"
            ) from exc

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=config.catboost_iterations,
            learning_rate=config.catboost_learning_rate,
            depth=config.catboost_depth,
            l2_leaf_reg=config.catboost_l2_leaf_reg,
            random_strength=config.catboost_random_strength,
            auto_class_weights="Balanced",
            thread_count=config.catboost_thread_count,
            random_seed=config.random_state,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(f"Unsupported model backend: {backend}")


def fit_binary_models(
    train: pd.DataFrame,
    backend: str,
    config: BacktestConfig,
) -> BinaryModels:
    x_train = train[FEATURE_COLUMNS]
    bottom_target = train["actual_bottom"].astype(int)
    top_target = train["actual_top"].astype(int)

    bottom_events = int(bottom_target.sum())
    top_events = int(top_target.sum())

    if bottom_events == 0 or top_events == 0:
        label_basis = (
            "closing prices"
            if is_swing_timeframe(config.timeframe)
            else "intraperiod highs and lows"
        )
        raise ValueError(
            f"Training label counts for {config.timeframe}: "
            f"rows={len(train)}, "
            f"BOTTOM={bottom_events}, "
            f"TOP={top_events}. "
            f"Label basis={label_basis}; "
            f"EXTREMA_LOOKBACK={config.extrema_lookback}; "
            f"REVERSAL_THRESHOLD={config.reversal_threshold:.4f}; "
            f"EXTREMA_TOLERANCE={config.extrema_tolerance:.4f}; "
            f"FUTURE_HORIZON={config.future_horizon}. "
            "Training must contain at least one BOTTOM and one TOP event."
        )

    print(
        f"Training labels for {config.timeframe}: "
        f"rows={len(train):,}, "
        f"BOTTOM={bottom_events:,}, "
        f"TOP={top_events:,}."
    )

    bottom_model = build_binary_classifier(backend, bottom_target, config)
    top_model = build_binary_classifier(backend, top_target, config)

    if backend == "histgb":
        bottom_weights = compute_sample_weight("balanced", bottom_target)
        top_weights = compute_sample_weight("balanced", top_target)
        bottom_model.fit(x_train, bottom_target, sample_weight=bottom_weights)
        top_model.fit(x_train, top_target, sample_weight=top_weights)
    else:
        bottom_model.fit(x_train, bottom_target)
        top_model.fit(x_train, top_target)

    return BinaryModels(bottom_model=bottom_model, top_model=top_model)


def positive_probability(model: ProbabilityClassifier, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    matches = np.where(classes == 1)[0]
    if len(matches) != 1:
        raise ValueError(f"Expected binary classes containing 1, received {classes.tolist()}")
    return probabilities[:, int(matches[0])]


def add_model_probabilities(
    data: pd.DataFrame,
    models: BinaryModels,
) -> pd.DataFrame:
    result = data.copy()
    x = result[FEATURE_COLUMNS]
    result["bottom_probability"] = positive_probability(models.bottom_model, x)
    result["top_probability"] = positive_probability(models.top_model, x)
    return result


def fit_single_target_model(
    train: pd.DataFrame,
    backend: str,
    config: BacktestConfig,
    *,
    target_column: str,
    feature_columns: list[str],
) -> ProbabilityClassifier:
    target = train[target_column].astype(int)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Training target {target_column} must contain both classes; "
            f"rows={len(train)}, positives={positives}, negatives={negatives}."
        )

    model = build_binary_classifier(backend, target, config)
    x_train = train[feature_columns]
    if backend == "histgb":
        weights = compute_sample_weight("balanced", target)
        model.fit(x_train, target, sample_weight=weights)
    else:
        model.fit(x_train, target)
    return model


def add_single_target_probability(
    data: pd.DataFrame,
    model: ProbabilityClassifier,
    *,
    feature_columns: list[str],
    probability_column: str,
) -> pd.DataFrame:
    result = data.copy()
    result[probability_column] = positive_probability(
        model, result[feature_columns]
    )
    return result


def calibrate_single_target(
    split: SplitData,
    backend: str,
    config: BacktestConfig,
    *,
    target_column: str,
    probability_column: str,
    feature_columns: list[str],
    minimum_precision: float,
    minimum_recall: float,
    maximum_threshold: float,
    minimum_signals: int,
    tolerance_bars: int,
) -> tuple[ThresholdResult, pd.DataFrame]:
    model = fit_single_target_model(
        split.train,
        backend,
        config,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    calibration = add_single_target_probability(
        split.calibration,
        model,
        feature_columns=feature_columns,
        probability_column=probability_column,
    )
    local_config = replace(config, event_tolerance_bars=tolerance_bars)
    threshold = calibrate_threshold(
        actual=calibration[target_column],
        probability=calibration[probability_column],
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
        maximum_threshold=maximum_threshold,
        minimum_signals=minimum_signals,
        config=local_config,
    )
    return threshold, calibration


def walk_forward_single_target(
    data: pd.DataFrame,
    split: SplitData,
    backend: str,
    config: BacktestConfig,
    *,
    target_column: str,
    probability_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    test_start = split.calibration_end_position
    frames: list[pd.DataFrame] = []

    for block_start in range(
        test_start, len(data), config.retrain_every_bars
    ):
        block_end = min(
            block_start + config.retrain_every_bars,
            len(data),
        )
        training_end = block_start - config.future_horizon
        if training_end < config.minimum_training_rows:
            raise ValueError("Walk-forward training window is too short.")

        training = data.iloc[:training_end].copy()
        block = data.iloc[block_start:block_end].copy()
        model = fit_single_target_model(
            training,
            backend,
            config,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        predicted = add_single_target_probability(
            block,
            model,
            feature_columns=feature_columns,
            probability_column=probability_column,
        )
        predicted["walk_forward_training_end"] = training.index.max()
        predicted["walk_forward_block"] = len(frames) + 1
        frames.append(predicted)

    if not frames:
        raise ValueError("Walk-forward prediction produced no test blocks.")
    return pd.concat(frames).sort_index()


def event_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    tolerance_bars: int,
) -> dict[str, float | int]:
    actual_values = actual.astype(bool).to_numpy()
    predicted_values = predicted.astype(bool).to_numpy()
    actual_positions = np.flatnonzero(actual_values)
    predicted_positions = np.flatnonzero(predicted_values)

    predicted_hits = sum(
        np.any(np.abs(actual_positions - position) <= tolerance_bars)
        for position in predicted_positions
    )
    actual_hits = sum(
        np.any(np.abs(predicted_positions - position) <= tolerance_bars)
        for position in actual_positions
    )

    precision = predicted_hits / len(predicted_positions) if len(predicted_positions) else 0.0
    recall = actual_hits / len(actual_positions) if len(actual_positions) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_events": int(len(predicted_positions)),
        "actual_events": int(len(actual_positions)),
    }


def calibrate_threshold(
    actual: pd.Series,
    probability: pd.Series,
    minimum_precision: float,
    minimum_recall: float,
    maximum_threshold: float,
    minimum_signals: int,
    config: BacktestConfig,
) -> ThresholdResult:
    candidates: list[ThresholdResult] = []
    effective_maximum = min(
        float(maximum_threshold),
        float(config.threshold_max),
    )
    thresholds = np.arange(
        config.threshold_min,
        effective_maximum + config.threshold_step / 2,
        config.threshold_step,
    )

    for threshold in thresholds:
        predicted = probability >= threshold
        metrics = event_metrics(actual, predicted, config.event_tolerance_bars)
        candidates.append(
            ThresholdResult(
                threshold=float(round(threshold, 6)),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                predicted_events=int(metrics["predicted_events"]),
                actual_events=int(metrics["actual_events"]),
            )
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate.precision >= minimum_precision
        and candidate.recall >= minimum_recall
        and candidate.predicted_events >= minimum_signals
    ]

    # Preserve a minimum event-count safeguard even when every strict
    # precision/recall constraint cannot be met.
    signal_pool = [
        candidate
        for candidate in candidates
        if candidate.predicted_events >= minimum_signals
    ]
    pool = eligible or signal_pool or candidates

    return max(
        pool,
        key=lambda candidate: (
            candidate.f1,
            candidate.recall >= minimum_recall,
            candidate.precision >= minimum_precision,
            candidate.precision,
            candidate.recall,
            -candidate.threshold,
        ),
    )


def calibrate_models(
    split: SplitData,
    backend: str,
    config: BacktestConfig,
) -> tuple[ThresholdResult, ThresholdResult, pd.DataFrame]:
    models = fit_binary_models(split.train, backend, config)
    calibration = add_model_probabilities(split.calibration, models)

    bottom_threshold = calibrate_threshold(
        actual=calibration["actual_bottom"],
        probability=calibration["bottom_probability"],
        minimum_precision=config.bottom_min_precision,
        minimum_recall=config.bottom_min_recall,
        maximum_threshold=config.bottom_threshold_max,
        minimum_signals=config.bottom_min_calibration_signals,
        config=config,
    )
    top_threshold = calibrate_threshold(
        actual=calibration["actual_top"],
        probability=calibration["top_probability"],
        minimum_precision=config.top_min_precision,
        minimum_recall=config.top_min_recall,
        maximum_threshold=config.top_threshold_max,
        minimum_signals=config.top_min_calibration_signals,
        config=config,
    )
    return bottom_threshold, top_threshold, calibration


def walk_forward_predict(
    data: pd.DataFrame,
    split: SplitData,
    backend: str,
    config: BacktestConfig,
) -> pd.DataFrame:
    test_start = split.calibration_end_position
    frames: list[pd.DataFrame] = []

    for block_start in range(test_start, len(data), config.retrain_every_bars):
        block_end = min(block_start + config.retrain_every_bars, len(data))
        training_end = block_start - config.future_horizon
        if training_end < config.minimum_training_rows:
            raise ValueError("Walk-forward training window is too short.")

        training = data.iloc[:training_end].copy()
        block = data.iloc[block_start:block_end].copy()
        models = fit_binary_models(training, backend, config)
        predicted = add_model_probabilities(block, models)
        predicted["walk_forward_training_end"] = training.index.max()
        predicted["walk_forward_block"] = len(frames) + 1
        frames.append(predicted)

    if not frames:
        raise ValueError("Walk-forward prediction produced no test blocks.")
    return pd.concat(frames).sort_index()


# -----------------------------------------------------------------------------
# Signal rules and local capital simulation
# -----------------------------------------------------------------------------


def confirmed_boolean_signal(
    raw_signal: pd.Series,
    confirmation_bars: int,
) -> pd.Series:
    bars = max(1, int(confirmation_bars))
    return (
        raw_signal.fillna(False).astype(int)
        .rolling(window=bars, min_periods=bars)
        .sum()
        .eq(bars)
    )


def add_decision_signals(
    predictions: pd.DataFrame,
    bottom_threshold: float,
    top_threshold: float,
    config: BacktestConfig,
) -> pd.DataFrame:
    result = predictions.copy()

    entry_filter = result["rsi_14"] <= config.entry_max_rsi
    if config.entry_require_above_ema50:
        entry_filter &= result["close"] >= result["ema_50"]

    result["predicted_bottom_signal"] = (
        (result["bottom_probability"] >= bottom_threshold)
        & entry_filter
    )
    result["predicted_top_signal"] = (
        result["top_probability"] >= top_threshold
    )

    available_emas = {5, 10, 20, 50}
    requested_emas = {
        config.trend_exit_ema_fast,
        config.trend_exit_ema_slow,
        config.trend_pullback_ema,
        config.bull_regime_ema_fast,
        config.bull_regime_ema_slow,
        config.bull_exit_ema_fast,
        config.bull_exit_ema_slow,
    }
    if not requested_emas.issubset(available_emas):
        raise ValueError(
            "Trend and pullback EMA parameters must be among "
            "5, 10, 20, 50."
        )

    regime_fast = result[f"ema_{config.bull_regime_ema_fast}"]
    regime_slow = result[f"ema_{config.bull_regime_ema_slow}"]
    bull_regime = regime_fast > regime_slow
    if config.bull_regime_require_price_above_slow:
        bull_regime &= result["close"] >= regime_slow
    if config.bull_regime_require_slow_ema_rising:
        bull_regime &= regime_slow.diff() > 0

    result["bull_regime_signal"] = (
        bull_regime
        if config.adaptive_bull_regime_enabled
        else pd.Series(False, index=result.index)
    )
    result["bull_regime_entry_signal"] = (
        bool(config.bull_regime_entry_enabled)
        & confirmed_boolean_signal(
            result["bull_regime_signal"],
            config.bull_regime_entry_confirmation_bars,
        )
    )

    pullback_ema = result[f"ema_{config.trend_pullback_ema}"]
    pullback_touched = (
        result["low"]
        <= pullback_ema
        * (1 + config.trend_pullback_touch_tolerance)
    )
    pullback_recovered = result["close"] >= pullback_ema
    pullback_rsi = result["rsi_14"].between(
        config.trend_pullback_rsi_min,
        config.trend_pullback_rsi_max,
        inclusive="both",
    )
    pullback_positive = (
        result["return_1"] > 0
        if config.trend_pullback_require_positive_return
        else pd.Series(True, index=result.index)
    )
    result["trend_pullback_signal"] = (
        bool(config.trend_pullback_entry_enabled)
        & result["bull_regime_signal"]
        & pullback_touched
        & pullback_recovered
        & pullback_rsi
        & pullback_positive
    )

    normal_fast = result[f"ema_{config.trend_exit_ema_fast}"]
    normal_slow = result[f"ema_{config.trend_exit_ema_slow}"]
    normal_breakdown_raw = (
        (result["close"] < normal_slow)
        & (normal_fast < normal_slow)
        & (result["return_5"] < 0)
    )
    if config.trend_breakdown_require_slow_ema_decline:
        normal_breakdown_raw &= normal_slow.diff() < 0

    bull_fast = result[f"ema_{config.bull_exit_ema_fast}"]
    bull_slow = result[f"ema_{config.bull_exit_ema_slow}"]
    bull_breakdown_raw = (
        (result["close"] < bull_slow)
        & (bull_fast < bull_slow)
        & (result["return_5"] < 0)
    )
    if config.bull_exit_require_slow_ema_decline:
        bull_breakdown_raw &= bull_slow.diff() < 0

    result["normal_trend_breakdown_raw_signal"] = (
        normal_breakdown_raw
    )
    result["bull_trend_breakdown_raw_signal"] = (
        bull_breakdown_raw
    )
    result["normal_trend_breakdown_signal"] = (
        confirmed_boolean_signal(
            normal_breakdown_raw,
            config.trend_breakdown_confirmation_bars,
        )
    )
    result["bull_trend_breakdown_signal"] = (
        confirmed_boolean_signal(
            bull_breakdown_raw,
            config.bull_exit_confirmation_bars,
        )
    )

    result["adaptive_exit_mode"] = np.where(
        bool(config.adaptive_bull_regime_enabled)
        & result["bull_regime_signal"],
        "BULL",
        "NORMAL",
    )
    use_bull_exit = result["adaptive_exit_mode"].eq("BULL")
    result["trend_breakdown_raw_signal"] = (
        result["normal_trend_breakdown_raw_signal"].where(
            ~use_bull_exit,
            result["bull_trend_breakdown_raw_signal"],
        )
    )
    result["trend_breakdown_signal"] = (
        result["normal_trend_breakdown_signal"].where(
            ~use_bull_exit,
            result["bull_trend_breakdown_signal"],
        )
    ).astype(bool)

    return result


def round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def calculate_reference_fees(
    side: str,
    quantity: float,
    price: float,
    config: BacktestConfig,
) -> dict[str, float]:
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }

    normalized_side = side.upper()
    trade_value = quantity * price
    commission = round_fee_to_cent(trade_value * config.commission_rate)
    cat = round_fee_to_cent(quantity * config.cat_fee_per_share)
    sec = 0.0
    taf = 0.0
    if normalized_side == "SELL":
        sec = round_fee_to_cent(trade_value * config.sec_fee_rate)
        taf = round_fee_to_cent(min(quantity * config.taf_fee_per_share, config.taf_fee_cap))
    elif normalized_side != "BUY":
        raise ValueError(f"Unsupported side: {side}")

    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def buy_quantity(
    cash: float,
    price: float,
    config: BacktestConfig,
) -> tuple[float, dict[str, float]]:
    quantity = cash / price
    for _ in range(25):
        fees = calculate_reference_fees("BUY", quantity, price, config)
        next_quantity = max(0.0, (cash - fees["total_fee"]) / price)
        if not config.fractional_shares:
            next_quantity = float(math.floor(next_quantity))
        if abs(next_quantity - quantity) < 1e-10:
            quantity = next_quantity
            break
        quantity = next_quantity
    fees = calculate_reference_fees("BUY", quantity, price, config)
    return quantity, fees


def apply_slippage(price: float, side: str, config: BacktestConfig) -> float:
    adjustment = config.slippage_bps / 10_000
    return price * (1 + adjustment if side == "BUY" else 1 - adjustment)




def trade_record(
    timestamp: Any,
    action: str,
    reason: str,
    execution_price: float,
    quantity: float,
    gross_value: float,
    fees: dict[str, float],
    realized_pnl: float,
    cash: float,
    shares: float,
    *,
    entry_timestamp: Any | None = None,
    entry_price: float = float("nan"),
    holding_bars: int = 0,
    position_return: float = 0.0,
) -> dict[str, Any]:
    """Create one MongoDB-friendly simulated trade record."""
    return {
        "timestamp": bson_value(pd.Timestamp(timestamp)),
        "action": action,
        "reason": reason,
        "execution_price": float(execution_price),
        "quantity": float(quantity),
        "gross_trade_value": float(gross_value),
        "commission_fee": float(fees.get("commission_fee", 0.0)),
        "sec_fee": float(fees.get("sec_fee", 0.0)),
        "taf_fee": float(fees.get("taf_fee", 0.0)),
        "cat_fee": float(fees.get("cat_fee", 0.0)),
        "total_fee": float(fees.get("total_fee", 0.0)),
        "realized_pnl": float(realized_pnl),
        "position_return": float(position_return),
        "holding_bars": int(holding_bars),
        "entry_timestamp": (
            bson_value(pd.Timestamp(entry_timestamp))
            if entry_timestamp is not None
            else None
        ),
        "entry_price": (
            float(entry_price)
            if np.isfinite(entry_price)
            else None
        ),
        "cash_after_trade": float(cash),
        "shares_after_trade": float(shares),
    }


def fibonacci_levels_at_signal(
    signals: pd.DataFrame,
    position: int,
    config: BacktestConfig,
) -> tuple[float, float, float]:
    """Calculate Fibonacci levels using only bars known at the signal close."""
    swing_start = max(0, position - config.fibonacci_swing_lookback + 1)
    low_start = max(0, position - config.fibonacci_low_lookback + 1)

    swing_frame = signals.iloc[swing_start : position + 1]
    low_frame = signals.iloc[low_start : position + 1]

    swing_low = float(low_frame["low"].min())
    swing_high = float(swing_frame["high"].max())
    amplitude = max(0.0, swing_high - swing_low)
    target = swing_low + amplitude * config.fibonacci_target_ratio
    return swing_low, swing_high, target



def first_touch_exit_risk_labels(
    bars: pd.DataFrame,
    horizon: int,
    down_barrier: float,
    up_barrier: float,
) -> pd.DataFrame:
    """Build a no-lookahead training label from future barrier first touches.

    1 = downside barrier is reached before upside continuation.
    0 = upside barrier is reached first, or neither barrier is reached within
        the horizon.
    NaN = both barriers are touched inside the same weekly OHLC candle, where
          intrabar order cannot be known safely.
    """
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)
    labels = np.full(len(bars), np.nan)
    first_touch = np.full(len(bars), "unavailable", dtype=object)
    weeks_to_touch = np.full(len(bars), np.nan)

    for position in range(0, len(bars) - horizon):
        entry = close[position]
        if not np.isfinite(entry) or entry <= 0:
            continue
        downside = entry * (1 - down_barrier)
        upside = entry * (1 + up_barrier)
        outcome = 0.0
        status = "none"
        touch_week = float(horizon)

        for step in range(1, horizon + 1):
            down_hit = low[position + step] <= downside
            up_hit = high[position + step] >= upside
            if down_hit and up_hit:
                outcome = np.nan
                status = "ambiguous_same_week"
                touch_week = float(step)
                break
            if down_hit:
                outcome = 1.0
                status = "downside_first"
                touch_week = float(step)
                break
            if up_hit:
                outcome = 0.0
                status = "upside_first"
                touch_week = float(step)
                break

        labels[position] = outcome
        first_touch[position] = status
        weeks_to_touch[position] = touch_week

    return pd.DataFrame(
        {
            "actual_exit_risk": labels,
            "exit_risk_first_touch": first_touch,
            "exit_risk_weeks_to_touch": weeks_to_touch,
        },
        index=bars.index,
    )


def build_exit_risk_dataset(
    weekly_bars: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    daily_swing = is_daily_swing_exit_risk(config)
    horizon = (
        config.swing_exit_horizon_days
        if daily_swing
        else config.exit_risk_horizon_weeks
    )
    minimum_rows = (
        config.swing_exit_minimum_training_rows
        if daily_swing
        else config.exit_risk_minimum_training_rows
    )
    label_config = replace(
        config,
        timeframe="1Day" if daily_swing else "1Week",
        future_horizon=horizon,
        minimum_training_rows=minimum_rows,
    )
    data = build_dataset(weekly_bars, label_config)
    source = weekly_bars.copy()
    close = source["close"]
    high = source["high"]

    data["exit_return_4"] = close.pct_change(4)
    data["exit_return_8"] = close.pct_change(8)
    data["exit_return_13"] = close.pct_change(13)
    data["exit_return_26"] = close.pct_change(26)

    high_26 = high.rolling(26).max()
    high_52 = high.rolling(52).max()
    data["exit_distance_from_high_26"] = safe_divide(close, high_26) - 1
    data["exit_distance_from_high_52"] = safe_divide(close, high_52) - 1

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    data["exit_ema10_slope_4"] = ema10.pct_change(4)
    data["exit_ema20_slope_4"] = ema20.pct_change(4)
    data["exit_momentum_deceleration"] = (
        close.pct_change(4) - close.pct_change(13) * (4 / 13)
    )
    weekly_rsi = rsi(close, 14)
    data["exit_rsi_change_4"] = weekly_rsi.diff(4)

    weekly_atr = true_range(source).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    atr_pct = safe_divide(weekly_atr, close)
    data["exit_atr_expansion_13"] = (
        safe_divide(atr_pct, atr_pct.rolling(13).mean()) - 1
    )
    upper_wick = high - pd.concat(
        [source["open"], source["close"]], axis=1
    ).max(axis=1)
    data["exit_upper_wick_atr"] = safe_divide(upper_wick, weekly_atr)

    labels = first_touch_exit_risk_labels(
        source,
        (
            config.swing_exit_horizon_days
            if daily_swing
            else config.exit_risk_horizon_weeks
        ),
        (
            config.swing_exit_down_barrier
            if daily_swing
            else config.exit_risk_down_barrier
        ),
        (
            config.swing_exit_up_barrier
            if daily_swing
            else config.exit_risk_up_barrier
        ),
    )
    for column in labels.columns:
        data[column] = labels[column]

    required = EXIT_RISK_FEATURE_COLUMNS + [
        "open", "high", "low", "close", "volume", "actual_exit_risk"
    ]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    data["actual_exit_risk"] = data["actual_exit_risk"].astype(bool)
    return data


def exit_risk_model_config(config: BacktestConfig) -> BacktestConfig:
    if is_daily_swing_exit_risk(config):
        return replace(
            config,
            timeframe="1Day",
            future_horizon=config.swing_exit_horizon_days,
            retrain_every_bars=config.swing_exit_retrain_every_bars,
            minimum_training_rows=config.swing_exit_minimum_training_rows,
        )
    return replace(
        config,
        timeframe="1Week",
        future_horizon=config.exit_risk_horizon_weeks,
        retrain_every_bars=config.exit_risk_retrain_every_bars,
        minimum_training_rows=config.exit_risk_minimum_training_rows,
    )


def exit_risk_calibration_gate(
    result: ThresholdResult,
    config: BacktestConfig,
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if result.precision < config.exit_risk_min_precision:
        failures.append(
            "precision "
            f"{result.precision:.3f} < {config.exit_risk_min_precision:.3f}"
        )
    if result.recall < config.exit_risk_min_recall:
        failures.append(
            "recall "
            f"{result.recall:.3f} < {config.exit_risk_min_recall:.3f}"
        )
    if result.predicted_events < config.exit_risk_min_calibration_signals:
        failures.append(
            "signals "
            f"{result.predicted_events} < "
            f"{config.exit_risk_min_calibration_signals}"
        )

    return not failures, failures


def add_exit_risk_signals(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    calibration_gate_passed: bool = True,
) -> pd.DataFrame:
    result = predictions.copy()
    candidate_signal = (
        result["exit_risk_probability"] >= float(threshold)
    )
    result["exit_risk_candidate_signal"] = candidate_signal

    if calibration_gate_passed:
        result["predicted_exit_risk_signal"] = candidate_signal
    else:
        result["predicted_exit_risk_signal"] = False

    result["exit_risk_calibration_gate_passed"] = bool(
        calibration_gate_passed
    )
    result["predicted_top_signal"] = result[
        "predicted_exit_risk_signal"
    ]
    result["top_probability"] = result["exit_risk_probability"]
    result["actual_top"] = result["actual_exit_risk"]
    result["baseline_top_signal"] = False

    # Re-entry is deliberately based only on information known at the
    # completed weekly close. It is not a future label.
    ema10 = result["ema_10"]
    ema20 = result["ema_20"]
    result["exit_risk_trend_resumption_signal"] = (
        (result["close"] > ema10)
        & (ema10 > ema20)
        & (ema10.pct_change(2) > 0)
        & (result["exit_return_4"] > 0)
    ).fillna(False)
    return result


def build_exit_risk_execution_frame(
    daily_bars: pd.DataFrame,
    structural_predictions: pd.DataFrame,
    weekly_predictions: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    frame = build_mtf_execution_frame(
        daily_bars,
        structural_predictions,
        weekly_predictions,
        config,
    )
    frame["exit_risk_probability"] = frame["top_probability"]
    frame["predicted_exit_risk_signal"] = frame["predicted_top_signal"]
    frame["predicted_exit_risk_signal_source_timestamp"] = frame[
        "predicted_top_signal_source_timestamp"
    ]
    frame = map_release_events_to_daily(
        frame,
        weekly_predictions,
        signal_column="exit_risk_trend_resumption_signal",
        probability_column="exit_risk_probability",
        target_signal_column="exit_risk_trend_resumption_signal",
        target_probability_column="exit_risk_reentry_probability",
    )
    return frame


def threshold_result_at(
    actual: pd.Series,
    probability: pd.Series,
    threshold: float,
    tolerance_bars: int,
) -> ThresholdResult:
    predicted = probability >= float(threshold)
    metrics = event_metrics(actual, predicted, tolerance_bars)
    return ThresholdResult(
        threshold=float(threshold),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        predicted_events=int(metrics["predicted_events"]),
        actual_events=int(metrics["actual_events"]),
    )


def mtf_top_model_config(config: BacktestConfig) -> BacktestConfig:
    return replace(
        config,
        timeframe=config.mtf_top_signal_timeframe,
        future_horizon=config.mtf_top_future_horizon,
        extrema_lookback=config.mtf_top_extrema_lookback,
        reversal_threshold=config.mtf_top_reversal_threshold,
        extrema_tolerance=config.mtf_top_extrema_tolerance,
        event_tolerance_bars=1,
        retrain_every_bars=config.mtf_top_retrain_every_bars,
        minimum_training_rows=config.mtf_top_minimum_training_rows,
        strategy_mode="BOTTOM_REVERSAL_TOP_EXIT",
        exit_top_probability=True,
        exit_fibonacci_target=False,
        exit_trend_breakdown=False,
        exit_atr_trailing_stop=False,
    )


def map_release_events_to_daily(
    daily: pd.DataFrame,
    source: pd.DataFrame,
    *,
    signal_column: str,
    probability_column: str,
    target_signal_column: str,
    target_probability_column: str,
) -> pd.DataFrame:
    result = daily.copy()
    result[target_signal_column] = False
    result[target_probability_column] = np.nan
    result[f"{target_signal_column}_source_timestamp"] = None

    if result.empty or source.empty:
        return result

    daily_index = pd.DatetimeIndex(result.index)
    for timestamp, row in source.iterrows():
        position = int(daily_index.searchsorted(pd.Timestamp(timestamp), side="left"))
        if position >= len(result):
            continue
        release_timestamp = result.index[position]
        result.at[release_timestamp, target_signal_column] = bool(
            row.get(signal_column, False)
        )
        result.at[release_timestamp, target_probability_column] = pd.to_numeric(
            pd.Series([row.get(probability_column)]), errors="coerce"
        ).iloc[0]
        result.at[
            release_timestamp,
            f"{target_signal_column}_source_timestamp",
        ] = bson_value(pd.Timestamp(timestamp))
    return result


def build_mtf_execution_frame(
    daily_bars: pd.DataFrame,
    structural_predictions: pd.DataFrame,
    weekly_predictions: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    start = max(
        pd.Timestamp(structural_predictions.index.min()),
        pd.Timestamp(weekly_predictions.index.min()),
    )

    # Calculate daily indicators on the complete source history before
    # trimming to the test period. This preserves the EMA warm-up without
    # exposing any future observation to a decision.
    daily = daily_bars.copy()
    ema_window = int(config.mtf_daily_confirmation_ema)
    daily_ema = daily["close"].ewm(span=ema_window, adjust=False).mean()
    daily["mtf_daily_ema"] = daily_ema
    daily["mtf_daily_return_1"] = daily["close"].pct_change()
    daily["mtf_daily_ema_declining"] = daily_ema.diff() < 0
    daily["mtf_daily_lower_high"] = daily["high"] < daily["high"].shift(1)

    rolling_high_window = int(config.mtf_top_high_lookback_weeks) * 5
    daily["mtf_rolling_high"] = daily["high"].rolling(
        window=rolling_high_window,
        min_periods=rolling_high_window,
    ).max()
    daily["mtf_distance_from_high"] = (
        daily["mtf_rolling_high"] - daily["close"]
    ) / daily["mtf_rolling_high"]
    daily["mtf_distance_from_high"] = daily[
        "mtf_distance_from_high"
    ].clip(lower=0)

    raw_confirmation = daily["close"] < daily_ema
    if config.mtf_daily_require_negative_return:
        raw_confirmation &= daily["mtf_daily_return_1"] < 0
    if config.mtf_daily_require_ema_decline:
        raw_confirmation &= daily["mtf_daily_ema_declining"]
    if config.mtf_daily_require_lower_high:
        raw_confirmation &= daily["mtf_daily_lower_high"]

    daily["mtf_daily_confirmation_raw"] = raw_confirmation.fillna(False)
    daily["mtf_daily_confirmation_signal"] = confirmed_boolean_signal(
        daily["mtf_daily_confirmation_raw"],
        config.mtf_daily_confirmation_bars,
    )
    daily = daily.loc[daily.index >= start].copy()
    if len(daily) < 30:
        raise ValueError(
            "The multi-timeframe daily execution period contains fewer than "
            "30 sessions."
        )

    daily = map_release_events_to_daily(
        daily,
        structural_predictions,
        signal_column="predicted_bottom_signal",
        probability_column="bottom_probability",
        target_signal_column="predicted_bottom_signal",
        target_probability_column="bottom_probability",
    )
    daily = map_release_events_to_daily(
        daily,
        weekly_predictions,
        signal_column="predicted_top_signal",
        probability_column="top_probability",
        target_signal_column="predicted_top_signal",
        target_probability_column="top_probability",
    )

    # Preserve the actual event labels only on their native release dates.
    daily["actual_bottom"] = False
    daily["actual_top"] = False
    daily["baseline_bottom_signal"] = False
    daily["baseline_top_signal"] = False
    daily["bull_regime_signal"] = False
    daily["trend_pullback_signal"] = False
    daily["bull_regime_entry_signal"] = False
    daily["trend_breakdown_signal"] = False
    daily["atr_14"] = true_range(daily).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    return daily


def simulate_mtf_strategy(
    daily_signals: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = daily_signals.copy()
    columns = [
        "trade_action", "trade_reason", "execution_price", "trade_quantity",
        "gross_trade_value", "commission_fee", "sec_fee", "taf_fee", "cat_fee",
        "total_fee", "cash", "shares", "market_value", "strategy_equity",
        "realized_pnl", "cumulative_realized_pnl", "bars_in_position",
        "trailing_stop_price", "active_trailing_multiplier",
        "fibonacci_swing_low", "fibonacci_swing_high",
        "fibonacci_target_price", "fibonacci_target_ratio",
        "top_signal_seen", "buy_hold_equity", "mtf_top_armed",
        "mtf_top_signal_age_days", "mtf_active_top_probability",
        "mtf_active_top_signal_timestamp", "mtf_position_return",
        "mtf_top_position_return_ok", "mtf_top_high_proximity_ok",
        "mtf_top_exit_guard_passed", "mtf_top_guard_blocked",
    ]
    for column in columns:
        result[column] = np.nan
    result["trade_action"] = ""
    result["trade_reason"] = ""
    result["top_signal_seen"] = False
    result["mtf_top_armed"] = False
    result["mtf_active_top_signal_timestamp"] = None
    result["mtf_top_position_return_ok"] = False
    result["mtf_top_high_proximity_ok"] = False
    result["mtf_top_exit_guard_passed"] = False
    result["mtf_top_guard_blocked"] = False

    cash = config.initial_capital
    shares = 0.0
    entry_total_cost = 0.0
    active_entry_timestamp: Any | None = None
    active_entry_price = np.nan
    cumulative_realized = 0.0
    bars_in_position = 0
    last_exit_position = -10_000

    pending_action: str | None = None
    pending_reason = ""
    pending_top_timestamp: Any | None = None
    pending_top_probability = np.nan
    pending_confirmation_position_return = np.nan
    pending_confirmation_distance_from_high = np.nan
    pending_confirmation_rolling_high = np.nan

    top_armed = False
    top_signal_position = -1
    active_top_timestamp: Any | None = None
    active_top_probability = np.nan
    trade_records: list[dict[str, Any]] = []
    exit_risk_mode = is_exit_risk_strategy(config)
    exit_signal_reason = (
        exit_risk_sell_reason(config)
        if exit_risk_mode
        else "MTF_TOP_WEEKLY_DAILY_CONFIRMATION"
    )
    awaiting_trend_reentry = False
    last_model_exit_price = np.nan

    first_open = float(result.iloc[0]["open"])
    bh_buy_price = apply_slippage(first_open, "BUY", config)
    bh_quantity, bh_buy_fees = buy_quantity(
        config.initial_capital, bh_buy_price, config
    )
    bh_cash = (
        config.initial_capital
        - bh_quantity * bh_buy_price
        - bh_buy_fees["total_fee"]
    )

    for position, (timestamp, row) in enumerate(result.iterrows()):
        action = "HOLD_POSITION" if shares > 0 else "HOLD_CASH"
        reason = ""
        execution_price = np.nan
        quantity = 0.0
        gross_value = 0.0
        fees = calculate_reference_fees("BUY", 0.0, 0.0, config)
        realized_pnl = 0.0

        if pending_action == "BUY" and shares <= 0:
            execution_price = apply_slippage(float(row["open"]), "BUY", config)
            quantity, fees = buy_quantity(cash, execution_price, config)
            if quantity > 0:
                gross_value = quantity * execution_price
                cash -= gross_value + fees["total_fee"]
                shares = quantity
                entry_total_cost = gross_value + fees["total_fee"]
                active_entry_timestamp = timestamp
                active_entry_price = execution_price
                bars_in_position = 0
                top_armed = False
                top_signal_position = -1
                active_top_timestamp = None
                active_top_probability = np.nan
                action = "BUY"
                reason = pending_reason
                awaiting_trend_reentry = False
                trade_records.append(
                    trade_record(
                        timestamp, action, reason, execution_price, quantity,
                        gross_value, fees, realized_pnl, cash, shares,
                        entry_timestamp=active_entry_timestamp,
                        entry_price=active_entry_price,
                        holding_bars=0,
                        position_return=0.0,
                    )
                )

        elif pending_action == "SELL" and shares > 0:
            execution_price = apply_slippage(float(row["open"]), "SELL", config)
            quantity = shares
            fees = calculate_reference_fees("SELL", quantity, execution_price, config)
            gross_value = quantity * execution_price
            proceeds = gross_value - fees["total_fee"]
            position_cost = entry_total_cost
            completed_holding_bars = bars_in_position
            completed_entry_timestamp = active_entry_timestamp
            completed_entry_price = active_entry_price
            realized_pnl = proceeds - position_cost
            position_return = realized_pnl / position_cost if position_cost > 0 else 0.0
            cumulative_realized += realized_pnl
            cash += proceeds
            shares = 0.0
            entry_total_cost = 0.0
            active_entry_timestamp = None
            active_entry_price = np.nan
            bars_in_position = 0
            last_exit_position = position
            action = "SELL"
            reason = pending_reason
            if exit_risk_mode and reason == exit_signal_reason:
                awaiting_trend_reentry = True
                last_model_exit_price = execution_price
            record = trade_record(
                timestamp, action, reason, execution_price, quantity,
                gross_value, fees, realized_pnl, cash, shares,
                entry_timestamp=completed_entry_timestamp,
                entry_price=completed_entry_price,
                holding_bars=completed_holding_bars,
                position_return=position_return,
            )
            record.update(
                {
                    "mtf_top_signal_timestamp": bson_value(
                        pd.Timestamp(pending_top_timestamp)
                    ) if pending_top_timestamp is not None else None,
                    "mtf_top_probability": (
                        float(pending_top_probability)
                        if np.isfinite(pending_top_probability)
                        else None
                    ),
                    "mtf_daily_confirmation_bars": int(
                        config.mtf_daily_confirmation_bars
                    ),
                    "mtf_confirmation_position_return": (
                        float(pending_confirmation_position_return)
                        if np.isfinite(pending_confirmation_position_return)
                        else None
                    ),
                    "mtf_confirmation_distance_from_high": (
                        float(pending_confirmation_distance_from_high)
                        if np.isfinite(pending_confirmation_distance_from_high)
                        else None
                    ),
                    "mtf_confirmation_rolling_high": (
                        float(pending_confirmation_rolling_high)
                        if np.isfinite(pending_confirmation_rolling_high)
                        else None
                    ),
                    "mtf_min_position_return_required": float(
                        config.mtf_top_min_position_return
                    ),
                    "mtf_max_distance_from_high_required": float(
                        config.mtf_top_max_distance_from_high
                    ),
                    "mtf_high_lookback_weeks": int(
                        config.mtf_top_high_lookback_weeks
                    ),
                    "exit_risk_probability": (
                        float(pending_top_probability)
                        if exit_risk_mode
                        and np.isfinite(pending_top_probability)
                        else None
                    ),
                    "exit_risk_down_barrier": (
                        float(config.exit_risk_down_barrier)
                        if exit_risk_mode else None
                    ),
                    "exit_risk_up_barrier": (
                        float(config.exit_risk_up_barrier)
                        if exit_risk_mode else None
                    ),
                    "exit_risk_horizon_weeks": (
                        int(config.exit_risk_horizon_weeks)
                        if exit_risk_mode else None
                    ),
                }
            )
            trade_records.append(record)
            top_armed = False
            top_signal_position = -1
            active_top_timestamp = None
            active_top_probability = np.nan

        pending_action = None
        pending_reason = ""
        pending_top_timestamp = None
        pending_top_probability = np.nan
        pending_confirmation_position_return = np.nan
        pending_confirmation_distance_from_high = np.nan
        pending_confirmation_rolling_high = np.nan

        current_position_return = np.nan
        current_rolling_high = pd.to_numeric(
            pd.Series([row.get("mtf_rolling_high")]),
            errors="coerce",
        ).iloc[0]
        current_distance_from_high = pd.to_numeric(
            pd.Series([row.get("mtf_distance_from_high")]),
            errors="coerce",
        ).iloc[0]
        position_return_ok = False
        high_proximity_ok = False
        exit_guard_passed = False
        guard_blocked = False

        if shares > 0:
            bars_in_position += 1
            current_market_value = shares * float(row["close"])
            current_position_return = (
                (current_market_value - entry_total_cost) / entry_total_cost
                if entry_total_cost > 0
                else np.nan
            )
            position_return_ok = (
                np.isfinite(current_position_return)
                and current_position_return
                >= config.mtf_top_min_position_return
            )
            high_proximity_ok = (
                np.isfinite(current_distance_from_high)
                and current_distance_from_high
                <= config.mtf_top_max_distance_from_high
            )
            exit_guard_passed = position_return_ok and high_proximity_ok
            if bool(row.get("predicted_top_signal", False)):
                top_armed = True
                top_signal_position = position
                source_timestamp = row.get(
                    "predicted_top_signal_source_timestamp"
                )
                active_top_timestamp = (
                    source_timestamp if source_timestamp is not None else timestamp
                )
                active_top_probability = pd.to_numeric(
                    pd.Series([row.get("top_probability")]), errors="coerce"
                ).iloc[0]

            if (
                top_armed
                and position - top_signal_position
                > config.mtf_top_signal_valid_days
            ):
                top_armed = False
                top_signal_position = -1
                active_top_timestamp = None
                active_top_probability = np.nan

            daily_confirmation = bool(
                row.get("mtf_daily_confirmation_signal", False)
            )
            guard_blocked = (
                top_armed
                and bars_in_position >= config.minimum_holding_bars
                and daily_confirmation
                and not exit_guard_passed
            )

            if (
                top_armed
                and bars_in_position >= config.minimum_holding_bars
                and daily_confirmation
                and exit_guard_passed
                and position < len(result) - 1
            ):
                pending_action = "SELL"
                pending_reason = exit_signal_reason
                pending_top_timestamp = active_top_timestamp
                pending_top_probability = active_top_probability
                pending_confirmation_position_return = (
                    current_position_return
                )
                pending_confirmation_distance_from_high = (
                    current_distance_from_high
                )
                pending_confirmation_rolling_high = current_rolling_high
        else:
            cooldown_ok = position - last_exit_position > config.entry_cooldown_bars
            reentry_cooldown_ok = (
                position - last_exit_position
                > config.exit_risk_reentry_cooldown_days
            )
            if (
                exit_risk_mode
                and config.exit_risk_reentry_enabled
                and awaiting_trend_reentry
                and reentry_cooldown_ok
                and bool(row.get("exit_risk_trend_resumption_signal", False))
                and position < len(result) - 1
            ):
                pending_action = "BUY"
                pending_reason = "TREND_RESUMPTION_REENTRY"
            elif (
                cooldown_ok
                and bool(row.get("predicted_bottom_signal", False))
                and position < len(result) - 1
            ):
                pending_action = "BUY"
                pending_reason = "BOTTOM_PROBABILITY"

        market_value = shares * float(row["close"])
        strategy_equity = cash + market_value
        bh_equity = bh_cash + bh_quantity * float(row["close"])
        age = position - top_signal_position if top_armed else np.nan

        result.at[timestamp, "trade_action"] = action
        result.at[timestamp, "trade_reason"] = reason
        result.at[timestamp, "execution_price"] = execution_price
        result.at[timestamp, "trade_quantity"] = quantity
        result.at[timestamp, "gross_trade_value"] = gross_value
        for fee_name, fee_value in fees.items():
            result.at[timestamp, fee_name] = fee_value
        result.at[timestamp, "cash"] = cash
        result.at[timestamp, "shares"] = shares
        result.at[timestamp, "market_value"] = market_value
        result.at[timestamp, "strategy_equity"] = strategy_equity
        result.at[timestamp, "realized_pnl"] = realized_pnl
        result.at[timestamp, "cumulative_realized_pnl"] = cumulative_realized
        result.at[timestamp, "bars_in_position"] = bars_in_position
        result.at[timestamp, "top_signal_seen"] = top_armed
        result.at[timestamp, "mtf_top_armed"] = top_armed
        result.at[timestamp, "mtf_top_signal_age_days"] = age
        result.at[timestamp, "mtf_active_top_probability"] = active_top_probability
        result.at[timestamp, "mtf_active_top_signal_timestamp"] = active_top_timestamp
        result.at[timestamp, "mtf_position_return"] = current_position_return
        result.at[timestamp, "mtf_rolling_high"] = current_rolling_high
        result.at[timestamp, "mtf_distance_from_high"] = current_distance_from_high
        result.at[timestamp, "mtf_top_position_return_ok"] = position_return_ok
        result.at[timestamp, "mtf_top_high_proximity_ok"] = high_proximity_ok
        result.at[timestamp, "mtf_top_exit_guard_passed"] = exit_guard_passed
        result.at[timestamp, "mtf_top_guard_blocked"] = guard_blocked
        result.at[timestamp, "buy_hold_equity"] = bh_equity

    if shares > 0:
        timestamp = result.index[-1]
        close_price = apply_slippage(float(result.iloc[-1]["close"]), "SELL", config)
        quantity = shares
        fees = calculate_reference_fees("SELL", quantity, close_price, config)
        gross_value = quantity * close_price
        proceeds = gross_value - fees["total_fee"]
        position_cost = entry_total_cost
        realized_pnl = proceeds - position_cost
        position_return = realized_pnl / position_cost if position_cost > 0 else 0.0
        cumulative_realized += realized_pnl
        cash += proceeds
        result.at[timestamp, "trade_action"] = "FINAL_SELL"
        result.at[timestamp, "trade_reason"] = "FINAL_LIQUIDATION"
        result.at[timestamp, "execution_price"] = close_price
        result.at[timestamp, "trade_quantity"] = quantity
        result.at[timestamp, "gross_trade_value"] = gross_value
        for fee_name, fee_value in fees.items():
            result.at[timestamp, fee_name] = fee_value
        result.at[timestamp, "cash"] = cash
        result.at[timestamp, "shares"] = 0.0
        result.at[timestamp, "market_value"] = 0.0
        result.at[timestamp, "strategy_equity"] = cash
        result.at[timestamp, "realized_pnl"] = realized_pnl
        result.at[timestamp, "cumulative_realized_pnl"] = cumulative_realized
        result.at[timestamp, "bars_in_position"] = 0
        result.at[timestamp, "mtf_top_armed"] = False
        result.at[timestamp, "top_signal_seen"] = False
        trade_records.append(
            trade_record(
                timestamp, "FINAL_SELL", "FINAL_LIQUIDATION", close_price,
                quantity, gross_value, fees, realized_pnl, cash, 0.0,
                entry_timestamp=active_entry_timestamp,
                entry_price=active_entry_price,
                holding_bars=bars_in_position,
                position_return=position_return,
            )
        )

    last_timestamp = result.index[-1]
    bh_sell_price = apply_slippage(float(result.iloc[-1]["close"]), "SELL", config)
    bh_sell_fees = calculate_reference_fees("SELL", bh_quantity, bh_sell_price, config)
    bh_final_cash = bh_cash + bh_quantity * bh_sell_price - bh_sell_fees["total_fee"]
    result.at[last_timestamp, "buy_hold_equity"] = bh_final_cash
    return result, pd.DataFrame(trade_records)


def simulate_strategy(
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = signals.copy()

    strategy_mode = config.strategy_mode
    bottom_entry_allowed = strategy_mode in {
        "ADAPTIVE_HYBRID",
        "BOTTOM_REVERSAL_FIBONACCI",
        "BOTTOM_REVERSAL_TOP_EXIT",
        "BOTTOM_ENTRY_MTF_TOP_EXIT",
    }
    pullback_entry_allowed = strategy_mode in {
        "ADAPTIVE_HYBRID",
        "STRUCTURAL_TREND",
    }
    bull_regime_entry_allowed = (
        strategy_mode in {"ADAPTIVE_HYBRID", "STRUCTURAL_TREND"}
        and config.bull_regime_entry_enabled
    )
    fibonacci_exit_allowed = (
        strategy_mode == "BOTTOM_REVERSAL_FIBONACCI"
        or (
            strategy_mode == "ADAPTIVE_HYBRID"
            and config.exit_fibonacci_target
        )
    )
    top_exit_allowed = (
        strategy_mode == "BOTTOM_REVERSAL_TOP_EXIT"
        or (
            strategy_mode == "ADAPTIVE_HYBRID"
            and config.exit_top_probability
        )
    )
    trend_exit_allowed = (
        strategy_mode in {"ADAPTIVE_HYBRID", "STRUCTURAL_TREND"}
        and config.exit_trend_breakdown
    )
    atr_exit_allowed = (
        strategy_mode == "ADAPTIVE_HYBRID"
        and config.exit_atr_trailing_stop
    )
    columns = [
        "trade_action", "trade_reason", "execution_price", "trade_quantity",
        "gross_trade_value", "commission_fee", "sec_fee", "taf_fee", "cat_fee",
        "total_fee", "cash", "shares", "market_value", "strategy_equity",
        "realized_pnl", "cumulative_realized_pnl", "bars_in_position",
        "trailing_stop_price", "active_trailing_multiplier",
        "fibonacci_swing_low", "fibonacci_swing_high",
        "fibonacci_target_price", "fibonacci_target_ratio",
        "top_signal_seen", "buy_hold_equity",
    ]
    for column in columns:
        result[column] = np.nan
    result["trade_action"] = ""
    result["trade_reason"] = ""
    result["top_signal_seen"] = False

    cash = config.initial_capital
    shares = 0.0
    entry_total_cost = 0.0
    active_entry_timestamp: Any | None = None
    active_entry_price = np.nan
    cumulative_realized = 0.0
    bars_in_position = 0
    peak_close = 0.0

    active_swing_low = np.nan
    active_swing_high = np.nan
    active_fibonacci_target = np.nan
    top_seen = False

    pending_action: str | None = None
    pending_reason = ""
    pending_swing_low = np.nan
    pending_swing_high = np.nan
    pending_fibonacci_target = np.nan

    last_exit_position = -10_000
    trade_records: list[dict[str, Any]] = []

    first_open = float(result.iloc[0]["open"])
    bh_buy_price = apply_slippage(first_open, "BUY", config)
    bh_quantity, bh_buy_fees = buy_quantity(config.initial_capital, bh_buy_price, config)
    bh_cash = config.initial_capital - bh_quantity * bh_buy_price - bh_buy_fees["total_fee"]

    for position, (timestamp, row) in enumerate(result.iterrows()):
        action = "HOLD_POSITION" if shares > 0 else "HOLD_CASH"
        reason = ""
        execution_price = np.nan
        quantity = 0.0
        gross_value = 0.0
        fees = calculate_reference_fees("BUY", 0.0, 0.0, config)
        realized_pnl = 0.0

        if pending_action == "BUY" and shares <= 0:
            execution_price = apply_slippage(float(row["open"]), "BUY", config)
            quantity, fees = buy_quantity(cash, execution_price, config)
            if quantity > 0:
                gross_value = quantity * execution_price
                cash -= gross_value + fees["total_fee"]
                shares = quantity
                entry_total_cost = gross_value + fees["total_fee"]
                active_entry_timestamp = timestamp
                active_entry_price = execution_price
                bars_in_position = 0
                peak_close = float(row["close"])
                active_swing_low = pending_swing_low
                active_swing_high = pending_swing_high
                active_fibonacci_target = pending_fibonacci_target
                top_seen = False
                action = "BUY"
                reason = pending_reason
                trade_records.append(
                    trade_record(
                        timestamp,
                        action,
                        reason,
                        execution_price,
                        quantity,
                        gross_value,
                        fees,
                        realized_pnl,
                        cash,
                        shares,
                        entry_timestamp=active_entry_timestamp,
                        entry_price=active_entry_price,
                        holding_bars=0,
                        position_return=0.0,
                    )
                )

        elif pending_action == "SELL" and shares > 0:
            execution_price = apply_slippage(float(row["open"]), "SELL", config)
            quantity = shares
            fees = calculate_reference_fees("SELL", quantity, execution_price, config)
            gross_value = quantity * execution_price
            proceeds = gross_value - fees["total_fee"]
            cash += proceeds
            position_cost = entry_total_cost
            completed_holding_bars = bars_in_position
            completed_entry_timestamp = active_entry_timestamp
            completed_entry_price = active_entry_price
            realized_pnl = proceeds - position_cost
            position_return = (
                realized_pnl / position_cost
                if position_cost > 0
                else 0.0
            )
            cumulative_realized += realized_pnl
            shares = 0.0
            entry_total_cost = 0.0
            active_entry_timestamp = None
            active_entry_price = np.nan
            bars_in_position = 0
            peak_close = 0.0
            active_swing_low = np.nan
            active_swing_high = np.nan
            active_fibonacci_target = np.nan
            top_seen = False
            last_exit_position = position
            action = "SELL"
            reason = pending_reason
            trade_records.append(
                trade_record(
                    timestamp,
                    action,
                    reason,
                    execution_price,
                    quantity,
                    gross_value,
                    fees,
                    realized_pnl,
                    cash,
                    shares,
                    entry_timestamp=completed_entry_timestamp,
                    entry_price=completed_entry_price,
                    holding_bars=completed_holding_bars,
                    position_return=position_return,
                )
            )

        pending_action = None
        pending_reason = ""
        pending_swing_low = np.nan
        pending_swing_high = np.nan
        pending_fibonacci_target = np.nan

        trailing_stop = np.nan
        active_multiplier = np.nan

        if shares > 0:
            bars_in_position += 1
            peak_close = max(peak_close, float(row["close"]))

            if bool(row["predicted_top_signal"]):
                top_seen = True

            active_multiplier = (
                config.tightened_atr_multiplier
                if config.top_tighten_trailing and top_seen
                else config.atr_trailing_multiplier
            )
            trailing_stop = peak_close - active_multiplier * float(row["atr_14"])

            exit_reasons: list[str] = []
            if bars_in_position >= config.minimum_holding_bars:
                if (
                    fibonacci_exit_allowed
                    and np.isfinite(active_fibonacci_target)
                    and float(row["high"]) >= active_fibonacci_target
                ):
                    exit_reasons.append(f"FIBONACCI_{config.fibonacci_target_ratio:.3f}")

                if top_exit_allowed and bool(row["predicted_top_signal"]):
                    exit_reasons.append("TOP_PROBABILITY")

                if trend_exit_allowed and bool(row["trend_breakdown_signal"]):
                    exit_reasons.append("TREND_BREAKDOWN")

                if (
                    atr_exit_allowed
                    and np.isfinite(trailing_stop)
                    and float(row["close"]) <= trailing_stop
                ):
                    exit_reasons.append(
                        "ATR_TRAILING_STOP_TIGHTENED" if top_seen
                        else "ATR_TRAILING_STOP"
                    )

            if exit_reasons and position < len(result) - 1:
                pending_action = "SELL"
                pending_reason = "+".join(exit_reasons)

        else:
            cooldown_ok = (
                position - last_exit_position
                > config.entry_cooldown_bars
            )
            bottom_entry = (
                bottom_entry_allowed
                and bool(row["predicted_bottom_signal"])
            )
            pullback_entry = (
                pullback_entry_allowed
                and bool(row["trend_pullback_signal"])
            )
            bull_regime_entry = (
                bull_regime_entry_allowed
                and bool(row["bull_regime_entry_signal"])
            )
            if (
                cooldown_ok
                and (bottom_entry or pullback_entry or bull_regime_entry)
                and position < len(result) - 1
            ):
                if fibonacci_exit_allowed:
                    swing_low, swing_high, fib_target = (
                        fibonacci_levels_at_signal(
                            result,
                            position,
                            config,
                        )
                    )
                else:
                    swing_low, swing_high, fib_target = (
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                pending_action = "BUY"
                pending_reason = (
                    "BOTTOM_PROBABILITY"
                    if bottom_entry
                    else (
                        "TREND_PULLBACK"
                        if pullback_entry
                        else "BULL_REGIME_ENTRY"
                    )
                )
                pending_swing_low = swing_low
                pending_swing_high = swing_high
                pending_fibonacci_target = fib_target

        market_value = shares * float(row["close"])
        strategy_equity = cash + market_value
        bh_equity = bh_cash + bh_quantity * float(row["close"])

        result.at[timestamp, "trade_action"] = action
        result.at[timestamp, "trade_reason"] = reason
        result.at[timestamp, "execution_price"] = execution_price
        result.at[timestamp, "trade_quantity"] = quantity
        result.at[timestamp, "gross_trade_value"] = gross_value
        for fee_name, fee_value in fees.items():
            result.at[timestamp, fee_name] = fee_value
        result.at[timestamp, "cash"] = cash
        result.at[timestamp, "shares"] = shares
        result.at[timestamp, "market_value"] = market_value
        result.at[timestamp, "strategy_equity"] = strategy_equity
        result.at[timestamp, "realized_pnl"] = realized_pnl
        result.at[timestamp, "cumulative_realized_pnl"] = cumulative_realized
        result.at[timestamp, "bars_in_position"] = bars_in_position
        result.at[timestamp, "trailing_stop_price"] = trailing_stop
        result.at[timestamp, "active_trailing_multiplier"] = active_multiplier
        result.at[timestamp, "fibonacci_swing_low"] = active_swing_low
        result.at[timestamp, "fibonacci_swing_high"] = active_swing_high
        result.at[timestamp, "fibonacci_target_price"] = active_fibonacci_target
        result.at[timestamp, "fibonacci_target_ratio"] = (
            config.fibonacci_target_ratio if shares > 0 else np.nan
        )
        result.at[timestamp, "top_signal_seen"] = top_seen
        result.at[timestamp, "buy_hold_equity"] = bh_equity

    if shares > 0:
        timestamp = result.index[-1]
        close_price = apply_slippage(float(result.iloc[-1]["close"]), "SELL", config)
        quantity = shares
        fees = calculate_reference_fees("SELL", quantity, close_price, config)
        gross_value = quantity * close_price
        proceeds = gross_value - fees["total_fee"]
        position_cost = entry_total_cost
        completed_holding_bars = bars_in_position
        completed_entry_timestamp = active_entry_timestamp
        completed_entry_price = active_entry_price
        realized_pnl = proceeds - position_cost
        position_return = (
            realized_pnl / position_cost
            if position_cost > 0
            else 0.0
        )
        cumulative_realized += realized_pnl
        cash += proceeds
        shares = 0.0

        result.at[timestamp, "trade_action"] = "FINAL_SELL"
        result.at[timestamp, "trade_reason"] = "FINAL_LIQUIDATION"
        result.at[timestamp, "execution_price"] = close_price
        result.at[timestamp, "trade_quantity"] = quantity
        result.at[timestamp, "gross_trade_value"] = gross_value
        for fee_name, fee_value in fees.items():
            result.at[timestamp, fee_name] = fee_value
        result.at[timestamp, "cash"] = cash
        result.at[timestamp, "shares"] = 0.0
        result.at[timestamp, "market_value"] = 0.0
        result.at[timestamp, "strategy_equity"] = cash
        result.at[timestamp, "realized_pnl"] = realized_pnl
        result.at[timestamp, "cumulative_realized_pnl"] = cumulative_realized
        result.at[timestamp, "bars_in_position"] = 0
        result.at[timestamp, "trailing_stop_price"] = np.nan
        result.at[timestamp, "active_trailing_multiplier"] = np.nan
        result.at[timestamp, "fibonacci_swing_low"] = np.nan
        result.at[timestamp, "fibonacci_swing_high"] = np.nan
        result.at[timestamp, "fibonacci_target_price"] = np.nan
        result.at[timestamp, "fibonacci_target_ratio"] = np.nan
        result.at[timestamp, "top_signal_seen"] = False

        trade_records.append(
            trade_record(
                timestamp,
                "FINAL_SELL",
                "FINAL_LIQUIDATION",
                close_price,
                quantity,
                gross_value,
                fees,
                realized_pnl,
                cash,
                0.0,
                entry_timestamp=completed_entry_timestamp,
                entry_price=completed_entry_price,
                holding_bars=completed_holding_bars,
                position_return=position_return,
            )
        )

    last_timestamp = result.index[-1]
    bh_sell_price = apply_slippage(float(result.iloc[-1]["close"]), "SELL", config)
    bh_sell_fees = calculate_reference_fees("SELL", bh_quantity, bh_sell_price, config)
    bh_final_cash = bh_cash + bh_quantity * bh_sell_price - bh_sell_fees["total_fee"]
    result.at[last_timestamp, "buy_hold_equity"] = bh_final_cash

    trades = pd.DataFrame(trade_records)
    return result, trades


# -----------------------------------------------------------------------------
# Metrics and reports
# -----------------------------------------------------------------------------


def safe_roc_auc(actual: pd.Series, probability: pd.Series) -> float:
    if actual.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(actual.astype(int), probability))


def safe_average_precision(actual: pd.Series, probability: pd.Series) -> float:
    if actual.sum() == 0:
        return float("nan")
    return float(average_precision_score(actual.astype(int), probability))


def safe_balanced_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_int = actual.astype(int)
    predicted_int = predicted.astype(int)

    # Balanced accuracy needs both real classes. With a single class the
    # metric is not informative, so return NaN and avoid sklearn's internal
    # confusion-matrix warning.
    if actual_int.nunique(dropna=False) < 2:
        return float("nan")

    return float(
        balanced_accuracy_score(actual_int, predicted_int)
    )


def classifier_metrics(
    actual: pd.Series,
    probability: pd.Series,
    predicted: pd.Series,
    tolerance: int,
) -> dict[str, float | int]:
    actual_int = actual.astype(int)
    predicted_int = predicted.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_int,
        predicted_int,
        average="binary",
        zero_division=0,
    )
    event = event_metrics(actual, predicted, tolerance)
    return {
        "precision_exact": float(precision),
        "recall_exact": float(recall),
        "f1_exact": float(f1),
        "balanced_accuracy": safe_balanced_accuracy(
            actual_int,
            predicted_int,
        ),
        "roc_auc": safe_roc_auc(actual, probability),
        "average_precision": safe_average_precision(actual, probability),
        "brier": float(brier_score_loss(actual_int, probability)),
        "precision_tolerant": float(event["precision"]),
        "recall_tolerant": float(event["recall"]),
        "f1_tolerant": float(event["f1"]),
        "predicted_events": int(event["predicted_events"]),
        "actual_events": int(event["actual_events"]),
    }


def total_return(curve: pd.Series, initial_capital: float) -> float:
    return float(curve.iloc[-1] / initial_capital - 1)


def maximum_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def sharpe_ratio(curve: pd.Series, timeframe: str) -> float:
    returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty or returns.std(ddof=1) == 0:
        return float("nan")
    annualization = {
        "1Day": 252,
        "1Hour": 252 * 6.5,
        "30Min": 252 * 13,
        "15Min": 252 * 26,
        "5Min": 252 * 78,
        "1Week": 52,
        "2Weeks": 26,
        "3Weeks": 52 / 3,
        "4Weeks": 13,
    }[timeframe]
    return float(np.sqrt(annualization) * returns.mean() / returns.std(ddof=1))


def full_series_buy_and_hold(
    bars: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float]:
    buy_price = apply_slippage(float(bars.iloc[0]["open"]), "BUY", config)
    quantity, buy_fees = buy_quantity(config.initial_capital, buy_price, config)
    cash = config.initial_capital - quantity * buy_price - buy_fees["total_fee"]
    curve = cash + quantity * bars["close"]
    sell_price = apply_slippage(float(bars.iloc[-1]["close"]), "SELL", config)
    sell_fees = calculate_reference_fees("SELL", quantity, sell_price, config)
    final_cash = cash + quantity * sell_price - sell_fees["total_fee"]
    curve = curve.copy()
    curve.iloc[-1] = final_cash
    return {
        "ending_capital": float(final_cash),
        "return": float(final_cash / config.initial_capital - 1),
        "maximum_drawdown": maximum_drawdown(curve),
        "sharpe": sharpe_ratio(curve, config.timeframe),
        "fees": float(buy_fees["total_fee"] + sell_fees["total_fee"]),
    }


def split_exit_reason_tokens(reason: Any) -> set[str]:
    return {
        token.strip()
        for token in str(reason or "").split("+")
        if token.strip()
    }


def exit_subset_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "total_realized_pnl": 0.0,
            "average_realized_pnl": 0.0,
            "median_realized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "average_position_return": 0.0,
            "median_position_return": 0.0,
            "average_holding_bars": 0.0,
        }

    pnl = pd.to_numeric(frame["realized_pnl"], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(
        frame.get("position_return", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    holding = pd.to_numeric(
        frame.get("holding_bars", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    return {
        "count": int(len(frame)),
        "total_realized_pnl": float(pnl.sum()),
        "average_realized_pnl": float(pnl.mean()),
        "median_realized_pnl": float(pnl.median()),
        "win_count": wins,
        "loss_count": losses,
        "win_rate": float(wins / len(frame)),
        "average_position_return": float(returns.mean()),
        "median_position_return": float(returns.median()),
        "average_holding_bars": float(holding.mean()),
    }



def annotate_exit_quality(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Add look-ahead evaluation fields to completed exits.

    These fields are created only after the simulation. They never participate
    in signal generation or order decisions.
    """
    result = trades.copy()
    quality_columns = [
        "evaluation_local_peak_price",
        "evaluation_local_peak_timestamp",
        "evaluation_exit_to_peak_gap",
        "evaluation_absolute_peak_gap",
        "evaluation_bars_from_peak",
        "evaluation_top_capture_ratio",
        "evaluation_post_exit_max_upside",
        "evaluation_post_exit_max_drawdown",
    ]
    for column in quality_columns:
        result[column] = np.nan
    result["evaluation_local_peak_timestamp"] = None

    if result.empty or predictions.empty:
        return result

    prediction_index = pd.DatetimeIndex(predictions.index)
    if prediction_index.tz is None:
        prediction_index = prediction_index.tz_localize("UTC")
    else:
        prediction_index = prediction_index.tz_convert("UTC")

    position_by_timestamp = {
        int(timestamp.value): position
        for position, timestamp in enumerate(prediction_index)
    }
    peak_column = "close" if is_swing_timeframe(config.timeframe) else "high"
    trough_column = "close" if is_swing_timeframe(config.timeframe) else "low"
    before_bars = max(1, int(config.event_tolerance_bars) + 1)
    after_bars = max(1, int(config.future_horizon))

    sell_mask = result["action"].isin(["SELL", "FINAL_SELL"])
    for trade_index, trade in result.loc[sell_mask].iterrows():
        timestamp = pd.Timestamp(trade["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        position = position_by_timestamp.get(int(timestamp.value))
        if position is None:
            continue

        execution_price = float(trade["execution_price"])
        if not np.isfinite(execution_price) or execution_price <= 0:
            continue

        window_start = max(0, position - before_bars)
        window_end = min(len(predictions), position + after_bars + 1)
        local_window = predictions.iloc[window_start:window_end]
        local_peak_series = pd.to_numeric(
            local_window[peak_column], errors="coerce"
        ).dropna()
        if local_peak_series.empty:
            continue

        local_peak_timestamp = local_peak_series.idxmax()
        local_peak_price = float(local_peak_series.loc[local_peak_timestamp])
        local_peak_position = position_by_timestamp.get(
            int(pd.Timestamp(local_peak_timestamp).value), position
        )
        peak_gap = execution_price / local_peak_price - 1

        entry_price = pd.to_numeric(
            pd.Series([trade.get("entry_price")]), errors="coerce"
        ).iloc[0]
        top_capture_ratio = np.nan
        if (
            np.isfinite(entry_price)
            and local_peak_price > float(entry_price)
        ):
            top_capture_ratio = (
                execution_price - float(entry_price)
            ) / (local_peak_price - float(entry_price))

        future_window = predictions.iloc[
            position + 1 : min(len(predictions), position + after_bars + 1)
        ]
        post_exit_upside = np.nan
        post_exit_drawdown = np.nan
        if not future_window.empty:
            future_peak = pd.to_numeric(
                future_window[peak_column], errors="coerce"
            ).max()
            future_trough = pd.to_numeric(
                future_window[trough_column], errors="coerce"
            ).min()
            if np.isfinite(future_peak):
                post_exit_upside = float(future_peak) / execution_price - 1
            if np.isfinite(future_trough):
                post_exit_drawdown = float(future_trough) / execution_price - 1

        result.at[trade_index, "evaluation_local_peak_price"] = local_peak_price
        result.at[trade_index, "evaluation_local_peak_timestamp"] = bson_value(
            pd.Timestamp(local_peak_timestamp)
        )
        result.at[trade_index, "evaluation_exit_to_peak_gap"] = peak_gap
        result.at[trade_index, "evaluation_absolute_peak_gap"] = abs(peak_gap)
        result.at[trade_index, "evaluation_bars_from_peak"] = (
            position - local_peak_position
        )
        result.at[trade_index, "evaluation_top_capture_ratio"] = top_capture_ratio
        result.at[trade_index, "evaluation_post_exit_max_upside"] = post_exit_upside
        result.at[trade_index, "evaluation_post_exit_max_drawdown"] = post_exit_drawdown

    return result


def exit_quality_subset_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "average_exit_to_peak_gap": 0.0,
            "median_exit_to_peak_gap": 0.0,
            "average_absolute_peak_gap": 0.0,
            "average_bars_from_peak": 0.0,
            "median_bars_from_peak": 0.0,
            "average_top_capture_ratio": 0.0,
            "median_top_capture_ratio": 0.0,
            "average_post_exit_max_upside": 0.0,
            "average_post_exit_max_drawdown": 0.0,
        }

    valid = frame.loc[
        pd.to_numeric(
            frame["evaluation_exit_to_peak_gap"], errors="coerce"
        ).notna()
    ].copy()
    if valid.empty:
        return exit_quality_subset_metrics(pd.DataFrame())

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(valid[column], errors="coerce").dropna()

    gap = numeric("evaluation_exit_to_peak_gap")
    absolute_gap = numeric("evaluation_absolute_peak_gap")
    bars = numeric("evaluation_bars_from_peak")
    capture = numeric("evaluation_top_capture_ratio")
    upside = numeric("evaluation_post_exit_max_upside")
    drawdown = numeric("evaluation_post_exit_max_drawdown")

    return {
        "count": int(len(valid)),
        "average_exit_to_peak_gap": float(gap.mean()) if not gap.empty else 0.0,
        "median_exit_to_peak_gap": float(gap.median()) if not gap.empty else 0.0,
        "average_absolute_peak_gap": (
            float(absolute_gap.mean()) if not absolute_gap.empty else 0.0
        ),
        "average_bars_from_peak": float(bars.mean()) if not bars.empty else 0.0,
        "median_bars_from_peak": float(bars.median()) if not bars.empty else 0.0,
        "average_top_capture_ratio": (
            float(capture.mean()) if not capture.empty else 0.0
        ),
        "median_top_capture_ratio": (
            float(capture.median()) if not capture.empty else 0.0
        ),
        "average_post_exit_max_upside": (
            float(upside.mean()) if not upside.empty else 0.0
        ),
        "average_post_exit_max_drawdown": (
            float(drawdown.mean()) if not drawdown.empty else 0.0
        ),
    }


def build_top_exit_quality(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        normal_sells = pd.DataFrame()
    else:
        normal_sells = trades.loc[trades["action"].eq("SELL")].copy()

    if normal_sells.empty:
        empty = exit_quality_subset_metrics(normal_sells)
        return {
            "all_model_exits": dict(empty),
            "top_probability_exits": dict(empty),
        }

    top_mask = normal_sells["reason"].fillna("").map(
        lambda reason: bool(
            {
                "TOP_PROBABILITY",
                "MTF_TOP_WEEKLY_DAILY_CONFIRMATION",
                "EXIT_RISK_WEEKLY_DAILY_CONFIRMATION",
            }
            & split_exit_reason_tokens(reason)
        )
    )
    return {
        "all_model_exits": exit_quality_subset_metrics(normal_sells),
        "top_probability_exits": exit_quality_subset_metrics(
            normal_sells.loc[top_mask]
        ),
    }


def build_exit_approximation_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    """Evaluate useful proximity, not exact peak prediction.

    This uses post-simulation look-ahead fields only. It never participates in
    signal generation. A good exit can be several days away from the exact
    maximum if it captures most of the move and avoids more downside than the
    upside it leaves on the table.
    """
    if trades.empty:
        exits = pd.DataFrame()
    else:
        exits = trades.loc[
            trades["action"].eq("SELL")
            & trades["reason"].fillna("").eq(
                "EXIT_RISK_WEEKLY_DAILY_CONFIRMATION"
            )
        ].copy()

    if exits.empty:
        return {
            "count": 0,
            "within_5pct_of_local_peak_rate": 0.0,
            "within_10pct_of_local_peak_rate": 0.0,
            "within_15pct_of_local_peak_rate": 0.0,
            "average_absolute_peak_gap": 0.0,
            "median_absolute_peak_gap": 0.0,
            "average_absolute_bars_from_peak": 0.0,
            "average_missed_upside": 0.0,
            "average_drawdown_avoided": 0.0,
            "average_exit_balance": 0.0,
        }

    gap = pd.to_numeric(
        exits["evaluation_absolute_peak_gap"], errors="coerce"
    )
    bars = pd.to_numeric(
        exits["evaluation_bars_from_peak"], errors="coerce"
    ).abs()
    upside = pd.to_numeric(
        exits["evaluation_post_exit_max_upside"], errors="coerce"
    ).clip(lower=0)
    drawdown = pd.to_numeric(
        exits["evaluation_post_exit_max_drawdown"], errors="coerce"
    )
    avoided = (-drawdown).clip(lower=0)
    valid_gap = gap.dropna()
    valid_bars = bars.dropna()
    valid_upside = upside.dropna()
    valid_avoided = avoided.dropna()

    missed = float(valid_upside.mean()) if not valid_upside.empty else 0.0
    drawdown_avoided = (
        float(valid_avoided.mean()) if not valid_avoided.empty else 0.0
    )
    return {
        "count": int(len(exits)),
        "within_5pct_of_local_peak_rate": float(
            valid_gap.le(0.05).mean() if not valid_gap.empty else 0.0
        ),
        "within_10pct_of_local_peak_rate": float(
            valid_gap.le(0.10).mean() if not valid_gap.empty else 0.0
        ),
        "within_15pct_of_local_peak_rate": float(
            valid_gap.le(0.15).mean() if not valid_gap.empty else 0.0
        ),
        "average_absolute_peak_gap": float(
            valid_gap.mean() if not valid_gap.empty else 0.0
        ),
        "median_absolute_peak_gap": float(
            valid_gap.median() if not valid_gap.empty else 0.0
        ),
        "average_absolute_bars_from_peak": float(
            valid_bars.mean() if not valid_bars.empty else 0.0
        ),
        "average_missed_upside": missed,
        "average_drawdown_avoided": drawdown_avoided,
        "average_exit_balance": drawdown_avoided - missed,
    }


def build_exit_performance(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        sells = pd.DataFrame()
    else:
        sells = trades.loc[
            trades["action"].isin(["SELL", "FINAL_SELL"])
        ].copy()

    if sells.empty:
        empty = exit_subset_metrics(sells)
        return {
            "all": dict(empty),
            "fibonacci": dict(empty),
            "trend_breakdown": dict(empty),
            "atr_trailing_stop": dict(empty),
            "atr_trailing_stop_tightened": dict(empty),
            "top_probability": dict(empty),
            "mtf_top_confirmation": dict(empty),
            "exit_risk_confirmation": dict(empty),
            "final_liquidation": dict(empty),
            "exact_reasons": {},
        }

    token_sets = sells["reason"].map(split_exit_reason_tokens)

    masks = {
        "fibonacci": token_sets.map(
            lambda tokens: any(
                token.startswith("FIBONACCI_")
                for token in tokens
            )
        ),
        "trend_breakdown": token_sets.map(
            lambda tokens: "TREND_BREAKDOWN" in tokens
        ),
        "atr_trailing_stop": token_sets.map(
            lambda tokens: "ATR_TRAILING_STOP" in tokens
        ),
        "atr_trailing_stop_tightened": token_sets.map(
            lambda tokens: "ATR_TRAILING_STOP_TIGHTENED" in tokens
        ),
        "top_probability": token_sets.map(
            lambda tokens: (
                "TOP_PROBABILITY" in tokens
                or "MTF_TOP_WEEKLY_DAILY_CONFIRMATION" in tokens
                or "EXIT_RISK_WEEKLY_DAILY_CONFIRMATION" in tokens
            )
        ),
        "mtf_top_confirmation": token_sets.map(
            lambda tokens: "MTF_TOP_WEEKLY_DAILY_CONFIRMATION" in tokens
        ),
        "exit_risk_confirmation": token_sets.map(
            lambda tokens: "EXIT_RISK_WEEKLY_DAILY_CONFIRMATION" in tokens
        ),
        "final_liquidation": token_sets.map(
            lambda tokens: "FINAL_LIQUIDATION" in tokens
        ),
    }

    exact_reasons = {
        str(reason): exit_subset_metrics(group)
        for reason, group in sells.groupby("reason", dropna=False)
    }

    return {
        "all": exit_subset_metrics(sells),
        **{
            name: exit_subset_metrics(sells.loc[mask])
            for name, mask in masks.items()
        },
        "exact_reasons": exact_reasons,
    }


def build_metrics(
    symbol: str,
    backend: str,
    bars: pd.DataFrame,
    split: SplitData,
    calibration: pd.DataFrame,
    bottom_calibration: ThresholdResult,
    top_calibration: ThresholdResult,
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any]:
    strategy_mode = config.strategy_mode
    bottom_entry_allowed = strategy_mode in {
        "ADAPTIVE_HYBRID",
        "BOTTOM_REVERSAL_FIBONACCI",
        "BOTTOM_REVERSAL_TOP_EXIT",
        "BOTTOM_ENTRY_MTF_TOP_EXIT",
    }
    pullback_entry_allowed = strategy_mode in {
        "ADAPTIVE_HYBRID",
        "STRUCTURAL_TREND",
    }
    bull_regime_entry_allowed = (
        strategy_mode in {"ADAPTIVE_HYBRID", "STRUCTURAL_TREND"}
        and config.bull_regime_entry_enabled
    )
    fibonacci_exit_allowed = (
        strategy_mode == "BOTTOM_REVERSAL_FIBONACCI"
        or (
            strategy_mode == "ADAPTIVE_HYBRID"
            and config.exit_fibonacci_target
        )
    )
    top_exit_allowed = (
        strategy_mode == "BOTTOM_REVERSAL_TOP_EXIT"
        or (
            strategy_mode == "ADAPTIVE_HYBRID"
            and config.exit_top_probability
        )
    )
    trend_exit_allowed = (
        strategy_mode in {"ADAPTIVE_HYBRID", "STRUCTURAL_TREND"}
        and config.exit_trend_breakdown
    )
    atr_exit_allowed = (
        strategy_mode == "ADAPTIVE_HYBRID"
        and config.exit_atr_trailing_stop
    )

    bottom_metrics = classifier_metrics(
        predictions["actual_bottom"],
        predictions["bottom_probability"],
        predictions["predicted_bottom_signal"],
        config.event_tolerance_bars,
    )
    top_metrics = classifier_metrics(
        predictions["actual_top"],
        predictions["top_probability"],
        predictions["predicted_top_signal"],
        config.event_tolerance_bars,
    )
    baseline_bottom = event_metrics(
        predictions["actual_bottom"],
        predictions["baseline_bottom_signal"],
        config.event_tolerance_bars,
    )
    baseline_top = event_metrics(
        predictions["actual_top"],
        predictions["baseline_top_signal"],
        config.event_tolerance_bars,
    )

    strategy_curve = predictions["strategy_equity"]
    buy_hold_curve = predictions["buy_hold_equity"]
    strategy_return = total_return(strategy_curve, config.initial_capital)
    buy_hold_return = total_return(buy_hold_curve, config.initial_capital)
    full_buy_hold = full_series_buy_and_hold(bars, config)

    trade_fees = float(trades["total_fee"].sum()) if not trades.empty else 0.0
    buys = int((trades["action"] == "BUY").sum()) if not trades.empty else 0
    sells = int(trades["action"].isin(["SELL", "FINAL_SELL"]).sum()) if not trades.empty else 0
    normal_sells = int((trades["action"] == "SELL").sum()) if not trades.empty else 0
    exposure = float((predictions["shares"] > 0).mean())

    exit_performance = build_exit_performance(trades)
    top_exit_quality = build_top_exit_quality(trades)
    fibonacci_exit_count = exit_performance["fibonacci"]["count"]
    trend_breakdown_exit_count = exit_performance["trend_breakdown"]["count"]
    atr_exit_count = exit_performance["atr_trailing_stop"]["count"]
    tightened_atr_exit_count = exit_performance[
        "atr_trailing_stop_tightened"
    ]["count"]
    top_probability_exit_count = exit_performance["top_probability"]["count"]
    final_liquidation_count = exit_performance["final_liquidation"]["count"]

    return {
        "symbol": symbol,
        "backend": backend,
        "timeframe": config.timeframe,
        "extrema_label_price_basis": (
            "close"
            if is_swing_timeframe(config.timeframe)
            else "high_low"
        ),
        "data_start": str(bars.index.min()),
        "data_end": str(bars.index.max()),
        "train_start": str(split.train.index.min()),
        "train_end": str(split.train.index.max()),
        "calibration_start": str(split.calibration.index.min()),
        "calibration_end": str(split.calibration.index.max()),
        "test_start": str(predictions.index.min()),
        "test_end": str(predictions.index.max()),
        "training_rows": int(len(split.train)),
        "calibration_rows": int(len(split.calibration)),
        "test_rows": int(len(predictions)),
        "bottom_calibration": asdict(bottom_calibration),
        "top_calibration": asdict(top_calibration),
        "calibration_constraints": {
            "threshold_min": float(config.threshold_min),
            "threshold_step": float(config.threshold_step),
            "bottom_threshold_max": float(config.bottom_threshold_max),
            "top_threshold_max": float(config.top_threshold_max),
            "bottom_min_precision": float(config.bottom_min_precision),
            "bottom_min_recall": float(config.bottom_min_recall),
            "top_min_precision": float(config.top_min_precision),
            "top_min_recall": float(config.top_min_recall),
            "bottom_min_calibration_signals": int(
                config.bottom_min_calibration_signals
            ),
            "top_min_calibration_signals": int(
                config.top_min_calibration_signals
            ),
        },
        "bottom_metrics": bottom_metrics,
        "top_metrics": top_metrics,
        "baseline_bottom": baseline_bottom,
        "baseline_top": baseline_top,
        "strategy_ending_capital": float(strategy_curve.iloc[-1]),
        "strategy_return": strategy_return,
        "strategy_maximum_drawdown": maximum_drawdown(strategy_curve),
        "strategy_sharpe": sharpe_ratio(strategy_curve, config.timeframe),
        "buy_hold_ending_capital": float(buy_hold_curve.iloc[-1]),
        "buy_hold_return": buy_hold_return,
        "buy_hold_maximum_drawdown": maximum_drawdown(buy_hold_curve),
        "buy_hold_sharpe": sharpe_ratio(buy_hold_curve, config.timeframe),
        "excess_return": strategy_return - buy_hold_return,
        "full_series_buy_hold": full_buy_hold,
        "market_exposure": exposure,
        "simulated_buys": buys,
        "simulated_sells": sells,
        "normal_model_sells": normal_sells,
        "total_executions": int(len(trades)),
        "total_transaction_fees": trade_fees,

        # Exit configuration and proof of execution.
        "strategy_mode": config.strategy_mode,
        "fibonacci_target_enabled": bool(fibonacci_exit_allowed),
        "effective_bottom_entry_enabled": bool(bottom_entry_allowed),
        "effective_trend_pullback_entry_enabled": bool(
            pullback_entry_allowed
        ),
        "effective_bull_regime_entry_enabled": bool(
            bull_regime_entry_allowed
        ),
        "effective_top_exit_enabled": bool(top_exit_allowed),
        "effective_trend_exit_enabled": bool(trend_exit_allowed),
        "effective_atr_exit_enabled": bool(atr_exit_allowed),
        "fibonacci_target_ratio": float(config.fibonacci_target_ratio),
        "fibonacci_swing_lookback": int(config.fibonacci_swing_lookback),
        "fibonacci_low_lookback": int(config.fibonacci_low_lookback),
        "fibonacci_exit_count": fibonacci_exit_count,
        "fibonacci_exit_rate": (
            fibonacci_exit_count / normal_sells
            if normal_sells > 0
            else 0.0
        ),
        "top_tighten_trailing_enabled": bool(config.top_tighten_trailing),
        "atr_trailing_multiplier": float(config.atr_trailing_multiplier),
        "tightened_atr_multiplier": float(config.tightened_atr_multiplier),
        "trend_breakdown_exit_count": trend_breakdown_exit_count,
        "bottom_probability_entry_count": int(
            (
                (trades["action"] == "BUY")
                & trades["reason"].fillna("").eq(
                    "BOTTOM_PROBABILITY"
                )
            ).sum()
        ) if not trades.empty else 0,
        "trend_pullback_entry_count": int(
            (
                (trades["action"] == "BUY")
                & trades["reason"].fillna("").eq(
                    "TREND_PULLBACK"
                )
            ).sum()
        ) if not trades.empty else 0,
        "bull_regime_entry_count": int(
            (
                (trades["action"] == "BUY")
                & trades["reason"].fillna("").eq("BULL_REGIME_ENTRY")
            ).sum()
        ) if not trades.empty else 0,
        "bull_regime_entry_enabled": bool(config.bull_regime_entry_enabled),
        "bull_regime_entry_confirmation_bars": int(
            config.bull_regime_entry_confirmation_bars
        ),
        "bull_regime_bar_count": int(
            predictions["bull_regime_signal"]
            .eq(True)
            .sum()
        ),
        "adaptive_bull_regime_enabled": bool(
            config.adaptive_bull_regime_enabled
        ),
        "trend_pullback_entry_enabled": bool(
            config.trend_pullback_entry_enabled
        ),
        "atr_exit_count": atr_exit_count,
        "tightened_atr_exit_count": tightened_atr_exit_count,
        "top_probability_exit_count": top_probability_exit_count,
        "final_liquidation_count": final_liquidation_count,
        "exit_performance": exit_performance,
        "top_exit_quality": top_exit_quality,
        "trend_breakdown_confirmation_bars": int(
            config.trend_breakdown_confirmation_bars
        ),
        "trend_breakdown_require_slow_ema_decline": bool(
            config.trend_breakdown_require_slow_ema_decline
        ),

        "calibration_bottom_positive_rate": float(calibration["actual_bottom"].mean()),
        "calibration_top_positive_rate": float(calibration["actual_top"].mean()),
    }



def build_mtf_metrics(
    symbol: str,
    backend: str,
    structural_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    structural_split: SplitData,
    weekly_split: SplitData,
    structural_calibration: pd.DataFrame,
    weekly_calibration: pd.DataFrame,
    bottom_calibration: ThresholdResult,
    weekly_top_calibration: ThresholdResult,
    structural_predictions: pd.DataFrame,
    weekly_predictions: pd.DataFrame,
    execution_predictions: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any]:
    bottom_metrics = classifier_metrics(
        structural_predictions["actual_bottom"],
        structural_predictions["bottom_probability"],
        structural_predictions["predicted_bottom_signal"],
        config.event_tolerance_bars,
    )
    top_metrics = classifier_metrics(
        weekly_predictions["actual_top"],
        weekly_predictions["top_probability"],
        weekly_predictions["predicted_top_signal"],
        1,
    )
    baseline_bottom = event_metrics(
        structural_predictions["actual_bottom"],
        structural_predictions["baseline_bottom_signal"],
        config.event_tolerance_bars,
    )
    baseline_top = event_metrics(
        weekly_predictions["actual_top"],
        weekly_predictions["baseline_top_signal"],
        1,
    )

    strategy_curve = execution_predictions["strategy_equity"]
    buy_hold_curve = execution_predictions["buy_hold_equity"]
    daily_config = replace(config, timeframe="1Day")
    strategy_return = total_return(strategy_curve, config.initial_capital)
    buy_hold_return = total_return(buy_hold_curve, config.initial_capital)
    full_buy_hold = full_series_buy_and_hold(daily_bars, daily_config)
    trade_fees = float(trades["total_fee"].sum()) if not trades.empty else 0.0
    buys = int((trades["action"] == "BUY").sum()) if not trades.empty else 0
    sells = int(trades["action"].isin(["SELL", "FINAL_SELL"]).sum()) if not trades.empty else 0
    normal_sells = int((trades["action"] == "SELL").sum()) if not trades.empty else 0
    exposure = float((execution_predictions["shares"] > 0).mean())
    exit_performance = build_exit_performance(trades)
    top_exit_quality = build_top_exit_quality(trades)
    mtf_count = exit_performance["mtf_top_confirmation"]["count"]

    return {
        "symbol": symbol,
        "backend": backend,
        "timeframe": config.timeframe,
        "execution_timeframe": "1Day",
        "top_signal_timeframe": config.mtf_top_signal_timeframe,
        "strategy_mode": config.strategy_mode,
        "extrema_label_price_basis": "structural_close_weekly_close",
        "data_start": str(structural_bars.index.min()),
        "data_end": str(structural_bars.index.max()),
        "train_start": str(structural_split.train.index.min()),
        "train_end": str(structural_split.train.index.max()),
        "calibration_start": str(structural_split.calibration.index.min()),
        "calibration_end": str(structural_split.calibration.index.max()),
        "test_start": str(execution_predictions.index.min()),
        "test_end": str(execution_predictions.index.max()),
        "training_rows": int(len(structural_split.train)),
        "calibration_rows": int(len(structural_split.calibration)),
        "test_rows": int(len(execution_predictions)),
        "weekly_top_training_rows": int(len(weekly_split.train)),
        "weekly_top_calibration_rows": int(len(weekly_split.calibration)),
        "weekly_top_test_rows": int(len(weekly_predictions)),
        "bottom_calibration": asdict(bottom_calibration),
        "top_calibration": asdict(weekly_top_calibration),
        "bottom_metrics": bottom_metrics,
        "top_metrics": top_metrics,
        "baseline_bottom": baseline_bottom,
        "baseline_top": baseline_top,
        "strategy_ending_capital": float(strategy_curve.iloc[-1]),
        "strategy_return": strategy_return,
        "strategy_maximum_drawdown": maximum_drawdown(strategy_curve),
        "strategy_sharpe": sharpe_ratio(strategy_curve, "1Day"),
        "buy_hold_ending_capital": float(buy_hold_curve.iloc[-1]),
        "buy_hold_return": buy_hold_return,
        "buy_hold_maximum_drawdown": maximum_drawdown(buy_hold_curve),
        "buy_hold_sharpe": sharpe_ratio(buy_hold_curve, "1Day"),
        "excess_return": strategy_return - buy_hold_return,
        "full_series_buy_hold": full_buy_hold,
        "market_exposure": exposure,
        "simulated_buys": buys,
        "simulated_sells": sells,
        "normal_model_sells": normal_sells,
        "total_executions": int(len(trades)),
        "total_transaction_fees": trade_fees,
        "fibonacci_target_enabled": False,
        "effective_bottom_entry_enabled": True,
        "effective_trend_pullback_entry_enabled": False,
        "effective_bull_regime_entry_enabled": False,
        "effective_top_exit_enabled": True,
        "effective_trend_exit_enabled": False,
        "effective_atr_exit_enabled": False,
        "fibonacci_target_ratio": float(config.fibonacci_target_ratio),
        "fibonacci_swing_lookback": int(config.fibonacci_swing_lookback),
        "fibonacci_low_lookback": int(config.fibonacci_low_lookback),
        "fibonacci_exit_count": 0,
        "fibonacci_exit_rate": 0.0,
        "top_tighten_trailing_enabled": False,
        "atr_trailing_multiplier": float(config.atr_trailing_multiplier),
        "tightened_atr_multiplier": float(config.tightened_atr_multiplier),
        "trend_breakdown_exit_count": 0,
        "bottom_probability_entry_count": int(
            ((trades["action"] == "BUY") & trades["reason"].fillna("").eq("BOTTOM_PROBABILITY")).sum()
        ) if not trades.empty else 0,
        "trend_pullback_entry_count": 0,
        "bull_regime_entry_count": 0,
        "bull_regime_entry_enabled": False,
        "bull_regime_entry_confirmation_bars": int(config.bull_regime_entry_confirmation_bars),
        "bull_regime_bar_count": 0,
        "adaptive_bull_regime_enabled": False,
        "trend_pullback_entry_enabled": False,
        "atr_exit_count": 0,
        "tightened_atr_exit_count": 0,
        "top_probability_exit_count": mtf_count,
        "mtf_top_exit_count": mtf_count,
        "final_liquidation_count": exit_performance["final_liquidation"]["count"],
        "exit_performance": exit_performance,
        "top_exit_quality": top_exit_quality,
        "trend_breakdown_confirmation_bars": int(config.trend_breakdown_confirmation_bars),
        "trend_breakdown_require_slow_ema_decline": bool(config.trend_breakdown_require_slow_ema_decline),
        "calibration_bottom_positive_rate": float(structural_calibration["actual_bottom"].mean()),
        "calibration_top_positive_rate": float(weekly_calibration["actual_top"].mean()),
        "mtf_top_probability_floor": float(config.mtf_top_probability_floor),
        "mtf_effective_top_threshold": float(weekly_top_calibration.threshold),
        "mtf_daily_confirmation_ema": int(config.mtf_daily_confirmation_ema),
        "mtf_daily_confirmation_bars": int(config.mtf_daily_confirmation_bars),
        "mtf_daily_require_negative_return": bool(config.mtf_daily_require_negative_return),
        "mtf_daily_require_ema_decline": bool(config.mtf_daily_require_ema_decline),
        "mtf_daily_require_lower_high": bool(config.mtf_daily_require_lower_high),
        "mtf_top_signal_valid_days": int(config.mtf_top_signal_valid_days),
        "mtf_top_min_position_return": float(
            config.mtf_top_min_position_return
        ),
        "mtf_top_high_lookback_weeks": int(
            config.mtf_top_high_lookback_weeks
        ),
        "mtf_top_max_distance_from_high": float(
            config.mtf_top_max_distance_from_high
        ),
        "mtf_top_guard_blocked_count": int(
            execution_predictions["mtf_top_guard_blocked"].eq(True).sum()
        ),
        "mtf_top_position_return_blocked_count": int(
            (
                execution_predictions["mtf_top_guard_blocked"].eq(True)
                & ~execution_predictions[
                    "mtf_top_position_return_ok"
                ].eq(True)
            ).sum()
        ),
        "mtf_top_high_proximity_blocked_count": int(
            (
                execution_predictions["mtf_top_guard_blocked"].eq(True)
                & ~execution_predictions[
                    "mtf_top_high_proximity_ok"
                ].eq(True)
            ).sum()
        ),
    }


def is_exit_risk_strategy(config: BacktestConfig) -> bool:
    return config.strategy_mode in {
        "BOTTOM_ENTRY_EXIT_RISK_V1",
        "BOTTOM_ENTRY_EXIT_RISK_SWING_1D",
    }


def is_daily_swing_exit_risk(config: BacktestConfig) -> bool:
    return config.strategy_mode == "BOTTOM_ENTRY_EXIT_RISK_SWING_1D"


def exit_risk_sell_reason(config: BacktestConfig) -> str:
    if is_daily_swing_exit_risk(config):
        return "EXIT_RISK_DAILY_CONFIRMATION"
    return "EXIT_RISK_WEEKLY_DAILY_CONFIRMATION"


def configured_exit_risk_backends(config: BacktestConfig) -> tuple[str, ...]:
    if config.exit_risk_compare_models:
        return tuple(dict.fromkeys(config.exit_risk_model_backends))
    return (str(config.exit_risk_model_backend).lower(),)


def prepare_exit_risk_bottom_context(
    structural_backend: str,
    structural_bars: pd.DataFrame,
    config: BacktestConfig,
) -> ExitRiskBottomContext:
    """Train the proven structural BOTTOM once per asset/backend."""
    structural_dataset = build_dataset(structural_bars, config)
    structural_split = split_dataset(structural_dataset, config)
    bottom_calibration, structural_calibration = calibrate_single_target(
        structural_split,
        structural_backend,
        config,
        target_column="actual_bottom",
        probability_column="bottom_probability",
        feature_columns=FEATURE_COLUMNS,
        minimum_precision=config.bottom_min_precision,
        minimum_recall=config.bottom_min_recall,
        maximum_threshold=config.bottom_threshold_max,
        minimum_signals=config.bottom_min_calibration_signals,
        tolerance_bars=config.event_tolerance_bars,
    )
    structural_predictions = walk_forward_single_target(
        structural_dataset,
        structural_split,
        structural_backend,
        config,
        target_column="actual_bottom",
        probability_column="bottom_probability",
        feature_columns=FEATURE_COLUMNS,
    )
    structural_predictions["predicted_bottom_signal"] = (
        structural_predictions["bottom_probability"]
        >= bottom_calibration.threshold
    )
    structural_predictions["predicted_top_signal"] = False
    return ExitRiskBottomContext(
        structural_dataset=structural_dataset,
        structural_split=structural_split,
        bottom_calibration=bottom_calibration,
        structural_calibration=structural_calibration,
        structural_predictions=structural_predictions,
    )


def run_asset_backend_exit_risk(
    symbol: str,
    structural_backend: str,
    structural_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: BacktestConfig,
    *,
    exit_backend: str | None = None,
    bottom_context: ExitRiskBottomContext | None = None,
) -> AssetRunResult:
    selected_exit_backend = str(
        exit_backend or config.exit_risk_model_backend
    ).lower()
    context = bottom_context or prepare_exit_risk_bottom_context(
        structural_backend,
        structural_bars,
        config,
    )
    structural_split = context.structural_split
    bottom_calibration = context.bottom_calibration
    structural_calibration = context.structural_calibration
    structural_predictions = context.structural_predictions.copy()

    exit_config = exit_risk_model_config(config)
    exit_calibration_config = replace(
        exit_config,
        threshold_min=max(
            float(exit_config.threshold_min),
            float(config.exit_risk_probability_floor),
        ),
    )

    weekly_bars = aggregate_swing_bars(daily_bars, exit_config)
    weekly_bars = validate_and_clean_bars(weekly_bars, exit_config)
    exit_dataset = build_exit_risk_dataset(weekly_bars, exit_config)
    exit_split = split_dataset(exit_dataset, exit_config)
    tolerance = int(
        config.swing_exit_event_tolerance_days
        if is_daily_swing_exit_risk(config)
        else config.exit_risk_event_tolerance_weeks
    )

    calibrated_exit, exit_calibration = calibrate_single_target(
        exit_split,
        selected_exit_backend,
        exit_calibration_config,
        target_column="actual_exit_risk",
        probability_column="exit_risk_probability",
        feature_columns=EXIT_RISK_FEATURE_COLUMNS,
        minimum_precision=config.exit_risk_min_precision,
        minimum_recall=config.exit_risk_min_recall,
        maximum_threshold=config.exit_risk_threshold_max,
        minimum_signals=config.exit_risk_min_calibration_signals,
        tolerance_bars=tolerance,
    )
    effective_threshold = float(calibrated_exit.threshold)
    exit_calibration_effective = threshold_result_at(
        exit_calibration["actual_exit_risk"],
        exit_calibration["exit_risk_probability"],
        effective_threshold,
        tolerance,
    )
    gate_requirements_passed, gate_failures = exit_risk_calibration_gate(
        exit_calibration_effective,
        config,
    )
    calibration_gate_passed = (
        gate_requirements_passed
        or not config.exit_risk_hard_calibration_gate
    )
    weekly_predictions = walk_forward_single_target(
        exit_dataset,
        exit_split,
        selected_exit_backend,
        exit_config,
        target_column="actual_exit_risk",
        probability_column="exit_risk_probability",
        feature_columns=EXIT_RISK_FEATURE_COLUMNS,
    )
    weekly_predictions = add_exit_risk_signals(
        weekly_predictions,
        effective_threshold,
        calibration_gate_passed=calibration_gate_passed,
    )

    execution_frame = build_exit_risk_execution_frame(
        daily_bars,
        structural_predictions,
        weekly_predictions,
        config,
    )
    execution_predictions, trades = simulate_mtf_strategy(
        execution_frame,
        config,
    )
    quality_config = replace(
        config,
        timeframe="1Day",
        future_horizon=config.mtf_exit_quality_horizon_days,
        event_tolerance_bars=min(5, config.mtf_exit_quality_horizon_days),
    )
    trades = annotate_exit_quality(
        execution_predictions,
        trades,
        quality_config,
    )

    run_backend = (
        f"{structural_backend}__exit_{selected_exit_backend}"
        if config.exit_risk_compare_models
        else structural_backend
    )
    metrics = build_mtf_metrics(
        symbol=symbol,
        backend=run_backend,
        structural_bars=structural_bars,
        daily_bars=daily_bars,
        structural_split=structural_split,
        weekly_split=exit_split,
        structural_calibration=structural_calibration,
        weekly_calibration=exit_calibration.assign(
            actual_top=exit_calibration["actual_exit_risk"],
            top_probability=exit_calibration["exit_risk_probability"],
            baseline_top_signal=False,
        ),
        bottom_calibration=bottom_calibration,
        weekly_top_calibration=exit_calibration_effective,
        structural_predictions=structural_predictions,
        weekly_predictions=weekly_predictions,
        execution_predictions=execution_predictions,
        trades=trades,
        config=config,
    )
    exit_metrics = classifier_metrics(
        weekly_predictions["actual_exit_risk"],
        weekly_predictions["exit_risk_probability"],
        weekly_predictions["predicted_exit_risk_signal"],
        tolerance,
    )
    approximation = build_exit_approximation_metrics(trades)
    metrics.update(
        {
            "backend": run_backend,
            "structural_backend": structural_backend,
            "exit_risk_backend": selected_exit_backend,
            "exit_risk_compare_models": bool(config.exit_risk_compare_models),
            "exit_risk_signal_timeframe": (
                "1Day"
                if is_daily_swing_exit_risk(config)
                else config.exit_risk_signal_timeframe
            ),
            "exit_risk_horizon_weeks": (
                None
                if is_daily_swing_exit_risk(config)
                else int(config.exit_risk_horizon_weeks)
            ),
            "exit_risk_event_tolerance_weeks": (
                None
                if is_daily_swing_exit_risk(config)
                else tolerance
            ),
            "swing_exit_horizon_days": (
                int(config.swing_exit_horizon_days)
                if is_daily_swing_exit_risk(config)
                else None
            ),
            "swing_exit_event_tolerance_days": (
                tolerance
                if is_daily_swing_exit_risk(config)
                else None
            ),
            "exit_risk_down_barrier": float(
                config.swing_exit_down_barrier
                if is_daily_swing_exit_risk(config)
                else config.exit_risk_down_barrier
            ),
            "exit_risk_up_barrier": float(
                config.swing_exit_up_barrier
                if is_daily_swing_exit_risk(config)
                else config.exit_risk_up_barrier
            ),
            "exit_risk_probability_floor": float(
                config.exit_risk_probability_floor
            ),
            "exit_risk_effective_threshold": float(effective_threshold),
            "exit_risk_calibration": asdict(exit_calibration_effective),
            "exit_risk_hard_calibration_gate": bool(
                config.exit_risk_hard_calibration_gate
            ),
            "exit_risk_calibration_gate_passed": bool(
                calibration_gate_passed
            ),
            "exit_risk_calibration_requirements_passed": bool(
                gate_requirements_passed
            ),
            "exit_risk_calibration_gate_failures": list(gate_failures),
            "exit_risk_calibration_gate_reason": (
                "PASS"
                if gate_requirements_passed
                else "; ".join(gate_failures)
            ),
            "exit_risk_metrics": exit_metrics,
            "exit_risk_approximation": approximation,
            "exit_risk_candidate_signal_count": int(
                weekly_predictions["exit_risk_candidate_signal"].eq(True).sum()
            ),
            "exit_risk_signal_count": int(
                weekly_predictions["predicted_exit_risk_signal"].eq(True).sum()
            ),
            "exit_risk_exit_count": int(
                trades["reason"].fillna("")
                .eq(exit_risk_sell_reason(config))
                .sum()
            ) if not trades.empty else 0,
            "trend_resumption_reentry_count": int(
                trades["reason"].fillna("")
                .eq("TREND_RESUMPTION_REENTRY")
                .sum()
            ) if not trades.empty else 0,
            "exit_risk_reentry_enabled": bool(
                config.exit_risk_reentry_enabled
            ),
            "exit_risk_reentry_cooldown_days": int(
                config.exit_risk_reentry_cooldown_days
            ),
        }
    )
    metrics["top_metrics"] = exit_metrics
    metrics["top_calibration"] = asdict(exit_calibration_effective)
    metrics["top_signal_timeframe"] = (
        "1Day"
        if is_daily_swing_exit_risk(config)
        else config.exit_risk_signal_timeframe
    )
    metrics["mtf_effective_top_threshold"] = effective_threshold
    metrics["mtf_top_exit_count"] = metrics["exit_risk_exit_count"]
    metrics["top_probability_exit_count"] = metrics["exit_risk_exit_count"]

    summary = build_summary(metrics, config)
    if is_daily_swing_exit_risk(config):
        tolerance_label = f"±{tolerance} trading days"
        barrier_label = (
            f"-{config.swing_exit_down_barrier:.0%} before "
            f"+{config.swing_exit_up_barrier:.0%} within "
            f"{config.swing_exit_horizon_days} trading days"
        )
    else:
        tolerance_label = f"±{tolerance} weekly bars"
        barrier_label = (
            f"-{config.exit_risk_down_barrier:.0%} before "
            f"+{config.exit_risk_up_barrier:.0%} within "
            f"{config.exit_risk_horizon_weeks} weeks"
        )

    summary += (
        "\nExit Risk model comparison:\n"
        f"- Structural BOTTOM backend: {structural_backend}\n"
        f"- Exit model backend: {selected_exit_backend}\n"
        f"- Approximation tolerance: {tolerance_label}\n"
        f"- Barrier target: {barrier_label}\n"
        f"- Effective exit-risk threshold: {effective_threshold:.3f}\n"
        f"- Hard calibration gate: "
        f"{'PASS' if calibration_gate_passed else 'BLOCKED'}\n"
        f"- Gate detail: {metrics['exit_risk_calibration_gate_reason']}\n"
        f"- Candidate exit-risk signals: "
        f"{metrics['exit_risk_candidate_signal_count']}\n"
        f"- Approved exit-risk signals: {metrics['exit_risk_signal_count']}\n"
        f"- Exit-risk sells: {metrics['exit_risk_exit_count']}\n"
        f"- Exits within 10% of local peak: "
        f"{approximation['within_10pct_of_local_peak_rate']:.1%}\n"
        f"- Average exit balance (drawdown avoided - upside missed): "
        f"{approximation['average_exit_balance']:.2%}\n"
        f"- Trend-resumption reentries: "
        f"{metrics['trend_resumption_reentry_count']}"
    )
    return AssetRunResult(
        symbol=symbol,
        backend=run_backend,
        predictions=execution_predictions,
        trades=trades,
        summary_text=summary,
        metrics=metrics,
    )


def run_asset_backend_mtf(
    symbol: str,
    backend: str,
    structural_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: BacktestConfig,
) -> AssetRunResult:
    structural_dataset = build_dataset(structural_bars, config)
    structural_split = split_dataset(structural_dataset, config)
    bottom_calibration, _ignored_top, structural_calibration = calibrate_models(
        structural_split, backend, config
    )
    structural_predictions = walk_forward_predict(
        structural_dataset, structural_split, backend, config
    )
    structural_predictions = add_decision_signals(
        structural_predictions,
        bottom_calibration.threshold,
        1.0,
        config,
    )
    structural_predictions["predicted_top_signal"] = False

    weekly_config = mtf_top_model_config(config)
    weekly_bars = aggregate_swing_bars(daily_bars, weekly_config)
    weekly_bars = validate_and_clean_bars(weekly_bars, weekly_config)
    weekly_dataset = build_dataset(weekly_bars, weekly_config)
    weekly_split = split_dataset(weekly_dataset, weekly_config)
    _weekly_bottom, calibrated_weekly_top, weekly_calibration = calibrate_models(
        weekly_split, backend, weekly_config
    )
    effective_threshold = max(
        float(calibrated_weekly_top.threshold),
        float(config.mtf_top_probability_floor),
    )
    weekly_top_calibration = threshold_result_at(
        weekly_calibration["actual_top"],
        weekly_calibration["top_probability"],
        effective_threshold,
        1,
    )
    weekly_predictions = walk_forward_predict(
        weekly_dataset, weekly_split, backend, weekly_config
    )
    weekly_predictions = add_decision_signals(
        weekly_predictions,
        1.0,
        effective_threshold,
        weekly_config,
    )
    weekly_predictions["predicted_bottom_signal"] = False

    execution_frame = build_mtf_execution_frame(
        daily_bars,
        structural_predictions,
        weekly_predictions,
        config,
    )
    execution_predictions, trades = simulate_mtf_strategy(
        execution_frame, config
    )
    quality_config = replace(
        config,
        timeframe="1Day",
        future_horizon=config.mtf_exit_quality_horizon_days,
        event_tolerance_bars=min(5, config.mtf_exit_quality_horizon_days),
    )
    trades = annotate_exit_quality(
        execution_predictions, trades, quality_config
    )
    metrics = build_mtf_metrics(
        symbol=symbol,
        backend=backend,
        structural_bars=structural_bars,
        daily_bars=daily_bars,
        structural_split=structural_split,
        weekly_split=weekly_split,
        structural_calibration=structural_calibration,
        weekly_calibration=weekly_calibration,
        bottom_calibration=bottom_calibration,
        weekly_top_calibration=weekly_top_calibration,
        structural_predictions=structural_predictions,
        weekly_predictions=weekly_predictions,
        execution_predictions=execution_predictions,
        trades=trades,
        config=config,
    )
    summary = build_summary(metrics, config)
    return AssetRunResult(
        symbol=symbol,
        backend=backend,
        predictions=execution_predictions,
        trades=trades,
        summary_text=summary,
        metrics=metrics,
    )


def percentage(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def decimal(value: float, places: int = 3) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{places}f}"


def build_summary(metrics: dict[str, Any], config: BacktestConfig) -> str:
    bottom = metrics["bottom_metrics"]
    top = metrics["top_metrics"]
    baseline_bottom = metrics["baseline_bottom"]
    baseline_top = metrics["baseline_top"]
    full_bh = metrics["full_series_buy_hold"]
    bottom_cal = metrics["bottom_calibration"]
    top_cal = metrics["top_calibration"]

    lines = [
        "=" * 88,
        "MULTI-ASSET LOCAL EXTREMA WALK-FORWARD BACKTEST",
        "=" * 88,
        "Execution mode: LOCAL SIMULATION ONLY - no real or paper orders",
        f"Symbol: {metrics['symbol']}",
        f"Model backend: {metrics['backend']}",
        "Price source: Yahoo Finance through yfinance (adjusted OHLC when enabled)",
        f"Timeframe: {config.timeframe}",
        f"Future event horizon: {config.future_horizon} bars",
        f"Historical raw period: {metrics['data_start']} -> {metrics['data_end']}",
        f"Training period: {metrics['train_start']} -> {metrics['train_end']}",
        f"Calibration period: {metrics['calibration_start']} -> {metrics['calibration_end']}",
        f"Final walk-forward test: {metrics['test_start']} -> {metrics['test_end']}",
        f"Rows: train={metrics['training_rows']:,}, calibration={metrics['calibration_rows']:,}, test={metrics['test_rows']:,}",
        "",
        "VALIDATION-CALIBRATED THRESHOLDS",
        f"BOTTOM threshold: {bottom_cal['threshold']:.3f} | precision={percentage(bottom_cal['precision'])} | recall={percentage(bottom_cal['recall'])} | f1={percentage(bottom_cal['f1'])}",
        f"TOP threshold:    {top_cal['threshold']:.3f} | precision={percentage(top_cal['precision'])} | recall={percentage(top_cal['recall'])} | f1={percentage(top_cal['f1'])}",
        "These thresholds are fixed before the final test and are not tuned on final-test results.",
        "",
        f"FINAL TEST EVENT DETECTION WITH +/- {config.event_tolerance_bars} BARS",
        "BOTTOM model:",
        f"  tolerant precision={percentage(bottom['precision_tolerant'])}, recall={percentage(bottom['recall_tolerant'])}, f1={percentage(bottom['f1_tolerant'])}",
        f"  average precision={percentage(bottom['average_precision'])}, ROC AUC={decimal(bottom['roc_auc'])}, Brier={decimal(bottom['brier'], 4)}",
        f"  predicted={bottom['predicted_events']}, actual={bottom['actual_events']}",
        "TOP model:",
        f"  tolerant precision={percentage(top['precision_tolerant'])}, recall={percentage(top['recall_tolerant'])}, f1={percentage(top['f1_tolerant'])}",
        f"  average precision={percentage(top['average_precision'])}, ROC AUC={decimal(top['roc_auc'])}, Brier={decimal(top['brier'], 4)}",
        f"  predicted={top['predicted_events']}, actual={top['actual_events']}",
        "",
        "NAIVE EVENT BASELINES",
        f"Rolling-low BOTTOM baseline: precision={percentage(baseline_bottom['precision'])}, recall={percentage(baseline_bottom['recall'])}, f1={percentage(baseline_bottom['f1'])}",
        f"Rolling-high TOP baseline:   precision={percentage(baseline_top['precision'])}, recall={percentage(baseline_top['recall'])}, f1={percentage(baseline_top['f1'])}",
        "",
        "LONG-ONLY CAPITAL SIMULATION",
        f"Strategy mode: {metrics['strategy_mode']}",
        f"Structural timeframe: {metrics['timeframe']}",
        f"TOP signal timeframe: {metrics.get('top_signal_timeframe', metrics['timeframe'])}",
        f"Execution timeframe: {metrics.get('execution_timeframe', metrics['timeframe'])}",
        f"Initial capital: ${config.initial_capital:,.2f}",
        f"Strategy ending capital: ${metrics['strategy_ending_capital']:,.2f}",
        f"Strategy total return: {percentage(metrics['strategy_return'])}",
        f"Same-period buy-and-hold ending capital: ${metrics['buy_hold_ending_capital']:,.2f}",
        f"Same-period buy-and-hold return: {percentage(metrics['buy_hold_return'])}",
        f"Excess return: {percentage(metrics['excess_return'])}",
        f"Strategy maximum drawdown: {percentage(metrics['strategy_maximum_drawdown'])}",
        f"Same-period buy-and-hold maximum drawdown: {percentage(metrics['buy_hold_maximum_drawdown'])}",
        f"Strategy Sharpe estimate: {decimal(metrics['strategy_sharpe'])}",
        f"Same-period buy-and-hold Sharpe estimate: {decimal(metrics['buy_hold_sharpe'])}",
        f"Market exposure: {percentage(metrics['market_exposure'])}",
        f"Simulated buys: {metrics['simulated_buys']}",
        f"Simulated sells including final liquidation: {metrics['simulated_sells']}",
        f"Normal model/hybrid sells: {metrics['normal_model_sells']}",
        f"Fibonacci target enabled: {metrics['fibonacci_target_enabled']}",
        f"Fibonacci target ratio: {metrics['fibonacci_target_ratio']:.3f}",
        f"Fibonacci exits: {metrics['fibonacci_exit_count']}",
        f"ATR exits: {metrics['atr_exit_count']}",
        f"Tightened ATR exits: {metrics['tightened_atr_exit_count']}",
        f"Trend-breakdown exits: {metrics['trend_breakdown_exit_count']}",
        f"BOTTOM entries: {metrics['bottom_probability_entry_count']}",
        f"Trend-pullback entries: {metrics['trend_pullback_entry_count']}",
        f"Bull-regime bars: {metrics['bull_regime_bar_count']}",
        f"TOP-probability/MTF-confirmed exits: {metrics['top_probability_exit_count']}",
        f"MTF weekly-TOP + daily-confirmation exits: {metrics.get('mtf_top_exit_count', 0)}",
        (
            "TOP-exit average absolute distance from local peak: "
            f"{percentage(metrics['top_exit_quality']['top_probability_exits']['average_absolute_peak_gap'])}"
        ),
        (
            "TOP-exit average captured move: "
            f"{percentage(metrics['top_exit_quality']['top_probability_exits']['average_top_capture_ratio'])}"
        ),
        (
            "TOP-exit average bars from local peak: "
            f"{decimal(metrics['top_exit_quality']['top_probability_exits']['average_bars_from_peak'])}"
        ),
        (
            "TOP-exit average upside missed after sale: "
            f"{percentage(metrics['top_exit_quality']['top_probability_exits']['average_post_exit_max_upside'])}"
        ),
        (
            "TOP-exit average post-sale drawdown: "
            f"{percentage(metrics['top_exit_quality']['top_probability_exits']['average_post_exit_max_drawdown'])}"
        ),
        f"Trend confirmation bars: {metrics['trend_breakdown_confirmation_bars']}",
        f"Require declining slow EMA: {metrics['trend_breakdown_require_slow_ema_decline']}",
        f"Total transaction fees: ${metrics['total_transaction_fees']:,.2f}",
        "",
        "REALIZED PERFORMANCE BY EXIT TRIGGER",
        (
            "Fibonacci: "
            f"count={metrics['exit_performance']['fibonacci']['count']}, "
            f"pnl=${metrics['exit_performance']['fibonacci']['total_realized_pnl']:,.2f}, "
            f"win_rate={percentage(metrics['exit_performance']['fibonacci']['win_rate'])}, "
            f"avg_return={percentage(metrics['exit_performance']['fibonacci']['average_position_return'])}"
        ),
        (
            "Trend breakdown: "
            f"count={metrics['exit_performance']['trend_breakdown']['count']}, "
            f"pnl=${metrics['exit_performance']['trend_breakdown']['total_realized_pnl']:,.2f}, "
            f"win_rate={percentage(metrics['exit_performance']['trend_breakdown']['win_rate'])}, "
            f"avg_return={percentage(metrics['exit_performance']['trend_breakdown']['average_position_return'])}"
        ),
        (
            "ATR trailing stop: "
            f"count={metrics['exit_performance']['atr_trailing_stop']['count']}, "
            f"pnl=${metrics['exit_performance']['atr_trailing_stop']['total_realized_pnl']:,.2f}, "
            f"win_rate={percentage(metrics['exit_performance']['atr_trailing_stop']['win_rate'])}, "
            f"avg_return={percentage(metrics['exit_performance']['atr_trailing_stop']['average_position_return'])}"
        ),
        (
            "Tightened ATR trailing stop: "
            f"count={metrics['exit_performance']['atr_trailing_stop_tightened']['count']}, "
            f"pnl=${metrics['exit_performance']['atr_trailing_stop_tightened']['total_realized_pnl']:,.2f}, "
            f"win_rate={percentage(metrics['exit_performance']['atr_trailing_stop_tightened']['win_rate'])}, "
            f"avg_return={percentage(metrics['exit_performance']['atr_trailing_stop_tightened']['average_position_return'])}"
        ),
        "",
        "FULL-SERIES BUY AND HOLD (REQUESTED REFERENCE, NOT A FAIR MODEL COMPARISON)",
        f"Ending capital: ${full_bh['ending_capital']:,.2f}",
        f"Return: {percentage(full_bh['return'])}",
        f"Maximum drawdown: {percentage(full_bh['maximum_drawdown'])}",
        "",
        "METHOD NOTES",
        "- BOTTOM and TOP are independent binary classifiers.",
        "- Future bars create labels only; they are never features.",
        "- Thresholds are calibrated on a validation period, not the final test.",
        f"- Final-test models are retrained every {config.retrain_every_bars} bars using only prior data.",
        "- BUY decisions use the calibrated BOTTOM probability and execute at the next bar open.",
        "- BOTTOM_REVERSAL_TOP_EXIT buys only on calibrated BOTTOM signals and sells only on calibrated TOP signals.",
        "- Fibonacci, trend-breakdown, pullback, bull-regime, and ATR rules are disabled in BOTTOM_REVERSAL_TOP_EXIT.",
        "- TOP proximity fields are post-trade evaluation metrics and never feed the decision model.",
        "- BOTTOM_ENTRY_MTF_TOP_EXIT keeps the structural BOTTOM entry, trains TOP on completed weekly bars, and sells only after daily reversal confirmation.",
        "- Weekly and structural signals are released conservatively on the first matching daily session; execution occurs at the next daily open.",
        "- Other strategy modes retain their own isolated or hybrid exit policies.",
        "- Signal is generated at close and executed at the next bar open.",
        "- Alpaca is not used for data, authentication, orders, or account access.",
        "- This is research code and not a trading recommendation.",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plotting and output
# -----------------------------------------------------------------------------


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    bottom = metrics["bottom_metrics"]
    top = metrics["top_metrics"]
    full = metrics["full_series_buy_hold"]
    return {
        "symbol": metrics["symbol"],
        "backend": metrics["backend"],
        "timeframe": metrics["timeframe"],
        "strategy_mode": metrics["strategy_mode"],
        "execution_timeframe": metrics.get("execution_timeframe", metrics["timeframe"]),
        "top_signal_timeframe": metrics.get("top_signal_timeframe", metrics["timeframe"]),
        "mtf_top_exit_count": metrics.get("mtf_top_exit_count", 0),
        "mtf_effective_top_threshold": metrics.get("mtf_effective_top_threshold"),
        "structural_backend": metrics.get("structural_backend", metrics["backend"]),
        "exit_risk_backend": metrics.get("exit_risk_backend"),
        "exit_risk_effective_threshold": metrics.get("exit_risk_effective_threshold"),
        "exit_risk_calibration_gate_passed": metrics.get(
            "exit_risk_calibration_gate_passed"
        ),
        "exit_risk_event_tolerance_weeks": metrics.get(
            "exit_risk_event_tolerance_weeks"
        ),
        "exit_risk_candidate_signal_count": metrics.get(
            "exit_risk_candidate_signal_count", 0
        ),
        "exit_risk_signal_count": metrics.get("exit_risk_signal_count", 0),
        "exit_risk_exit_count": metrics.get("exit_risk_exit_count", 0),
        "trend_resumption_reentry_count": metrics.get(
            "trend_resumption_reentry_count", 0
        ),
        "exit_approx_within_10pct_rate": metrics.get(
            "exit_risk_approximation", {}
        ).get("within_10pct_of_local_peak_rate", 0.0),
        "exit_approx_average_peak_gap": metrics.get(
            "exit_risk_approximation", {}
        ).get("average_absolute_peak_gap", 0.0),
        "exit_approx_average_missed_upside": metrics.get(
            "exit_risk_approximation", {}
        ).get("average_missed_upside", 0.0),
        "exit_approx_average_drawdown_avoided": metrics.get(
            "exit_risk_approximation", {}
        ).get("average_drawdown_avoided", 0.0),
        "exit_approx_average_balance": metrics.get(
            "exit_risk_approximation", {}
        ).get("average_exit_balance", 0.0),
        "extrema_label_price_basis": metrics[
            "extrema_label_price_basis"
        ],
        "test_start": metrics["test_start"],
        "test_end": metrics["test_end"],
        "bottom_threshold": metrics["bottom_calibration"]["threshold"],
        "top_threshold": metrics["top_calibration"]["threshold"],
        "bottom_precision": bottom["precision_tolerant"],
        "bottom_recall": bottom["recall_tolerant"],
        "bottom_f1": bottom["f1_tolerant"],
        "top_precision": top["precision_tolerant"],
        "top_recall": top["recall_tolerant"],
        "top_f1": top["f1_tolerant"],
        "strategy_ending_capital": metrics["strategy_ending_capital"],
        "strategy_return": metrics["strategy_return"],
        "buy_hold_ending_capital": metrics["buy_hold_ending_capital"],
        "buy_hold_return": metrics["buy_hold_return"],
        "excess_return": metrics["excess_return"],
        "strategy_maximum_drawdown": metrics["strategy_maximum_drawdown"],
        "buy_hold_maximum_drawdown": metrics["buy_hold_maximum_drawdown"],
        "strategy_sharpe": metrics["strategy_sharpe"],
        "buy_hold_sharpe": metrics["buy_hold_sharpe"],
        "market_exposure": metrics["market_exposure"],
        "simulated_buys": metrics["simulated_buys"],
        "simulated_sells": metrics["simulated_sells"],
        "total_transaction_fees": metrics["total_transaction_fees"],

        "fibonacci_target_enabled": metrics["fibonacci_target_enabled"],
        "fibonacci_target_ratio": metrics["fibonacci_target_ratio"],
        "fibonacci_swing_lookback": metrics["fibonacci_swing_lookback"],
        "fibonacci_low_lookback": metrics["fibonacci_low_lookback"],
        "fibonacci_exit_count": metrics["fibonacci_exit_count"],
        "fibonacci_exit_rate": metrics["fibonacci_exit_rate"],
        "top_tighten_trailing_enabled": metrics["top_tighten_trailing_enabled"],
        "atr_trailing_multiplier": metrics["atr_trailing_multiplier"],
        "tightened_atr_multiplier": metrics["tightened_atr_multiplier"],
        "trend_breakdown_exit_count": metrics["trend_breakdown_exit_count"],
        "bottom_probability_entry_count": metrics[
            "bottom_probability_entry_count"
        ],
        "trend_pullback_entry_count": metrics[
            "trend_pullback_entry_count"
        ],
        "bull_regime_bar_count": metrics["bull_regime_bar_count"],
        "adaptive_bull_regime_enabled": metrics[
            "adaptive_bull_regime_enabled"
        ],
        "trend_pullback_entry_enabled": metrics[
            "trend_pullback_entry_enabled"
        ],
        "atr_exit_count": metrics["atr_exit_count"],
        "tightened_atr_exit_count": metrics["tightened_atr_exit_count"],
        "top_probability_exit_count": metrics["top_probability_exit_count"],
        "final_liquidation_count": metrics["final_liquidation_count"],
        "trend_breakdown_confirmation_bars": metrics[
            "trend_breakdown_confirmation_bars"
        ],
        "trend_breakdown_require_slow_ema_decline": metrics[
            "trend_breakdown_require_slow_ema_decline"
        ],

        "fibonacci_exit_total_pnl": metrics["exit_performance"]["fibonacci"][
            "total_realized_pnl"
        ],
        "fibonacci_exit_average_pnl": metrics["exit_performance"]["fibonacci"][
            "average_realized_pnl"
        ],
        "fibonacci_exit_win_rate": metrics["exit_performance"]["fibonacci"][
            "win_rate"
        ],
        "fibonacci_exit_average_return": metrics["exit_performance"]["fibonacci"][
            "average_position_return"
        ],
        "fibonacci_exit_average_holding_bars": metrics["exit_performance"][
            "fibonacci"
        ]["average_holding_bars"],

        "trend_breakdown_exit_total_pnl": metrics["exit_performance"][
            "trend_breakdown"
        ]["total_realized_pnl"],
        "trend_breakdown_exit_average_pnl": metrics["exit_performance"][
            "trend_breakdown"
        ]["average_realized_pnl"],
        "trend_breakdown_exit_win_rate": metrics["exit_performance"][
            "trend_breakdown"
        ]["win_rate"],
        "trend_breakdown_exit_average_return": metrics["exit_performance"][
            "trend_breakdown"
        ]["average_position_return"],
        "trend_breakdown_exit_average_holding_bars": metrics["exit_performance"][
            "trend_breakdown"
        ]["average_holding_bars"],

        "atr_exit_total_pnl": metrics["exit_performance"]["atr_trailing_stop"][
            "total_realized_pnl"
        ],
        "atr_exit_average_pnl": metrics["exit_performance"]["atr_trailing_stop"][
            "average_realized_pnl"
        ],
        "atr_exit_win_rate": metrics["exit_performance"]["atr_trailing_stop"][
            "win_rate"
        ],
        "atr_exit_average_return": metrics["exit_performance"][
            "atr_trailing_stop"
        ]["average_position_return"],
        "atr_exit_average_holding_bars": metrics["exit_performance"][
            "atr_trailing_stop"
        ]["average_holding_bars"],

        "tightened_atr_exit_total_pnl": metrics["exit_performance"][
            "atr_trailing_stop_tightened"
        ]["total_realized_pnl"],
        "tightened_atr_exit_average_pnl": metrics["exit_performance"][
            "atr_trailing_stop_tightened"
        ]["average_realized_pnl"],
        "tightened_atr_exit_win_rate": metrics["exit_performance"][
            "atr_trailing_stop_tightened"
        ]["win_rate"],
        "tightened_atr_exit_average_return": metrics["exit_performance"][
            "atr_trailing_stop_tightened"
        ]["average_position_return"],
        "tightened_atr_exit_average_holding_bars": metrics["exit_performance"][
            "atr_trailing_stop_tightened"
        ]["average_holding_bars"],

        "top_probability_exit_total_pnl": metrics["exit_performance"][
            "top_probability"
        ]["total_realized_pnl"],
        "top_probability_exit_win_rate": metrics["exit_performance"][
            "top_probability"
        ]["win_rate"],
        "top_exit_average_peak_gap": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_exit_to_peak_gap"],
        "top_exit_average_absolute_peak_gap": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_absolute_peak_gap"],
        "top_exit_average_bars_from_peak": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_bars_from_peak"],
        "top_exit_average_capture_ratio": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_top_capture_ratio"],
        "top_exit_average_post_sale_upside": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_post_exit_max_upside"],
        "top_exit_average_post_sale_drawdown": metrics["top_exit_quality"][
            "top_probability_exits"
        ]["average_post_exit_max_drawdown"],
        "final_liquidation_total_pnl": metrics["exit_performance"][
            "final_liquidation"
        ]["total_realized_pnl"],

        "full_series_buy_hold_return": full["return"],
    }


def serializable_config(config: BacktestConfig) -> dict[str, Any]:
    data = asdict(config)
    data["assets"] = list(config.assets)
    data["model_backends"] = list(config.model_backends)
    data["exit_risk_model_backends"] = list(config.exit_risk_model_backends)
    data["rotation_models"] = list(config.rotation_models)
    data["rotation_switch_margin_candidates"] = list(
        config.rotation_switch_margin_candidates
    )
    return bson_value(data)


def flatten_rotation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": "PORTFOLIO",
        "backend": metrics["backend"],
        "model_family": metrics.get("model_family", metrics["backend"]),
        "random_seed": metrics.get("random_seed"),
        "repetition_index": metrics.get("repetition_index", 1),
        "repetition_count": metrics.get("repetition_count", 1),
        "strategy_mode": metrics["strategy_mode"],
        "strategy_label": metrics["strategy_label"],
        "portfolio_rotation": True,
        "assets": metrics["assets"],
        "decision_horizon_days": metrics["decision_horizon_days"],
        "decision_horizon_label": metrics["decision_horizon_label"],
        "benchmark_name": metrics["benchmark_name"],
        "strategy_ending_capital": metrics["strategy_ending_capital"],
        "strategy_return": metrics["strategy_return"],
        "buy_hold_ending_capital": metrics["buy_hold_ending_capital"],
        "buy_hold_return": metrics["buy_hold_return"],
        "excess_return": metrics["excess_return"],
        "strategy_maximum_drawdown": metrics["strategy_maximum_drawdown"],
        "buy_hold_maximum_drawdown": metrics["buy_hold_maximum_drawdown"],
        "strategy_sharpe": metrics["strategy_sharpe"],
        "buy_hold_sharpe": metrics["buy_hold_sharpe"],
        "strategy_cagr": metrics["strategy_cagr"],
        "buy_hold_cagr": metrics["buy_hold_cagr"],
        "compound_log_growth": metrics["compound_log_growth"],
        "risk_adjusted_compound_score": metrics.get("risk_adjusted_compound_score"),
        "walk_forward_fold_count": metrics.get("walk_forward_fold_count"),
        "walk_forward_purge_days": metrics.get("walk_forward_purge_days"),
        "downside_penalty": metrics.get("downside_penalty"),
        "drawdown_penalty": metrics.get("drawdown_penalty"),
        "effective_switch_margin": metrics.get("effective_switch_margin"),
        "market_exposure": metrics["market_exposure"],
        "cash_days": metrics["cash_days"],
        "simulated_buys": metrics["simulated_buys"],
        "simulated_sells": metrics["simulated_sells"],
        "capital_rotations": metrics["capital_rotations"],
        "cycles_per_year": metrics["cycles_per_year"],
        "average_holding_days": metrics["average_holding_days"],
        "average_holding_bars": metrics.get("average_holding_bars"),
        "average_holding_minutes": metrics.get("average_holding_minutes"),
        "overnight_positions_allowed": metrics.get("overnight_positions_allowed"),
        "intraday_rotations_allowed": metrics.get("intraday_rotations_allowed"),
        "maximum_entries_per_session": metrics.get("maximum_entries_per_session"),
        "maximum_exits_per_session": metrics.get("maximum_exits_per_session"),
        "invested_sessions": metrics.get("invested_sessions"),
        "winning_sessions": metrics.get("winning_sessions"),
        "session_win_rate": metrics.get("session_win_rate"),
        "reference_benchmark_name": metrics.get("reference_benchmark_name"),
        "reference_buy_hold_ending_capital": metrics.get("reference_buy_hold_ending_capital"),
        "reference_buy_hold_return": metrics.get("reference_buy_hold_return"),
        "reference_buy_hold_maximum_drawdown": metrics.get("reference_buy_hold_maximum_drawdown"),
        "reference_buy_hold_sharpe": metrics.get("reference_buy_hold_sharpe"),
        "reference_buy_hold_cagr": metrics.get("reference_buy_hold_cagr"),
        "geometric_trade_return": metrics["geometric_trade_return"],
        "total_transaction_fees": metrics["total_transaction_fees"],
        "turnover_ratio": metrics["turnover_ratio"],
        "effective_compute_device": metrics.get("effective_compute_device"),
        "gpu_name": metrics.get("gpu_name"),
        "qrdqn_parallel_folds_effective": metrics.get("qrdqn_parallel_folds_effective"),
        "qrdqn_training_steps_mean_used": metrics.get("qrdqn_training_steps_mean_used"),
        "qrdqn_early_stopped_folds": metrics.get("qrdqn_early_stopped_folds"),
    }


def run_compound_rotation_job(
    job_id: str,
    config: BacktestConfig,
    demo: bool,
    db: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    from engine.capital_rotation import run_rotation_models
    from engine.day_trade_open_close import run_open_close_models

    def emit_progress(
        percent: float,
        stage: str,
        completed_runs: int = 0,
    ) -> None:
        safe_stage = str(stage).replace("|", "/").strip()
        print(
            f"JOB_PROGRESS|{float(percent):.1f}|"
            f"{int(completed_runs)}|{safe_stage}",
            flush=True,
        )

    def emit_trade(trade: dict[str, Any]) -> None:
        normalized = bson_value(trade)
        for key in ("timestamp", "entry_timestamp"):
            value = normalized.get(key)
            if isinstance(value, (datetime, pd.Timestamp)):
                normalized[key] = pd.Timestamp(value).isoformat()
        print(
            "JOB_TRADE|"
            + json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
            flush=True,
        )

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    is_day_trade = config.strategy_mode == "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE"
    source_timeframe_value = "15Min" if is_day_trade else "1Day"
    source_config = replace(config, timeframe=source_timeframe_value)

    if is_day_trade:
        print(
            "Compound Capital Rotation — Day Trade Open→Close: one shared capital pool; "
            "15-minute source bars aggregated to one decision per session; "
            "one entry maximum; same-session close exit; no overnight exposure; "
            f"provider={config.market_data_provider}" + (f"/{config.alpaca_feed}" if config.market_data_provider == "alpaca" else "") + "; "
            f"purge={config.rotation_purge_days}; "
            f"downside_penalty={config.rotation_downside_penalty}; "
            f"drawdown_penalty={config.rotation_drawdown_penalty}."
        )
    else:
        print(
            "Compound capital rotation Swing: one shared capital pool; "
            f"daily candles; {config.rotation_horizon_days}-session utility; expanding walk-forward; "
            f"provider={config.market_data_provider}" + (f"/{config.alpaca_feed}" if config.market_data_provider == "alpaca" else "") + "; "
            f"purge={config.rotation_purge_days}; "
            f"downside_penalty={config.rotation_downside_penalty}; "
            f"drawdown_penalty={config.rotation_drawdown_penalty}."
        )
    total_assets = max(1, len(config.assets))
    emit_progress(2.0, "Preparing shared-capital rotation")
    for asset_position, symbol in enumerate(config.assets, start=1):
        loading_percent = 3.0 + 12.0 * (
            (asset_position - 1) / total_assets
        )
        emit_progress(
            loading_percent,
            f"Loading market data {asset_position}/{total_assets} — {symbol}",
        )
        if is_day_trade:
            print(
                f"Loading {symbol} 15Min source bars for Day Trade Open→Close session aggregation..."
            )
        else:
            print(
                f"Loading {symbol} 1Day bars for Swing capital rotation..."
            )
        try:
            raw = (
                generate_demo_data(symbol, source_config)
                if demo
                else load_market_bars(symbol, source_config)
            )
            bars_by_symbol[symbol] = validate_and_clean_bars(
                raw, source_config
            )
            emit_progress(
                3.0 + 12.0 * (asset_position / total_assets),
                f"Loaded market data {asset_position}/{total_assets} — {symbol}",
            )
        except Exception as exc:
            failures.append(
                {
                    "symbol": symbol,
                    "backend": "data_load",
                    "error": str(exc),
                }
            )
            print(
                f"ERROR loading {symbol}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if len(bars_by_symbol) < 2:
        raise ValueError(
            "Compound rotation needs at least two successfully loaded assets."
        )

    emit_progress(
        17.0,
        (
            "Building aligned Open→Close session panel and walk-forward folds"
            if is_day_trade
            else "Building aligned daily panel and walk-forward folds"
        ),
    )
    model_runner = run_open_close_models if is_day_trade else run_rotation_models
    results = model_runner(
        bars_by_symbol,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=emit_progress,
        trade_callback=emit_trade,
    )
    comparisons: list[dict[str, Any]] = []
    total_results = max(1, len(results))
    for result_position, result in enumerate(results, start=1):
        backend = result.backend
        emit_progress(
            92.0 + 6.0 * ((result_position - 1) / total_results),
            f"Saving {result.metrics.get('strategy_label', backend)} results to MongoDB",
            len(results),
        )
        update_asset_status(
            db,
            job_id,
            "PORTFOLIO",
            backend,
            "running",
        )
        replace_run_result(
            db,
            job_id=job_id,
            symbol="PORTFOLIO",
            backend=backend,
            metrics=result.metrics,
            summary=result.summary,
            predictions=result.predictions,
            trades=result.trades,
            batch_size=config.mongo_write_batch_size,
        )
        comparisons.append(
            bson_value(flatten_rotation_metrics(result.metrics))
        )
        update_asset_status(
            db,
            job_id,
            "PORTFOLIO",
            backend,
            "completed",
        )
        print(
            f"PORTFOLIO/{backend}: Strategy="
            f"{result.metrics['strategy_return']:.2%} | "
            f"Benchmark={result.metrics['buy_hold_return']:.2%}",
            flush=True,
        )
    emit_progress(
        99.0,
        "Finalizing comparison and reports",
        len(results),
    )
    return comparisons, failures


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def run_asset_backend(
    symbol: str,
    backend: str,
    bars: pd.DataFrame,
    config: BacktestConfig,
    daily_bars: pd.DataFrame | None = None,
) -> AssetRunResult:
    if config.strategy_mode == "BOTTOM_ENTRY_MTF_TOP_EXIT":
        if daily_bars is None or daily_bars.empty:
            raise ValueError(
                "Multi-timeframe TOP exit requires daily source bars."
            )
        return run_asset_backend_mtf(
            symbol, backend, bars, daily_bars, config
        )
    if is_exit_risk_strategy(config):
        if daily_bars is None or daily_bars.empty:
            raise ValueError(
                "Exit Risk strategies require daily source bars."
            )
        return run_asset_backend_exit_risk(
            symbol, backend, bars, daily_bars, config
        )

    dataset = build_dataset(bars, config)
    split = split_dataset(dataset, config)

    bottom_calibration, top_calibration, calibration = calibrate_models(
        split, backend, config
    )
    predictions = walk_forward_predict(dataset, split, backend, config)
    predictions = add_decision_signals(
        predictions,
        bottom_calibration.threshold,
        top_calibration.threshold,
        config,
    )
    predictions, trades = simulate_strategy(predictions, config)
    trades = annotate_exit_quality(predictions, trades, config)

    metrics = build_metrics(
        symbol=symbol,
        backend=backend,
        bars=bars,
        split=split,
        calibration=calibration,
        bottom_calibration=bottom_calibration,
        top_calibration=top_calibration,
        predictions=predictions,
        trades=trades,
        config=config,
    )
    summary = build_summary(metrics, config)
    return AssetRunResult(
        symbol=symbol,
        backend=backend,
        predictions=predictions,
        trades=trades,
        summary_text=summary,
        metrics=metrics,
    )




def resolve_symbol_config(
    base_config: BacktestConfig,
    symbol: str,
) -> BacktestConfig:
    if base_config.parameter_mode != "asset_profiles":
        return base_config

    override = base_config.asset_overrides.get(
        str(symbol).upper()
    )
    if not override:
        return base_config

    merged = serializable_config(base_config)
    merged.update(override)
    merged["assets"] = [str(symbol).upper()]
    merged["model_backends"] = list(
        base_config.model_backends
    )
    merged["parameter_mode"] = base_config.parameter_mode
    merged["asset_overrides"] = base_config.asset_overrides
    return config_from_mapping(merged)


def effective_parallel_workers(config: BacktestConfig) -> int:
    workers = max(1, int(config.max_parallel_workers))
    uses_xgboost = (
        "xgboost" in config.model_backends
        or (
            is_exit_risk_strategy(config)
            and "xgboost" in configured_exit_risk_backends(config)
        )
    )
    if (
        str(config.xgb_device).lower() == "cuda"
        and uses_xgboost
    ):
        workers = min(
            workers,
            max(1, int(config.cuda_parallel_workers)),
        )
    return min(workers, max(1, len(config.assets)))


def update_asset_status(
    db: Any,
    job_id: str,
    symbol: str,
    backend: str,
    status: str,
    error: str | None = None,
) -> None:
    key = f"{symbol}__{backend}"
    value: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
    }
    if error:
        value["error"] = error
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {"$set": {f"asset_status.{key}": value}},
    )


def run_symbol_worker(
    job_id: str,
    symbol: str,
    config_payload: dict[str, Any],
    demo: bool,
) -> dict[str, Any]:
    base_config = config_from_mapping(config_payload)
    config = resolve_symbol_config(base_config, symbol)
    if (
        config.xgb_device == "cpu"
        and config.xgb_n_jobs == -1
        and config.max_parallel_workers > 1
    ):
        config = replace(config, xgb_n_jobs=1)
    if (
        config.catboost_thread_count == -1
        and config.max_parallel_workers > 1
    ):
        config = replace(config, catboost_thread_count=1)

    client = create_client()
    db = get_database(client)
    ensure_database(db)
    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def run_keys_for_structural_backend(backend: str) -> list[tuple[str, str | None]]:
        if not is_exit_risk_strategy(config):
            return [(backend, None)]
        return [
            (f"{backend}__exit_{exit_backend}", exit_backend)
            for exit_backend in configured_exit_risk_backends(config)
        ]

    try:
        all_run_keys = [
            run_key
            for backend in config.model_backends
            for run_key, _ in run_keys_for_structural_backend(backend)
        ]
        for run_key in all_run_keys:
            update_asset_status(
                db, job_id, symbol, run_key, "loading"
            )

        print(f"Loading {symbol}...")
        try:
            daily_source: pd.DataFrame | None = None
            if demo:
                raw = generate_demo_data(symbol, config)
                prepared = aggregate_swing_bars(raw, config)
                if config.strategy_mode in {
                    "BOTTOM_ENTRY_MTF_TOP_EXIT",
                    "BOTTOM_ENTRY_EXIT_RISK_V1",
                    "BOTTOM_ENTRY_EXIT_RISK_SWING_1D",
                }:
                    daily_config = replace(config, timeframe="1Day")
                    daily_source = validate_and_clean_bars(raw, daily_config)
            else:
                prepared = load_market_bars(symbol, config)
                if config.strategy_mode in {
                    "BOTTOM_ENTRY_MTF_TOP_EXIT",
                    "BOTTOM_ENTRY_EXIT_RISK_V1",
                    "BOTTOM_ENTRY_EXIT_RISK_SWING_1D",
                }:
                    print(
                        f"Loading {symbol} daily confirmation bars for "
                        "multi-timeframe exit strategy..."
                    )
                    daily_config = replace(config, timeframe="1Day")
                    daily_source = validate_and_clean_bars(
                        load_yfinance_bars(symbol, daily_config),
                        daily_config,
                    )
            bars = validate_and_clean_bars(prepared, config)
        except Exception as exc:
            error = str(exc)
            for run_key in all_run_keys:
                update_asset_status(
                    db, job_id, symbol, run_key, "failed", error
                )
                failures.append(
                    {
                        "symbol": symbol,
                        "backend": run_key,
                        "error": error,
                    }
                )
            return {"comparisons": comparisons, "failures": failures}

        for backend in config.model_backends:
            # In comparison mode, the structural BOTTOM is intentionally
            # trained once and reused by all Exit Risk candidates.
            bottom_context: ExitRiskBottomContext | None = None
            if is_exit_risk_strategy(config):
                try:
                    print(
                        f"Preparing {symbol} structural BOTTOM with {backend}..."
                    )
                    bottom_context = prepare_exit_risk_bottom_context(
                        backend, bars, config
                    )
                except Exception as exc:
                    error = str(exc)
                    for run_key, _ in run_keys_for_structural_backend(backend):
                        update_asset_status(
                            db, job_id, symbol, run_key, "failed", error
                        )
                        failures.append(
                            {
                                "symbol": symbol,
                                "backend": run_key,
                                "error": error,
                            }
                        )
                    continue

            for run_key, exit_backend in run_keys_for_structural_backend(backend):
                try:
                    update_asset_status(
                        db, job_id, symbol, run_key, "running"
                    )
                    if exit_backend is None:
                        print(f"Running {symbol} with {backend}...")
                        result = run_asset_backend(
                            symbol, backend, bars, config, daily_source
                        )
                    else:
                        assert daily_source is not None
                        print(
                            f"Running {symbol}: BOTTOM={backend}, "
                            f"EXIT={exit_backend}..."
                        )
                        result = run_asset_backend_exit_risk(
                            symbol,
                            backend,
                            bars,
                            daily_source,
                            config,
                            exit_backend=exit_backend,
                            bottom_context=bottom_context,
                        )

                    replace_run_result(
                        db,
                        job_id=job_id,
                        symbol=symbol,
                        backend=run_key,
                        metrics=result.metrics,
                        summary=result.summary_text,
                        predictions=result.predictions,
                        trades=result.trades,
                        batch_size=config.mongo_write_batch_size,
                    )
                    comparison_row = flatten_metrics(result.metrics)
                    comparison_row["parameter_mode"] = config.parameter_mode
                    comparison_row["profile_symbol"] = (
                        symbol
                        if config.parameter_mode == "asset_profiles"
                        else None
                    )
                    comparison_row["profile_timeframe"] = config.timeframe
                    comparisons.append(bson_value(comparison_row))
                    update_asset_status(
                        db, job_id, symbol, run_key, "completed"
                    )
                    print(
                        f"{symbol}/{run_key}: Strategy="
                        f"{result.metrics['strategy_return']:.2%} | "
                        f"BuyHold={result.metrics['buy_hold_return']:.2%}"
                    )
                except Exception as exc:
                    error = str(exc)
                    update_asset_status(
                        db, job_id, symbol, run_key, "failed", error
                    )
                    failures.append(
                        {
                            "symbol": symbol,
                            "backend": run_key,
                            "error": error,
                        }
                    )
                    print(
                        f"ERROR {symbol}/{run_key}: {error}",
                        file=sys.stderr,
                    )

        return {"comparisons": comparisons, "failures": failures}
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.job_id)
    validate_config(config)

    client = create_client()
    db = get_database(client)
    ensure_database(db)
    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        print("Local simulation only. No broker order will be created.")
        print(f"Assets: {', '.join(config.assets)}")
        print(f"Structural backends: {', '.join(config.model_backends)}")
        if is_exit_risk_strategy(config):
            print(
                "Exit Risk backends: "
                + ", ".join(configured_exit_risk_backends(config))
            )
        print("Storage: MongoDB only. No runtime result files will be created.")

        if config.strategy_mode in {
            "COMPOUND_ROTATION_SWING_XGBOOST",
            "COMPOUND_ROTATION_SWING_QRDQN",
            "COMPOUND_ROTATION_SWING_1W",
            "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
        }:
            comparisons, failures = run_compound_rotation_job(
                args.job_id,
                config,
                bool(args.demo),
                db,
            )
            comparisons.sort(
                key=lambda item: str(item.get("backend", ""))
            )
            replace_comparison(
                db,
                job_id=args.job_id,
                comparison=comparisons,
                failures=failures,
                effective_config=serializable_config(config),
            )
            if not comparisons:
                raise SystemExit(1)
            return

        workers = effective_parallel_workers(config)
        print(
            f"Parallel asset workers: {workers} "
            f"(requested={config.max_parallel_workers}, "
            f"device={config.xgb_device})."
        )

        payload = serializable_config(config)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_symbol_worker,
                    args.job_id,
                    symbol,
                    payload,
                    bool(args.demo),
                ): symbol
                for symbol in config.assets
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    comparisons.extend(
                        result.get("comparisons", [])
                    )
                    worker_failures = result.get("failures", [])
                    failures.extend(worker_failures)
                    for failure in worker_failures:
                        print(
                            "ERROR "
                            f"{failure.get('symbol', symbol)}/"
                            f"{failure.get('backend', 'unknown')}: "
                            f"{failure.get('error', 'Unknown error')}",
                            file=sys.stderr,
                            flush=True,
                        )
                except Exception as exc:
                    error = str(exc)
                    print(
                        f"ERROR loading {symbol}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    for backend in config.model_backends:
                        update_asset_status(
                            db,
                            args.job_id,
                            symbol,
                            backend,
                            "failed",
                            error,
                        )
                        failures.append(
                            {
                                "symbol": symbol,
                                "backend": backend,
                                "error": error,
                            }
                        )

        comparisons.sort(
            key=lambda item: (
                str(item.get("symbol", "")),
                str(item.get("backend", "")),
            )
        )
        replace_comparison(
            db,
            job_id=args.job_id,
            comparison=comparisons,
            failures=failures,
            effective_config=serializable_config(config),
        )
        if not comparisons:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
