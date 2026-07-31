from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pymongo.database import Database

from ..core.config import ACTIVE_STRATEGY_MODE
from ..infrastructure.persistence.mongo_repository import DEFAULT_SETTINGS, get_settings

def normalize_assets(value: list[str]) -> list[str]:
    cleaned: list[str] = []
    for asset in value:
        symbol = str(asset).strip().upper()
        if symbol and re.fullmatch(r"[A-Z0-9.\-^=]+", symbol):
            cleaned.append(symbol)
    if not cleaned:
        raise ValueError("At least one valid asset is required.")
    return list(dict.fromkeys(cleaned))


def normalize_backends(value: list[str]) -> list[str]:
    allowed = {"histgb", "xgboost"}
    cleaned = [
        str(item).strip().lower()
        for item in value
        if str(item).strip()
    ]
    invalid = sorted(set(cleaned) - allowed)
    if invalid:
        raise ValueError(f"Unsupported model backends: {invalid}")
    if not cleaned:
        raise ValueError("Select at least one model backend.")
    return list(dict.fromkeys(cleaned))


def normalize_exit_risk_backends(value: list[str]) -> list[str]:
    allowed = {"xgboost", "histgb", "catboost"}
    cleaned = [
        str(item).strip().lower()
        for item in value
        if str(item).strip()
    ]
    invalid = sorted(set(cleaned) - allowed)
    if invalid:
        raise ValueError(f"Unsupported Exit Risk backends: {invalid}")
    if not cleaned:
        raise ValueError("Select at least one Exit Risk model backend.")
    return list(dict.fromkeys(cleaned))




class ParameterProfileRequest(BaseModel):
    symbol: str
    timeframe: str
    parameters: dict[str, Any]
    profile_name: str | None = None
    source_job_id: str | None = None
    validation_status: Literal[
        "candidate",
        "validated",
        "deprecated",
    ] = "candidate"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-^=]+", normalized):
            raise ValueError("Invalid profile symbol.")
        return normalized


CLASSIC_FIBONACCI_RATIOS = (
    1.272,
    1.414,
    1.618,
    2.000,
    2.618,
    3.618,
    4.236,
)


class AlpacaCredentialsRequest(BaseModel):
    api_key_id: str = Field(..., min_length=1, max_length=512)
    secret_key: str = Field(..., min_length=1, max_length=1024)


class AlpacaConnectionTestRequest(BaseModel):
    feed: Literal["iex", "sip"] = "iex"


class BacktestRequest(BaseModel):
    assets: list[str] = Field(
        default_factory=lambda: ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY"]
    )
    model_backends: list[Literal["histgb", "xgboost"]] = Field(
        default_factory=lambda: ["histgb"]
    )
    parameter_mode: Literal["general", "asset_profiles"] = "general"
    asset_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    strategy_mode: Literal[
        "COMPOUND_ROTATION_SWING_1W",
        "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
    ] = ACTIVE_STRATEGY_MODE
    start_date: str = "2000-01-01"
    end_date: str | None = None
    timeframe: Literal[
        "1Day",
        "1Hour",
        "30Min",
        "15Min",
        "5Min",
        "1Week",
        "2Weeks",
        "3Weeks",
        "4Weeks",
    ] = "1Day"
    market_data_provider: Literal["yahoo", "alpaca"] = "alpaca"
    alpaca_feed: Literal["iex", "sip"] = "iex"
    alpaca_adjustment: Literal["raw", "split", "dividend", "all"] = "all"

    future_horizon: int = Field(5, ge=1, le=60)
    extrema_lookback: int = Field(10, ge=2, le=250)
    reversal_threshold: float = Field(0.03, gt=0, lt=1)
    extrema_tolerance: float = Field(0.01, ge=0, lt=1)
    event_tolerance_bars: int = Field(2, ge=0, le=20)

    calibration_fraction: float = Field(0.15, gt=0, lt=0.50)
    test_fraction: float = Field(0.20, gt=0, lt=0.50)
    retrain_every_bars: int = Field(63, ge=1, le=1000)
    minimum_training_rows: int = Field(500, ge=100)

    threshold_min: float = Field(0.25, gt=0, lt=1)
    threshold_max: float = Field(0.85, gt=0, lt=1)
    threshold_step: float = Field(0.025, gt=0, lt=1)
    bottom_threshold_max: float = Field(0.75, gt=0, lt=1)
    top_threshold_max: float = Field(0.85, gt=0, lt=1)
    bottom_min_precision: float = Field(0.60, gt=0, lt=1)
    bottom_min_recall: float = Field(0.30, ge=0, lt=1)
    top_min_precision: float = Field(0.45, gt=0, lt=1)
    top_min_recall: float = Field(0.00, ge=0, lt=1)
    minimum_calibration_signals: int = Field(3, ge=1, le=10_000)
    bottom_min_calibration_signals: int = Field(3, ge=1, le=10_000)
    top_min_calibration_signals: int = Field(3, ge=1, le=10_000)

    entry_max_rsi: float = Field(60.0, ge=0, le=100)
    entry_require_above_ema50: bool = False
    entry_cooldown_bars: int = Field(3, ge=0, le=100)

    trend_pullback_entry_enabled: bool = True
    trend_pullback_ema: int = Field(20)
    trend_pullback_rsi_min: float = Field(40.0, ge=0, le=100)
    trend_pullback_rsi_max: float = Field(65.0, ge=0, le=100)
    trend_pullback_touch_tolerance: float = Field(0.02, ge=0, lt=1)
    trend_pullback_require_positive_return: bool = True

    adaptive_bull_regime_enabled: bool = True
    bull_regime_ema_fast: int = Field(20)
    bull_regime_ema_slow: int = Field(50)
    bull_regime_require_price_above_slow: bool = True
    bull_regime_require_slow_ema_rising: bool = True
    bull_regime_entry_enabled: bool = False
    bull_regime_entry_confirmation_bars: int = Field(3, ge=1, le=50)

    exit_top_probability: bool = False
    exit_trend_breakdown: bool = True
    exit_atr_trailing_stop: bool = True
    minimum_holding_bars: int = Field(3, ge=0, le=250)
    atr_trailing_multiplier: float = Field(3.0, gt=0, le=20)
    top_tighten_trailing: bool = True
    tightened_atr_multiplier: float = Field(1.5, gt=0, le=20)
    trend_exit_ema_fast: int = Field(5, ge=1, le=500)
    trend_exit_ema_slow: int = Field(20, ge=2, le=1000)
    trend_breakdown_confirmation_bars: int = Field(2, ge=1, le=50)
    trend_breakdown_require_slow_ema_decline: bool = True

    bull_exit_ema_fast: int = Field(20)
    bull_exit_ema_slow: int = Field(50)
    bull_exit_confirmation_bars: int = Field(3, ge=1, le=50)
    bull_exit_require_slow_ema_decline: bool = True

    exit_fibonacci_target: bool = True
    fibonacci_target_ratio: float = Field(1.618)
    fibonacci_swing_lookback: int = Field(50, ge=2, le=1000)
    fibonacci_low_lookback: int = Field(5, ge=1, le=250)

    # Multi-timeframe TOP exit: structural BOTTOM entry, weekly TOP model,
    # and daily reversal confirmation.
    mtf_top_signal_timeframe: Literal["1Week"] = "1Week"
    mtf_top_confirmation_timeframe: Literal["1Day"] = "1Day"
    mtf_top_future_horizon: int = Field(4, ge=1, le=20)
    mtf_top_extrema_lookback: int = Field(10, ge=2, le=100)
    mtf_top_reversal_threshold: float = Field(0.10, gt=0, lt=1)
    mtf_top_extrema_tolerance: float = Field(0.03, ge=0, lt=1)
    mtf_top_probability_floor: float = Field(0.60, gt=0, lt=1)
    mtf_top_retrain_every_bars: int = Field(13, ge=1, le=260)
    mtf_top_minimum_training_rows: int = Field(500, ge=100, le=10_000)
    mtf_daily_confirmation_ema: Literal[5, 10, 20, 50] = 20
    mtf_daily_confirmation_bars: int = Field(2, ge=1, le=20)
    mtf_daily_require_negative_return: bool = True
    mtf_daily_require_ema_decline: bool = True
    mtf_daily_require_lower_high: bool = False
    mtf_top_signal_valid_days: int = Field(20, ge=1, le=120)
    mtf_top_min_position_return: float = Field(0.0, ge=-1, le=10)
    mtf_top_high_lookback_weeks: int = Field(26, ge=1, le=260)
    mtf_top_max_distance_from_high: float = Field(
        0.10, ge=0, lt=1
    )
    mtf_exit_quality_horizon_days: int = Field(20, ge=1, le=120)

    exit_risk_model_backend: Literal["xgboost", "histgb", "catboost"] = "xgboost"
    exit_risk_compare_models: bool = True
    exit_risk_model_backends: list[
        Literal["xgboost", "histgb", "catboost"]
    ] = Field(
        default_factory=lambda: ["xgboost", "histgb", "catboost"]
    )
    exit_risk_signal_timeframe: Literal["1Week"] = "1Week"
    exit_risk_horizon_weeks: int = Field(8, ge=2, le=26)
    exit_risk_event_tolerance_weeks: int = Field(2, ge=0, le=8)
    exit_risk_down_barrier: float = Field(0.12, gt=0, lt=1)
    exit_risk_up_barrier: float = Field(0.08, gt=0, lt=1)
    exit_risk_probability_floor: float = Field(0.60, gt=0, lt=1)
    exit_risk_threshold_max: float = Field(0.85, gt=0, lt=1)
    exit_risk_min_precision: float = Field(0.55, gt=0, lt=1)
    exit_risk_min_recall: float = Field(0.20, ge=0, lt=1)
    exit_risk_min_calibration_signals: int = Field(5, ge=1, le=10_000)
    exit_risk_hard_calibration_gate: bool = True
    exit_risk_retrain_every_bars: int = Field(26, ge=1, le=260)
    exit_risk_minimum_training_rows: int = Field(300, ge=100, le=10_000)
    exit_risk_reentry_enabled: bool = True
    exit_risk_reentry_cooldown_days: int = Field(5, ge=0, le=120)

    swing_exit_horizon_days: int = Field(10, ge=2, le=120)
    swing_exit_event_tolerance_days: int = Field(3, ge=0, le=20)
    swing_exit_down_barrier: float = Field(0.06, gt=0, lt=1)
    swing_exit_up_barrier: float = Field(0.04, gt=0, lt=1)
    swing_exit_retrain_every_bars: int = Field(20, ge=1, le=260)
    swing_exit_minimum_training_rows: int = Field(
        500, ge=100, le=10_000
    )

    rotation_models: list[
        Literal["xgboost_utility", "qrdqn"]
    ] = Field(
        default_factory=lambda: ["xgboost_utility"]
    )
    rotation_horizon_days: int = Field(40, ge=1, le=260)
    rotation_minimum_training_rows: int = Field(700, ge=300, le=20_000)
    rotation_walk_forward_enabled: bool = True
    rotation_walk_forward_calibration_days: int = Field(126, ge=40, le=2_000)
    rotation_walk_forward_test_days: int = Field(504, ge=63, le=5_000)
    rotation_walk_forward_min_test_days: int = Field(126, ge=20, le=2_000)
    rotation_purge_days: int = Field(60, ge=1, le=260)
    rotation_downside_penalty: float = Field(0.20, ge=0, le=10)
    rotation_drawdown_penalty: float = Field(0.35, ge=0, le=10)
    rotation_min_holding_days: int = Field(2, ge=0, le=60)
    rotation_min_expected_edge: float = Field(0.001, ge=0, le=0.50)
    rotation_cash_threshold: float = Field(0.0, ge=-0.50, le=0.50)
    rotation_switch_margin: float = Field(0.005, ge=0, le=0.50)
    rotation_switch_margin_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.0025, 0.005, 0.01]
    )
    rotation_xgb_n_estimators: int = Field(300, ge=10, le=100_000)
    rotation_xgb_learning_rate: float = Field(0.035, gt=0, le=1)
    rotation_xgb_max_depth: int = Field(3, ge=1, le=20)
    rotation_accelerator: Literal["auto", "cpu", "cuda"] = "auto"
    rotation_allow_cpu_fallback: bool = True
    rotation_parallel_models: bool = True
    rotation_xgb_repetitions: int = Field(1, ge=1, le=100)
    rotation_qrdqn_repetitions: int = Field(1, ge=1, le=100)
    rotation_seed_step: int = Field(1_000, ge=1, le=10_000_000)

    qrdqn_training_steps: int = Field(15_000, ge=500, le=2_000_000)
    qrdqn_parallel_folds: int = Field(2, ge=1, le=32)
    qrdqn_early_stopping_enabled: bool = False
    qrdqn_early_stopping_patience: int = Field(4, ge=1, le=100)
    qrdqn_min_training_steps: int = Field(5_000, ge=500, le=2_000_000)
    qrdqn_episode_days: int = Field(252, ge=20, le=2_000)
    qrdqn_replay_size: int = Field(30_000, ge=1_000, le=2_000_000)
    qrdqn_learning_starts: int = Field(750, ge=100, le=100_000)
    qrdqn_batch_size: int = Field(128, ge=16, le=4_096)
    qrdqn_learning_rate: float = Field(0.0003, gt=0, le=1)
    qrdqn_gamma: float = Field(0.99, ge=0, le=1)
    qrdqn_n_quantiles: int = Field(25, ge=5, le=200)
    qrdqn_hidden_dim: int = Field(128, ge=16, le=2_048)
    qrdqn_target_update_steps: int = Field(250, ge=10, le=100_000)
    qrdqn_eval_every_steps: int = Field(1000, ge=100, le=100_000)
    qrdqn_epsilon_start: float = Field(1.0, ge=0, le=1)
    qrdqn_epsilon_end: float = Field(0.05, ge=0, le=1)
    qrdqn_device: Literal["cpu", "cuda"] = "cpu"

    initial_capital: float = Field(10_000.0, gt=0)
    whole_shares: bool = False
    slippage_bps: float = Field(0.0, ge=0, le=500)

    commission_rate: float = Field(0.0, ge=0, le=1)
    sec_fee_rate: float = Field(0.0000206, ge=0, le=1)
    taf_fee_per_share: float = Field(0.000195, ge=0)
    taf_fee_cap: float = Field(9.79, ge=0)
    cat_fee_per_share: float = Field(0.000003, ge=0)

    hist_max_iter: int = Field(300, ge=1, le=100_000)
    hist_learning_rate: float = Field(0.04, gt=0, le=1)
    hist_max_leaf_nodes: int = Field(15, ge=2, le=10_000)
    hist_min_samples_leaf: int = Field(25, ge=1, le=100_000)
    hist_l2_regularization: float = Field(2.0, ge=0)

    xgb_n_estimators: int = Field(350, ge=1, le=100_000)
    xgb_learning_rate: float = Field(0.035, gt=0, le=1)
    xgb_max_depth: int = Field(3, ge=1, le=100)
    xgb_min_child_weight: float = Field(5.0, ge=0)
    xgb_subsample: float = Field(0.85, gt=0, le=1)
    xgb_colsample_bytree: float = Field(0.85, gt=0, le=1)
    xgb_gamma: float = Field(0.0, ge=0)
    xgb_reg_alpha: float = Field(0.10, ge=0)
    xgb_reg_lambda: float = Field(2.0, ge=0)
    xgb_n_jobs: int = Field(-1, ge=-1)
    xgb_device: Literal["cpu", "cuda"] = "cpu"

    catboost_iterations: int = Field(350, ge=1, le=100_000)
    catboost_learning_rate: float = Field(0.035, gt=0, le=1)
    catboost_depth: int = Field(6, ge=1, le=16)
    catboost_l2_leaf_reg: float = Field(3.0, ge=0)
    catboost_random_strength: float = Field(1.0, ge=0)
    catboost_thread_count: int = Field(-1, ge=-1)

    max_parallel_workers: int = Field(3, ge=1, le=32)
    cuda_parallel_workers: int = Field(1, ge=1, le=8)

    yfinance_auto_adjust: bool = True
    yfinance_repair: bool = False
    yfinance_timeout: int = Field(30, ge=1, le=600)
    yfinance_fallback_period: str = "max"

    mongo_cache_enabled: bool = True
    mongo_refresh_overlap_days: int = Field(7, ge=0, le=365)
    mongo_write_batch_size: int = Field(1000, ge=1, le=100_000)
    random_state: int = 42

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, value: list[str]) -> list[str]:
        return normalize_assets(value)

    @field_validator("model_backends")
    @classmethod
    def validate_backends(cls, value: list[str]) -> list[str]:
        return normalize_backends(value)

    @field_validator("exit_risk_model_backends")
    @classmethod
    def validate_exit_backends(cls, value: list[str]) -> list[str]:
        return normalize_exit_risk_backends(value)

    @field_validator("rotation_models")
    @classmethod
    def validate_rotation_models(cls, value: list[str]) -> list[str]:
        allowed = {"xgboost_utility", "qrdqn"}
        cleaned = [
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        ]
        invalid = sorted(set(cleaned) - allowed)
        if invalid:
            raise ValueError(f"Unsupported rotation models: {invalid}")
        if not cleaned:
            raise ValueError("Select at least one capital-rotation model.")
        return list(dict.fromkeys(cleaned))

    @field_validator("fibonacci_target_ratio")
    @classmethod
    def validate_fibonacci_target_ratio(cls, value: float) -> float:
        normalized = round(float(value), 3)
        if normalized not in CLASSIC_FIBONACCI_RATIOS:
            allowed = ", ".join(
                f"{ratio:.3f}"
                for ratio in CLASSIC_FIBONACCI_RATIOS
            )
            raise ValueError(
                "Fibonacci target ratio must be one of the classic "
                f"extensions: {allowed}."
            )
        return normalized

    @model_validator(mode="after")
    def validate_parameter_relationships(self) -> "BacktestRequest":
        if self.calibration_fraction + self.test_fraction >= 1:
            raise ValueError(
                "Calibration fraction plus test fraction must be below 1."
            )
        if self.threshold_min > self.threshold_max:
            raise ValueError(
                "Threshold minimum cannot be greater than threshold maximum."
            )
        if self.qrdqn_min_training_steps > self.qrdqn_training_steps:
            raise ValueError(
                "QR-DQN minimum training steps cannot exceed total training steps."
            )
        if self.threshold_min >= self.bottom_threshold_max:
            raise ValueError(
                "Threshold minimum must be lower than BOTTOM threshold maximum."
            )
        if self.threshold_min >= self.top_threshold_max:
            raise ValueError(
                "Threshold minimum must be lower than TOP threshold maximum."
            )
        if self.exit_risk_probability_floor > self.exit_risk_threshold_max:
            raise ValueError(
                "Exit-risk probability floor cannot exceed "
                "exit-risk threshold maximum."
            )
        allowed_signal_emas = {5, 10, 20, 50}
        signal_emas = {
            self.trend_pullback_ema,
            self.bull_regime_ema_fast,
            self.bull_regime_ema_slow,
            self.bull_exit_ema_fast,
            self.bull_exit_ema_slow,
        }
        if not signal_emas.issubset(allowed_signal_emas):
            raise ValueError(
                "Trend-mode EMA values must be one of 5, 10, 20, 50."
            )
        if self.bull_regime_ema_fast >= self.bull_regime_ema_slow:
            raise ValueError(
                "Bull-regime EMA fast must be smaller than EMA slow."
            )
        if self.bull_exit_ema_fast >= self.bull_exit_ema_slow:
            raise ValueError(
                "Bull-exit EMA fast must be smaller than EMA slow."
            )
        if self.trend_pullback_rsi_min > self.trend_pullback_rsi_max:
            raise ValueError(
                "Pullback RSI minimum cannot exceed maximum."
            )
        if self.trend_exit_ema_fast >= self.trend_exit_ema_slow:
            raise ValueError(
                "Trend EMA fast must be smaller than Trend EMA slow."
            )
        if self.fibonacci_low_lookback > self.fibonacci_swing_lookback:
            raise ValueError(
                "Fibonacci low lookback cannot exceed swing lookback."
            )
        if self.strategy_mode in {
            "BOTTOM_ENTRY_MTF_TOP_EXIT",
            "BOTTOM_ENTRY_EXIT_RISK_V1",
        }:
            if self.timeframe not in {"2Weeks", "3Weeks", "4Weeks"}:
                raise ValueError(
                    "Multi-timeframe exit strategies require a structural "
                    "timeframe of 2Weeks, 3Weeks, or 4Weeks."
                )
            if self.mtf_top_confirmation_timeframe != "1Day":
                raise ValueError("MTF confirmation timeframe must be 1Day.")
        if self.strategy_mode == "COMPOUND_ROTATION_SWING_1W":
            if self.market_data_provider not in {"alpaca", "yahoo"}:
                raise ValueError("Swing Capital Rotation supports Alpaca or Yahoo Finance market data.")
            if self.market_data_provider == "alpaca":
                if self.alpaca_feed not in {"iex", "sip"}:
                    raise ValueError("Alpaca feed must be iex or sip.")
                if self.alpaca_adjustment not in {"raw", "split", "dividend", "all"}:
                    raise ValueError("Alpaca adjustment must be raw, split, dividend or all.")
            if self.timeframe != "1Day":
                raise ValueError(
                    "Compound Capital Rotation Swing uses daily candles."
                )
            allowed_swing_horizons = {5, 10, 20, 40, 60}
            if self.rotation_horizon_days not in allowed_swing_horizons:
                raise ValueError(
                    "Swing Capital Rotation utility horizon must be one of "
                    "5, 10, 20, 40, or 60 trading sessions."
                )
            if not self.rotation_walk_forward_enabled:
                raise ValueError(
                    "V8.1.0 requires expanding walk-forward validation."
                )
            if self.rotation_purge_days < self.rotation_horizon_days:
                raise ValueError(
                    "rotation_purge_days must be >= rotation_horizon_days."
                )
            if len(self.assets) < 2:
                raise ValueError(
                    "Compound Capital Rotation requires at least two assets."
                )
        if self.strategy_mode == "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE":
            if self.market_data_provider not in {"alpaca", "yahoo"}:
                raise ValueError("Day Trade Open→Close supports Alpaca or Yahoo Finance market data.")
            if self.timeframe != "15Min":
                raise ValueError(
                    "Day Trade Open→Close uses 15-minute source bars, aggregated to one decision per session."
                )
            if self.market_data_provider == "yahoo":
                start = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
                yahoo_cutoff = datetime.now(timezone.utc) - timedelta(days=60)
                if start < yahoo_cutoff:
                    raise ValueError(
                        "Yahoo Finance 15-minute history is limited to roughly the last 60 days. "
                        "Choose Alpaca for long Day Trade training windows or move Start date into the recent Yahoo window."
                    )
            if not self.rotation_models:
                raise ValueError("Select XGBoost Utility and/or QR-DQN for Day Trade Open→Close.")
            if self.rotation_horizon_days != 1:
                raise ValueError(
                    "Day Trade Open→Close uses a fixed one-session utility horizon."
                )
            if not self.rotation_walk_forward_enabled:
                raise ValueError(
                    "Day Trade Open→Close requires expanding walk-forward validation."
                )
            if self.rotation_purge_days < 1:
                raise ValueError("Day Trade Open→Close requires at least one purge session.")
            if len(self.assets) < 2:
                raise ValueError(
                    "Compound Capital Rotation requires at least two assets."
                )

        if self.strategy_mode == "BOTTOM_ENTRY_EXIT_RISK_SWING_1D":
            if self.timeframe != "1Day":
                raise ValueError(
                    "Bottom Entry + Exit Risk Swing 1D requires timeframe 1Day."
                )
            if self.mtf_top_confirmation_timeframe != "1Day":
                raise ValueError(
                    "Swing 1D confirmation timeframe must be 1Day."
                )
        if self.strategy_mode == "BOTTOM_ENTRY_MTF_TOP_EXIT":
            if self.mtf_top_signal_timeframe != "1Week":
                raise ValueError("MTF TOP signal timeframe must be 1Week.")
        return self


def validate_json_configuration(
    db: Database,
    changes: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a partial JSON configuration against the effective settings.

    The JSON endpoint accepts only persisted backtest settings. Alpaca credentials
    and other integration secrets intentionally live outside DEFAULT_SETTINGS and
    therefore cannot be imported through this path.
    """
    if not isinstance(changes, dict):
        raise HTTPException(
            status_code=422,
            detail="Configuration JSON must be an object.",
        )
    if not changes:
        raise HTTPException(
            status_code=422,
            detail="Configuration JSON cannot be empty.",
        )

    allowed = set(DEFAULT_SETTINGS)
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported configuration keys: "
                + ", ".join(unknown)
            ),
        )

    current = get_settings(db)
    candidate = {**current, **changes}
    model_fields = set(BacktestRequest.model_fields)
    model_payload = {
        key: value
        for key, value in candidate.items()
        if key in model_fields
    }

    try:
        validated = BacktestRequest.model_validate(model_payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json()),
        ) from exc

    normalized_model = validated.model_dump(mode="python")
    effective = dict(candidate)
    effective.update(normalized_model)

    normalized_changes: dict[str, Any] = {}
    for key in changes:
        if key in normalized_model:
            normalized_changes[key] = normalized_model[key]
        else:
            # Storage-only settings such as mongo_collection are already
            # whitelisted by DEFAULT_SETTINGS. Keep their supplied value.
            normalized_changes[key] = changes[key]

    return normalized_changes, effective
