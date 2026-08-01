from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.config import SWING_STRATEGY_MODES

StrategyMode = Literal[
    "COMPOUND_ROTATION_SWING_XGBOOST",
    "COMPOUND_ROTATION_SWING_QRDQN",
    "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
]
Timeframe = Literal["1Day", "15Min"]
MarketDataProvider = Literal["yahoo", "alpaca"]
AlpacaFeed = Literal["iex", "sip"]
AlpacaAdjustment = Literal["raw", "split", "dividend", "all"]
RotationModel = Literal["xgboost_utility", "qrdqn"]
RotationAccelerator = Literal["auto", "cpu", "cuda"]


def normalize_assets(value: list[str]) -> list[str]:
    cleaned: list[str] = []
    for asset in value:
        symbol = str(asset).strip().upper()
        if symbol and re.fullmatch(r"[A-Z0-9.\-^=]+", symbol):
            cleaned.append(symbol)
    result = list(dict.fromkeys(cleaned))
    if len(result) < 2:
        raise ValueError("Compound Capital Rotation requires at least two valid assets.")
    return result


def normalize_iso_date(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


class PublicBacktestRequest(BaseModel):
    """Only the historical date range may be supplied by the public client."""

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "PublicBacktestRequest":
        market_today = datetime.now(ZoneInfo("America/New_York")).date()
        if self.start_date > market_today:
            raise ValueError("Start date cannot be later than today.")
        if self.end_date is not None:
            if self.end_date > market_today:
                raise ValueError("End date cannot be later than today.")
            if self.end_date < self.start_date:
                raise ValueError("End date cannot be earlier than start date.")
        return self


class BacktestRequest(BaseModel):
    """Complete locked execution configuration loaded from MongoDB.

    Operational values intentionally have no Python defaults. A missing MongoDB
    field is therefore a configuration error instead of being silently replaced
    by a value embedded in the application binary.
    """

    model_config = ConfigDict(extra="forbid")

    assets: list[str]
    strategy_mode: StrategyMode
    start_date: str
    end_date: str | None
    timeframe: Timeframe
    market_data_provider: MarketDataProvider
    alpaca_feed: AlpacaFeed
    alpaca_adjustment: AlpacaAdjustment

    rotation_models: list[RotationModel]
    rotation_horizon_days: int = Field(ge=1, le=260)
    rotation_minimum_training_rows: int = Field(ge=300, le=100_000)
    rotation_walk_forward_enabled: bool
    rotation_walk_forward_calibration_days: int = Field(ge=40, le=5_000)
    rotation_walk_forward_test_days: int = Field(ge=20, le=10_000)
    rotation_walk_forward_min_test_days: int = Field(ge=20, le=10_000)
    rotation_purge_days: int = Field(ge=1, le=2_000)
    rotation_downside_penalty: float = Field(ge=0, le=10)
    rotation_drawdown_penalty: float = Field(ge=0, le=10)
    rotation_min_holding_days: int = Field(ge=0, le=260)
    rotation_min_expected_edge: float = Field(ge=-0.50, le=0.50)
    rotation_cash_threshold: float = Field(ge=-0.50, le=0.50)
    rotation_switch_margin: float = Field(ge=0, le=0.50)
    rotation_switch_margin_candidates: list[float]
    rotation_xgb_n_estimators: int = Field(ge=10, le=100_000)
    rotation_xgb_learning_rate: float = Field(gt=0, le=1)
    rotation_xgb_max_depth: int = Field(ge=1, le=20)
    rotation_accelerator: RotationAccelerator
    rotation_allow_cpu_fallback: bool
    rotation_parallel_models: bool
    rotation_xgb_repetitions: int = Field(ge=1, le=100)
    rotation_qrdqn_repetitions: int = Field(ge=1, le=100)
    rotation_seed_step: int = Field(ge=1, le=10_000_000)

    qrdqn_training_steps: int = Field(ge=500, le=2_000_000)
    qrdqn_parallel_folds: int = Field(ge=1, le=32)
    qrdqn_parallel_repetitions: int = Field(ge=1, le=16)
    qrdqn_torch_num_threads: int = Field(ge=0, le=64)
    qrdqn_early_stopping_enabled: bool
    qrdqn_early_stopping_patience: int = Field(ge=1, le=100)
    qrdqn_min_training_steps: int = Field(ge=500, le=2_000_000)
    qrdqn_episode_days: int = Field(ge=20, le=2_000)
    qrdqn_replay_size: int = Field(ge=1_000, le=2_000_000)
    qrdqn_learning_starts: int = Field(ge=100, le=100_000)
    qrdqn_batch_size: int = Field(ge=16, le=4_096)
    qrdqn_learning_rate: float = Field(gt=0, le=1)
    qrdqn_gamma: float = Field(ge=0, le=1)
    qrdqn_n_step: int = Field(ge=1, le=60)
    qrdqn_n_quantiles: int = Field(ge=5, le=200)
    qrdqn_hidden_dim: int = Field(ge=16, le=2_048)
    qrdqn_target_update_steps: int = Field(ge=10, le=100_000)
    qrdqn_eval_every_steps: int = Field(ge=100, le=100_000)
    qrdqn_epsilon_start: float = Field(ge=0, le=1)
    qrdqn_epsilon_end: float = Field(ge=0, le=1)

    initial_capital: float = Field(gt=0)
    whole_shares: bool
    slippage_bps: float = Field(ge=0, le=500)
    commission_rate: float = Field(ge=0, le=1)
    sec_fee_rate: float = Field(ge=0, le=1)
    taf_fee_per_share: float = Field(ge=0)
    taf_fee_cap: float = Field(ge=0)
    cat_fee_per_share: float = Field(ge=0)

    xgb_min_child_weight: float = Field(ge=0)
    xgb_subsample: float = Field(gt=0, le=1)
    xgb_colsample_bytree: float = Field(gt=0, le=1)
    xgb_reg_alpha: float = Field(ge=0)
    xgb_reg_lambda: float = Field(ge=0)
    xgb_n_jobs: int = Field(ge=-1)

    yfinance_auto_adjust: bool
    yfinance_repair: bool
    yfinance_timeout: int = Field(ge=1, le=600)
    yfinance_fallback_period: str = Field(min_length=1, max_length=50)
    mongo_cache_enabled: bool
    mongo_refresh_overlap_days: int = Field(ge=0, le=365)
    mongo_write_batch_size: int = Field(ge=1, le=100_000)
    random_state: int

    @property
    def fractional_shares(self) -> bool:
        return not self.whole_shares

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, value: list[str]) -> list[str]:
        return normalize_assets(value)

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, value: str) -> str:
        return normalize_iso_date(value, field_name="start_date")

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return normalize_iso_date(value, field_name="end_date")

    @field_validator("rotation_models")
    @classmethod
    def validate_rotation_models(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip()))
        invalid = sorted(set(cleaned) - {"xgboost_utility", "qrdqn"})
        if invalid:
            raise ValueError(f"Unsupported rotation models: {invalid}")
        if not cleaned:
            raise ValueError("Select at least one capital-rotation model.")
        return cleaned

    @field_validator("rotation_switch_margin_candidates")
    @classmethod
    def validate_switch_margins(cls, value: list[float]) -> list[float]:
        cleaned = list(dict.fromkeys(float(item) for item in value))
        if not cleaned or any(item < 0 or item > 0.50 for item in cleaned):
            raise ValueError("Switch-margin candidates must contain values between 0 and 0.50.")
        return cleaned

    @model_validator(mode="after")
    def validate_relationships(self) -> "BacktestRequest":
        market_today = datetime.now(ZoneInfo("America/New_York")).date()
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date) if self.end_date else None
        if start > market_today:
            raise ValueError("Start date cannot be later than today.")
        if end is not None:
            if end > market_today:
                raise ValueError("End date cannot be later than today.")
            if end < start:
                raise ValueError("End date cannot be earlier than start date.")

        if self.qrdqn_min_training_steps > self.qrdqn_training_steps:
            raise ValueError("QR-DQN minimum training steps cannot exceed total training steps.")
        if self.qrdqn_learning_starts >= self.qrdqn_replay_size:
            raise ValueError("QR-DQN learning starts must be smaller than replay size.")
        if self.qrdqn_batch_size > self.qrdqn_replay_size:
            raise ValueError("QR-DQN batch size cannot exceed replay size.")
        if self.qrdqn_epsilon_end > self.qrdqn_epsilon_start:
            raise ValueError("QR-DQN epsilon end cannot exceed epsilon start.")
        if self.xgb_n_jobs == 0:
            raise ValueError("xgb_n_jobs must be -1 or a positive integer.")

        if not self.rotation_walk_forward_enabled:
            raise ValueError("Compound Capital Rotation requires expanding walk-forward validation.")
        if self.rotation_walk_forward_min_test_days > self.rotation_walk_forward_test_days:
            raise ValueError("Minimum test sessions cannot exceed the configured test window.")

        if self.strategy_mode in SWING_STRATEGY_MODES:
            if self.timeframe != "1Day":
                raise ValueError("Swing rotation strategies require timeframe=1Day.")
            if self.rotation_purge_days < self.rotation_horizon_days:
                raise ValueError("Swing rotation purge must be at least the decision horizon.")
            expected_models = (
                ["qrdqn"]
                if self.strategy_mode == "COMPOUND_ROTATION_SWING_QRDQN"
                else ["xgboost_utility"]
            )
            if self.rotation_models != expected_models:
                raise ValueError(
                    f"{self.strategy_mode} requires rotation_models={expected_models}."
                )
        else:
            if self.timeframe != "15Min":
                raise ValueError("Day Trade Open→Close requires timeframe=15Min.")
            if self.rotation_horizon_days != 1:
                raise ValueError("Day Trade Open→Close requires a one-session decision horizon.")
            if self.market_data_provider == "yahoo":
                start_timestamp = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
                if start_timestamp < datetime.now(timezone.utc) - timedelta(days=60):
                    raise ValueError(
                        "Yahoo Finance 15-minute history is limited to roughly the last 60 days. "
                        "Use Alpaca for longer Day Trade windows."
                    )
        return self


LOCKED_CONFIGURATION_FIELDS = frozenset(BacktestRequest.model_fields)
