from __future__ import annotations

import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_assets(value: list[str]) -> list[str]:
    cleaned: list[str] = []
    for asset in value:
        symbol = str(asset).strip().upper()
        if symbol and re.fullmatch(r"[A-Z0-9.\-^=]+", symbol):
            cleaned.append(symbol)
    result = list(dict.fromkeys(cleaned))
    if len(result) < 2:
        raise ValueError("At least two valid assets are required.")
    return result


def normalize_iso_date(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


def normalize_required_text(value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("A non-empty value is required.")
    return cleaned


class PublicBacktestRequest(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    assets: list[str]
    strategy_mode: str
    timeframe: str
    alpaca_adjustment: str
    market_data_history_backfill_enabled: bool
    market_data_history_backfill_provider: str
    market_data_history_start_tolerance_days: int = Field(ge=0, le=365)
    market_data_require_complete_history: bool

    rotation_models: list[str]
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
    rotation_accelerator: str
    rotation_allow_cpu_fallback: bool
    rotation_xgb_repetitions: int = Field(ge=1, le=100)
    rotation_seed_step: int = Field(ge=1, le=10_000_000)

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
    deterministic_execution: bool
    numeric_thread_limit: int = Field(ge=1, le=128)

    mongo_cache_enabled: bool
    mongo_refresh_overlap_days: int = Field(ge=0, le=365)
    mongo_write_batch_size: int = Field(ge=1, le=100_000)
    random_state: int

    @property
    def fractional_shares(self) -> bool:
        return not self.whole_shares

    @property
    def ensemble_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(self.random_state) + index * int(self.rotation_seed_step)
            for index in range(int(self.rotation_xgb_repetitions))
        )

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, value: list[str]) -> list[str]:
        return normalize_assets(value)

    @field_validator(
        "strategy_mode",
        "timeframe",
        "alpaca_adjustment",
        "market_data_history_backfill_provider",
        "rotation_accelerator",
    )
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        return normalize_required_text(value)

    @field_validator("rotation_models")
    @classmethod
    def validate_rotation_models(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if not cleaned:
            raise ValueError("At least one model is required.")
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
        if self.xgb_n_jobs == 0:
            raise ValueError("xgb_n_jobs must be -1 or a positive integer.")
        if self.deterministic_execution:
            if self.xgb_n_jobs != 1 or self.numeric_thread_limit != 1:
                raise ValueError("Deterministic execution requires single-threaded settings.")
        if not self.rotation_walk_forward_enabled:
            raise ValueError("Walk-forward validation must be enabled.")
        if self.rotation_walk_forward_min_test_days > self.rotation_walk_forward_test_days:
            raise ValueError("Minimum test sessions cannot exceed the configured test window.")
        if self.rotation_purge_days < self.rotation_horizon_days:
            raise ValueError("The purge window cannot be shorter than the decision horizon.")
        return self


class BacktestExecutionRequest(BacktestRequest):
    start_date: str
    end_date: str | None
    market_data_provider: str
    alpaca_historical_feed: str
    alpaca_live_feed: str
    analysis_start_date: str
    analysis_end_date: str | None

    @field_validator("start_date", "analysis_start_date")
    @classmethod
    def validate_required_dates(cls, value: str, info) -> str:
        return normalize_iso_date(value, field_name=info.field_name)

    @field_validator("end_date", "analysis_end_date")
    @classmethod
    def validate_optional_dates(cls, value: str | None, info) -> str | None:
        if value is None or not str(value).strip():
            return None
        return normalize_iso_date(value, field_name=info.field_name)

    @field_validator("market_data_provider", "alpaca_historical_feed", "alpaca_live_feed")
    @classmethod
    def validate_policy_text(cls, value: str) -> str:
        return normalize_required_text(value)

    @model_validator(mode="after")
    def validate_analysis_window(self) -> "BacktestExecutionRequest":
        market_today = datetime.now(ZoneInfo("America/New_York")).date()
        history_start = date.fromisoformat(self.start_date)
        analysis_start = date.fromisoformat(self.analysis_start_date)
        analysis_end = date.fromisoformat(self.analysis_end_date) if self.analysis_end_date else None
        if analysis_start < history_start:
            raise ValueError("The requested analysis window is outside the available policy range.")
        if analysis_start > market_today:
            raise ValueError("Analysis start date cannot be later than today.")
        if analysis_end is not None:
            if analysis_end > market_today:
                raise ValueError("Analysis end date cannot be later than today.")
            if analysis_end < analysis_start:
                raise ValueError("Analysis end date cannot be earlier than analysis start date.")
        return self

    @property
    def effective_analysis_end_date(self) -> str | None:
        return self.analysis_end_date


LOCKED_CONFIGURATION_FIELDS = frozenset(BacktestRequest.model_fields)
