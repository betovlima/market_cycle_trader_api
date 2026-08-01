from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.system_rules import (
    ALPACA_HISTORICAL_FEED,
    ALPACA_LIVE_FEED,
    MARKET_DATA_PROVIDER,
    TRAINING_HISTORY_END,
    TRAINING_HISTORY_START,
)

StrategyMode = Literal["COMPOUND_ROTATION_SWING_XGBOOST"]
Timeframe = Literal["1Day"]
HistoryBackfillProvider = Literal["alpaca"]
RotationModel = Literal["xgboost_utility"]
RotationAccelerator = Literal["auto", "cpu", "cuda"]
SeedEnsembleMethod = Literal["majority_vote"]


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
    """Public analysis window.

    ``start_date`` and ``end_date`` preserve the existing frontend contract, but
    they select only the simulated account/report interval. They never change
    the fixed model-training history that begins on 2016-01-01.
    """

    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "PublicBacktestRequest":
        market_today = datetime.now(ZoneInfo("America/New_York")).date()
        training_start = date.fromisoformat(TRAINING_HISTORY_START)
        if self.start_date < training_start:
            raise ValueError(
                f"Analysis start date cannot be earlier than {TRAINING_HISTORY_START}."
            )
        if self.start_date > market_today:
            raise ValueError("Start date cannot be later than today.")
        if self.end_date is not None:
            if self.end_date > market_today:
                raise ValueError("End date cannot be later than today.")
            if self.end_date < self.start_date:
                raise ValueError("End date cannot be earlier than start date.")
        return self


class BacktestRequest(BaseModel):
    """Editable XGBoost strategy parameters loaded from MongoDB.

    Training start, provider and feeds are system rules and intentionally do not
    belong to this model. This prevents the browser and administration API from
    changing the production research protocol.
    """

    model_config = ConfigDict(extra="forbid")

    assets: list[str]
    strategy_mode: StrategyMode
    timeframe: Timeframe
    alpaca_adjustment: Literal["raw", "split", "dividend", "all"]
    market_data_history_backfill_enabled: bool
    market_data_history_backfill_provider: HistoryBackfillProvider
    market_data_history_start_tolerance_days: int = Field(ge=0, le=365)
    market_data_require_complete_history: bool

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
    rotation_xgb_repetitions: int = Field(ge=1, le=100)
    rotation_seed_step: int = Field(ge=1, le=10_000_000)
    rotation_seed_ensemble_enabled: bool
    rotation_seed_ensemble_method: SeedEnsembleMethod
    rotation_seed_ensemble_min_agreement: float = Field(gt=0, le=1)

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
    def start_date(self) -> str:
        return TRAINING_HISTORY_START

    @property
    def end_date(self) -> str | None:
        return TRAINING_HISTORY_END

    @property
    def market_data_provider(self) -> str:
        return MARKET_DATA_PROVIDER

    @property
    def alpaca_historical_feed(self) -> str:
        return ALPACA_HISTORICAL_FEED

    @property
    def alpaca_live_feed(self) -> str:
        return ALPACA_LIVE_FEED

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

    @field_validator("rotation_models")
    @classmethod
    def validate_rotation_models(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip()))
        if cleaned != ["xgboost_utility"]:
            raise ValueError("This version supports only rotation_models=['xgboost_utility'].")
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
            if self.xgb_n_jobs != 1:
                raise ValueError("Deterministic execution requires xgb_n_jobs=1 in MongoDB.")
            if self.numeric_thread_limit != 1:
                raise ValueError("Deterministic execution requires numeric_thread_limit=1 in MongoDB.")
        if not self.rotation_walk_forward_enabled:
            raise ValueError("Compound Capital Rotation requires expanding walk-forward validation.")
        if self.rotation_walk_forward_min_test_days > self.rotation_walk_forward_test_days:
            raise ValueError("Minimum test sessions cannot exceed the configured test window.")
        if self.rotation_purge_days < self.rotation_horizon_days:
            raise ValueError("Swing rotation purge must be at least the decision horizon.")
        if self.rotation_seed_ensemble_enabled and self.rotation_xgb_repetitions < 3:
            raise ValueError("The production seed ensemble requires at least three repetitions.")
        return self


class BacktestExecutionRequest(BacktestRequest):
    """Editable strategy settings plus a public out-of-sample report window."""

    analysis_start_date: str
    analysis_end_date: str | None

    @field_validator("analysis_start_date")
    @classmethod
    def validate_analysis_start_date(cls, value: str) -> str:
        return normalize_iso_date(value, field_name="analysis_start_date")

    @field_validator("analysis_end_date")
    @classmethod
    def validate_analysis_end_date(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return normalize_iso_date(value, field_name="analysis_end_date")

    @model_validator(mode="after")
    def validate_analysis_window(self) -> "BacktestExecutionRequest":
        market_today = datetime.now(ZoneInfo("America/New_York")).date()
        history_start = date.fromisoformat(TRAINING_HISTORY_START)
        analysis_start = date.fromisoformat(self.analysis_start_date)
        analysis_end = date.fromisoformat(self.analysis_end_date) if self.analysis_end_date else None

        if analysis_start < history_start:
            raise ValueError(
                "Analysis start date cannot be earlier than the fixed training-history start "
                f"({TRAINING_HISTORY_START})."
            )
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
