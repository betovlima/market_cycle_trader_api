from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StrategyMode = Literal["COMPOUND_ROTATION_SWING_XGBOOST"]
Timeframe = Literal["1Day"]
MarketDataProvider = Literal["alpaca"]
AlpacaHistoricalFeed = Literal["sip", "iex"]
AlpacaLiveFeed = Literal["iex", "sip"]
AlpacaAdjustment = Literal["raw", "split", "dividend", "all"]
HistoryBackfillProvider = Literal["alpaca"]
RotationModel = Literal["xgboost_utility"]
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


class BacktestRequest(BaseModel):
    """Complete XGBoost-only configuration loaded from MongoDB.

    Operational values intentionally have no Python defaults. Missing or unknown
    MongoDB fields are configuration errors instead of being silently replaced.
    """

    model_config = ConfigDict(extra="forbid")

    assets: list[str]
    strategy_mode: StrategyMode
    start_date: str
    end_date: str | None
    timeframe: Timeframe
    market_data_provider: MarketDataProvider
    alpaca_historical_feed: AlpacaHistoricalFeed
    alpaca_live_feed: AlpacaLiveFeed
    alpaca_adjustment: AlpacaAdjustment
    market_data_history_backfill_enabled: bool
    market_data_history_backfill_provider: HistoryBackfillProvider
    market_data_history_start_tolerance_days: int = Field(ge=0, le=365)
    market_data_require_complete_history: bool

    rotation_models: list[RotationModel]
    rotation_horizon_days: int = Field(ge=1, le=260)
    rotation_target_horizons: list[int]
    rotation_target_horizon_weights: list[float]
    rotation_movement_capture_weight: float = Field(ge=0, le=10)
    rotation_trend_persistence_weight: float = Field(ge=0, le=10)
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
        if cleaned != ["xgboost_utility"]:
            raise ValueError("This version supports only rotation_models=['xgboost_utility'].")
        return cleaned


    @field_validator("rotation_target_horizons")
    @classmethod
    def validate_target_horizons(cls, value: list[int]) -> list[int]:
        cleaned = sorted(dict.fromkeys(int(item) for item in value))
        if not cleaned or any(item < 2 or item > 260 for item in cleaned):
            raise ValueError("Target horizons must contain unique values between 2 and 260 sessions.")
        return cleaned

    @field_validator("rotation_target_horizon_weights")
    @classmethod
    def validate_target_weights(cls, value: list[float]) -> list[float]:
        cleaned = [float(item) for item in value]
        if not cleaned or any(item < 0 for item in cleaned) or sum(cleaned) <= 0:
            raise ValueError("Target-horizon weights must be non-negative and have a positive sum.")
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

        if self.xgb_n_jobs == 0:
            raise ValueError("xgb_n_jobs must be -1 or a positive integer.")
        if self.deterministic_execution:
            if self.xgb_n_jobs != 1:
                raise ValueError(
                    "Deterministic execution requires xgb_n_jobs=1 in MongoDB."
                )
            if self.numeric_thread_limit != 1:
                raise ValueError(
                    "Deterministic execution requires numeric_thread_limit=1 in MongoDB."
                )
        if not self.rotation_walk_forward_enabled:
            raise ValueError("Compound Capital Rotation requires expanding walk-forward validation.")
        if self.rotation_walk_forward_min_test_days > self.rotation_walk_forward_test_days:
            raise ValueError("Minimum test sessions cannot exceed the configured test window.")
        if len(self.rotation_target_horizons) != len(self.rotation_target_horizon_weights):
            raise ValueError("Target horizons and target-horizon weights must have the same length.")
        if self.rotation_horizon_days not in self.rotation_target_horizons:
            raise ValueError("rotation_horizon_days must be included in rotation_target_horizons.")
        if self.rotation_purge_days < max(self.rotation_target_horizons):
            raise ValueError("Swing rotation purge must be at least the maximum target horizon.")
        return self


class BacktestExecutionRequest(BacktestRequest):
    """Immutable execution snapshot derived entirely from the locked MongoDB configuration."""

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
        history_start = date.fromisoformat(self.start_date)
        history_end = date.fromisoformat(self.end_date) if self.end_date else None
        analysis_start = date.fromisoformat(self.analysis_start_date)
        analysis_end = date.fromisoformat(self.analysis_end_date) if self.analysis_end_date else None

        if analysis_start < history_start:
            raise ValueError(
                "Analysis start date cannot be earlier than the locked historical data start "
                f"({self.start_date})."
            )
        if analysis_start > market_today:
            raise ValueError("Analysis start date cannot be later than today.")
        if analysis_end is not None:
            if analysis_end > market_today:
                raise ValueError("Analysis end date cannot be later than today.")
            if analysis_end < analysis_start:
                raise ValueError("Analysis end date cannot be earlier than analysis start date.")
        if history_end is not None:
            if analysis_start > history_end:
                raise ValueError(
                    "Analysis start date cannot be later than the locked historical data end "
                    f"({self.end_date})."
                )
            if analysis_end is not None and analysis_end > history_end:
                raise ValueError(
                    "Analysis end date cannot be later than the locked historical data end "
                    f"({self.end_date})."
                )
        return self

    @property
    def effective_analysis_end_date(self) -> str | None:
        return self.analysis_end_date or self.end_date


LOCKED_CONFIGURATION_FIELDS = frozenset(BacktestRequest.model_fields)
