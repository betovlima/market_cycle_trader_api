from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pymongo.database import Database

from ..core.config import ACTIVE_STRATEGY_MODE, SWING_STRATEGY_MODES
from ..infrastructure.persistence.mongo_repository import DEFAULT_SETTINGS, get_settings


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


class AlpacaCredentialsRequest(BaseModel):
    api_key_id: str = Field(..., min_length=1, max_length=512)
    secret_key: str = Field(..., min_length=1, max_length=1024)


class AlpacaConnectionTestRequest(BaseModel):
    feed: Literal["iex", "sip"] = "iex"


class BacktestRequest(BaseModel):
    assets: list[str] = Field(default_factory=lambda: ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY"])
    strategy_mode: Literal[
        "COMPOUND_ROTATION_SWING_XGBOOST",
        "COMPOUND_ROTATION_SWING_QRDQN",
        "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
    ] = ACTIVE_STRATEGY_MODE
    start_date: str = "2016-01-01"
    end_date: str | None = None
    timeframe: Literal["1Day", "15Min"] = "1Day"
    market_data_provider: Literal["yahoo", "alpaca"] = "alpaca"
    alpaca_feed: Literal["iex", "sip"] = "iex"
    alpaca_adjustment: Literal["raw", "split", "dividend", "all"] = "all"

    rotation_models: list[Literal["xgboost_utility", "qrdqn"]] = Field(default_factory=lambda: ["xgboost_utility"])
    rotation_horizon_days: int = Field(40, ge=1, le=260)
    rotation_minimum_training_rows: int = Field(700, ge=300, le=100_000)
    rotation_walk_forward_enabled: bool = True
    rotation_walk_forward_calibration_days: int = Field(126, ge=40, le=5_000)
    rotation_walk_forward_test_days: int = Field(504, ge=63, le=10_000)
    rotation_walk_forward_min_test_days: int = Field(126, ge=20, le=10_000)
    rotation_purge_days: int = Field(60, ge=1, le=2_000)
    rotation_downside_penalty: float = Field(0.20, ge=0, le=10)
    rotation_drawdown_penalty: float = Field(0.35, ge=0, le=10)
    rotation_min_holding_days: int = Field(2, ge=0, le=260)
    rotation_min_expected_edge: float = Field(0.001, ge=-0.50, le=0.50)
    rotation_cash_threshold: float = Field(0.0, ge=-0.50, le=0.50)
    rotation_switch_margin: float = Field(0.005, ge=0, le=0.50)
    rotation_switch_margin_candidates: list[float] = Field(default_factory=lambda: [0.0, 0.0025, 0.005, 0.01])
    rotation_xgb_n_estimators: int = Field(300, ge=10, le=100_000)
    rotation_xgb_learning_rate: float = Field(0.035, gt=0, le=1)
    rotation_xgb_max_depth: int = Field(3, ge=1, le=20)
    rotation_accelerator: Literal["auto", "cpu", "cuda"] = "auto"
    rotation_allow_cpu_fallback: bool = True
    rotation_parallel_models: bool = True
    rotation_xgb_repetitions: int = Field(1, ge=1, le=100)
    rotation_qrdqn_repetitions: int = Field(1, ge=1, le=100)
    rotation_seed_step: int = Field(1000, ge=1, le=10_000_000)

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
    qrdqn_n_step: int = Field(10, ge=1, le=60)
    qrdqn_n_quantiles: int = Field(25, ge=5, le=200)
    qrdqn_hidden_dim: int = Field(128, ge=16, le=2_048)
    qrdqn_target_update_steps: int = Field(250, ge=10, le=100_000)
    qrdqn_eval_every_steps: int = Field(1000, ge=100, le=100_000)
    qrdqn_epsilon_start: float = Field(1.0, ge=0, le=1)
    qrdqn_epsilon_end: float = Field(0.05, ge=0, le=1)

    initial_capital: float = Field(10_000.0, gt=0)
    whole_shares: bool = False
    slippage_bps: float = Field(0.0, ge=0, le=500)
    commission_rate: float = Field(0.0, ge=0, le=1)
    sec_fee_rate: float = Field(0.0000206, ge=0, le=1)
    taf_fee_per_share: float = Field(0.000195, ge=0)
    taf_fee_cap: float = Field(9.79, ge=0)
    cat_fee_per_share: float = Field(0.000003, ge=0)

    xgb_min_child_weight: float = Field(5.0, ge=0)
    xgb_subsample: float = Field(0.85, gt=0, le=1)
    xgb_colsample_bytree: float = Field(0.85, gt=0, le=1)
    xgb_reg_alpha: float = Field(0.10, ge=0)
    xgb_reg_lambda: float = Field(2.0, ge=0)
    xgb_n_jobs: int = Field(-1, ge=-1)

    yfinance_auto_adjust: bool = True
    yfinance_repair: bool = False
    yfinance_timeout: int = Field(30, ge=1, le=600)
    yfinance_fallback_period: str = "max"
    mongo_cache_enabled: bool = True
    mongo_refresh_overlap_days: int = Field(7, ge=0, le=365)
    mongo_write_batch_size: int = Field(1000, ge=1, le=100_000)
    random_state: int = 42

    @property
    def fractional_shares(self) -> bool:
        return not self.whole_shares

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, value: list[str]) -> list[str]:
        return normalize_assets(value)

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
        if self.qrdqn_min_training_steps > self.qrdqn_training_steps:
            raise ValueError("QR-DQN minimum training steps cannot exceed total training steps.")
        if not self.rotation_walk_forward_enabled:
            raise ValueError("Compound Capital Rotation requires expanding walk-forward validation.")
        if self.rotation_walk_forward_min_test_days > self.rotation_walk_forward_test_days:
            raise ValueError("Minimum test sessions cannot exceed the configured test window.")
        if self.strategy_mode in SWING_STRATEGY_MODES:
            self.timeframe = "1Day"
            self.rotation_horizon_days = 40
            if self.rotation_purge_days < 40:
                raise ValueError("Swing rotation purge must be at least 40 sessions.")
            self.rotation_models = ["qrdqn"] if self.strategy_mode == "COMPOUND_ROTATION_SWING_QRDQN" else ["xgboost_utility"]
        else:
            self.timeframe = "15Min"
            self.rotation_horizon_days = 1
            if self.rotation_purge_days < 1:
                raise ValueError("Day Trade Open→Close requires at least one purge session.")
            if self.market_data_provider == "yahoo":
                start = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
                if start < datetime.now(timezone.utc) - timedelta(days=60):
                    raise ValueError("Yahoo Finance 15-minute history is limited to roughly the last 60 days. Use Alpaca for longer Day Trade windows.")
        return self


def validate_json_configuration(db: Database, changes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(status_code=422, detail="Configuration JSON must be a non-empty object.")
    unknown = sorted(set(changes) - set(DEFAULT_SETTINGS))
    if unknown:
        raise HTTPException(status_code=422, detail="Unsupported configuration keys: " + ", ".join(unknown))
    candidate = {**get_settings(db), **changes}
    try:
        validated = BacktestRequest.model_validate(candidate)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
    effective = validated.model_dump(mode="python")
    normalized_changes = {key: effective[key] for key in changes if key in effective}
    return normalized_changes, effective
