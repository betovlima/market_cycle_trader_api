from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PaperTradingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    paper_account_id: str | None = None
    client_order_id_prefix: str = Field(min_length=3, max_length=32)
    market_open_delay_seconds: int = Field(ge=0, le=1_800)
    market_execution_window_seconds: int = Field(ge=60, le=7_200)
    order_fill_timeout_seconds: int = Field(ge=10, le=900)
    order_poll_interval_seconds: float = Field(ge=0.5, le=30)
    cash_reserve_dollars: float = Field(ge=0, le=1_000)
    automatic_continuation_enabled: bool
    scheduler_poll_seconds: float = Field(ge=1, le=300)
    preparation_retry_seconds: float = Field(ge=10, le=3_600)

    @field_validator("client_order_id_prefix")
    @classmethod
    def validate_client_order_id_prefix(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,31}", normalized):
            raise ValueError(
                "client_order_id_prefix must use 3-32 lowercase letters, numbers, '_' or '-'."
            )
        return normalized

    @model_validator(mode="after")
    def validate_execution_window(self) -> "PaperTradingSettings":
        if self.market_execution_window_seconds <= self.market_open_delay_seconds:
            raise ValueError(
                "market_execution_window_seconds must be greater than market_open_delay_seconds."
            )
        return self


class PaperTradingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_capital: float = Field(gt=0)
    strategy_cash: float = Field(ge=0)
    managed_symbol: str | None
    managed_quantity: float = Field(ge=0)
    average_entry_price: float | None = Field(default=None, gt=0)
    holding_sessions: int = Field(ge=0)
    realized_pnl: float
    last_decision_date: str | None
    last_execution_session: str | None

    @model_validator(mode="after")
    def validate_position(self) -> "PaperTradingState":
        if self.managed_symbol is None:
            if self.managed_quantity != 0:
                raise ValueError("managed_quantity must be zero when managed_symbol is empty.")
            if self.average_entry_price is not None:
                raise ValueError("average_entry_price must be empty when no position is managed.")
            if self.holding_sessions != 0:
                raise ValueError("holding_sessions must be zero when no position is managed.")
        else:
            symbol = str(self.managed_symbol).strip().upper()
            if not re.fullmatch(r"[A-Z0-9.\-]+", symbol):
                raise ValueError("managed_symbol is invalid.")
            self.managed_symbol = symbol
            if self.managed_quantity <= 0:
                raise ValueError("managed_quantity must be positive for an open position.")
            if self.average_entry_price is None:
                raise ValueError("average_entry_price is required for an open position.")
        return self


class PaperTradePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    status: Literal["prepared", "executing", "executed", "failed", "cancelled"]
    decision_date: str
    expected_market_open: str
    execution_session: str
    current_asset: str
    target_asset: str
    action: Literal["hold", "buy", "sell_to_cash", "rotate", "stay_in_cash"]
    strategy_configuration_sha256: str
    state_snapshot: dict
    created_at: str
