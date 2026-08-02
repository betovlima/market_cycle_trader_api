from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paper_trading import PaperTradingSettings
from .requests import BacktestRequest


class ParameterBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_apply: Literal[True]
    strategy_configuration: BacktestRequest | None = None
    paper_trading_configuration: PaperTradingSettings | None = None
    replace_existing: bool = False
    note: str = Field(min_length=3, max_length=500)

    @model_validator(mode="after")
    def require_configuration(self) -> "ParameterBootstrapRequest":
        if (
            self.strategy_configuration is None
            and self.paper_trading_configuration is None
        ):
            raise ValueError("Supply at least one configuration document.")
        self.note = " ".join(self.note.split())
        return self
