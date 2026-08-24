from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StrategyTransferExportRequest(BaseModel):
    strategy_id: str | None = Field(default=None, min_length=1, max_length=200)
    strategy_sequence: int | None = Field(default=None, ge=1)
    include_market_snapshot: bool = True


class StrategyTransferImportRequest(BaseModel):
    confirm: Literal["IMPORT"]
