from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class StartNextSessionRequest(BaseModel):
    """Explicit confirmation required before arming paper-market automation."""

    model_config = ConfigDict(extra="forbid")

    confirm_paper: bool

    @field_validator("confirm_paper")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm_paper must be true to arm Alpaca paper execution.")
        return value


class CancelPaperMarketRequest(BaseModel):
    """Explicit confirmation required before cancelling an armed run."""

    model_config = ConfigDict(extra="forbid")

    confirm_cancel: bool

    @field_validator("confirm_cancel")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm_cancel must be true.")
        return value
