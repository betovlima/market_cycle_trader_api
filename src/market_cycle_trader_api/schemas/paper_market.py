from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class StartNextSessionRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_paper: bool

    @field_validator("confirm_paper")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm_paper must be true to arm Alpaca paper execution.")
        return value


class CancelPaperMarketRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_cancel: bool

    @field_validator("confirm_cancel")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm_cancel must be true.")
        return value


class StopPaperRobotRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_stop: bool
    cancel_pending_run: bool = True

    @field_validator("confirm_stop")
    @classmethod
    def require_stop_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm_stop must be true.")
        return value
