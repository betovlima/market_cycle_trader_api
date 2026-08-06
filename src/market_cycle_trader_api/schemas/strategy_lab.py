from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .requests import BacktestRequest


class StrategyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=500)
    clone_from_strategy_id: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("name", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    configuration: BacktestRequest
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=500)
    note: str = Field(min_length=3, max_length=500)

    @field_validator("name", "description", "note")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategySelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_control_revision: int = Field(ge=1)
    note: str = Field(min_length=3, max_length=500)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyCandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_mark_as_candidate: Literal[True]
    expected_strategy_revision: int = Field(ge=1)
    note: str = Field(min_length=3, max_length=500)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyPromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_promote_to_trader: Literal[True]
    expected_control_revision: int = Field(ge=1)
    expected_strategy_revision: int = Field(ge=1)
    note: str = Field(min_length=3, max_length=500)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_delete: Literal[True]
    note: str = Field(min_length=3, max_length=500)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())
