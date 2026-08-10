from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .requests import BacktestRequest, normalize_assets_input


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
    configuration: dict[str, Any]
    assets_input: str | None = Field(default=None, min_length=1, max_length=20_000)
    name: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=500)
    note: str = Field(min_length=3, max_length=500)

    @field_validator("name", "description", "note")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(str(value).split())

    def build_configuration(self) -> BacktestRequest:
        payload = dict(self.configuration)
        if self.assets_input is not None:
            payload["assets"] = normalize_assets_input(self.assets_input)
        return BacktestRequest.model_validate(payload)

    @model_validator(mode="after")
    def validate_strategy_configuration(self) -> "StrategyUpdateRequest":
        # Validate the effective configuration only after the API has constructed
        # the canonical assets list from plain text. Old clients that still send
        # configuration.assets remain compatible while the new UI no longer does.
        self.build_configuration()
        return self


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
    confirm_market_closed: Literal[True]
    confirm_preserve_operational_state: Literal[True]
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
