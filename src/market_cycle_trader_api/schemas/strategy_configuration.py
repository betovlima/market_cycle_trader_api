from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .requests import BacktestRequest


class StrategyConfigurationPatchRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_update: Literal[True]
    changes: dict[str, Any]
    note: str = Field(min_length=3, max_length=500)
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("changes")
    @classmethod
    def validate_changes(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("At least one strategy parameter must be supplied.")
        allowed = set(BacktestRequest.model_fields)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "Unknown or non-operational strategy fields: " + ", ".join(unknown)
            )
        return value

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyConfigurationReplaceRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_replace: Literal[True]
    configuration: BacktestRequest
    note: str = Field(min_length=3, max_length=500)
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyWinnerInstallRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_delete_existing_strategy_data: Literal[True]
    confirm_install_winner_v1_13_2: Literal[True]
    note: str = Field(
        default=(
            "Delete old strategy configuration data and install winner-v1.13.2.json."
        ),
        min_length=3,
        max_length=500,
    )

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyConfigurationResetRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_reset: Literal[True]
    note: str = Field(
        default="Restore the bundled canonical strategy configuration.",
        min_length=3,
        max_length=500,
    )
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())


class StrategyConfigurationRestoreRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_restore: Literal[True]
    note: str = Field(min_length=3, max_length=500)
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return " ".join(str(value).split())
