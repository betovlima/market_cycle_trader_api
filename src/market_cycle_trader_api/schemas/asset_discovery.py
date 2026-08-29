from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AssetDiscoveryStartRequest(BaseModel):
    research_size: int = Field(default=64, ge=1)

class AssetDiscoveryValidateSelectionRequest(BaseModel):
    run_id: str | None = Field(default=None, max_length=160)
    symbols: list[str] = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def normalize_run_id(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(str(value or "").strip().upper() for value in values if str(value or "").strip()))
        if not normalized:
            raise ValueError("Select at least one Asset Discovery symbol.")
        return normalized


class AssetDiscoveryCreateStrategyRequest(BaseModel):
    run_id: str | None = Field(default=None, max_length=160)
    symbols: list[str] = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def normalize_run_id(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(str(value or "").strip().upper() for value in values if str(value or "").strip()))
        if not normalized:
            raise ValueError("Select at least one Asset Discovery symbol.")
        return normalized
