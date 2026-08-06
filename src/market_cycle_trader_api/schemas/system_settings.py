from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrainingSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    automatic_training_enabled: bool | None = None
    model_threads: int | None = Field(default=None, ge=1, le=64)
    numeric_threads: int | None = Field(default=None, ge=1, le=64)
    max_concurrent_jobs: int | None = Field(default=None, ge=1, le=1)
    timeout_seconds: int | None = Field(default=None, ge=300, le=86_400)

    @model_validator(mode="after")
    def require_change(self) -> "TrainingSettingsPatch":
        if not self.model_dump(exclude_none=True):
            raise ValueError("At least one training setting is required.")
        return self


class SystemSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)
    training: TrainingSettingsPatch

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A change reason is required.")
        return normalized


class TrainingSettingsResponse(BaseModel):
    enabled: bool
    automatic_training_enabled: bool
    model_threads: int
    numeric_threads: int
    max_concurrent_jobs: int
    timeout_seconds: int


class RuntimeCapacityResponse(BaseModel):
    detected_cpu_count: int
    configured_model_threads: int
    configured_numeric_threads: int
    winner_compute_locked: bool = True


class SystemSettingsResponse(BaseModel):
    revision: int
    training: TrainingSettingsResponse
    runtime: RuntimeCapacityResponse
    updated_at: datetime | None = None
    updated_by: str | None = None


class SystemSettingsHistoryItem(BaseModel):
    revision: int
    previous_revision: int
    reason: str
    updated_at: datetime
    updated_by: str | None = None
    training: TrainingSettingsResponse
