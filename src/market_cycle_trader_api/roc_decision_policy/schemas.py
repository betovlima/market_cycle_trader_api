from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RocDecisionPolicySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_metric: str = Field(min_length=3, max_length=64)
    minimum_calibration_samples: int = Field(ge=20, le=1_000_000)
    minimum_class_samples: int = Field(ge=2, le=500_000)
    max_curve_points: int = Field(ge=21, le=1001)

    @field_validator("selection_metric")
    @classmethod
    def validate_selection_metric(cls, value: str) -> str:
        normalized = value.strip().lower()
        supported = {"youden_j", "balanced_accuracy", "distance_to_top_left"}
        if normalized not in supported:
            raise ValueError(f"selection_metric must be one of {sorted(supported)}.")
        return normalized


class RocDecisionPolicySettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)
    settings: RocDecisionPolicySettings

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A change reason is required.")
        return normalized


class RocDecisionPolicyRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processing_id: str = Field(min_length=1, max_length=256)
    start_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    end_month: str = Field(pattern=r"^\d{4}-\d{2}$")
