from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MilpDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    end_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    processing_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_period(self) -> "MilpDecisionRequest":
        if self.end_month < self.start_month:
            raise ValueError("end_month must be greater than or equal to start_month.")
        return self
