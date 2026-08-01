from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrategyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_start_date: date
    training_end_date: date | None
    market_data_provider: str
    historical_feed: str
    live_feed: str

    @field_validator("market_data_provider", "historical_feed", "live_feed")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = str(value).strip().lower()
        if not cleaned:
            raise ValueError("A non-empty value is required.")
        return cleaned

    @model_validator(mode="after")
    def validate_range(self) -> "StrategyPolicy":
        if self.training_end_date is not None and self.training_end_date < self.training_start_date:
            raise ValueError("The training end date cannot be earlier than the training start date.")
        return self
