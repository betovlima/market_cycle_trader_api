from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WinnerTransitionRiskSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severe_threshold: float = Field(gt=-1.0, lt=0.0)
    risk_quantiles: list[float] = Field(min_length=1, max_length=20)
    default_risk_quantile: float = Field(gt=0.0, lt=1.0)
    min_outer_train_rows: int = Field(ge=2, le=100_000)
    min_inner_train_rows: int = Field(ge=2, le=100_000)
    min_train_severe: int = Field(ge=1, le=10_000)

    @field_validator("risk_quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[float]) -> list[float]:
        normalized = sorted({float(item) for item in value})
        if any(item <= 0.0 or item >= 1.0 for item in normalized):
            raise ValueError("risk_quantiles must contain values strictly between 0 and 1.")
        return normalized

    @model_validator(mode="after")
    def validate_training_contract(self) -> "WinnerTransitionRiskSettings":
        if self.min_inner_train_rows > self.min_outer_train_rows:
            raise ValueError("min_inner_train_rows cannot exceed min_outer_train_rows.")
        if self.min_train_severe * 2 > self.min_inner_train_rows:
            raise ValueError("min_inner_train_rows must support both severe and non-severe minimum counts.")
        if self.default_risk_quantile not in self.risk_quantiles:
            raise ValueError("default_risk_quantile must be included in risk_quantiles.")
        return self


class WinnerTransitionConfidenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    margin_quantiles: list[float] = Field(min_length=1, max_length=20)
    min_alerts: int = Field(ge=1, le=100_000)

    @field_validator("margin_quantiles")
    @classmethod
    def validate_margin_quantiles(cls, value: list[float]) -> list[float]:
        normalized = sorted({float(item) for item in value})
        if any(item < 0.0 or item > 1.0 for item in normalized):
            raise ValueError("margin_quantiles must contain values between 0 and 1.")
        return normalized


class TemporalWinnerTransitionResearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: WinnerTransitionRiskSettings
    confidence: WinnerTransitionConfidenceSettings


class TemporalResearchSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: WinnerTransitionRiskSettings | None = None
    confidence: WinnerTransitionConfidenceSettings | None = None

    @model_validator(mode="after")
    def require_change(self) -> "TemporalResearchSettingsPatch":
        if self.risk is None and self.confidence is None:
            raise ValueError("At least one temporal research settings group is required.")
        return self


class TemporalResearchSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)
    settings: TemporalResearchSettingsPatch

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A change reason is required.")
        return normalized
