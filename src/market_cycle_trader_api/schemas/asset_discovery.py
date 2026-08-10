from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AssetDiscoverySettingsPatch(BaseModel):
    automatic_enabled: bool | None = None
    batch_size: int | None = Field(default=None, ge=1, le=50)
    schedule_hours_et: list[int] | None = None
    recheck_days: int | None = Field(default=None, ge=1, le=365)
    min_price: float | None = Field(default=None, ge=0.5, le=10_000)
    min_median_dollar_volume: float | None = Field(default=None, ge=0, le=10_000_000_000)
    min_nonzero_volume_ratio: float | None = Field(default=None, ge=0, le=1)
    behavior_lookback_days: int | None = Field(default=None, ge=365, le=3650)
    behavior_lookback_sessions: int | None = Field(default=None, ge=63, le=2520)
    behavior_min_sessions: int | None = Field(default=None, ge=20, le=252)
    behavior_max_downside_tail_1pct: float | None = Field(default=None, ge=0.01, le=1)
    behavior_max_gap_downside_tail_1pct: float | None = Field(default=None, ge=0.01, le=1)
    behavior_max_annualized_volatility: float | None = Field(default=None, ge=0.10, le=5)
    behavior_max_drawdown: float | None = Field(default=None, ge=0.10, le=0.99)
    behavior_max_single_day_loss: float | None = Field(default=None, ge=0.05, le=0.99)
    behavior_max_single_gap_loss: float | None = Field(default=None, ge=0.05, le=0.99)
    behavior_max_10_session_loss: float | None = Field(default=None, ge=0.10, le=0.99)

    @field_validator("schedule_hours_et")
    @classmethod
    def normalize_schedule_hours(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        normalized = sorted(set(int(hour) for hour in value))
        if not normalized:
            raise ValueError("At least one schedule hour is required.")
        if any(hour < 0 or hour > 23 for hour in normalized):
            raise ValueError("Schedule hours must be between 0 and 23 Eastern Time.")
        return normalized


class AssetDiscoverySettingsUpdateRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)
    settings: AssetDiscoverySettingsPatch
