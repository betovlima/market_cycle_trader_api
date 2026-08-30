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


class StatisticalMlControlSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_sessions: int = Field(ge=20, le=1000)
    minimum_history_sessions: int = Field(ge=10, le=500)
    horizons_sessions: list[int] = Field(min_length=1, max_length=10)
    horizon_weights: dict[str, float]
    downside_penalty: float = Field(ge=0.0, le=10.0)
    minimum_cash_edge: float = Field(ge=0.0, le=1.0)
    minimum_rotation_edge: float = Field(ge=0.0, le=1.0)
    extreme_tail_percentile: float = Field(gt=0.5, lt=1.0)
    extreme_robust_z: float = Field(gt=0.0, le=20.0)
    opportunity_conflict_min_percentile: float = Field(ge=0.0, le=1.0)
    risk_conflict_max_safety: float = Field(ge=0.0, le=1.0)
    min_train_rows: int = Field(ge=30, le=1_000_000)
    inner_validation_share: float = Field(gt=0.0, lt=0.5)
    probability_threshold_candidates: list[float] = Field(min_length=1, max_length=30)
    default_probability_threshold: float = Field(gt=0.0, lt=1.0)
    default_rotation_probability_threshold: float = Field(gt=0.0, lt=1.0)
    action_probability_margin: float = Field(ge=0.0, le=1.0)
    regime_context_enabled: bool
    regime_window_sessions: int = Field(ge=5, le=252)
    regime_min_clusters: int = Field(ge=2, le=10)
    regime_max_clusters: int = Field(ge=2, le=10)
    regime_min_train_rows: int = Field(ge=30, le=100_000)
    regime_distance_temperature: float = Field(gt=0.0, le=100.0)
    regime_trajectory_enabled: bool
    regime_trajectory_window_sessions: int = Field(ge=3, le=60)
    regime_trajectory_warning_quantile: float = Field(gt=0.5, lt=1.0)
    regime_trajectory_target_horizon_sessions: int = Field(ge=1, le=20)
    regime_trajectory_severe_loss_threshold: float = Field(gt=-1.0, lt=0.0)
    n_estimators: int = Field(ge=50, le=5000)
    max_depth: int = Field(ge=2, le=30)
    min_samples_leaf: int = Field(ge=1, le=10_000)
    random_state: int = Field(ge=0, le=2_147_483_647)
    min_interventions: int = Field(ge=1, le=100_000)
    min_capital_lift: float = Field(ge=0.0, le=10.0)
    min_mean_open_auc: float = Field(ge=0.5, le=1.0)
    min_positive_oos_years: int = Field(ge=1, le=100)
    max_drawdown_degradation: float = Field(ge=0.0, le=1.0)
    max_worst_month_degradation: float = Field(ge=0.0, le=1.0)

    @field_validator("horizons_sessions")
    @classmethod
    def validate_horizons(cls, value: list[int]) -> list[int]:
        normalized = sorted({int(item) for item in value})
        if any(item < 1 or item > 60 for item in normalized):
            raise ValueError("horizons_sessions must contain values between 1 and 60.")
        return normalized

    @field_validator("probability_threshold_candidates")
    @classmethod
    def validate_probability_thresholds(cls, value: list[float]) -> list[float]:
        normalized = sorted({float(item) for item in value})
        if any(item <= 0.0 or item >= 1.0 for item in normalized):
            raise ValueError("probability_threshold_candidates must contain values strictly between 0 and 1.")
        return normalized

    @model_validator(mode="after")
    def validate_control_contract(self) -> "StatisticalMlControlSettings":
        if self.minimum_history_sessions > self.lookback_sessions:
            raise ValueError("minimum_history_sessions cannot exceed lookback_sessions.")
        if self.default_probability_threshold not in self.probability_threshold_candidates:
            raise ValueError("default_probability_threshold must be included in probability_threshold_candidates.")
        if self.default_rotation_probability_threshold not in self.probability_threshold_candidates:
            raise ValueError("default_rotation_probability_threshold must be included in probability_threshold_candidates.")
        if self.regime_min_clusters > self.regime_max_clusters:
            raise ValueError("regime_min_clusters cannot exceed regime_max_clusters.")
        expected = {str(item) for item in self.horizons_sessions}
        supplied = {str(key) for key in self.horizon_weights}
        if expected != supplied:
            raise ValueError("horizon_weights must define exactly one weight for each horizon.")
        if any(float(value) < 0.0 for value in self.horizon_weights.values()) or sum(float(value) for value in self.horizon_weights.values()) <= 0.0:
            raise ValueError("horizon_weights must be non-negative and sum to a positive value.")
        return self


class TemporalWinnerTransitionResearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: WinnerTransitionRiskSettings
    confidence: WinnerTransitionConfidenceSettings
    statistical_ml_control: StatisticalMlControlSettings


class TemporalResearchSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: WinnerTransitionRiskSettings | None = None
    confidence: WinnerTransitionConfidenceSettings | None = None
    statistical_ml_control: StatisticalMlControlSettings | None = None

    @model_validator(mode="after")
    def require_change(self) -> "TemporalResearchSettingsPatch":
        if self.risk is None and self.confidence is None and self.statistical_ml_control is None:
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
