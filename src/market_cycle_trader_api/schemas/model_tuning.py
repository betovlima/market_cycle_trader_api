from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChampionProbabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Kept for backward compatibility with v3.10 clients. Unified CARO treats this
    # as a minimum exploration floor instead of a fixed warm-up boundary.
    startup_trials: int | None = Field(default=None, ge=4, le=24)
    minimum_exploration_trials: int | None = Field(default=None, ge=4, le=24)
    min_capital_improvement: float = Field(default=0.03, ge=0.0, le=1.0)
    sharpe_tolerance: float = Field(default=0.05, ge=0.0, le=2.0)
    drawdown_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)
    min_worst_fold_return: float = Field(default=0.0, ge=-1.0, le=10.0)
    candidate_pool_size: int = Field(default=2048, ge=256, le=16384)
    exploration_weight: float = Field(default=0.15, ge=0.0, le=2.0)
    initial_exploration_fraction: float = Field(default=0.45, ge=0.10, le=0.90)
    minimum_exploration_fraction: float = Field(default=0.20, ge=0.05, le=0.60)
    stagnation_recovery_trials: int = Field(default=4, ge=2, le=12)
    adaptive_stopping_enabled: bool = True
    no_improvement_trial_limit: int = Field(default=100, ge=10, le=5000)
    minimum_meaningful_improvement: float = Field(default=0.0025, ge=0.0, le=0.25)

    @model_validator(mode="after")
    def validate_exploration_policy(self) -> "ChampionProbabilityConfig":
        if self.minimum_exploration_fraction > self.initial_exploration_fraction:
            raise ValueError("Minimum exploration fraction cannot exceed the initial exploration fraction.")
        return self


class ModelTuningFoldProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_folds: int = Field(default=3, ge=2)
    validation_folds: int = Field(default=5, ge=2)
    certification_folds: int = Field(default=7, ge=2)

    @model_validator(mode="after")
    def validate_stage_order(self) -> "ModelTuningFoldProtocol":
        if self.validation_folds < self.research_folds:
            raise ValueError("Validation folds must be greater than or equal to research folds.")
        if self.certification_folds < self.validation_folds:
            raise ValueError("Certification folds must be greater than or equal to validation folds.")
        return self


class ModelTuningStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # latin_hypercube_then_caro remains accepted only for backward compatibility
    # with v3.10 clients/history. New UI campaigns use Unified CARO.
    method: Literal["latin_hypercube", "champion_probability", "latin_hypercube_then_caro"] = "champion_probability"
    candidate_count: int = Field(default=20, ge=4, le=2000)
    caro_candidate_count: int | None = Field(default=None, ge=1, le=2000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    baseline_job_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_tuning_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    anchor_candidate_id: int | None = Field(default=None, ge=0)
    tuning_target: Literal["temporal_model", "temporal_policy"] | None = None
    probability: ChampionProbabilityConfig | None = None
    fold_protocol: ModelTuningFoldProtocol | None = None
    explicit_start_confirmation: bool = False

    @model_validator(mode="after")
    def validate_probability_startup(self) -> "ModelTuningStartRequest":
        if self.method == "latin_hypercube":
            if self.source_tuning_run_id is not None or self.anchor_candidate_id is not None:
                raise ValueError("A prior tuning campaign and anchor candidate are only valid for Adaptive CARO.")
            if self.caro_candidate_count is not None:
                raise ValueError("CARO candidate count is only valid for the legacy Latin Hypercube → CARO pipeline.")
            return self
        if self.method == "latin_hypercube_then_caro":
            if self.source_tuning_run_id is not None or self.anchor_candidate_id is not None:
                raise ValueError("The legacy Latin Hypercube → CARO pipeline always starts from the active Candidate baseline.")
            return self
        if self.caro_candidate_count is not None:
            raise ValueError("CARO candidate count is only valid for the legacy Latin Hypercube → CARO pipeline.")
        if self.anchor_candidate_id is not None and self.source_tuning_run_id is None:
            raise ValueError("An anchor candidate requires a source tuning campaign.")
        return self


class ModelTuningAdoptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=500)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
