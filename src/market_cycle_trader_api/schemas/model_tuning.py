from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChampionProbabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    startup_trials: int = Field(default=8, ge=4, le=24)
    min_capital_improvement: float = Field(default=0.03, ge=0.0, le=1.0)
    sharpe_tolerance: float = Field(default=0.05, ge=0.0, le=2.0)
    drawdown_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)
    min_worst_fold_return: float = Field(default=0.0, ge=-1.0, le=10.0)
    candidate_pool_size: int = Field(default=2048, ge=256, le=16384)
    exploration_weight: float = Field(default=0.15, ge=0.0, le=2.0)


class ModelTuningStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["latin_hypercube", "champion_probability"] = "latin_hypercube"
    candidate_count: int = Field(default=20, ge=4, le=60)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    baseline_job_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_tuning_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    anchor_candidate_id: int | None = Field(default=None, ge=0)
    probability: ChampionProbabilityConfig | None = None

    @model_validator(mode="after")
    def validate_probability_startup(self) -> "ModelTuningStartRequest":
        if self.method != "champion_probability":
            if self.source_tuning_run_id is not None or self.anchor_candidate_id is not None:
                raise ValueError("A prior tuning campaign and anchor candidate are only valid for CARO Probability.")
            return self
        config = self.probability or ChampionProbabilityConfig()
        # When a completed exploration campaign is imported, its observations replace
        # the standalone startup design and candidate_count means new adaptive trials.
        if self.source_tuning_run_id is None and config.startup_trials >= self.candidate_count:
            raise ValueError("Probabilistic startup trials must be smaller than the total candidate count.")
        if self.anchor_candidate_id is not None and self.source_tuning_run_id is None:
            raise ValueError("An anchor candidate requires a source tuning campaign.")
        return self


class ModelTuningAdoptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=3, max_length=500)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A change reason is required.")
        return normalized
