from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ResearchModelFamily = Literal["xgboost_utility", "lightgbm_utility", "iqn"]


class ModelResearchJobRequest(BaseModel):
    model_family: ResearchModelFamily


class XGBoostResearchSettings(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(ge=10, le=100_000)
    learning_rate: float = Field(gt=0, le=1)
    max_depth: int = Field(ge=1, le=20)
    min_child_weight: float = Field(ge=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    n_jobs: int = Field(ge=-1, le=256)
    repetitions: int = Field(ge=1, le=100)
    seed_step: int = Field(ge=1, le=10_000_000)
    random_state: int

    @model_validator(mode="after")
    def validate_xgboost_contract(self) -> "XGBoostResearchSettings":
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        return self


class LightGBMResearchSettings(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(ge=10, le=100_000)
    learning_rate: float = Field(gt=0, le=1)
    max_depth: int = Field(ge=-1, le=64)
    num_leaves: int = Field(ge=2, le=4_096)
    min_child_samples: int = Field(ge=1, le=100_000)
    min_child_weight: float = Field(ge=0)
    subsample: float = Field(gt=0, le=1)
    subsample_freq: int = Field(ge=0, le=100)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    max_bin: int = Field(ge=16, le=65_535)
    n_jobs: int = Field(ge=-1, le=256)
    repetitions: int = Field(ge=1, le=100)
    seed_step: int = Field(ge=1, le=10_000_000)
    random_state: int

    @model_validator(mode="after")
    def validate_lightgbm_contract(self) -> "LightGBMResearchSettings":
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if self.max_depth > 0 and self.num_leaves > 2 ** self.max_depth:
            raise ValueError("num_leaves cannot exceed 2 ** max_depth when max_depth is positive.")
        return self


class IQNResearchSettings(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    training_steps: int = Field(ge=500, le=2_000_000)
    episode_days: int = Field(ge=20, le=2_000)
    replay_size: int = Field(ge=1_000, le=2_000_000)
    learning_starts: int = Field(ge=100, le=100_000)
    batch_size: int = Field(ge=16, le=4_096)
    learning_rate: float = Field(gt=0, le=1)
    gamma: float = Field(ge=0, le=1)
    n_step: int = Field(ge=1, le=60)
    quantile_samples: int = Field(ge=4, le=256)
    target_quantile_samples: int = Field(ge=4, le=256)
    action_quantile_samples: int = Field(ge=4, le=256)
    evaluation_quantiles: int = Field(ge=8, le=512)
    hidden_dim: int = Field(ge=16, le=2_048)
    cosine_embedding_dim: int = Field(ge=8, le=512)
    target_update_steps: int = Field(ge=10, le=100_000)
    eval_every_steps: int = Field(ge=100, le=100_000)
    epsilon_start: float = Field(ge=0, le=1)
    epsilon_end: float = Field(ge=0, le=1)
    early_stopping_enabled: bool
    early_stopping_patience: int = Field(ge=1, le=100)
    minimum_training_steps: int = Field(ge=500, le=2_000_000)
    gradient_clip_norm: float = Field(gt=0, le=1_000)
    huber_kappa: float = Field(gt=0, le=100)
    repetitions: int = Field(ge=1, le=100)
    seed_step: int = Field(ge=1, le=10_000_000)
    random_state: int

    @model_validator(mode="after")
    def validate_training_window(self) -> "IQNResearchSettings":
        if self.minimum_training_steps > self.training_steps:
            raise ValueError("minimum_training_steps cannot exceed training_steps.")
        if self.learning_starts >= self.replay_size:
            raise ValueError("learning_starts must be smaller than replay_size.")
        if self.batch_size > self.replay_size:
            raise ValueError("batch_size cannot exceed replay_size.")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start.")
        return self


class ModelResearchSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)
    values: dict[str, Any]

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A change reason is required.")
        return normalized

    @field_validator("values")
    @classmethod
    def require_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("At least one model setting is required.")
        return value
