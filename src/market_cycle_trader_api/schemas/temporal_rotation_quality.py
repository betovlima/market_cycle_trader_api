from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_DRAWDOWN_TRIGGERS = [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12]
DEFAULT_ROTATION_SCORE_TOLERANCES = [-0.150, -0.125, -0.100, -0.075, -0.050, -0.025, 0.000]
DEFAULT_CHALLENGER_QUALITY_FLOORS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


class TemporalRotationQualityManualCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drawdown_trigger: float = Field(gt=-0.95, lt=0.0)
    rotation_score_tolerance: float = Field(ge=-1.0, le=1.0)
    challenger_quality_floor: float | None = Field(default=None, ge=0.0, le=1.0)


class TemporalRotationQualityResearchGate(BaseModel):
    """Exploratory selection gate. Values are supplied by the client and persisted with the run."""

    model_config = ConfigDict(extra="forbid")

    minimum_capital_lift: float = Field(default=0.0, ge=-1.0, le=20.0)
    minimum_sharpe_delta: float = Field(default=0.0, ge=-10.0, le=10.0)
    minimum_max_drawdown_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    required_fold_wins: int | None = Field(default=None, ge=0, le=50)


class TemporalRotationQualityCaroConfig(BaseModel):
    """Unified Adaptive CARO search controls for the two-dimensional Rotation Quality surface."""

    model_config = ConfigDict(extra="forbid")

    drawdown_trigger_min: float = Field(default=-0.15, gt=-0.95, lt=0.0)
    drawdown_trigger_max: float = Field(default=-0.01, gt=-0.95, lt=0.0)
    rotation_score_tolerance_min: float = Field(default=-0.20, ge=-1.0, le=1.0)
    rotation_score_tolerance_max: float = Field(default=0.00, ge=-1.0, le=1.0)
    challenger_quality_floor_min: float = Field(default=0.35, ge=0.0, le=1.0)
    challenger_quality_floor_max: float = Field(default=0.85, ge=0.0, le=1.0)
    trials: int = Field(default=100, ge=4, le=2000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    candidate_pool_size: int = Field(default=2048, ge=256, le=16384)
    space_filling_pool_size: int = Field(default=1024, ge=256, le=8192)
    exploration_weight: float = Field(default=0.15, ge=0.0, le=2.0)
    minimum_exploration_trials: int | None = Field(default=None, ge=4, le=24)
    initial_exploration_fraction: float = Field(default=0.45, ge=0.10, le=0.90)
    minimum_exploration_fraction: float = Field(default=0.20, ge=0.05, le=0.90)
    stagnation_recovery_trials: int = Field(default=4, ge=2, le=12)
    minimum_capital_improvement: float = Field(default=0.0, ge=0.0, le=20.0)
    sharpe_tolerance: float = Field(default=0.0, ge=0.0, le=10.0)
    drawdown_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_worst_fold_return: float = Field(default=-1.0, ge=-1.0, le=20.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "TemporalRotationQualityCaroConfig":
        if self.drawdown_trigger_min >= self.drawdown_trigger_max:
            raise ValueError("drawdown_trigger_min must be lower than drawdown_trigger_max.")
        if self.rotation_score_tolerance_min >= self.rotation_score_tolerance_max:
            raise ValueError("rotation_score_tolerance_min must be lower than rotation_score_tolerance_max.")
        if self.challenger_quality_floor_min >= self.challenger_quality_floor_max:
            raise ValueError("challenger_quality_floor_min must be lower than challenger_quality_floor_max.")
        if self.minimum_exploration_fraction > self.initial_exploration_fraction:
            raise ValueError("minimum_exploration_fraction cannot exceed initial_exploration_fraction.")
        return self


class TemporalRotationQualityResearchStartRequest(BaseModel):
    """Simple request used by the backward-compatible /docs research flow."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source_run_id": "20260816T181543-temporal-a5afd924"
            }
        },
    )

    source_run_id: str = Field(min_length=1, max_length=160)
    focus_month: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}$")
    control_tolerance_usd: float = Field(default=1.0, ge=0.0, le=100.0)

    def to_research_request(self) -> "TemporalRotationQualityResearchRequest":
        return TemporalRotationQualityResearchRequest(
            source_run_id=self.source_run_id,
            focus_month=self.focus_month,
            control_tolerance_usd=self.control_tolerance_usd,
        )


class TemporalRotationQualityResearchRequest(BaseModel):
    """Configurable Rotation Quality research request used by the UI and advanced API."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source_run_id": "20260816T181543-temporal-a5afd924",
                "search_method": "grid",
                "focus_month": None,
                "drawdown_triggers": DEFAULT_DRAWDOWN_TRIGGERS,
                "rotation_score_tolerances": DEFAULT_ROTATION_SCORE_TOLERANCES,
                "control_tolerance_usd": 1.0,
                "research_gate": {
                    "minimum_capital_lift": 0.0,
                    "minimum_sharpe_delta": 0.0,
                    "minimum_max_drawdown_delta": 0.0,
                    "required_fold_wins": None,
                },
            }
        },
    )

    source_run_id: str = Field(min_length=1, max_length=160)
    search_method: Literal["grid", "caro", "manual"] = "grid"
    focus_month: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}$")
    strong_challenger_override: bool = False
    baseline_drawdown_trigger: float | None = Field(default=None, gt=-0.95, lt=0.0)
    baseline_rotation_score_tolerance: float | None = Field(default=None, ge=-1.0, le=1.0)
    challenger_quality_floors: list[float] = Field(default_factory=lambda: list(DEFAULT_CHALLENGER_QUALITY_FLOORS), min_length=1, max_length=64)
    drawdown_triggers: list[float] = Field(default_factory=lambda: list(DEFAULT_DRAWDOWN_TRIGGERS), min_length=1, max_length=64)
    rotation_score_tolerances: list[float] = Field(
        default_factory=lambda: list(DEFAULT_ROTATION_SCORE_TOLERANCES),
        min_length=1,
        max_length=64,
    )
    manual_candidates: list[TemporalRotationQualityManualCandidate] = Field(default_factory=list, max_length=2000)
    caro: TemporalRotationQualityCaroConfig = Field(default_factory=TemporalRotationQualityCaroConfig)
    research_gate: TemporalRotationQualityResearchGate = Field(default_factory=TemporalRotationQualityResearchGate)
    control_tolerance_usd: float = Field(default=1.0, ge=0.0, le=100.0)

    @field_validator("drawdown_triggers")
    @classmethod
    def validate_drawdown_triggers(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if any(item >= 0.0 or item < -0.95 for item in normalized):
            raise ValueError("Drawdown triggers must be negative fractions between -0.95 and 0.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Drawdown triggers must not contain duplicates.")
        return normalized

    @field_validator("rotation_score_tolerances")
    @classmethod
    def validate_rotation_tolerances(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if any(item < -1.0 or item > 1.0 for item in normalized):
            raise ValueError("Rotation score tolerances must be between -1 and 1.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Rotation score tolerances must not contain duplicates.")
        return normalized


    @field_validator("challenger_quality_floors")
    @classmethod
    def validate_challenger_quality_floors(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if any(item < 0.0 or item > 1.0 for item in normalized):
            raise ValueError("Challenger quality floors must be between 0 and 1.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Challenger quality floors must not contain duplicates.")
        return normalized

    @model_validator(mode="after")
    def validate_method_specific_configuration(self) -> "TemporalRotationQualityResearchRequest":
        if self.search_method == "manual" and not self.manual_candidates:
            raise ValueError("manual_candidates is required when search_method=manual.")
        if self.strong_challenger_override and self.search_method in {"grid", "caro"}:
            if self.baseline_drawdown_trigger is None or self.baseline_rotation_score_tolerance is None:
                raise ValueError("Strong Challenger Override requires baseline_drawdown_trigger and baseline_rotation_score_tolerance for grid/CARO research.")
        if self.strong_challenger_override and self.search_method == "manual":
            if any(item.challenger_quality_floor is None for item in self.manual_candidates):
                raise ValueError("Strong Challenger Override manual candidates require challenger_quality_floor.")
        if not self.strong_challenger_override and self.search_method == "manual":
            if any(item.challenger_quality_floor is not None for item in self.manual_candidates):
                raise ValueError("challenger_quality_floor requires strong_challenger_override=true.")
        return self


class TemporalRotationQualityValidationRequest(BaseModel):
    """Validate or certify frozen Rotation Quality candidates on a new walk-forward protocol."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "kind": "validation",
                "fold_count": 5,
                "required_fold_wins": 4,
                "candidate_ids": ["RQ-017", "RQ-038", "RQ-053"],
                "minimum_capital_lift": 0.0,
                "minimum_sharpe_delta": 0.0,
                "minimum_max_drawdown_delta": 0.0,
            }
        },
    )

    kind: Literal["validation", "certification"] = "validation"
    fold_count: int = Field(default=5, ge=2, le=20)
    required_fold_wins: int | None = Field(default=None, ge=0, le=20)
    candidate_ids: list[str] = Field(min_length=1, max_length=20)
    minimum_capital_lift: float = Field(default=0.0, ge=-1.0, le=20.0)
    minimum_sharpe_delta: float = Field(default=0.0, ge=-10.0, le=10.0)
    minimum_max_drawdown_delta: float = Field(default=0.0, ge=-1.0, le=1.0)

    @field_validator("candidate_ids")
    @classmethod
    def validate_candidate_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("candidate_ids must contain non-empty candidate identifiers.")
        if any(item == "CONTROL" for item in normalized):
            raise ValueError("CONTROL is always evaluated automatically and must not be listed in candidate_ids.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate_ids must not contain duplicates.")
        return normalized

    @model_validator(mode="after")
    def validate_fold_wins(self) -> "TemporalRotationQualityValidationRequest":
        if self.required_fold_wins is not None and self.required_fold_wins > self.fold_count:
            raise ValueError("required_fold_wins cannot exceed fold_count.")
        return self

    def resolved_required_fold_wins(self) -> int:
        if self.required_fold_wins is not None:
            return int(self.required_fold_wins)
        return max(1, int(self.fold_count) - 1)


class TemporalRotationQualityDiagnosticRequest(BaseModel):
    """Diagnose completed frozen Rotation Quality evidence using only decision-time features."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "candidate_id": "RQ-017",
                "lookback_sessions": 5,
                "feature_names": [
                    "entry_rank_score",
                    "hold_score",
                    "incumbent_persistence_score",
                    "short_profit_consensus",
                    "short_risk_safety",
                    "long_profit_confirmation",
                    "long_trend_support",
                ],
                "minimum_group_samples": 3,
                "outcome_neutral_band": 0.0,
                "top_feature_count": 20,
            }
        },
    )

    candidate_id: str = Field(min_length=1, max_length=80)
    lookback_sessions: int = Field(default=5, ge=1, le=60)
    feature_names: list[str] = Field(min_length=1, max_length=30)
    minimum_group_samples: int = Field(default=3, ge=2, le=100)
    outcome_neutral_band: float = Field(default=0.0, ge=0.0, le=0.20)
    top_feature_count: int = Field(default=20, ge=1, le=100)

    @field_validator("candidate_id")
    @classmethod
    def normalize_candidate_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized or normalized == "CONTROL":
            raise ValueError("candidate_id must identify a non-Control Rotation Quality candidate.")
        return normalized

    @field_validator("feature_names")
    @classmethod
    def normalize_feature_names(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("feature_names must contain non-empty feature identifiers.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("feature_names must not contain duplicates.")
        return normalized
