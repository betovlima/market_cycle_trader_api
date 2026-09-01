from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReviewerAccessCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_name: str = Field(min_length=1, max_length=120)
    duration_seconds: int = Field(ge=300, le=31_536_000)
    max_active_sessions: int = Field(default=2, ge=1, le=5)


class ReviewerAccessRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(ge=300, le=31_536_000)


class ReviewerAccessPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_id: str = Field(min_length=8, max_length=128)


class ReviewerAccessLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_id: str = Field(min_length=8, max_length=128)
    code: str = Field(min_length=8, max_length=128)


class ReviewerAccessResponse(BaseModel):
    id: str
    guest_name: str
    role: str = "reviewer"
    status: str
    created_at: datetime
    expires_at: datetime
    last_access_at: datetime | None = None
    max_active_sessions: int
    active_sessions: int = 0
    revoked_at: datetime | None = None


class ReviewerAccessSecretResponse(ReviewerAccessResponse):
    access_url: str
    access_code: str


class ReviewerAccessListResponse(BaseModel):
    items: list[ReviewerAccessResponse]


class ReviewerAccessSessionTerminationResponse(BaseModel):
    terminated_sessions: int


class ReviewerAccessPreviewResponse(BaseModel):
    access_id: str
    guest_name: str
    role: str = "reviewer"
    status: str
    expires_at: datetime
    max_active_sessions: int
