from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AdminLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=512)


class AccessPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation_id: str = Field(min_length=8, max_length=128)
    token: str | None = Field(default=None, min_length=12, max_length=256)


class GoogleAccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential: str = Field(min_length=100, max_length=16_384)
    invitation_id: str | None = Field(default=None, min_length=8, max_length=128)
    token: str | None = Field(default=None, min_length=12, max_length=256)


class AccessPreviewResponse(BaseModel):
    invitation_id: str
    guest_name: str
    role: str
    masked_email: str
    status: str
    expires_at: datetime
    requires_token: bool


class SessionResponse(BaseModel):
    authenticated: bool
    role: str | None = None
    expires_in_seconds: int
    expires_at: datetime | None = None
    idle_expires_at: datetime | None = None
    display_name: str | None = None
    email: str | None = None
    capabilities: dict[str, bool] = Field(default_factory=dict)
