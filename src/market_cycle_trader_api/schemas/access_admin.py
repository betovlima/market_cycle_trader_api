from __future__ import annotations

from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InvitationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_name: str = Field(min_length=1, max_length=120)
    role: Literal["viewer", "trader"] = "viewer"
    duration_seconds: int = Field(ge=300, le=31_536_000)


class InvitationLinkRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(ge=300, le=31_536_000)


class InvitationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int | None = Field(default=None, ge=300, le=31_536_000)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def exactly_one_expiration(self):
        if (self.duration_seconds is None) == (self.expires_at is None):
            raise ValueError("Provide exactly one of duration_seconds or expires_at.")
        return self


class InvitationResponse(BaseModel):
    id: str
    guest_name: str
    role: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_access_at: datetime | None = None
    revoked_at: datetime | None = None


class InvitationLinkResponse(InvitationResponse):
    access_url: str


class InvitationListResponse(BaseModel):
    items: list[InvitationResponse]


class SessionTerminationResponse(BaseModel):
    terminated_sessions: int


class AccessLogResponse(BaseModel):
    id: str
    event: str
    invitation_id: str | None = None
    guest_name: str | None = None
    role: str | None = None
    success: bool
    client_ip: str
    created_at: datetime


class AccessLogListResponse(BaseModel):
    items: list[AccessLogResponse]
