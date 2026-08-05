from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class InvitationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_name: str = Field(min_length=1, max_length=120)
    authorized_email: EmailStr
    role: Literal["viewer", "trader", "admin"] = "viewer"
    duration_seconds: int = Field(ge=300, le=31_536_000)
    max_active_sessions: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def default_session_limit(self):
        if self.max_active_sessions is None:
            self.max_active_sessions = 1 if self.role in {"trader", "admin"} else 2
        return self


class InvitationLinkRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(ge=300, le=31_536_000)


class InvitationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int | None = Field(default=None, ge=300, le=31_536_000)
    expires_at: datetime | None = None
    max_active_sessions: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def require_an_update(self):
        expiration_fields = int(self.duration_seconds is not None) + int(self.expires_at is not None)
        if expiration_fields > 1:
            raise ValueError("Provide only one of duration_seconds or expires_at.")
        if expiration_fields == 0 and self.max_active_sessions is None:
            raise ValueError("Provide an expiration or max_active_sessions update.")
        return self


class InvitationResponse(BaseModel):
    id: str
    guest_name: str
    authorized_email: EmailStr | None = None
    role: str
    status: str
    created_at: datetime
    expires_at: datetime
    last_access_at: datetime | None = None
    claimed_at: datetime | None = None
    claimed_email: EmailStr | None = None
    max_active_sessions: int
    active_sessions: int = 0
    revoked_at: datetime | None = None
    primary_administrator: bool = False


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
    identity_email: EmailStr | None = None
    role: str | None = None
    success: bool
    client_ip: str
    created_at: datetime


class AccessLogListResponse(BaseModel):
    items: list[AccessLogResponse]
