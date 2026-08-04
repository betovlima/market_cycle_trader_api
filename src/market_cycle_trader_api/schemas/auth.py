from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AdminLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=512)


class ViewerAccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=12, max_length=256)


class SessionResponse(BaseModel):
    authenticated: bool
    role: str | None = None
    expires_in_seconds: int
    expires_at: datetime | None = None
    display_name: str | None = None
