from __future__ import annotations

from pydantic import BaseModel


class InitializeApplicationRequest(BaseModel):
    confirm_paper: bool
    arm_next_session: bool = True
