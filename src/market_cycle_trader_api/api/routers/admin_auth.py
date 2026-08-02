from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status


def _configured_admin_token() -> str:
    value = str(os.getenv("PARAMETER_BOOTSTRAP_API_TOKEN") or "").strip()
    if len(value) < 24:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The administrative API token is not configured.",
        )
    return value


def require_admin_token(
    supplied: Annotated[str, Header(alias="X-Parameter-Bootstrap-Token")],
) -> None:
    if not secrets.compare_digest(str(supplied), _configured_admin_token()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid administrative API token.",
        )
