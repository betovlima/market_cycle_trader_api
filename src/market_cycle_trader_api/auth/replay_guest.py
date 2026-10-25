"""Fail-closed HTTP surface for identity-verified annual replay guests.

The frontend hides other tabs, but only the server-side guard protects data.
"""
from __future__ import annotations

import re

_BACKTEST_DETAIL = re.compile(r"^/api/analytics/backtests/[^/]+$")


def replay_guest_request_allowed(method: str, path: str) -> bool:
    operation = str(method).upper()
    if operation == "OPTIONS":
        return True
    if operation == "POST":
        return path == "/api/auth/logout"
    if operation != "GET":
        return False
    return path in {"/api/auth/session", "/api/analytics/backtests"} or bool(
        _BACKTEST_DETAIL.fullmatch(path)
    )


def replay_guest_denied(role: str | None, method: str, path: str) -> bool:
    """Leave all existing roles unchanged and deny guests outside replay."""
    return role == "replay_guest" and not replay_guest_request_allowed(method, path)
