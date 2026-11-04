"""HTTP visibility boundary for invited Traders: Live Alpaca only.

The browser navigation is not an access control. Direct API requests by
Trader sessions must not expose research, backtests, administration, the
paper portfolio, other users' balances or administrative Winner controls.
The OAuth broker, Live scheduler and admin API-key operations are unchanged.
"""
from __future__ import annotations


_ALLOWED_GET = frozenset({
    "/api/auth/session",
    "/api/alpaca-oauth/status",
    "/api/alpaca-oauth/verify",
    "/api/alpaca-oauth/portfolio",
    "/api/alpaca-oauth/winner/readiness",
    "/api/alpaca-oauth/callback",
})
_ALLOWED_POST = frozenset({
    "/api/auth/logout",
    "/api/alpaca-oauth/start",
    "/api/alpaca-oauth/start-trading",
    "/api/alpaca-oauth/disconnect",
})


def trader_live_request_allowed(method: str, path: str) -> bool:
    operation = str(method).upper()
    if operation == "OPTIONS":
        return True
    if operation == "GET":
        return path in _ALLOWED_GET
    if operation == "POST":
        return path in _ALLOWED_POST
    return False


def trader_live_denied(role: str | None, method: str, path: str) -> bool:
    """Enforce a Live-only API surface for Trader, without affecting admin."""
    return role == "trader" and not trader_live_request_allowed(method, path)
