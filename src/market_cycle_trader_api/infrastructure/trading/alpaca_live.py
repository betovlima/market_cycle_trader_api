"""Dedicated Alpaca Live connection for observations and Winner execution.

Keep trading credentials, broker account and storage independent of Alpaca Paper.
"""
from __future__ import annotations

import os
from typing import Any

from .canonical_live_oauth import (
    canonical_live_auth_mode, canonical_oauth_orders_enabled,
    create_canonical_live_oauth_client,
)


def live_credentials() -> tuple[str, str]:
    key = str(os.getenv("ALPACA_LIVE_API_KEY_ID") or "").strip()
    secret = str(os.getenv("ALPACA_LIVE_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise RuntimeError(
            "Alpaca Live credentials are not configured. Set "
            "ALPACA_LIVE_API_KEY_ID and ALPACA_LIVE_SECRET_KEY on the API service."
        )
    paper_key = str(os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID") or "").strip()
    paper_secret = str(
        os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or ""
    ).strip()
    if (paper_key and key == paper_key) or (paper_secret and secret == paper_secret):
        raise RuntimeError("Live credentials must be different from Alpaca Paper credentials.")
    return key, secret


def create_live_readonly_client() -> Any:
    """Single canonical broker transport for both reads and execution.

    OAuth mode *never* falls back to API keys, even if valid keys exist.
    """
    if canonical_live_auth_mode() == "oauth":
        return create_canonical_live_oauth_client()
    from alpaca.trading.client import TradingClient

    key, secret = live_credentials()
    client = TradingClient(key, secret, paper=False)
    # Prevent accidentally querying a different live account after a key rotation.
    expected = str(os.getenv("ALPACA_LIVE_ACCOUNT_ID") or "").strip()
    if expected:
        actual = str(getattr(client.get_account(), "id", "") or "").strip()
        if actual != expected:
            raise RuntimeError("Live account identity does not match ALPACA_LIVE_ACCOUNT_ID.")
    return client


def create_live_execution_client() -> Any:
    """Fail-closed Live writer; the dev OAuth migration is read-only by default."""
    if canonical_live_auth_mode() == "oauth" and not canonical_oauth_orders_enabled():
        raise RuntimeError(
            "Canonical OAuth Live order submission is disabled in this environment."
        )
    return create_live_readonly_client()
