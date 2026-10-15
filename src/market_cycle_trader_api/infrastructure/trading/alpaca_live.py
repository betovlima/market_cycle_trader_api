"""Live Alpaca read-only integration. Never exposes order submission.

Keep trading credentials, broker account and storage independent of Alpaca Paper.
"""
from __future__ import annotations

import os
from typing import Any


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
