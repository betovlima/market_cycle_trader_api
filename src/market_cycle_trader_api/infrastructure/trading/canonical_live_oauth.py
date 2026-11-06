"""Explicit, fail-closed OAuth authentication for the canonical Alpaca Live account.

This module is intentionally separate from per-user OAuth trading fan-out.
An opted-in canonical Winner uses ONE authenticated broker client for
positions, account state, validation and order submission. Nothing here
changes the Live operating controls or sends orders.

MCT_CANONICAL_LIVE_AUTH_MODE defaults to api_keys (existing production path).
When set to oauth there is no API-key fallback. A missing, revoked, ambiguous
or mismatched OAuth grant blocks the canonical Live operation.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

_CONNECTIONS = "alpaca_oauth_connections"
_MODE_ENV = "MCT_CANONICAL_LIVE_AUTH_MODE"
_SUBJECT_ENV = "MCT_CANONICAL_LIVE_OAUTH_SUBJECT"


def canonical_live_auth_mode() -> str:
    mode = str(os.getenv(_MODE_ENV, "api_keys") or "api_keys").strip().lower()
    if mode not in {"api_keys", "oauth"}:
        raise RuntimeError("Invalid canonical Live authentication mode.")
    return mode


def canonical_oauth_orders_enabled() -> bool:
    """An additional explicit dev migration gate; off means NO OAuth orders."""
    return canonical_live_auth_mode() == "oauth" and (
        str(os.getenv("MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS") or "").strip() == "1"
    )


def _owner_key(subject: str) -> str:
    return hashlib.sha256(subject.encode("utf-8")).hexdigest() + ":live"


def _read_canonical_grant(db: Any, expected_account: str) -> dict[str, Any]:
    """Use an admin-owned grant for exactly the configured canonical account."""
    filters = {
        "account_id": expected_account,
        "environment": "live",
        "role": "admin",
        "trading_authorized": True,
        "oauth_scope": "trading",
        "read_only": False,
    }
    selected_subject = str(os.getenv(_SUBJECT_ENV) or "").strip()
    if selected_subject:
        # The explicit subject is helpful if an administrator connected through
        # more than one authenticated MCT login. No cross-account fallback.
        grant = db[_CONNECTIONS].find_one({
            **filters, "_id": _owner_key(selected_subject), "subject": selected_subject,
        })
        if not grant:
            raise RuntimeError("Canonical Live OAuth administrator grant is unavailable.")
    else:
        # Avoid choosing an arbitrary administrator token when more than one
        # subject has authorized the same canonical Alpaca account.
        matches = list(db[_CONNECTIONS].find(filters).limit(2))
        if len(matches) != 1:
            raise RuntimeError("Canonical Live OAuth grant is missing or ambiguous.")
        grant = matches[0]
    subject = str(grant.get("subject") or "").strip()
    if not subject or str(grant.get("_id") or "") != _owner_key(subject):
        raise RuntimeError("Canonical Live OAuth owner binding is invalid.")
    return grant


def create_canonical_live_oauth_client(db: Any | None = None) -> Any:
    """Resolve a verified OAuth client, with no API-key or Paper fallback."""
    expected_account = str(os.getenv("ALPACA_LIVE_ACCOUNT_ID") or "").strip()
    if not expected_account:
        raise RuntimeError("Canonical Live OAuth requires ALPACA_LIVE_ACCOUNT_ID.")
    encryption_key = str(os.getenv("ALPACA_OAUTH_ENCRYPTION_KEY") or "").strip()
    if not encryption_key:
        raise RuntimeError("Canonical Live OAuth encryption is not configured.")
    if db is None:
        from ...core.runtime import database
        db = database()
    grant = _read_canonical_grant(db, expected_account)
    try:
        encrypted = str(grant["encrypted_token"]).encode("ascii")
        token = Fernet(encryption_key.encode("ascii")).decrypt(encrypted).decode("utf-8")
    except (KeyError, ValueError, TypeError, UnicodeError, InvalidToken) as exc:
        raise RuntimeError("Canonical Live OAuth token cannot be decrypted.") from exc
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(oauth_token=token, paper=False)
        # An actual broker round trip is required on every client creation.
        account_id = str(getattr(client.get_account(), "id", "") or "").strip()
    except Exception as exc:
        raise RuntimeError("Canonical Live OAuth account verification failed.") from exc
    if account_id != expected_account or account_id != str(grant["account_id"]):
        raise RuntimeError("Canonical Live OAuth broker account mismatch.")
    return client
