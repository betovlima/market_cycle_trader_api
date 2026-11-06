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


def canonical_live_account_id() -> str:
    """Resolve the same explicit canonical ID used by the shared Winner signal.

    Dev setups with only ALPACA_OAUTH_EXPECTED_ACCOUNT_ID remain supported.
    Two inconsistent bindings are never silently reconciled.
    """
    primary = str(os.getenv("ALPACA_LIVE_ACCOUNT_ID") or "").strip()
    oauth_expected = str(os.getenv("ALPACA_OAUTH_EXPECTED_ACCOUNT_ID") or "").strip()
    if primary and oauth_expected and primary != oauth_expected:
        raise RuntimeError("Canonical Live OAuth account configuration conflicts.")
    expected = primary or oauth_expected
    if not expected:
        raise RuntimeError("Canonical Live OAuth account ID is not configured.")
    return expected


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
        if not matches:
            raise RuntimeError("Canonical Live OAuth administrator grant is missing.")
        if len(matches) > 1:
            raise RuntimeError("Canonical Live OAuth administrator grants are ambiguous.")
        grant = matches[0]
    subject = str(grant.get("subject") or "").strip()
    if not subject or str(grant.get("_id") or "") != _owner_key(subject):
        raise RuntimeError("Canonical Live OAuth owner binding is invalid.")
    return grant


def create_canonical_live_oauth_client(db: Any | None = None) -> Any:
    """Resolve a verified OAuth client, with no API-key or Paper fallback."""
    expected_account = canonical_live_account_id()
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



def _current_admin_grant_failure(db: Any, identity: Any) -> str | None:
    """Compare only the calling administrator's OAuth grant with the canonical criteria.

    Never expose account IDs, owner subjects, tokens or grant contents.
    The check is diagnostic only: it cannot change how an executor finds grants.
    """
    if identity is None or str(getattr(identity, "role", "")).lower() != "admin":
        return None
    subject = str(getattr(identity, "subject", "") or "").strip()
    if not subject:
        return None
    selected = str(os.getenv(_SUBJECT_ENV) or "").strip()
    if selected and selected != subject:
        return "OAUTH_ADMIN_SUBJECT_SELECTION_MISMATCH"
    saved = db[_CONNECTIONS].find_one(
        {"_id": _owner_key(subject)},
        {"subject": 1, "account_id": 1, "environment": 1, "role": 1,
         "trading_authorized": 1, "oauth_scope": 1, "read_only": 1},
    )
    if not isinstance(saved, dict):
        return "OAUTH_CURRENT_ADMIN_GRANT_MISSING"
    if saved.get("subject") != subject:
        return "OAUTH_OWNER_BINDING_INVALID"
    if saved.get("environment") != "live":
        return "OAUTH_CURRENT_ADMIN_ENVIRONMENT_MISMATCH"
    if str(saved.get("account_id") or "").strip() != canonical_live_account_id():
        return "OAUTH_CURRENT_ADMIN_ACCOUNT_MISMATCH"
    if saved.get("role") != "admin":
        return "OAUTH_CURRENT_ADMIN_SAVED_ROLE_MISMATCH"
    if (saved.get("trading_authorized") is not True
        or saved.get("oauth_scope") != "trading"
        or saved.get("read_only") is not False):
        return "OAUTH_CURRENT_ADMIN_TRADING_GRANT_MISSING"
    return "OAUTH_GRANT_SELECTION_UNAVAILABLE"


def canonical_live_oauth_diagnostic(db: Any, identity: Any = None) -> dict[str, Any]:
    """Admin-only, on-demand read-only diagnosis. No provider errors or tokens.

    The ordinary Live portfolio route deliberately keeps its generic 503.
    This endpoint returns only predefined, non-sensitive failure codes.
    """
    mode = canonical_live_auth_mode()
    result: dict[str, Any] = {
        "authentication": mode,
        "orders_enabled": canonical_oauth_orders_enabled(),
        "connected_to_canonical_account": False,
        "status": "not_applicable" if mode != "oauth" else "blocked",
        "stage": "configuration" if mode == "oauth" else None,
        "failure_code": None,
    }
    if mode != "oauth":
        return result
    failures = {
        "Canonical Live OAuth account ID is not configured.": "CANONICAL_ACCOUNT_ID_MISSING",
        "Canonical Live OAuth account configuration conflicts.": "CANONICAL_ACCOUNT_ID_CONFLICT",
        "Canonical Live OAuth encryption is not configured.": "OAUTH_ENCRYPTION_KEY_MISSING",
        "Canonical Live OAuth administrator grant is unavailable.": "OAUTH_ADMIN_GRANT_UNAVAILABLE",
        "Canonical Live OAuth administrator grant is missing.": "OAUTH_ADMIN_GRANT_MISSING",
        "Canonical Live OAuth administrator grants are ambiguous.": "OAUTH_ADMIN_GRANT_AMBIGUOUS",
        "Canonical Live OAuth owner binding is invalid.": "OAUTH_OWNER_BINDING_INVALID",
        "Canonical Live OAuth token cannot be decrypted.": "OAUTH_TOKEN_UNREADABLE",
        "Canonical Live OAuth account verification failed.": "OAUTH_BROKER_VERIFICATION_FAILED",
        "Canonical Live OAuth broker account mismatch.": "OAUTH_ACCOUNT_MISMATCH",
    }
    try:
        broker = create_canonical_live_oauth_client(db)
    except RuntimeError as exc:
        code = failures.get(str(exc), "OAUTH_CANONICAL_VERIFICATION_FAILED")
        if code in {"OAUTH_ADMIN_GRANT_MISSING", "OAUTH_ADMIN_GRANT_UNAVAILABLE"}:
            try:
                code = _current_admin_grant_failure(db, identity) or code
            except Exception:
                pass  # Do not turn diagnostics into another source of private errors.
        result["failure_code"] = code
        return result
    except Exception:
        result["failure_code"] = "OAUTH_CANONICAL_VERIFICATION_FAILED"
        return result
    result["connected_to_canonical_account"] = True
    # Verify separately the broker reads used by the full Live portfolio.
    # Do not reveal response bodies or exception messages.
    checks = (
        ("positions", lambda: broker.get_all_positions()),
        ("orders", lambda: _read_live_orders(broker)),
        ("clock", lambda: _read_live_clock(broker)),
    )
    for name, check in checks:
        result["stage"] = name
        try:
            check()
        except Exception:
            result["failure_code"] = "OAUTH_BROKER_" + name.upper() + "_UNAVAILABLE"
            return result
    result["status"] = "ready"
    result["stage"] = "broker_reads"
    return result


def _read_live_orders(broker: Any) -> None:
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100))


def _read_live_clock(broker: Any) -> None:
    clock = broker.get_clock()
    if getattr(clock, "timestamp", None) is None or clock.timestamp.tzinfo is None:
        raise RuntimeError("Broker clock timestamp is unavailable.")
