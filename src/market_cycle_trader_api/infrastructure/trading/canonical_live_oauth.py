"""Explicit, fail-closed OAuth authentication for the canonical Alpaca Live account.

This module is intentionally separate from per-user OAuth trading fan-out.
An opted-in canonical Winner uses ONE authenticated broker client for
positions, account state, validation and order submission. Nothing here
changes the Live operating controls or sends orders.

Operational authentication and real-order authorization are stored in the
versioned Live control document. No authentication-mode or trading-toggle
environment variables are used. OAuth always fails closed without API-key fallback.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

_CONNECTIONS = "alpaca_oauth_connections"
_LIVE_CONTROL = "live_trader_control"


def _stored_configuration(db: Any | None = None) -> dict[str, Any]:
    """Operational authentication is a versioned MongoDB setting, not an env flag."""
    if db is None:
        from ...core.runtime import database
        db = database()
    doc = db[_LIVE_CONTROL].find_one({"_id": "default"})
    return doc if isinstance(doc, dict) else {}


def canonical_live_auth_mode(db: Any | None = None) -> str:
    mode = str(_stored_configuration(db).get("broker_auth_mode") or "api_keys")
    if mode not in {"api_keys", "oauth"}:
        raise RuntimeError("Invalid stored canonical Live authentication mode.")
    return mode


def canonical_oauth_orders_enabled(db: Any | None = None) -> bool:
    state = _stored_configuration(db)
    return bool(
        state.get("broker_auth_mode") == "oauth"
        and state.get("oauth_orders_allowed") is True
    )


def canonical_live_account_id(db: Any | None = None) -> str:
    """Use the account explicitly verified and bound by an administrator.

    The legacy API-key account ID remains relevant only in legacy mode.
    No client-supplied account ID is ever allowed for order execution.
    """
    if db is None:
        from ...core.runtime import database
        db = database()
    state = _stored_configuration(db)
    if state.get("broker_auth_mode") == "oauth":
        account = str(state.get("oauth_account_id") or "").strip()
        if not account:
            raise RuntimeError("Canonical Live OAuth account is not linked.")
        return account
    # This fallback is for the unmodified legacy API-key mode only.
    expected = str(os.getenv("ALPACA_LIVE_ACCOUNT_ID") or
                   os.getenv("ALPACA_OAUTH_EXPECTED_ACCOUNT_ID") or "").strip()
    if not expected:
        raise RuntimeError("Legacy Live account identity is not configured.")
    return expected



def _owner_key(subject: str) -> str:
    return hashlib.sha256(subject.encode("utf-8")).hexdigest() + ":live"


def _read_canonical_grant(db: Any, expected_account: str) -> dict[str, Any]:
    """Resolve the administrator's explicitly bound OAuth authorization."""
    state = _stored_configuration(db)
    subject = str(state.get("oauth_subject") or "").strip()
    if not subject:
        raise RuntimeError("Canonical Live OAuth administrator is not linked.")
    grant = db[_CONNECTIONS].find_one({
        "_id": _owner_key(subject),
        "subject": subject,
        "account_id": expected_account,
        "environment": "live",
        "role": "admin",
        "trading_authorized": True,
        "oauth_scope": "trading",
        "read_only": False,
    })
    if not isinstance(grant, dict):
        raise RuntimeError("Canonical Live OAuth administrator grant is unavailable.")
    if grant.get("_id") != _owner_key(subject) or grant.get("subject") != subject:
        raise RuntimeError("Canonical Live OAuth owner binding is invalid.")
    return grant



def create_canonical_live_oauth_client(db: Any | None = None) -> Any:
    """Resolve a verified OAuth client, with no API-key or Paper fallback."""
    expected_account = canonical_live_account_id(db)
    encryption_key = str(os.getenv("ALPACA_OAUTH_ENCRYPTION_KEY") or "").strip()
    if not encryption_key:
        raise RuntimeError("Canonical Live OAuth encryption is not configured.")
    if db is None:
        from ...core.runtime import database
        db = database()
    if canonical_live_auth_mode(db) != "oauth":
        raise RuntimeError("Canonical Live is not configured for OAuth.")
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
    """Return safe diagnostics scoped to the authenticated administrator."""
    if identity is None or not bool(getattr(identity, "is_admin", False)):
        # Some OAuth identities use the literal 'admin' role instead of a property.
        if str(getattr(identity, "role", "")).lower() != "admin":
            return None
    subject = str(getattr(identity, "subject", "") or "").strip()
    if not subject:
        return None
    state = _stored_configuration(db)
    selected = str(state.get("oauth_subject") or "").strip()
    if not selected:
        return "OAUTH_CANONICAL_OWNER_NOT_SELECTED"
    if selected != subject:
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
    if str(saved.get("account_id") or "").strip() != str(state.get("oauth_account_id") or ""):
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
    mode = canonical_live_auth_mode(db)
    result: dict[str, Any] = {
        "authentication": mode,
        "orders_enabled": canonical_oauth_orders_enabled(db),
        "connected_to_canonical_account": False,
        "status": "not_applicable" if mode != "oauth" else "blocked",
        "stage": "configuration" if mode == "oauth" else None,
        "failure_code": None,
    }
    if mode != "oauth":
        return result
    failures = {
        "Canonical Live OAuth account is not linked.": "CANONICAL_ACCOUNT_ID_MISSING",
        "Canonical Live OAuth administrator is not linked.": "OAUTH_CANONICAL_OWNER_NOT_SELECTED",
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
        if code in {"OAUTH_ADMIN_GRANT_MISSING", "OAUTH_ADMIN_GRANT_UNAVAILABLE",
                    "OAUTH_CANONICAL_OWNER_NOT_SELECTED"}:
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
