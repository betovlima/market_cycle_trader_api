"""Alpaca Connect: administrator-bound Live OAuth and optional trading consent.

Trading grants can be requested separately, but this module never submits
broker orders or enables the Winner executor or background scheduler.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlencode, urlsplit

import requests
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database

router = APIRouter(prefix="/api/alpaca-oauth", tags=["alpaca-oauth-readonly"])
PENDING = "alpaca_oauth_pending"
CONNECTIONS = "alpaca_oauth_connections"
AUTHORIZE = "https://app.alpaca.markets/oauth/authorize"
TOKEN = "https://api.alpaca.markets/oauth/token"
ACCOUNT = "https://api.alpaca.markets/v2/account"


def _config():
    c = {k: os.getenv(env, "").strip() for k, env in {
        "client": "ALPACA_OAUTH_CLIENT_ID",
        "secret": "ALPACA_OAUTH_CLIENT_SECRET",
        "redirect": "ALPACA_OAUTH_REDIRECT_URI",
        "frontend": "ALPACA_OAUTH_FRONTEND_URL",
        "key": "ALPACA_OAUTH_ENCRYPTION_KEY",
    }.items()}
    c["expected"] = (os.getenv("ALPACA_OAUTH_EXPECTED_ACCOUNT_ID")
                     or os.getenv("ALPACA_LIVE_ACCOUNT_ID") or "").strip()
    if not all(c.values()):
        raise RuntimeError("Alpaca OAuth configuration is missing.")
    for value in (c["redirect"], c["frontend"]):
        p = urlsplit(value)
        if (not p.netloc or p.query or p.fragment or p.username or p.password
            or not (p.scheme == "https" or
                    (p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1")))):
            raise RuntimeError("Invalid Alpaca OAuth redirect/frontend URL.")
    try:
        Fernet(c["key"].encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Invalid Alpaca OAuth encryption key.") from exc
    return c


def _key(subject: str) -> str:
    return hashlib.sha256(subject.encode("utf-8")).hexdigest() + ":live"


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _back(frontend, result):
    response = RedirectResponse(frontend + "?" + urlencode({"alpaca_oauth": result}),
                                status_code=303)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _same_origin(request, frontend):
    parts = urlsplit(frontend)
    if request.headers.get("origin") != f"{parts.scheme}://{parts.netloc}":
        raise HTTPException(403, "Invalid request origin.")


def _read_account(token):
    try:
        r = requests.get(ACCOUNT, headers={"Authorization": f"Bearer {token}"},
                         timeout=15, allow_redirects=False)
        if r.status_code != 200:
            raise RuntimeError("Alpaca rejected the account query.")
        account = r.json()
        if not isinstance(account, dict) or not account.get("id"):
            raise RuntimeError("Alpaca account identity is missing.")
        return account
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("Alpaca account lookup failed.") from exc


def _trading_consent_enabled() -> bool:
    """Default-off gate for requesting the privileged OAuth trading scope."""
    return os.getenv("ALPACA_OAUTH_TRADING_CONSENT_ENABLED", "").strip() == "1"


def _status(db, identity):
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)},
                                      {"encrypted_token": 0})
    authorized = bool(saved and saved.get("trading_authorized") is True)
    return {"configured": True, "connected": bool(saved),
            "environment": "live", "read_only": not authorized,
            "trading_authorized": authorized,
            "trading_consent_available": _trading_consent_enabled(),
            "trading_enabled": False,  # A grant never arms the Winner executor.
            "account_id_suffix": str(saved.get("account_id") or "")[-6:] if saved else None,
            "account_status": saved.get("account_status") if saved else None}


@router.get("/status")
def connection_status(identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    try:
        _config()
    except RuntimeError:
        return {"configured": False, "connected": False, "environment": "live",
                "read_only": True, "trading_authorized": False,
                "trading_consent_available": False, "trading_enabled": False}
    return _status(database(), identity)


@router.post("/start")
def start(request: Request,
          identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    _same_origin(request, c["frontend"])
    state = secrets.token_urlsafe(40)
    now = datetime.now(timezone.utc)
    db = database()
    db[PENDING].create_index("expires", expireAfterSeconds=0)
    db[PENDING].insert_one({"_id": _digest(state), "subject": identity.subject,
                            "requested_scope": "read",
                            "expires": now + timedelta(minutes=10)})
    # Alpaca defaults to read-only when the optional scope is not requested.
    url = AUTHORIZE + "?" + urlencode({
        "response_type": "code", "client_id": c["client"],
        "redirect_uri": c["redirect"], "state": state, "env": "live",
    })
    return {"authorization_url": url}


@router.post("/start-trading")
def start_trading(request: Request,
                  identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    """Explicit fresh user consent, separate from the read-only OAuth flow."""
    if not _trading_consent_enabled():
        raise HTTPException(409, "Alpaca trading consent is not enabled.")
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    _same_origin(request, c["frontend"])
    db = database()
    linked = db[CONNECTIONS].find_one({"_id": _key(identity.subject)})
    if (not isinstance(linked, dict)
        or linked.get("subject") != identity.subject
        or linked.get("environment") != "live"
        or linked.get("account_id") != c["expected"]):
        raise HTTPException(409, "Connect and verify the existing Live account first.")
    state = secrets.token_urlsafe(40)
    db[PENDING].create_index("expires", expireAfterSeconds=0)
    db[PENDING].insert_one({
        "_id": _digest(state), "subject": identity.subject,
        "requested_scope": "trading",
        # A disconnected/replaced grant must invalidate an old browser flow.
        "linked_token_hash": _digest(str(linked.get("encrypted_token") or "")),
        "expires": datetime.now(timezone.utc) + timedelta(minutes=10),
    })
    url = AUTHORIZE + "?" + urlencode({
        "response_type": "code", "client_id": c["client"],
        "redirect_uri": c["redirect"], "state": state,
        "scope": "trading", "env": "live",
    })
    return {"authorization_url": url, "trading_enabled": False}


@router.get("/callback")
def callback(code: str | None = Query(default=None, max_length=2048),
             state: str | None = Query(default=None, max_length=256),
             error: str | None = Query(default=None, max_length=256)):
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not state:
        return _back(c["frontend"], "invalid_state")
    db = database()
    saved = db[PENDING].find_one_and_delete({"_id": _digest(state)})
    # Alpaca returns via a top-level cross-site redirect, where the MCT
    # session cookie may be absent. The one-use state identifies the
    # administrator who initiated /start (which requires an admin session).
    if (not saved or not isinstance(saved.get("subject"), str)
        or not saved["subject"] or not isinstance(saved.get("expires"), datetime)):
        return _back(c["frontend"], "invalid_state")
    subject = saved["subject"]
    requested_scope = saved.get("requested_scope", "read")
    if requested_scope not in {"read", "trading"}:
        return _back(c["frontend"], "invalid_state")
    if requested_scope == "trading":
        if not _trading_consent_enabled():
            return _back(c["frontend"], "trading_unavailable")
        current = db[CONNECTIONS].find_one({"_id": _key(subject)})
        if (not isinstance(current, dict) or current.get("subject") != subject
            or current.get("environment") != "live"
            or current.get("account_id") != c["expected"]
            or not isinstance(saved.get("linked_token_hash"), str)
            or _digest(str(current.get("encrypted_token") or "")) != saved["linked_token_hash"]):
            return _back(c["frontend"], "invalid_state")
    expires = saved["expires"]
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= datetime.now(timezone.utc):
        return _back(c["frontend"], "invalid_state")
    if error or not code:
        return _back(c["frontend"], "denied")
    try:
        r = requests.post(TOKEN, data={
            "grant_type": "authorization_code", "code": code,
            "client_id": c["client"], "client_secret": c["secret"],
            "redirect_uri": c["redirect"],
        }, timeout=15, allow_redirects=False)
        if r.status_code != 200:
            return _back(c["frontend"], "token_failed")
        grant = r.json()
        token = str(grant.get("access_token") or "")
        scope = set(str(grant.get("scope") or "").split())
        if not token or str(grant.get("token_type") or "").lower() != "bearer":
            return _back(c["frontend"], "permission_failed")
        if "account:write" in scope:
            return _back(c["frontend"], "permission_failed")
        if requested_scope == "trading":
            if "trading" not in scope:
                return _back(c["frontend"], "trading_permission_failed")
        elif "trading" in scope:
            # Read-only authorizations must never silently escalate.
            return _back(c["frontend"], "permission_failed")
        account = _read_account(token)
    except (requests.RequestException, RuntimeError, ValueError, AttributeError, TypeError):
        return _back(c["frontend"], "verification_failed")
    if str(account["id"]) != c["expected"]:
        return _back(c["frontend"], "account_mismatch")
    trade_authorized = requested_scope == "trading"
    encrypted = Fernet(c["key"].encode("ascii")).encrypt(token.encode("utf-8")).decode("ascii")
    update_query = {"_id": _key(subject)}
    if trade_authorized:
        # Compare-and-swap: do not restore a grant that was disconnected or
        # replaced between the initial check and token verification.
        update_query.update({
            "subject": subject, "encrypted_token": current["encrypted_token"],
        })
    updated = db[CONNECTIONS].update_one(update_query, {
        "$set": {"subject": subject, "environment": "live",
                 "account_id": str(account["id"]),
                 "account_status": str(account.get("status") or ""),
                 "encrypted_token": encrypted, "read_only": not trade_authorized,
                 "trading_authorized": trade_authorized,
                 "oauth_scope": "trading" if trade_authorized else "read",
                 "verified_at": datetime.now(timezone.utc)},
    }, upsert=not trade_authorized)
    if trade_authorized and updated.matched_count != 1:
        return _back(c["frontend"], "invalid_state")
    return _back(c["frontend"], "trading_connected" if trade_authorized else "connected")


@router.get("/verify")
def verify(identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    db = database()
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)})
    if not saved:
        raise HTTPException(404, "Alpaca OAuth is not connected.")
    try:
        token = Fernet(c["key"].encode("ascii")).decrypt(
            saved["encrypted_token"].encode("ascii")).decode("utf-8")
        account = _read_account(token)
    except Exception as exc:
        raise HTTPException(503, "Alpaca OAuth verification failed.") from exc
    if str(account["id"]) != c["expected"] or str(account["id"]) != saved["account_id"]:
        raise HTTPException(409, "Alpaca account identity changed.")
    db[CONNECTIONS].update_one({"_id": _key(identity.subject)}, {"$set": {
        "account_status": str(account.get("status") or ""),
        "verified_at": datetime.now(timezone.utc),
    }})
    return _status(db, identity)


@router.get("/winner/readiness")
def winner_oauth_readiness(
    identity: Annotated[SessionIdentity, Depends(require_admin_session)],
):
    """Only inspect the authorized account and Winner metadata; no orders."""
    from ...services.live_oauth_winner import oauth_winner_readiness
    try:
        return oauth_winner_readiness(database(), identity)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/disconnect")
def disconnect(request: Request,
               identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    _same_origin(request, c["frontend"])
    database()[CONNECTIONS].delete_one({"_id": _key(identity.subject)})
    # Only local token is removed; the Alpaca-side grant can be revoked separately.
    return {"connected": False}


def connected_readonly_client(db, identity: SessionIdentity):
    """Broker for the requesting admin's OAuth Live portfolio, if linked.

    Never return an API-key client on an invalid/expired OAuth grant. The
    Winner scheduler and order executor must not import this function.
    """
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)})
    if not saved:
        return None
    try:
        config = _config()
        if (saved.get("subject") != identity.subject
            or saved.get("environment") != "live"
            or (saved.get("read_only") is not True
                and not (saved.get("trading_authorized") is True
                         and saved.get("oauth_scope") == "trading"))
            or str(saved.get("account_id") or "") != config["expected"]):
            raise RuntimeError("Alpaca OAuth account binding is invalid. Reconnect Alpaca.")
        token = Fernet(config["key"].encode("ascii")).decrypt(
            str(saved["encrypted_token"]).encode("ascii")
        ).decode("utf-8")
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, InvalidToken) as exc:
        raise RuntimeError("Alpaca OAuth authorization cannot be read. Reconnect Alpaca.") from exc

    from alpaca.trading.client import TradingClient
    try:
        client = TradingClient(oauth_token=token, paper=False)
        account_id = str(getattr(client.get_account(), "id", "") or "")
    except Exception as exc:
        raise RuntimeError("Alpaca OAuth authorization cannot be verified. Reconnect Alpaca.") from exc
    if account_id != saved["account_id"] or account_id != config["expected"]:
        raise RuntimeError("Alpaca OAuth account identity mismatch. Reconnect Alpaca.")
    return client


def connected_trading_client(db, identity: SessionIdentity):
    """Resolve the admin's verified trading grant for read-only Winner preflight.

    A returned client is never used to submit or cancel orders in this module.
    OAuth account identity is always checked live; no API-key fallback.
    """
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)})
    if (not isinstance(saved, dict)
        or saved.get("subject") != identity.subject
        or saved.get("trading_authorized") is not True
        or saved.get("oauth_scope") != "trading"
        or saved.get("read_only") is not False):
        raise RuntimeError("Alpaca OAuth trading permission has not been authorized.")
    client = connected_readonly_client(db, identity)
    if client is None:
        raise RuntimeError("Alpaca OAuth account is disconnected.")
    return client
