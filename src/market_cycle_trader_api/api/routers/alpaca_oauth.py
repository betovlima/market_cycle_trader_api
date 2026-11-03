"""Experimental Alpaca Connect OAuth: existing administrator, read-only Live.

No trading access and no modification to the MCT Winner order engine.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlencode, urlsplit

import requests
from cryptography.fernet import Fernet
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


def _status(db, identity):
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)},
                                      {"encrypted_token": 0})
    return {"configured": True, "connected": bool(saved),
            "environment": "live", "read_only": True, "trading_enabled": False,
            "account_id_suffix": str(saved.get("account_id") or "")[-6:] if saved else None,
            "account_status": saved.get("account_status") if saved else None}


@router.get("/status")
def connection_status(identity: Annotated[SessionIdentity, Depends(require_admin_session)]):
    try:
        _config()
    except RuntimeError:
        return {"configured": False, "connected": False, "environment": "live",
                "read_only": True, "trading_enabled": False}
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
                            "expires": now + timedelta(minutes=10)})
    # Alpaca defaults to read-only when the optional scope is not requested.
    url = AUTHORIZE + "?" + urlencode({
        "response_type": "code", "client_id": c["client"],
        "redirect_uri": c["redirect"], "state": state, "env": "live",
    })
    return {"authorization_url": url}


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
        if (not token or str(grant.get("token_type") or "").lower() != "bearer"
            or scope.intersection({"trading", "account:write"})):
            return _back(c["frontend"], "permission_failed")
        account = _read_account(token)
    except (requests.RequestException, RuntimeError, ValueError, AttributeError, TypeError):
        return _back(c["frontend"], "verification_failed")
    if str(account["id"]) != c["expected"]:
        return _back(c["frontend"], "account_mismatch")
    encrypted = Fernet(c["key"].encode("ascii")).encrypt(token.encode("utf-8")).decode("ascii")
    db[CONNECTIONS].update_one({"_id": _key(subject)}, {
        "$set": {"subject": subject, "environment": "live",
                 "account_id": str(account["id"]),
                 "account_status": str(account.get("status") or ""),
                 "encrypted_token": encrypted, "read_only": True,
                 "verified_at": datetime.now(timezone.utc)},
    }, upsert=True)
    return _back(c["frontend"], "connected")


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
