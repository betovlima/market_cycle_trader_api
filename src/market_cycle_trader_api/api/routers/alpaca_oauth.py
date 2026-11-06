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
from pymongo.errors import DuplicateKeyError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from ...auth.security import SessionIdentity, require_portfolio_session
from ...core.runtime import database
from ...infrastructure.trading.canonical_live_oauth import canonical_live_auth_mode

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
    if not all(c[name] for name in ("client", "secret", "redirect", "frontend", "key")):
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


def _expected_account_id(config, identity) -> str:
    """Preserve the existing admin Live binding; traders link their own Alpaca."""
    role = str(getattr(identity, "role", "admin") or "admin")
    return str(config["expected"] or "") if role == "admin" else ""


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


def _oauth_winner_operational(db, saved) -> bool:
    """Current existing Live operating state, not a new activation switch."""
    if not isinstance(saved, dict) or saved.get("trading_authorized") is not True:
        return False
    from ...services.live_control import live_control_status
    from ...services.live_shared_winner_signal import _canonical_account_id
    canonical = _canonical_account_id()
    if canonical and str(saved.get("account_id") or "") == canonical:
        # The canonical account only uses OAuth when the migration is opted in.
        # The fan-out scheduler always excludes the canonical account.
        if canonical_live_auth_mode() != "oauth":
            return False
    control = live_control_status(db)
    return bool(control.get("order_submission_enabled", False))


def _status(db, identity):
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)},
                                      {"encrypted_token": 0})
    authorized = bool(saved and saved.get("trading_authorized") is True)
    return {"configured": True, "connected": bool(saved),
            "canonical_live_auth_mode": canonical_live_auth_mode(),
            "environment": "live", "read_only": not authorized,
            "trading_authorized": authorized,
            "trading_consent_available": _trading_consent_enabled(),
            "trading_enabled": _oauth_winner_operational(db, saved),
            "execution_source": "oauth" if authorized else None,
            "account_id_suffix": str(saved.get("account_id") or "")[-6:] if saved else None,
            "account_status": saved.get("account_status") if saved else None}


@router.get("/status")
def connection_status(identity: Annotated[SessionIdentity, Depends(require_portfolio_session)]):
    try:
        _config()
    except RuntimeError:
        return {"configured": False, "connected": False, "environment": "live",
                "read_only": True, "trading_authorized": False,
                "trading_consent_available": False, "trading_enabled": False}
    return _status(database(), identity)


@router.post("/start")
def start(request: Request,
          identity: Annotated[SessionIdentity, Depends(require_portfolio_session)]):
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
                            "expected_account_id": _expected_account_id(c, identity),
                            "role": str(getattr(identity, "role", "admin") or "admin"),
                            "expires": now + timedelta(minutes=10)})
    # Alpaca defaults to read-only when the optional scope is not requested.
    url = AUTHORIZE + "?" + urlencode({
        "response_type": "code", "client_id": c["client"],
        "redirect_uri": c["redirect"], "state": state, "env": "live",
    })
    return {"authorization_url": url}


@router.post("/start-trading")
def start_trading(request: Request,
                  identity: Annotated[SessionIdentity, Depends(require_portfolio_session)]):
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
    role = str(getattr(identity, "role", "admin") or "admin")
    # An invited trader may grant trading on the FIRST Alpaca connection.
    # The existing admin Live account still requires its verified read-only
    # linkage before any escalation; an invited trader is not bound to the
    # admin's ALPACA_LIVE_ACCOUNT_ID.
    first_trader_connection = linked is None and role == "trader"
    if not first_trader_connection and (
        not isinstance(linked, dict)
        or linked.get("subject") != identity.subject
        or linked.get("environment") != "live"
        or (_expected_account_id(c, identity)
            and linked.get("account_id") != _expected_account_id(c, identity))
    ):
        raise HTTPException(409, "Connect and verify the existing Live account first.")
    state = secrets.token_urlsafe(40)
    db[PENDING].create_index("expires", expireAfterSeconds=0)
    pending = {
        "_id": _digest(state), "subject": identity.subject,
        "requested_scope": "trading",
        "expected_account_id": _expected_account_id(c, identity),
        "role": role,
        "first_trader_connection": first_trader_connection,
        "expires": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    if not first_trader_connection:
        pending.update({
            "linked_account_id": linked["account_id"],
            # A disconnected/replaced grant invalidates a prior browser flow.
            "linked_token_hash": _digest(str(linked.get("encrypted_token") or "")),
        })
    db[PENDING].insert_one(pending)
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
    # Alpaca redirects cross-site; the MCT cookie may not travel with it.
    # The one-use state binds this callback to the authenticated user at /start.
    if (not saved or not isinstance(saved.get("subject"), str)
        or not saved["subject"] or not isinstance(saved.get("expires"), datetime)):
        return _back(c["frontend"], "invalid_state")
    subject = saved["subject"]
    requested_scope = saved.get("requested_scope", "read")
    if requested_scope not in {"read", "trading"}:
        return _back(c["frontend"], "invalid_state")
    first_trader_connection = (
        requested_scope == "trading"
        and saved.get("first_trader_connection") is True
    )
    if requested_scope == "trading":
        if not _trading_consent_enabled():
            return _back(c["frontend"], "trading_unavailable")
        current = db[CONNECTIONS].find_one({"_id": _key(subject)})
        if first_trader_connection:
            # Claim token was issued after the invited Trader signed in.
            # A new grant must never replace a connection added meanwhile.
            if (saved.get("role") != "trader"
                or saved.get("expected_account_id") not in ("", None)
                or current is not None):
                return _back(c["frontend"], "invalid_state")
        elif (not isinstance(current, dict) or current.get("subject") != subject
              or current.get("environment") != "live"
              or current.get("account_id") != saved.get("linked_account_id", c["expected"])
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
    expected_account = str(saved.get("expected_account_id", c["expected"]) or "")
    if ((expected_account and str(account["id"]) != expected_account)
        or (requested_scope == "trading" and not first_trader_connection
            and str(account["id"]) != str(current["account_id"]))):
        return _back(c["frontend"], "account_mismatch")
    trade_authorized = requested_scope == "trading"
    encrypted = Fernet(c["key"].encode("ascii")).encrypt(token.encode("utf-8")).decode("ascii")
    connection_data = {
        "subject": subject, "environment": "live",
        "role": saved.get("role", "admin"),
        "account_id": str(account["id"]),
        "account_status": str(account.get("status") or ""),
        "encrypted_token": encrypted, "read_only": not trade_authorized,
        "trading_authorized": trade_authorized,
        "oauth_scope": "trading" if trade_authorized else "read",
        "verified_at": datetime.now(timezone.utc),
    }
    if first_trader_connection:
        try:
            db[CONNECTIONS].insert_one({
                "_id": _key(subject), **connection_data,
            })
        except DuplicateKeyError:
            # An authorization from another tab or a subsequent disconnect
            # must never be overwritten by this older callback.
            return _back(c["frontend"], "invalid_state")
    else:
        update_query = {"_id": _key(subject)}
        if trade_authorized:
            # Compare-and-swap: keep admin/upgrade revocation protection.
            update_query.update({
                "subject": subject, "encrypted_token": current["encrypted_token"],
            })
        updated = db[CONNECTIONS].update_one(
            update_query, {"$set": connection_data}, upsert=not trade_authorized,
        )
        if trade_authorized and updated.matched_count != 1:
            return _back(c["frontend"], "invalid_state")
    return _back(c["frontend"], "trading_connected" if trade_authorized else "connected")


@router.get("/verify")
def verify(identity: Annotated[SessionIdentity, Depends(require_portfolio_session)]):
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
    expected_account = _expected_account_id(c, identity)
    if ((expected_account and str(account["id"]) != expected_account)
        or str(account["id"]) != saved["account_id"]):
        raise HTTPException(409, "Alpaca account identity changed.")
    db[CONNECTIONS].update_one({"_id": _key(identity.subject)}, {"$set": {
        "account_status": str(account.get("status") or ""),
        "verified_at": datetime.now(timezone.utc),
    }})
    return _status(db, identity)


@router.get("/portfolio")
def personal_portfolio(
    identity: Annotated[SessionIdentity, Depends(require_portfolio_session)],
):
    """Individual Alpaca portfolio, with NO API-key fallback and no MCT balance writes."""
    from ...services.live_personal_portfolio import personal_live_portfolio_snapshot
    try:
        db = database()
        broker = connected_readonly_client(db, identity)
        if broker is None:
            raise HTTPException(404, "Connect your Alpaca account first.")
        return personal_live_portfolio_snapshot(broker)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/winner/readiness")
def winner_oauth_readiness(
    identity: Annotated[SessionIdentity, Depends(require_portfolio_session)],
):
    """Only inspect the authorized account and Winner metadata; no orders."""
    from ...services.live_oauth_winner import oauth_winner_readiness
    try:
        return oauth_winner_readiness(database(), identity)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/disconnect")
def disconnect(request: Request,
               identity: Annotated[SessionIdentity, Depends(require_portfolio_session)]):
    try:
        c = _config()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    _same_origin(request, c["frontend"])
    database()[CONNECTIONS].delete_one({"_id": _key(identity.subject)})
    # Only local token is removed; the Alpaca-side grant can be revoked separately.
    return {"connected": False}


def connected_readonly_client(db, identity: SessionIdentity):
    """Broker for the requesting user's OAuth Live portfolio, if linked.

    Never return an API-key client on an invalid/expired OAuth grant.
    The legacy API-key Winner executor remains independent.
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
            or (_expected_account_id(config, identity)
                and str(saved.get("account_id") or "") != _expected_account_id(config, identity))):
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
    if (account_id != saved["account_id"]
        or (_expected_account_id(config, identity)
            and account_id != _expected_account_id(config, identity))):
        raise RuntimeError("Alpaca OAuth account identity mismatch. Reconnect Alpaca.")
    return client


def connected_trading_client(db, identity: SessionIdentity):
    """Resolve the requesting user's verified trading grant for read-only Winner preflight.

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
