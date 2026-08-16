from __future__ import annotations

import hmac
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeSerializer

from market_cycle_trader_api.auth.config import AuthSettings as Settings, get_auth_settings as get_settings
from market_cycle_trader_api.auth.capabilities import capabilities_for_role

SESSION_COOKIE_NAME = "market_cycle_trader_session"


@dataclass(frozen=True)
class SessionIdentity:
    subject: str
    role: str
    scope: str
    expires_at: datetime
    session_id: str | None = None
    display_name: str | None = None
    email: str | None = None

    def has_capability(self, name: str) -> bool:
        return bool(capabilities_for_role(self.role).get(name))

    @property
    def is_admin(self) -> bool:
        return self.has_capability("admin.manage")

    @property
    def can_view_portfolio(self) -> bool:
        return self.has_capability("portfolio.view")


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.serializer = URLSafeSerializer(
            secret_key=settings.session_secret,
            salt="market-cycle-trader-session-v1",
        )

    def password_matches(self, submitted_password: str) -> bool:
        expected = self.settings.admin_password.encode("utf-8")
        submitted = submitted_password.encode("utf-8")
        return bool(expected) and hmac.compare_digest(expected, submitted)

    def create_admin_identity(self) -> SessionIdentity:
        
        return SessionIdentity(
            subject="trader-admin",
            role="admin",
            scope="trader:read portfolio:read admin:manage",
            expires_at=datetime.now(UTC) + timedelta(seconds=self.settings.session_max_age_for_role("admin")),
            display_name="Administrator",
            email=None,
        )

    def create_access_identity(self, session: dict) -> SessionIdentity:
        role = str(session.get("role") or "viewer")
        scope_by_role = {
            "viewer": "trader:read",
            "trader": "trader:read portfolio:read",
            "admin": "trader:read portfolio:read admin:manage",
        }
        if role not in scope_by_role:
            raise HTTPException(status_code=401, detail="Invalid access role.")
        identity_subject = str(session.get("identity_subject") or "")
        return SessionIdentity(
            subject=f"google:{identity_subject}",
            role=role,
            scope=scope_by_role[role],
            expires_at=session["expires_at"],
            session_id=session["_id"],
            display_name=session.get("display_name"),
            email=session.get("identity_email"),
        )

    def create_guest_identity(self, session: dict) -> SessionIdentity:
        return self.create_access_identity(session)

    def create_viewer_identity(self, session: dict) -> SessionIdentity:
        return self.create_access_identity(session)

    def create_session_token(self, identity: SessionIdentity) -> str:
        return self.serializer.dumps(
            {
                "subject": identity.subject,
                "role": identity.role,
                "scope": identity.scope,
                "expires_at": identity.expires_at.isoformat(),
                "session_id": identity.session_id,
                "display_name": identity.display_name,
                "email": identity.email,
            }
        )

    def decode_session_token(self, token: str) -> SessionIdentity:
        try:
            payload = self.serializer.loads(token)
        except BadSignature as exc:
            raise HTTPException(status_code=401, detail="Invalid session.") from exc

        try:
            expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=401, detail="Invalid session.") from exc
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Session expired.")

        role = str(payload.get("role") or "")
        scope = str(payload.get("scope") or "")
        if role not in {"admin", "viewer", "trader"} or "trader:read" not in scope.split():
            raise HTTPException(status_code=401, detail="Invalid session.")
        identity = SessionIdentity(
            subject=str(payload.get("subject") or ""),
            role=role,
            scope=scope,
            expires_at=expires_at,
            session_id=payload.get("session_id"),
            display_name=payload.get("display_name"),
            email=payload.get("email"),
        )

        if identity.session_id:
            from market_cycle_trader_api.auth.access_service import get_access_service

            current = get_access_service().validate_access_session(identity.session_id)
            if str(current.get("role") or "viewer") != role:
                raise HTTPException(status_code=401, detail="Invalid identity-bound session.")
            return self.create_access_identity(current)

        if identity.subject != "trader-admin" or role != "admin" or "admin:manage" not in scope.split():
            raise HTTPException(status_code=401, detail="Invalid administrator session.")
        return identity

    def set_cookie(self, response: Response, identity: SessionIdentity) -> None:
        max_age = max(1, int((identity.expires_at - datetime.now(UTC)).total_seconds()))
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=self.create_session_token(identity),
            max_age=max_age,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/",
        )

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/",
        )

    def require_session(self, request: Request) -> SessionIdentity:
        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return self.decode_session_token(token)


def get_session_manager() -> SessionManager:
    return SessionManager(get_settings())


def require_trader_session(request: Request) -> SessionIdentity:
    return get_session_manager().require_session(request)


def require_capability(capability: str):
    name = str(capability).strip()
    if not name:
        raise ValueError("Capability name is required.")

    def dependency(request: Request) -> SessionIdentity:
        identity = require_trader_session(request)
        if not identity.has_capability(name):
            raise HTTPException(status_code=403, detail="Required capability is not available for this session.")
        return identity

    return dependency


def require_trader_access(request: Request) -> SessionIdentity:
    identity = require_trader_session(request)
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"} and not identity.has_capability("research.manage"):
        raise HTTPException(status_code=403, detail="Administrator access required for this operation.")
    return identity


def require_backtest_access(request: Request) -> SessionIdentity:
    identity = require_trader_session(request)
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"} and not identity.has_capability("backtest.start"):
        raise HTTPException(status_code=403, detail="Trader or administrator access required for this operation.")
    return identity


def require_portfolio_session(request: Request) -> SessionIdentity:
    identity = require_trader_session(request)
    if not identity.has_capability("portfolio.view"):
        raise HTTPException(status_code=403, detail="Trader or administrator access required.")
    return identity


def require_admin_session(request: Request) -> SessionIdentity:
    identity = require_trader_session(request)
    if not identity.has_capability("admin.manage"):
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return identity


class LoginAttemptLimiter:
    def __init__(self, *, maximum_attempts: int = 5, window_seconds: int = 300) -> None:
        self.maximum_attempts = maximum_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def ensure_allowed(self, client_key: str) -> None:
        now = monotonic()
        with self._lock:
            attempts = self._attempts[client_key]
            while attempts and now - attempts[0] > self.window_seconds:
                attempts.popleft()
            if len(attempts) >= self.maximum_attempts:
                raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")

    def register_failure(self, client_key: str) -> None:
        with self._lock:
            self._attempts[client_key].append(monotonic())

    def clear(self, client_key: str) -> None:
        with self._lock:
            self._attempts.pop(client_key, None)


login_attempt_limiter = LoginAttemptLimiter()
