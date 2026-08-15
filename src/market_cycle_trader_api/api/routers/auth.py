from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response

from market_cycle_trader_api.auth.access_service import get_access_service
from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.auth.security import (
    SESSION_COOKIE_NAME,
    get_session_manager,
    login_attempt_limiter,
)
from market_cycle_trader_api.schemas.auth import (
    AccessPreviewRequest,
    AccessPreviewResponse,
    AdminLoginRequest,
    GoogleAccessRequest,
    SessionResponse,
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


def _session_response(identity, idle_expires_at=None) -> SessionResponse:
    return SessionResponse(
        authenticated=True,
        role=identity.role,
        expires_in_seconds=max(
            0, int((identity.expires_at - datetime.now(UTC)).total_seconds())
        ),
        expires_at=identity.expires_at,
        idle_expires_at=idle_expires_at,
        display_name=identity.display_name,
        email=identity.email,
        capabilities=capabilities_for_role(identity.role),
    )


@router.post("/admin/login", response_model=SessionResponse)
@router.post("/login", response_model=SessionResponse, include_in_schema=False)
def admin_login(
    payload: AdminLoginRequest,
    request: Request,
    response: Response,
) -> SessionResponse:
    manager = get_session_manager()
    client_key = request.client.host if request.client else "unknown"
    login_attempt_limiter.ensure_allowed(client_key)
    if not manager.password_matches(payload.password):
        login_attempt_limiter.register_failure(client_key)
        raise HTTPException(status_code=401, detail="Invalid administrator credentials.")
    login_attempt_limiter.clear(client_key)
    identity = manager.create_admin_identity()
    manager.set_cookie(response, identity)
    return _session_response(identity)


@router.post("/access/preview", response_model=AccessPreviewResponse)
def access_preview(
    payload: AccessPreviewRequest,
    request: Request,
) -> AccessPreviewResponse:
    client_key = request.client.host if request.client else "unknown"
    login_attempt_limiter.ensure_allowed(client_key)
    try:
        result = get_access_service().preview_access(payload, request)
    except HTTPException:
        login_attempt_limiter.register_failure(client_key)
        raise
    login_attempt_limiter.clear(client_key)
    return result


@router.post("/access", response_model=SessionResponse)
@router.post("/viewer/access", response_model=SessionResponse, include_in_schema=False)
def verified_access(
    payload: GoogleAccessRequest,
    request: Request,
    response: Response,
) -> SessionResponse:
    manager = get_session_manager()
    client_key = request.client.host if request.client else "unknown"
    login_attempt_limiter.ensure_allowed(client_key)
    try:
        session = get_access_service().create_google_session(payload, request)
    except HTTPException:
        login_attempt_limiter.register_failure(client_key)
        raise
    login_attempt_limiter.clear(client_key)
    identity = manager.create_access_identity(session)
    manager.set_cookie(response, identity)
    return _session_response(identity, session.get("idle_expires_at"))


@router.get("/session", response_model=SessionResponse)
def session(request: Request) -> SessionResponse:
    manager = get_session_manager()
    token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if not token:
        return SessionResponse(authenticated=False, expires_in_seconds=0)
    try:
        identity = manager.decode_session_token(token)
    except HTTPException:
        return SessionResponse(authenticated=False, expires_in_seconds=0)
    idle_expires_at = None
    if identity.session_id:
        current = get_access_service().validate_access_session(identity.session_id)
        idle_expires_at = current.get("idle_expires_at")
    return _session_response(identity, idle_expires_at)


@router.post("/logout", response_model=SessionResponse)
def logout(request: Request, response: Response) -> SessionResponse:
    manager = get_session_manager()
    token = request.cookies.get(SESSION_COOKIE_NAME, "")
    if token:
        try:
            identity = manager.decode_session_token(token)
            if identity.session_id:
                get_access_service().revoke_access_session(identity.session_id)
        except HTTPException:
            pass
    manager.clear_cookie(response)
    return SessionResponse(authenticated=False, expires_in_seconds=0)
