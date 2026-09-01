from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.auth.reviewer_access_service import get_reviewer_access_service
from market_cycle_trader_api.auth.security import SessionIdentity, get_session_manager, login_attempt_limiter, require_admin_session
from market_cycle_trader_api.schemas.auth import SessionResponse
from market_cycle_trader_api.schemas.reviewer_access import (
    ReviewerAccessCreateRequest,
    ReviewerAccessListResponse,
    ReviewerAccessLoginRequest,
    ReviewerAccessPreviewRequest,
    ReviewerAccessPreviewResponse,
    ReviewerAccessRegenerateRequest,
    ReviewerAccessResponse,
    ReviewerAccessSecretResponse,
    ReviewerAccessSessionTerminationResponse,
)

public_router = APIRouter(prefix="/api/auth/reviewer", tags=["reviewer-access"])
admin_router = APIRouter(prefix="/api/admin/reviewer-access", tags=["reviewer-access-administration"])
AdminSession = Annotated[SessionIdentity, Depends(require_admin_session)]


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _session_response(identity: SessionIdentity, idle_expires_at=None) -> SessionResponse:
    return SessionResponse(
        authenticated=True,
        role=identity.role,
        expires_in_seconds=max(0, int((identity.expires_at - datetime.now(UTC)).total_seconds())),
        expires_at=identity.expires_at,
        idle_expires_at=idle_expires_at,
        display_name=identity.display_name,
        email=identity.email,
        capabilities=capabilities_for_role(identity.role),
    )


@public_router.post("/preview", response_model=ReviewerAccessPreviewResponse)
def preview_reviewer_access(payload: ReviewerAccessPreviewRequest, request: Request) -> ReviewerAccessPreviewResponse:
    client_key = _client_ip(request)
    login_attempt_limiter.ensure_allowed(client_key)
    try:
        value = get_reviewer_access_service().preview(payload.access_id)
    except Exception:
        login_attempt_limiter.register_failure(client_key)
        raise
    login_attempt_limiter.clear(client_key)
    return ReviewerAccessPreviewResponse(**value)


@public_router.post("/access", response_model=SessionResponse)
def reviewer_access(
    payload: ReviewerAccessLoginRequest,
    request: Request,
    response: Response,
) -> SessionResponse:
    client_key = _client_ip(request)
    login_attempt_limiter.ensure_allowed(client_key)
    try:
        session = get_reviewer_access_service().create_session(
            access_id=payload.access_id,
            code=payload.code,
            client_ip=client_key,
        )
    except Exception:
        login_attempt_limiter.register_failure(client_key)
        raise
    login_attempt_limiter.clear(client_key)
    manager = get_session_manager()
    identity = manager.create_access_identity(session)
    manager.set_cookie(response, identity)
    return _session_response(identity, session.get("idle_expires_at"))


@admin_router.get("", response_model=ReviewerAccessListResponse)
def list_reviewer_access(_: AdminSession) -> ReviewerAccessListResponse:
    return ReviewerAccessListResponse(items=get_reviewer_access_service().list_accesses())


@admin_router.post("", response_model=ReviewerAccessSecretResponse, status_code=status.HTTP_201_CREATED)
def create_reviewer_access(
    payload: ReviewerAccessCreateRequest,
    request: Request,
    _: AdminSession,
) -> ReviewerAccessSecretResponse:
    value = get_reviewer_access_service().create_access(
        guest_name=payload.guest_name,
        duration_seconds=payload.duration_seconds,
        max_active_sessions=payload.max_active_sessions,
        client_ip=_client_ip(request),
    )
    return ReviewerAccessSecretResponse(**value)


@admin_router.post("/{access_id}/regenerate", response_model=ReviewerAccessSecretResponse)
def regenerate_reviewer_access(
    access_id: str,
    payload: ReviewerAccessRegenerateRequest,
    request: Request,
    _: AdminSession,
) -> ReviewerAccessSecretResponse:
    value = get_reviewer_access_service().regenerate_code(
        access_id,
        duration_seconds=payload.duration_seconds,
        client_ip=_client_ip(request),
    )
    return ReviewerAccessSecretResponse(**value)


@admin_router.post("/{access_id}/terminate-sessions", response_model=ReviewerAccessSessionTerminationResponse)
def terminate_reviewer_sessions(
    access_id: str,
    request: Request,
    _: AdminSession,
) -> ReviewerAccessSessionTerminationResponse:
    count = get_reviewer_access_service().terminate_sessions(access_id, client_ip=_client_ip(request))
    return ReviewerAccessSessionTerminationResponse(terminated_sessions=count)


@admin_router.post("/{access_id}/revoke", response_model=ReviewerAccessResponse)
def revoke_reviewer_access(
    access_id: str,
    request: Request,
    _: AdminSession,
) -> ReviewerAccessResponse:
    value = get_reviewer_access_service().revoke_access(access_id, client_ip=_client_ip(request))
    return ReviewerAccessResponse(**value)


@admin_router.delete("/{access_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reviewer_access(
    access_id: str,
    request: Request,
    _: AdminSession,
) -> Response:
    get_reviewer_access_service().delete_access(access_id, client_ip=_client_ip(request))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
