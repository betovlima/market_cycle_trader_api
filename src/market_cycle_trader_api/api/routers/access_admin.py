from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from market_cycle_trader_api.auth.security import SessionIdentity, require_admin_session
from market_cycle_trader_api.schemas.access_admin import (
    AccessLogListResponse,
    InvitationCreateRequest,
    InvitationLinkRegenerateRequest,
    InvitationLinkResponse,
    InvitationListResponse,
    InvitationResponse,
    InvitationUpdateRequest,
    SessionTerminationResponse,
)
from market_cycle_trader_api.auth.access_service import get_access_service

router = APIRouter(prefix="/api/admin", tags=["trader administration"])
AdminSession = Annotated[SessionIdentity, Depends(require_admin_session)]


@router.get("/invitations", response_model=InvitationListResponse)
def list_invitations(_: AdminSession) -> InvitationListResponse:
    return InvitationListResponse(items=get_access_service().list_invitations())


@router.post(
    "/invitations",
    response_model=InvitationLinkResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invitation(
    payload: InvitationCreateRequest,
    request: Request,
    _: AdminSession,
) -> InvitationLinkResponse:
    return get_access_service().create_invitation(payload, request)


@router.patch("/invitations/{invitation_id}", response_model=InvitationResponse)
def update_invitation(
    invitation_id: str,
    payload: InvitationUpdateRequest,
    request: Request,
    _: AdminSession,
) -> InvitationResponse:
    return get_access_service().update_invitation(invitation_id, payload, request)


@router.post(
    "/invitations/{invitation_id}/regenerate-link",
    response_model=InvitationLinkResponse,
)
def regenerate_access_link(
    invitation_id: str,
    payload: InvitationLinkRegenerateRequest,
    request: Request,
    _: AdminSession,
) -> InvitationLinkResponse:
    return get_access_service().regenerate_access_link(
        invitation_id,
        payload,
        request,
    )


@router.post("/invitations/{invitation_id}/revoke", response_model=InvitationResponse)
def revoke_invitation(
    invitation_id: str,
    request: Request,
    _: AdminSession,
) -> InvitationResponse:
    return get_access_service().revoke_invitation(invitation_id, request)


@router.post(
    "/invitations/{invitation_id}/terminate-sessions",
    response_model=SessionTerminationResponse,
)
def terminate_sessions(
    invitation_id: str,
    request: Request,
    _: AdminSession,
) -> SessionTerminationResponse:
    count = get_access_service().terminate_sessions(invitation_id, request)
    return SessionTerminationResponse(terminated_sessions=count)


@router.delete("/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_invitation(
    invitation_id: str,
    request: Request,
    _: AdminSession,
) -> Response:
    get_access_service().delete_invitation(invitation_id, request)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/access-logs", response_model=AccessLogListResponse)
def access_logs(
    _: AdminSession,
    limit: int = Query(default=100, ge=1, le=500),
) -> AccessLogListResponse:
    return AccessLogListResponse(items=get_access_service().list_logs(limit))
