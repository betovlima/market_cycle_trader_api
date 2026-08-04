from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from fastapi import HTTPException, Request

from market_cycle_trader_api.auth.config import get_auth_settings as get_settings
from market_cycle_trader_api.schemas.access_admin import (
    AccessLogResponse,
    InvitationCreateRequest,
    InvitationLinkRegenerateRequest,
    InvitationLinkResponse,
    InvitationResponse,
    InvitationUpdateRequest,
)
from market_cycle_trader_api.auth.access_store import get_access_store


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """Normalize MongoDB datetimes to timezone-aware UTC values.

    PyMongo returns BSON dates as naive UTC datetimes unless the client is
    configured with ``tz_aware=True``. Authentication code compares stored
    dates with aware UTC values, so every persisted date is normalized at the
    service boundary before comparison or serialization.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _new_access_token() -> str:
    encoded = base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")
    return "MCT-" + "-".join(
        encoded[index:index + 4] for index in range(0, len(encoded), 4)
    )


def token_digest(token: str) -> str:
    settings = get_settings()
    normalized = token.strip().upper().encode("utf-8")
    return hmac.new(
        settings.session_secret.encode("utf-8"),
        normalized,
        hashlib.sha256,
    ).hexdigest()


def _effective_status(document: dict, now: datetime | None = None) -> str:
    current = as_utc(now) or utc_now()
    if document.get("status") == "revoked":
        return "revoked"
    expires_at = as_utc(document.get("expires_at"))
    if expires_at is None or expires_at <= current:
        return "expired"
    return "active"


def invitation_response(document: dict) -> InvitationResponse:
    created_at = as_utc(document.get("created_at"))
    expires_at = as_utc(document.get("expires_at"))
    if created_at is None or expires_at is None:
        raise RuntimeError("Stored access record is missing required timestamps.")
    return InvitationResponse(
        id=document["_id"],
        guest_name=document.get("guest_name") or "Viewer",
        role=str(document.get("role") or "viewer"),
        status=_effective_status(document),
        created_at=created_at,
        expires_at=expires_at,
        last_access_at=as_utc(document.get("last_access_at")),
        revoked_at=as_utc(document.get("revoked_at")),
    )


def invitation_link_response(document: dict, raw_token: str) -> InvitationLinkResponse:
    response = invitation_response(document)
    access_url = (
        f"{get_settings().frontend_base_url}/access#token={quote(raw_token, safe='')}"
    )
    return InvitationLinkResponse(**response.model_dump(), access_url=access_url)


def _client_metadata(request: Request) -> tuple[str, str]:
    client_ip = request.client.host if request.client else "unknown"
    user_agent = str(request.headers.get("user-agent") or "")[:512]
    return client_ip, user_agent


class AccessService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = get_access_store()

    def ensure_storage(self) -> None:
        self.store.ensure_indexes()

    def _record_log(
        self,
        *,
        event: str,
        request: Request,
        invitation_id: str | None = None,
        guest_name: str | None = None,
        role: str | None = None,
        success: bool = True,
    ) -> None:
        client_ip, user_agent = _client_metadata(request)
        self.store.record_log(
            {
                "_id": str(uuid.uuid4()),
                "event": event,
                "invitation_id": invitation_id,
                "guest_name": guest_name,
                "role": role,
                "success": success,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "created_at": utc_now(),
            }
        )

    def create_invitation(
        self,
        payload: InvitationCreateRequest,
        request: Request,
    ) -> InvitationLinkResponse:
        now = utc_now()
        raw_token = _new_access_token()
        invitation_id = str(uuid.uuid4())
        document = {
            "_id": invitation_id,
            "guest_name": payload.guest_name.strip(),
            "role": payload.role,
            "token_hash": token_digest(raw_token),
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(seconds=payload.duration_seconds),
            "last_access_at": None,
            "revoked_at": None,
        }
        self.store.create_invitation(document)
        self._record_log(
            event="access_link_created",
            request=request,
            invitation_id=invitation_id,
            guest_name=document["guest_name"],
            role=document["role"],
        )
        return invitation_link_response(document, raw_token)

    def list_invitations(self) -> list[InvitationResponse]:
        return [invitation_response(item) for item in self.store.list_invitations()]

    def update_invitation(
        self,
        invitation_id: str,
        payload: InvitationUpdateRequest,
        request: Request,
    ) -> InvitationResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if document.get("status") == "revoked":
            raise HTTPException(status_code=409, detail="Revoked access cannot be extended.")
        now = utc_now()
        expires_at = (
            now + timedelta(seconds=payload.duration_seconds)
            if payload.duration_seconds is not None
            else as_utc(payload.expires_at)
        )
        updated = self.store.update_invitation(
            invitation_id,
            {
                "expires_at": expires_at,
                "status": "active",
                "updated_at": now,
            },
        )
        self._record_log(
            event="access_extended",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )
        return invitation_response(updated or document)

    def regenerate_access_link(
        self,
        invitation_id: str,
        payload: InvitationLinkRegenerateRequest,
        request: Request,
    ) -> InvitationLinkResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if document.get("status") == "revoked":
            raise HTTPException(
                status_code=409,
                detail="Revoked access cannot generate a new link.",
            )
        now = utc_now()
        raw_token = _new_access_token()
        updated = self.store.update_invitation(
            invitation_id,
            {
                "token_hash": token_digest(raw_token),
                "expires_at": now + timedelta(seconds=payload.duration_seconds),
                "status": "active",
                "updated_at": now,
            },
        ) or document
        self._record_log(
            event="access_link_regenerated",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )
        return invitation_link_response(updated, raw_token)

    def revoke_invitation(self, invitation_id: str, request: Request) -> InvitationResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        now = utc_now()
        updated = self.store.update_invitation(
            invitation_id,
            {"status": "revoked", "revoked_at": now, "updated_at": now},
        ) or document
        self.store.terminate_sessions(invitation_id)
        self._record_log(
            event="access_revoked",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )
        return invitation_response(updated)

    def terminate_sessions(self, invitation_id: str, request: Request) -> int:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        count = self.store.terminate_sessions(invitation_id)
        self._record_log(
            event="guest_sessions_terminated",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )
        return count

    def delete_invitation(self, invitation_id: str, request: Request) -> None:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if _effective_status(document) == "active":
            raise HTTPException(
                status_code=409,
                detail="Revoke or wait for the access to expire before deleting it.",
            )
        self.store.terminate_sessions(invitation_id)
        self.store.delete_invitation(invitation_id)
        self._record_log(
            event="access_deleted",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )

    def create_viewer_session(self, raw_token: str, request: Request) -> dict:
        digest = token_digest(raw_token)
        document = self.store.get_invitation_by_token_hash(digest)
        if not document:
            self._record_log(event="guest_access_denied", request=request, success=False)
            raise HTTPException(status_code=401, detail="Invalid or expired access token.")
        if _effective_status(document) != "active":
            self._record_log(
                event="guest_access_denied",
                request=request,
                invitation_id=document["_id"],
                guest_name=document.get("guest_name"),
                role=str(document.get("role") or "viewer"),
                success=False,
            )
            raise HTTPException(status_code=401, detail="Invalid or expired access token.")
        now = utc_now()
        session_id = str(uuid.uuid4())
        invitation_expires_at = as_utc(document.get("expires_at"))
        if invitation_expires_at is None:
            raise HTTPException(status_code=401, detail="Invalid or expired access token.")
        session = {
            "_id": session_id,
            "invitation_id": document["_id"],
            "role": str(document.get("role") or "viewer"),
            "display_name": document.get("guest_name") or "Viewer",
            "created_at": now,
            "expires_at": invitation_expires_at,
            "revoked": False,
        }
        self.store.create_session(session)
        self.store.update_invitation(
            document["_id"],
            {"last_access_at": now, "updated_at": now},
        )
        self._record_log(
            event="guest_access_granted",
            request=request,
            invitation_id=document["_id"],
            guest_name=document.get("guest_name"),
            role=str(document.get("role") or "viewer"),
        )
        return session

    def validate_viewer_session(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        now = utc_now()
        session_expires_at = as_utc(session.get("expires_at")) if session else None
        if (
            not session
            or session.get("revoked")
            or session_expires_at is None
            or session_expires_at <= now
        ):
            raise HTTPException(status_code=401, detail="Guest session expired or revoked.")
        invitation = self.store.get_invitation(session["invitation_id"])
        if not invitation or _effective_status(invitation, now) != "active":
            raise HTTPException(status_code=401, detail="Guest access expired or revoked.")
        session["expires_at"] = session_expires_at
        session["created_at"] = as_utc(session.get("created_at")) or now
        return session

    def revoke_viewer_session(self, session_id: str) -> None:
        self.store.revoke_session(session_id)

    def list_logs(self, limit: int) -> list[AccessLogResponse]:
        return [
            AccessLogResponse(
                id=item["_id"],
                event=item["event"],
                invitation_id=item.get("invitation_id"),
                guest_name=item.get("guest_name"),
                role=item.get("role"),
                success=bool(item.get("success", True)),
                client_ip=item.get("client_ip", "unknown"),
                created_at=as_utc(item.get("created_at")) or utc_now(),
            )
            for item in self.store.list_logs(limit)
        ]


_service: AccessService | None = None


def get_access_service() -> AccessService:
    global _service
    if _service is None:
        _service = AccessService()
    return _service


def reset_access_service() -> None:
    global _service
    _service = None
