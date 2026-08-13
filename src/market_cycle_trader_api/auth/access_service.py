from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from fastapi import HTTPException, Request

from market_cycle_trader_api.auth.access_store import get_access_store
from market_cycle_trader_api.auth.config import get_auth_settings as get_settings
from market_cycle_trader_api.auth.google_identity import (
    GoogleIdentityVerifier,
    ProductionGoogleIdentityVerifier,
    VerifiedGoogleIdentity,
    normalize_email,
)
from market_cycle_trader_api.schemas.access_admin import (
    AccessLogResponse,
    InvitationCreateRequest,
    InvitationLinkRegenerateRequest,
    InvitationLinkResponse,
    InvitationResponse,
    InvitationUpdateRequest,
)
from market_cycle_trader_api.schemas.auth import AccessPreviewRequest, AccessPreviewResponse, GoogleAccessRequest


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
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


def token_digest(token: str, invitation_id: str | None = None) -> str:
    





    settings = get_settings()
    normalized = str(token or "").strip().upper()
    material = f"{invitation_id}:{normalized}" if invitation_id else normalized
    return hmac.new(
        settings.session_secret.encode("utf-8"),
        material.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _authorization_status(document: dict, now: datetime | None = None) -> str:
    current = as_utc(now) or utc_now()
    stored = str(document.get("status") or "")
    if stored == "revoked":
        return "revoked"
    if stored == "legacy_unverified" or not document.get("authorized_email"):
        return "legacy_unverified"
    expires_at = as_utc(document.get("expires_at"))
    if expires_at is None or expires_at <= current:
        return "expired"
    if stored == "blocked":
        return "blocked"
    if stored == "claimed" and document.get("claimed_subject"):
        return "claimed"
    return "pending_verification"


def _masked_email(email: str) -> str:
    normalized = normalize_email(email)
    local, separator, domain = normalized.partition("@")
    if not separator:
        return "hidden"
    if len(local) <= 2:
        masked_local = local[:1] + "*"
    else:
        masked_local = local[:2] + "*" * min(6, len(local) - 2)
    return f"{masked_local}@{domain}"


def _client_metadata(request: Request) -> tuple[str, str]:
    forwarded = str(request.headers.get("x-forwarded-for") or "").split(",", 1)[0].strip()
    client_ip = forwarded or (request.client.host if request.client else "unknown")
    user_agent = str(request.headers.get("user-agent") or "")[:512]
    return client_ip, user_agent


class AccessService:
    def __init__(self, identity_verifier: GoogleIdentityVerifier | None = None) -> None:
        self.settings = get_settings()
        self.store = get_access_store()
        self.identity_verifier = identity_verifier or ProductionGoogleIdentityVerifier()

    def ensure_storage(self) -> None:
        self.store.ensure_indexes()

    def _record_log(
        self,
        *,
        event: str,
        request: Request,
        invitation_id: str | None = None,
        guest_name: str | None = None,
        identity_email: str | None = None,
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
                "identity_email": normalize_email(identity_email or "") or None,
                "role": role,
                "success": success,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "created_at": utc_now(),
            }
        )

    def _active_session_count(self, invitation_id: str, now: datetime | None = None) -> int:
        return self.store.count_active_sessions(invitation_id, as_utc(now) or utc_now())

    def _invitation_response(self, document: dict) -> InvitationResponse:
        created_at = as_utc(document.get("created_at"))
        expires_at = as_utc(document.get("expires_at"))
        if created_at is None or expires_at is None:
            raise RuntimeError("Stored access record is missing required timestamps.")
        active_sessions = self._active_session_count(document["_id"])
        status = _authorization_status(document)
        if status == "claimed" and active_sessions > 0:
            status = "active"
        return InvitationResponse(
            id=document["_id"],
            guest_name=document.get("guest_name") or "Guest",
            authorized_email=document.get("authorized_email") or None,
            role=str(document.get("role") or "viewer"),
            status=status,
            created_at=created_at,
            expires_at=expires_at,
            last_access_at=as_utc(document.get("last_access_at")),
            claimed_at=as_utc(document.get("claimed_at")),
            claimed_email=document.get("claimed_email") or None,
            max_active_sessions=int(
                document.get("max_active_sessions")
                or (1 if document.get("role") in {"trader", "admin"} else 2)
            ),
            active_sessions=active_sessions,
            revoked_at=as_utc(document.get("revoked_at")),
            primary_administrator=bool(document.get("primary_administrator")),
        )

    def _invitation_link_response(
        self,
        document: dict,
        raw_token: str,
    ) -> InvitationLinkResponse:
        response = self._invitation_response(document)
        access_url = (
            f"{self.settings.frontend_base_url}/access"
            f"#invitation={quote(document['_id'], safe='')}"
            f"&token={quote(raw_token, safe='')}"
        )
        return InvitationLinkResponse(**response.model_dump(), access_url=access_url)

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
            "authorized_email": normalize_email(str(payload.authorized_email)),
            "role": payload.role,
            "max_active_sessions": int(payload.max_active_sessions or 1),
            "token_hash": token_digest(raw_token, invitation_id),
            "token_version": 1,
            "status": "pending_verification",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(seconds=payload.duration_seconds),
            "last_access_at": None,
            "claimed_at": None,
            "claimed_subject": None,
            "claimed_email": None,
            "revoked_at": None,
        }
        self.store.create_invitation(document)
        self._record_log(
            event="verified_access_link_created",
            request=request,
            invitation_id=invitation_id,
            guest_name=document["guest_name"],
            identity_email=document["authorized_email"],
            role=document["role"],
        )
        return self._invitation_link_response(document, raw_token)

    def list_invitations(self) -> list[InvitationResponse]:
        return [self._invitation_response(item) for item in self.store.list_invitations()]

    def update_invitation(
        self,
        invitation_id: str,
        payload: InvitationUpdateRequest,
        request: Request,
    ) -> InvitationResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        state = _authorization_status(document)
        if state in {"revoked", "legacy_unverified"}:
            raise HTTPException(status_code=409, detail="This access record cannot be updated.")
        if document.get("primary_administrator") and (
            payload.duration_seconds is not None or payload.expires_at is not None
        ):
            raise HTTPException(
                status_code=409,
                detail="The primary Google administrator expiration cannot be changed.",
            )

        now = utc_now()
        updates: dict = {"updated_at": now}
        if payload.duration_seconds is not None:
            updates["expires_at"] = now + timedelta(seconds=payload.duration_seconds)
        elif payload.expires_at is not None:
            updates["expires_at"] = as_utc(payload.expires_at)
        if payload.max_active_sessions is not None:
            updates["max_active_sessions"] = int(payload.max_active_sessions)

        updated = self.store.update_invitation(invitation_id, updates) or document
        if payload.max_active_sessions is not None:
            terminated = self.store.trim_active_sessions(
                invitation_id,
                int(payload.max_active_sessions),
                now,
            )
            if terminated:
                self._record_log(
                    event="session_limit_enforced",
                    request=request,
                    invitation_id=invitation_id,
                    guest_name=document.get("guest_name"),
                    identity_email=document.get("authorized_email"),
                    role=str(document.get("role") or "viewer"),
                )
        self._record_log(
            event="verified_access_updated",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            identity_email=document.get("authorized_email"),
            role=str(document.get("role") or "viewer"),
        )
        return self._invitation_response(updated)

    def regenerate_access_link(
        self,
        invitation_id: str,
        payload: InvitationLinkRegenerateRequest,
        request: Request,
    ) -> InvitationLinkResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if document.get("primary_administrator"):
            raise HTTPException(
                status_code=409,
                detail="The primary Google administrator does not use a claim link.",
            )
        state = _authorization_status(document)
        if state == "revoked":
            raise HTTPException(status_code=409, detail="Revoked access cannot generate a new link.")
        if state == "legacy_unverified":
            raise HTTPException(
                status_code=409,
                detail="Create a new identity-verified invitation for this legacy access.",
            )

        now = utc_now()
        raw_token = _new_access_token()
        self.store.terminate_sessions(invitation_id)
        updated = self.store.update_invitation(
            invitation_id,
            {
                "token_hash": token_digest(raw_token, invitation_id),
                "token_version": int(document.get("token_version") or 0) + 1,
                "expires_at": now + timedelta(seconds=payload.duration_seconds),
                "status": "pending_verification",
                "claimed_at": None,
                "claimed_subject": None,
                "claimed_email": None,
                "token_consumed_at": None,
                "last_access_at": None,
                "updated_at": now,
            },
        ) or document
        self._record_log(
            event="verified_access_link_regenerated",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            identity_email=document.get("authorized_email"),
            role=str(document.get("role") or "viewer"),
        )
        return self._invitation_link_response(updated, raw_token)

    def revoke_invitation(self, invitation_id: str, request: Request) -> InvitationResponse:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if document.get("primary_administrator"):
            raise HTTPException(
                status_code=409,
                detail="The primary Google administrator cannot be revoked.",
            )
        now = utc_now()
        updated = self.store.update_invitation(
            invitation_id,
            {"status": "revoked", "revoked_at": now, "updated_at": now},
        ) or document
        self.store.terminate_sessions(invitation_id)
        self._record_log(
            event="verified_access_revoked",
            request=request,
            invitation_id=invitation_id,
            guest_name=document.get("guest_name"),
            identity_email=document.get("authorized_email"),
            role=str(document.get("role") or "viewer"),
        )
        return self._invitation_response(updated)

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
            identity_email=document.get("authorized_email"),
            role=str(document.get("role") or "viewer"),
        )
        return count

    def delete_invitation(self, invitation_id: str, request: Request) -> None:
        document = self.store.get_invitation(invitation_id)
        if not document:
            raise HTTPException(status_code=404, detail="Access record not found.")
        if document.get("primary_administrator"):
            raise HTTPException(
                status_code=409,
                detail="The primary Google administrator cannot be deleted.",
            )
        state = self._invitation_response(document).status
        if state in {"pending_verification", "claimed", "active"}:
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
            identity_email=document.get("authorized_email"),
            role=str(document.get("role") or "viewer"),
        )

    def _locate_invitation(self, payload: AccessPreviewRequest) -> dict:
        document = self.store.get_invitation(payload.invitation_id)
        if not document:
            raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")
        return document

    def preview_access(self, payload: AccessPreviewRequest, request: Request) -> AccessPreviewResponse:
        document = self._locate_invitation(payload)
        state = _authorization_status(document)
        if state == "legacy_unverified":
            raise HTTPException(status_code=410, detail="A new identity-verified invitation is required.")
        if state in {"revoked", "expired", "blocked"}:
            raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")
        if state == "pending_verification":
            if not payload.token or not hmac.compare_digest(
                str(document.get("token_hash") or ""),
                token_digest(payload.token, document["_id"]),
            ):
                self._record_log(
                    event="invitation_preview_denied",
                    request=request,
                    invitation_id=document.get("_id"),
                    guest_name=document.get("guest_name"),
                    role=str(document.get("role") or "viewer"),
                    success=False,
                )
                raise HTTPException(status_code=401, detail="The complete invitation link is required.")

        expires_at = as_utc(document.get("expires_at"))
        if expires_at is None:
            raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")
        return AccessPreviewResponse(
            invitation_id=document["_id"],
            guest_name=document.get("guest_name") or "Guest",
            role=str(document.get("role") or "viewer"),
            masked_email=_masked_email(document.get("authorized_email") or ""),
            status=state,
            expires_at=expires_at,
            requires_token=state == "pending_verification",
        )

    def _verify_invited_identity(
        self,
        document: dict,
        identity: VerifiedGoogleIdentity,
        request: Request,
    ) -> None:
        authorized_email = normalize_email(document.get("authorized_email") or "")
        if not authorized_email or identity.email != authorized_email:
            self._record_log(
                event="google_identity_mismatch",
                request=request,
                invitation_id=document.get("_id"),
                guest_name=document.get("guest_name"),
                identity_email=identity.email,
                role=str(document.get("role") or "viewer"),
                success=False,
            )
            raise HTTPException(
                status_code=403,
                detail="This Google account is not authorized for this invitation.",
            )

    def _primary_administrator_document(
        self,
        identity: VerifiedGoogleIdentity,
        now: datetime,
    ) -> dict:
        identifier = hashlib.sha256(identity.email.encode("utf-8")).hexdigest()[:32]
        invitation_id = f"google-admin-{identifier}"
        existing = self.store.get_invitation(invitation_id)
        document = {
            "_id": invitation_id,
            "guest_name": identity.display_name or "Administrator",
            "authorized_email": identity.email,
            "role": "admin",
            "max_active_sessions": 1,
            "token_hash": f"primary-admin:{identifier}",
            "token_version": 0,
            "status": "claimed",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(days=36_500),
            "last_access_at": None,
            "claimed_at": now,
            "claimed_subject": identity.subject,
            "claimed_email": identity.email,
            "google_display_name": identity.display_name,
            "google_hosted_domain": identity.hosted_domain,
            "token_consumed_at": now,
            "revoked_at": None,
            "primary_administrator": True,
        }
        stored = self.store.upsert_primary_administrator(document)
        if existing is None:
            return stored
        if (
            str(stored.get("claimed_subject") or "") != identity.subject
            or normalize_email(stored.get("claimed_email") or "") != identity.email
        ):
            raise HTTPException(
                status_code=403,
                detail="The primary administrator email is already linked to another Google identity.",
            )
        return stored

    def _direct_access_document(
        self,
        identity: VerifiedGoogleIdentity,
        request: Request,
        now: datetime,
    ) -> dict:
        matches = self.store.find_claimed_invitations(
            identity.subject,
            identity.email,
            now,
        )
        if not matches and identity.email == self.settings.admin_google_email:
            primary = self._primary_administrator_document(identity, now)
            matches = [primary]
            self._record_log(
                event="primary_google_administrator_verified",
                request=request,
                invitation_id=primary["_id"],
                guest_name=primary.get("guest_name"),
                identity_email=identity.email,
                role="admin",
            )

        active = [item for item in matches if _authorization_status(item, now) == "claimed"]
        if not active:
            self._record_log(
                event="google_direct_login_denied",
                request=request,
                identity_email=identity.email,
                success=False,
            )
            raise HTTPException(
                status_code=403,
                detail="This Google account does not have active access to Market Cycle Trader.",
            )

        role_priority = {"viewer": 1, "trader": 2, "admin": 3}
        return max(
            active,
            key=lambda item: (
                role_priority.get(str(item.get("role") or "viewer"), 0),
                as_utc(item.get("created_at")) or now,
            ),
        )

    def _create_identity_bound_session(
        self,
        document: dict,
        identity: VerifiedGoogleIdentity,
        request: Request,
        now: datetime,
    ) -> dict:
        invitation_expires_at = as_utc(document.get("expires_at"))
        if invitation_expires_at is None or invitation_expires_at <= now:
            raise HTTPException(status_code=401, detail="Access expired or revoked.")
        role = str(document.get("role") or "viewer")
        if role not in {"viewer", "trader", "admin"}:
            raise HTTPException(status_code=401, detail="Invalid access role.")
        absolute_lifetime = self.settings.session_max_age_for_role(role)
        idle_lifetime = self.settings.session_idle_for_role(role)
        session_expires_at = min(
            invitation_expires_at,
            now + timedelta(seconds=absolute_lifetime),
        )
        idle_expires_at = min(
            session_expires_at,
            now + timedelta(seconds=idle_lifetime),
        )
        session = {
            "_id": str(uuid.uuid4()),
            "invitation_id": document["_id"],
            "role": role,
            "display_name": document.get("guest_name") or identity.display_name or "User",
            "identity_subject": identity.subject,
            "identity_email": identity.email,
            "created_at": now,
            "last_activity_at": now,
            "expires_at": session_expires_at,
            "idle_expires_at": idle_expires_at,
            "revoked": False,
        }
        max_sessions = int(
            document.get("max_active_sessions")
            or (1 if role in {"trader", "admin"} else 2)
        )
        terminated = self.store.create_limited_session(session, max_sessions, now)
        self.store.update_invitation(
            document["_id"],
            {"last_access_at": now, "updated_at": now},
        )
        if terminated:
            self._record_log(
                event="older_session_replaced",
                request=request,
                invitation_id=document["_id"],
                guest_name=document.get("guest_name"),
                identity_email=identity.email,
                role=role,
            )
        self._record_log(
            event="google_access_granted",
            request=request,
            invitation_id=document["_id"],
            guest_name=document.get("guest_name"),
            identity_email=identity.email,
            role=role,
        )
        return session

    def create_google_session(self, payload: GoogleAccessRequest, request: Request) -> dict:
        identity = self.identity_verifier.verify(payload.credential)
        now = utc_now()

        if not payload.invitation_id:
            if payload.token:
                raise HTTPException(status_code=401, detail="Invalid invitation locator.")
            document = self._direct_access_document(identity, request, now)
            return self._create_identity_bound_session(document, identity, request, now)

        document = self.store.get_invitation(payload.invitation_id)
        if not document:
            raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")
        state = _authorization_status(document)
        if state == "legacy_unverified":
            raise HTTPException(status_code=410, detail="A new identity-verified invitation is required.")
        if state in {"revoked", "expired", "blocked"}:
            raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")

        self._verify_invited_identity(document, identity, request)
        if state == "pending_verification":
            if not payload.token:
                raise HTTPException(
                    status_code=401,
                    detail="Open the complete invitation link generated in Administration.",
                )
            expected_digest = token_digest(payload.token, document["_id"])
            if not hmac.compare_digest(
                str(document.get("token_hash") or ""),
                expected_digest,
            ):
                raise HTTPException(status_code=401, detail="Invalid or expired access invitation.")
            claimed = self.store.claim_invitation(
                document["_id"],
                expected_digest,
                now,
                {
                    "status": "claimed",
                    "claimed_at": now,
                    "claimed_subject": identity.subject,
                    "claimed_email": identity.email,
                    "google_display_name": identity.display_name,
                    "google_hosted_domain": identity.hosted_domain,
                    "token_consumed_at": now,
                    "token_hash": token_digest(_new_access_token(), document["_id"]),
                    "updated_at": now,
                },
            )
            if not claimed:
                raise HTTPException(
                    status_code=409,
                    detail="This invitation was already claimed or replaced.",
                )
            document = claimed
            self._record_log(
                event="google_identity_claimed",
                request=request,
                invitation_id=document["_id"],
                guest_name=document.get("guest_name"),
                identity_email=identity.email,
                role=str(document.get("role") or "viewer"),
            )
        else:
            claimed_subject = str(document.get("claimed_subject") or "")
            claimed_email = normalize_email(document.get("claimed_email") or "")
            if claimed_subject != identity.subject or claimed_email != identity.email:
                self._record_log(
                    event="claimed_identity_rejected",
                    request=request,
                    invitation_id=document.get("_id"),
                    guest_name=document.get("guest_name"),
                    identity_email=identity.email,
                    role=str(document.get("role") or "viewer"),
                    success=False,
                )
                raise HTTPException(
                    status_code=403,
                    detail="This invitation is already linked to another Google account.",
                )

        return self._create_identity_bound_session(document, identity, request, now)

    def validate_access_session(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        now = utc_now()
        session_expires_at = as_utc(session.get("expires_at")) if session else None
        idle_expires_at = as_utc(session.get("idle_expires_at")) if session else None
        if session and idle_expires_at is None and session_expires_at is not None:
            last_activity = as_utc(session.get("last_activity_at")) or as_utc(session.get("created_at")) or now
            idle_expires_at = min(
                session_expires_at,
                last_activity + timedelta(seconds=get_settings().session_idle_for_role(str(session.get("role") or "viewer"))),
            )
        if (
            not session
            or session.get("revoked")
            or session_expires_at is None
            or session_expires_at <= now
            or idle_expires_at is None
            or idle_expires_at <= now
        ):
            if session and not session.get("revoked") and hasattr(self.store, "revoke_session"):
                self.store.revoke_session(session_id)
            raise HTTPException(status_code=401, detail="Identity-bound session expired or revoked.")
        invitation = self.store.get_invitation(session["invitation_id"])
        if not invitation or _authorization_status(invitation, now) != "claimed":
            raise HTTPException(status_code=401, detail="Identity-bound access expired or revoked.")
        if (
            str(session.get("identity_subject") or "")
            != str(invitation.get("claimed_subject") or "")
            or normalize_email(session.get("identity_email") or "")
            != normalize_email(invitation.get("claimed_email") or "")
            or str(session.get("role") or "viewer")
            != str(invitation.get("role") or "viewer")
        ):
            raise HTTPException(status_code=401, detail="Google identity access is no longer valid.")
        next_idle_expiration = min(
            session_expires_at,
            now + timedelta(seconds=get_settings().session_idle_for_role(str(session.get("role") or "viewer"))),
        )
        touched = self.store.touch_session(session_id, now, next_idle_expiration) if hasattr(self.store, "touch_session") else None
        if touched:
            session = touched
        session["expires_at"] = session_expires_at
        session["idle_expires_at"] = next_idle_expiration
        session["last_activity_at"] = now
        session["created_at"] = as_utc(session.get("created_at")) or now
        return session

    def validate_guest_session(self, session_id: str) -> dict:
        return self.validate_access_session(session_id)

    def validate_viewer_session(self, session_id: str) -> dict:
        return self.validate_access_session(session_id)

    def revoke_access_session(self, session_id: str) -> None:
        self.store.revoke_session(session_id)

    def revoke_guest_session(self, session_id: str) -> None:
        self.revoke_access_session(session_id)

    def revoke_viewer_session(self, session_id: str) -> None:
        self.revoke_access_session(session_id)

    def list_logs(self, limit: int) -> list[AccessLogResponse]:
        return [
            AccessLogResponse(
                id=item["_id"],
                event=item["event"],
                invitation_id=item.get("invitation_id"),
                guest_name=item.get("guest_name"),
                identity_email=item.get("identity_email"),
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
