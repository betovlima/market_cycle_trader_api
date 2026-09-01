from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException

from market_cycle_trader_api.auth.access_store import get_access_store
from market_cycle_trader_api.auth.config import get_auth_settings
from market_cycle_trader_api.auth.reviewer_access_store import get_reviewer_access_store


def _normalize_code(value: str) -> str:
    return str(value or "").strip().upper()


def _code_digest(access_id: str, code: str) -> str:
    secret = get_auth_settings().session_secret.encode("utf-8")
    payload = f"reviewer-access-v1:{access_id}:{_normalize_code(code)}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _new_code() -> str:
    raw = base64.b32encode(secrets.token_bytes(5)).decode("ascii").rstrip("=")[:8]
    return f"MCT-{raw[:4]}-{raw[4:]}"


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ReviewerAccessService:
    def __init__(self) -> None:
        self.settings = get_auth_settings()
        self.store = get_reviewer_access_store()
        self.session_store = get_access_store()

    def ensure_storage(self) -> None:
        self.store.ensure_indexes()

    def _status(self, record: dict[str, Any], now: datetime | None = None) -> str:
        moment = now or datetime.now(UTC)
        if record.get("status") == "revoked" or record.get("revoked_at"):
            return "revoked"
        expires_at = _aware(record.get("expires_at"))
        if expires_at is None or expires_at <= moment:
            return "expired"
        return "active"

    def _public(self, record: dict[str, Any], *, include_secret: bool = False, access_code: str = "") -> dict[str, Any]:
        now = datetime.now(UTC)
        status = self._status(record, now)
        response = {
            "id": str(record["_id"]),
            "guest_name": str(record.get("guest_name") or "Reviewer"),
            "role": "reviewer",
            "status": status,
            "created_at": record["created_at"],
            "expires_at": record["expires_at"],
            "last_access_at": record.get("last_access_at"),
            "max_active_sessions": int(record.get("max_active_sessions") or 1),
            "active_sessions": self.session_store.count_active_sessions(str(record["_id"]), now),
            "revoked_at": record.get("revoked_at"),
        }
        if include_secret:
            response["access_url"] = f"{self.settings.frontend_base_url}/?reviewer={quote(str(record['_id']))}"
            response["access_code"] = access_code
        return response

    def _log(self, *, event: str, record: dict[str, Any] | None, success: bool, client_ip: str) -> None:
        self.session_store.record_log({
            "_id": f"reviewer-log-{uuid.uuid4().hex}",
            "event": event,
            "invitation_id": None if record is None else str(record.get("_id") or ""),
            "guest_name": None if record is None else record.get("guest_name"),
            "identity_email": None,
            "role": "reviewer",
            "success": bool(success),
            "client_ip": str(client_ip or "unknown"),
            "created_at": datetime.now(UTC),
        })

    def create_access(self, *, guest_name: str, duration_seconds: int, max_active_sessions: int, client_ip: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        access_id = f"reviewer-access-{uuid.uuid4().hex}"
        access_code = _new_code()
        document = {
            "_id": access_id,
            "guest_name": str(guest_name).strip(),
            "role": "reviewer",
            "status": "active",
            "code_hash": _code_digest(access_id, access_code),
            "code_version": 1,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(seconds=int(duration_seconds)),
            "last_access_at": None,
            "max_active_sessions": int(max_active_sessions),
            "revoked_at": None,
        }
        created = self.store.create(document)
        self._log(event="reviewer_access_created", record=created, success=True, client_ip=client_ip)
        return self._public(created, include_secret=True, access_code=access_code)

    def list_accesses(self) -> list[dict[str, Any]]:
        return [self._public(item) for item in self.store.list()]

    def preview(self, access_id: str) -> dict[str, Any]:
        record = self.store.get(access_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        status = self._status(record)
        if status != "active":
            raise HTTPException(status_code=403, detail=f"Reviewer access is {status}.")
        return {
            "access_id": str(record["_id"]),
            "guest_name": str(record.get("guest_name") or "Reviewer"),
            "role": "reviewer",
            "status": status,
            "expires_at": record["expires_at"],
            "max_active_sessions": int(record.get("max_active_sessions") or 1),
        }

    def create_session(self, *, access_id: str, code: str, client_ip: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        record = self.store.get(access_id)
        if record is None:
            self._log(event="reviewer_access_denied", record=None, success=False, client_ip=client_ip)
            raise HTTPException(status_code=401, detail="Invalid reviewer access.")
        if self._status(record, now) != "active":
            self._log(event="reviewer_access_denied", record=record, success=False, client_ip=client_ip)
            raise HTTPException(status_code=403, detail="Reviewer access is no longer active.")
        submitted_hash = _code_digest(access_id, code)
        if not hmac.compare_digest(str(record.get("code_hash") or ""), submitted_hash):
            self._log(event="reviewer_access_denied", record=record, success=False, client_ip=client_ip)
            raise HTTPException(status_code=401, detail="Invalid reviewer access code.")

        expires_at = _aware(record.get("expires_at"))
        if expires_at is None:
            raise HTTPException(status_code=403, detail="Reviewer access is no longer active.")
        session_lifetime = min(
            int(self.settings.session_max_age_for_role("reviewer")),
            max(1, int((expires_at - now).total_seconds())),
        )
        idle_lifetime = min(int(self.settings.session_idle_for_role("reviewer")), session_lifetime)
        session = {
            "_id": f"reviewer-session-{uuid.uuid4().hex}",
            "invitation_id": access_id,
            "role": "reviewer",
            "display_name": str(record.get("guest_name") or "Reviewer"),
            "identity_subject": f"reviewer:{access_id}",
            "identity_email": None,
            "access_method": "reviewer_code",
            "created_at": now,
            "last_activity_at": now,
            "expires_at": now + timedelta(seconds=session_lifetime),
            "idle_expires_at": now + timedelta(seconds=idle_lifetime),
            "revoked": False,
            "client_ip": str(client_ip or "unknown"),
        }
        self.session_store.create_limited_session(
            session,
            max_active_sessions=int(record.get("max_active_sessions") or 1),
            now=now,
        )
        self.store.update(access_id, {"last_access_at": now, "updated_at": now})
        self._log(event="reviewer_access_granted", record=record, success=True, client_ip=client_ip)
        return session

    def validate_session(self, session_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        session = self.session_store.get_session(session_id)
        if (
            session is None
            or bool(session.get("revoked"))
            or str(session.get("role") or "") != "reviewer"
            or str(session.get("access_method") or "") != "reviewer_code"
        ):
            raise HTTPException(status_code=401, detail="Reviewer session is no longer active.")
        expires_at = _aware(session.get("expires_at"))
        idle_expires_at = _aware(session.get("idle_expires_at"))
        if expires_at is None or expires_at <= now or idle_expires_at is None or idle_expires_at <= now:
            self.session_store.revoke_session(session_id)
            raise HTTPException(status_code=401, detail="Reviewer session expired.")
        access_id = str(session.get("invitation_id") or "")
        record = self.store.get(access_id)
        if record is None or self._status(record, now) != "active":
            self.session_store.revoke_session(session_id)
            raise HTTPException(status_code=401, detail="Reviewer access is no longer active.")
        refreshed_idle = min(
            expires_at,
            now + timedelta(seconds=int(self.settings.session_idle_for_role("reviewer"))),
        )
        updated = self.session_store.touch_session(session_id, now, refreshed_idle)
        if updated is None:
            raise HTTPException(status_code=401, detail="Reviewer session is no longer active.")
        return updated

    def regenerate_code(self, access_id: str, *, duration_seconds: int, client_ip: str) -> dict[str, Any]:
        record = self.store.get(access_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        if self._status(record) == "revoked":
            raise HTTPException(status_code=409, detail="Revoked reviewer access cannot be regenerated.")
        now = datetime.now(UTC)
        access_code = _new_code()
        self.session_store.terminate_sessions(access_id)
        updated = self.store.update(access_id, {
            "status": "active",
            "code_hash": _code_digest(access_id, access_code),
            "code_version": int(record.get("code_version") or 0) + 1,
            "expires_at": now + timedelta(seconds=int(duration_seconds)),
            "updated_at": now,
            "revoked_at": None,
        })
        if updated is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        self._log(event="reviewer_access_code_regenerated", record=updated, success=True, client_ip=client_ip)
        return self._public(updated, include_secret=True, access_code=access_code)

    def terminate_sessions(self, access_id: str, *, client_ip: str) -> int:
        record = self.store.get(access_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        count = self.session_store.terminate_sessions(access_id)
        self._log(event="reviewer_sessions_terminated", record=record, success=True, client_ip=client_ip)
        return int(count)

    def revoke_access(self, access_id: str, *, client_ip: str) -> dict[str, Any]:
        record = self.store.get(access_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        now = datetime.now(UTC)
        self.session_store.terminate_sessions(access_id)
        updated = self.store.update(access_id, {"status": "revoked", "revoked_at": now, "updated_at": now})
        if updated is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        self._log(event="reviewer_access_revoked", record=updated, success=True, client_ip=client_ip)
        return self._public(updated)

    def delete_access(self, access_id: str, *, client_ip: str) -> None:
        record = self.store.get(access_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        if self._status(record) == "active":
            raise HTTPException(status_code=409, detail="Revoke or let reviewer access expire before deleting it.")
        self.session_store.terminate_sessions(access_id)
        if not self.store.delete(access_id):
            raise HTTPException(status_code=404, detail="Reviewer access was not found.")
        self._log(event="reviewer_access_deleted", record=record, success=True, client_ip=client_ip)

    def revoke_session(self, session_id: str) -> None:
        self.session_store.revoke_session(session_id)


@lru_cache(maxsize=1)
def get_reviewer_access_service() -> ReviewerAccessService:
    return ReviewerAccessService()
