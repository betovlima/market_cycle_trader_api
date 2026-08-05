from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_cycle_trader_api.auth.access_service import (
    AccessService,
    _authorization_status,
    as_utc,
)


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_as_utc_attaches_utc_to_naive_mongodb_datetime() -> None:
    raw = datetime(2026, 8, 3, 20, 0, 0)
    normalized = as_utc(raw)

    assert normalized is not None
    assert normalized.tzinfo is UTC
    assert normalized.hour == 20


def test_authorization_status_accepts_naive_mongodb_expiration() -> None:
    now = _naive_utc_now()
    document = {
        "status": "pending_verification",
        "authorized_email": "viewer@example.com",
        "expires_at": now + timedelta(hours=1),
    }

    assert _authorization_status(document, now=now) == "pending_verification"


class _NaiveMongoStore:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def list_invitations(self):
        return [self.get_invitation("invitation-1")]

    def count_active_sessions(self, invitation_id: str, now: datetime) -> int:
        return 1

    def get_session(self, session_id: str):
        return {
            "_id": session_id,
            "invitation_id": "invitation-1",
            "role": "viewer",
            "display_name": "Viewer",
            "identity_subject": "google-subject",
            "identity_email": "viewer@example.com",
            "created_at": self.now,
            "expires_at": self.now + timedelta(hours=1),
            "revoked": False,
        }

    def get_invitation(self, invitation_id: str):
        return {
            "_id": invitation_id,
            "guest_name": "Viewer",
            "authorized_email": "viewer@example.com",
            "role": "viewer",
            "status": "claimed",
            "claimed_subject": "google-subject",
            "claimed_email": "viewer@example.com",
            "claimed_at": self.now,
            "max_active_sessions": 2,
            "created_at": self.now,
            "expires_at": self.now + timedelta(hours=1),
            "last_access_at": self.now + timedelta(minutes=5),
            "revoked_at": None,
        }


def test_list_invitations_accepts_naive_dates_from_mongodb() -> None:
    now = _naive_utc_now()
    service = AccessService.__new__(AccessService)
    service.store = _NaiveMongoStore(now)

    items = service.list_invitations()

    assert len(items) == 1
    assert items[0].status == "active"
    assert items[0].expires_at.tzinfo is UTC
    assert items[0].claimed_at is not None
    assert items[0].claimed_at.tzinfo is UTC


def test_validate_guest_session_accepts_naive_dates_from_mongodb() -> None:
    now = _naive_utc_now()
    service = AccessService.__new__(AccessService)
    service.store = _NaiveMongoStore(now)

    session = service.validate_guest_session("session-1")

    assert session["expires_at"].tzinfo is UTC
    assert session["created_at"].tzinfo is UTC
