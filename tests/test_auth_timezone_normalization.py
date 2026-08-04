from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_cycle_trader_api.auth.access_service import (
    _effective_status,
    as_utc,
    invitation_response,
)


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_as_utc_attaches_utc_to_naive_mongodb_datetime() -> None:
    raw = datetime(2026, 8, 3, 20, 0, 0)
    normalized = as_utc(raw)

    assert normalized is not None
    assert normalized.tzinfo is UTC
    assert normalized.hour == 20


def test_effective_status_accepts_naive_mongodb_expiration() -> None:
    now = _naive_utc_now()
    document = {
        "status": "active",
        "expires_at": now + timedelta(hours=1),
    }

    assert _effective_status(document, now=now) == "active"


def test_invitation_response_serializes_naive_mongodb_dates_as_utc() -> None:
    now = _naive_utc_now()
    response = invitation_response(
        {
            "_id": "invitation-1",
            "guest_name": "Viewer",
            "status": "active",
            "created_at": now,
            "expires_at": now + timedelta(hours=1),
            "last_access_at": now + timedelta(minutes=5),
            "revoked_at": None,
        }
    )

    assert response.status == "active"
    assert response.created_at.tzinfo is UTC
    assert response.expires_at.tzinfo is UTC
    assert response.last_access_at is not None
    assert response.last_access_at.tzinfo is UTC


class _NaiveMongoStore:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def list_invitations(self):
        return [
            {
                "_id": "invitation-1",
                "guest_name": "Viewer",
                "status": "active",
                "created_at": self.now,
                "expires_at": self.now + timedelta(hours=1),
                "last_access_at": None,
                "revoked_at": None,
            }
        ]

    def get_session(self, session_id: str):
        return {
            "_id": session_id,
            "invitation_id": "invitation-1",
            "role": "viewer",
            "display_name": "Viewer",
            "created_at": self.now,
            "expires_at": self.now + timedelta(hours=1),
            "revoked": False,
        }

    def get_invitation(self, invitation_id: str):
        return {
            "_id": invitation_id,
            "guest_name": "Viewer",
            "status": "active",
            "created_at": self.now,
            "expires_at": self.now + timedelta(hours=1),
        }


def test_list_invitations_accepts_naive_dates_from_mongodb() -> None:
    from market_cycle_trader_api.auth.access_service import AccessService

    now = _naive_utc_now()
    service = AccessService.__new__(AccessService)
    service.store = _NaiveMongoStore(now)

    items = service.list_invitations()

    assert len(items) == 1
    assert items[0].status == "active"
    assert items[0].expires_at.tzinfo is UTC


def test_validate_viewer_session_accepts_naive_dates_from_mongodb() -> None:
    from market_cycle_trader_api.auth.access_service import AccessService

    now = _naive_utc_now()
    service = AccessService.__new__(AccessService)
    service.store = _NaiveMongoStore(now)

    session = service.validate_viewer_session("session-1")

    assert session["expires_at"].tzinfo is UTC
    assert session["created_at"].tzinfo is UTC
