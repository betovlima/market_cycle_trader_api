from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import HTTPException, Request

from market_cycle_trader_api.auth.access_service import AccessService, token_digest
from market_cycle_trader_api.auth.access_store import get_access_store
from market_cycle_trader_api.auth.config import get_auth_settings
from market_cycle_trader_api.auth.google_identity import VerifiedGoogleIdentity
from market_cycle_trader_api.schemas.access_admin import InvitationCreateRequest
from market_cycle_trader_api.schemas.auth import AccessPreviewRequest, GoogleAccessRequest


class FakeGoogleVerifier:
    def __init__(self, identity: VerifiedGoogleIdentity) -> None:
        self.identity = identity

    def verify(self, credential: str) -> VerifiedGoogleIdentity:
        assert credential
        return self.identity


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/access",
            "headers": [(b"user-agent", b"pytest-browser")],
            "client": ("127.0.0.1", 50000),
        }
    )


def _locator(access_url: str) -> tuple[str, str]:
    values = parse_qs(urlsplit(access_url).fragment)
    return values["invitation"][0], values["token"][0]


@pytest.fixture()
def service(monkeypatch) -> tuple[AccessService, FakeGoogleVerifier]:
    monkeypatch.setenv("TRADER_ADMIN_PASSWORD", "admin-password")
    monkeypatch.setenv("TRADER_ADMIN_GOOGLE_EMAIL", "admin@example.com")
    monkeypatch.setenv("TRADER_SESSION_SECRET", "s" * 64)
    monkeypatch.setenv("TRADER_SESSION_MAX_AGE_SECONDS", "3600")
    monkeypatch.setenv("TRADER_COOKIE_SECURE", "false")
    monkeypatch.setenv("TRADER_COOKIE_SAMESITE", "lax")
    monkeypatch.setenv("TRADER_AUTH_STORAGE", "memory")
    monkeypatch.setenv("TRADER_FRONTEND_BASE_URL", "http://localhost:5173")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-client-id")
    get_auth_settings.cache_clear()
    get_access_store.cache_clear()
    verifier = FakeGoogleVerifier(
        VerifiedGoogleIdentity(
            subject="google-subject-b",
            email="user-b@example.com",
            display_name="User B",
        )
    )
    value = AccessService(identity_verifier=verifier)
    value.ensure_storage()
    yield value, verifier
    get_access_store.cache_clear()
    get_auth_settings.cache_clear()


def test_wrong_google_email_cannot_claim_another_users_token(service) -> None:
    access_service, verifier = service
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="User B",
            authorized_email="user-b@example.com",
            role="trader",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)

    verifier.identity = VerifiedGoogleIdentity(
        subject="google-subject-a",
        email="user-a@example.com",
        display_name="User A",
    )
    with pytest.raises(HTTPException) as error:
        access_service.create_google_session(
            GoogleAccessRequest(
                invitation_id=invitation_id,
                token=token,
                credential="a" * 120,
            ),
            _request(),
        )
    assert error.value.status_code == 403
    assert access_service.store.get_invitation(invitation_id)["status"] == "pending_verification"


def test_token_is_consumed_and_claim_is_bound_to_google_subject(service) -> None:
    access_service, verifier = service
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="User B",
            authorized_email="user-b@example.com",
            role="trader",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)
    original_digest = access_service.store.get_invitation(invitation_id)["token_hash"]
    assert original_digest == token_digest(token, invitation_id)
    assert original_digest != token_digest(token)

    first_session = access_service.create_google_session(
        GoogleAccessRequest(
            invitation_id=invitation_id,
            token=token,
            credential="b" * 120,
        ),
        _request(),
    )
    claimed = access_service.store.get_invitation(invitation_id)
    assert claimed["status"] == "claimed"
    assert claimed["claimed_subject"] == "google-subject-b"
    assert claimed["claimed_email"] == "user-b@example.com"
    assert claimed["token_hash"] != original_digest
    assert first_session["identity_subject"] == "google-subject-b"

    verifier.identity = VerifiedGoogleIdentity(
        subject="different-google-subject",
        email="user-b@example.com",
        display_name="Different account",
    )
    with pytest.raises(HTTPException) as error:
        access_service.create_google_session(
            GoogleAccessRequest(
                invitation_id=invitation_id,
                credential="c" * 120,
            ),
            _request(),
        )
    assert error.value.status_code == 403

    verifier.identity = VerifiedGoogleIdentity(
        subject="google-subject-b",
        email="user-b@example.com",
        display_name="User B",
    )
    second_session = access_service.create_google_session(
        GoogleAccessRequest(
            invitation_id=invitation_id,
            credential="d" * 120,
        ),
        _request(),
    )
    assert second_session["identity_subject"] == "google-subject-b"
    assert access_service.store.get_session(first_session["_id"])["revoked"] is True
    assert access_service.store.count_active_sessions(invitation_id, second_session["created_at"]) == 1


def test_viewer_default_allows_two_active_sessions(service) -> None:
    access_service, verifier = service
    verifier.identity = VerifiedGoogleIdentity(
        subject="viewer-subject",
        email="viewer@example.com",
        display_name="Viewer",
    )
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="Viewer",
            authorized_email="viewer@example.com",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)
    first = access_service.create_google_session(
        GoogleAccessRequest(invitation_id=invitation_id, token=token, credential="e" * 120),
        _request(),
    )
    second = access_service.create_google_session(
        GoogleAccessRequest(invitation_id=invitation_id, credential="f" * 120),
        _request(),
    )
    third = access_service.create_google_session(
        GoogleAccessRequest(invitation_id=invitation_id, credential="g" * 120),
        _request(),
    )
    assert access_service.store.get_session(first["_id"])["revoked"] is True
    assert access_service.store.get_session(second["_id"])["revoked"] is False
    assert access_service.store.get_session(third["_id"])["revoked"] is False
    assert access_service.store.count_active_sessions(invitation_id, third["created_at"]) == 2


def test_preview_masks_email_and_legacy_access_is_disabled(service) -> None:
    access_service, _ = service
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="Masked User",
            authorized_email="masked.user@example.com",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)
    preview = access_service.preview_access(
        AccessPreviewRequest(invitation_id=invitation_id, token=token),
        _request(),
    )
    assert preview.masked_email.startswith("ma")
    assert preview.masked_email.endswith("@example.com")
    assert preview.status == "pending_verification"

    legacy_id = "legacy-invitation"
    legacy_hash = "legacy-token-hash"
    access_service.store.create_invitation(
        {
            "_id": legacy_id,
            "guest_name": "Legacy",
            "role": "viewer",
            "token_hash": legacy_hash,
            "status": "active",
            "created_at": preview.expires_at,
            "updated_at": preview.expires_at,
            "expires_at": preview.expires_at,
        }
    )
    access_service.ensure_storage()
    migrated = access_service.store.get_invitation(legacy_id)
    assert migrated["status"] == "legacy_unverified"
    assert migrated["token_hash"] != legacy_hash
    with pytest.raises(HTTPException) as error:
        access_service.preview_access(AccessPreviewRequest(invitation_id=legacy_id), _request())
    assert error.value.status_code == 410


def test_claimed_identity_can_sign_in_without_reusing_invitation_link(service) -> None:
    access_service, verifier = service
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="Direct Trader",
            authorized_email="user-b@example.com",
            role="trader",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)
    claimed = access_service.create_google_session(
        GoogleAccessRequest(
            invitation_id=invitation_id,
            token=token,
            credential="h" * 120,
        ),
        _request(),
    )

    direct = access_service.create_google_session(
        GoogleAccessRequest(credential="i" * 120),
        _request(),
    )

    assert direct["role"] == "trader"
    assert direct["identity_subject"] == verifier.identity.subject
    assert access_service.store.get_session(claimed["_id"])["revoked"] is True


def test_administrator_invitation_creates_google_admin_access(service) -> None:
    access_service, verifier = service
    invitation = access_service.create_invitation(
        InvitationCreateRequest(
            guest_name="Invited Administrator",
            authorized_email="user-b@example.com",
            role="admin",
            duration_seconds=3600,
        ),
        _request(),
    )
    invitation_id, token = _locator(invitation.access_url)

    session = access_service.create_google_session(
        GoogleAccessRequest(
            invitation_id=invitation_id,
            token=token,
            credential="j" * 120,
        ),
        _request(),
    )

    assert session["role"] == "admin"
    assert session["identity_subject"] == verifier.identity.subject
    assert access_service.validate_access_session(session["_id"])["role"] == "admin"


def test_configured_google_email_bootstraps_primary_administrator(service) -> None:
    access_service, verifier = service
    verifier.identity = VerifiedGoogleIdentity(
        subject="primary-admin-subject",
        email="admin@example.com",
        display_name="Primary Administrator",
    )

    session = access_service.create_google_session(
        GoogleAccessRequest(credential="k" * 120),
        _request(),
    )
    document = access_service.store.get_invitation(session["invitation_id"])

    assert session["role"] == "admin"
    assert document["primary_administrator"] is True
    assert document["claimed_email"] == "admin@example.com"
    with pytest.raises(HTTPException) as error:
        access_service.revoke_invitation(document["_id"], _request())
    assert error.value.status_code == 409


def test_unregistered_google_account_cannot_use_direct_login(service) -> None:
    access_service, verifier = service
    verifier.identity = VerifiedGoogleIdentity(
        subject="unknown-subject",
        email="unknown@example.com",
        display_name="Unknown User",
    )

    with pytest.raises(HTTPException) as error:
        access_service.create_google_session(
            GoogleAccessRequest(credential="l" * 120),
            _request(),
        )

    assert error.value.status_code == 403
