from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_cycle_trader_api.auth.config import AuthSettings
from market_cycle_trader_api.auth.security import SessionManager
from market_cycle_trader_api.schemas.access_admin import InvitationCreateRequest


def _settings() -> AuthSettings:
    return AuthSettings(
        admin_password="admin-password",
        admin_google_email="admin@example.com",
        session_secret="x" * 48,
        session_max_age_seconds=3600,
        viewer_session_max_age_seconds=43_200,
        viewer_session_idle_seconds=7_200,
        trader_session_max_age_seconds=28_800,
        trader_session_idle_seconds=3_600,
        admin_session_max_age_seconds=7_200,
        admin_session_idle_seconds=1_800,
        cookie_secure=False,
        cookie_samesite="lax",
        auth_storage="memory",
        mongo_url="",
        mongo_database="",
        frontend_base_url="http://localhost:5173",
        google_client_id="google-client-id",
    )


def test_invitation_role_defaults_to_viewer_and_accepts_trader_and_admin() -> None:
    viewer = InvitationCreateRequest(guest_name="Viewer", authorized_email="viewer@example.com", duration_seconds=3600)
    trader = InvitationCreateRequest(guest_name="Trader", authorized_email="trader@example.com", role="trader", duration_seconds=3600)
    admin = InvitationCreateRequest(guest_name="Administrator", authorized_email="admin@example.com", role="admin", duration_seconds=3600)
    assert viewer.role == "viewer"
    assert trader.role == "trader"
    assert admin.role == "admin"
    assert viewer.max_active_sessions == 2
    assert trader.max_active_sessions == 1
    assert admin.max_active_sessions == 1


def test_trader_session_gets_portfolio_scope() -> None:
    manager = SessionManager(_settings())
    identity = manager.create_guest_identity(
        {
            "_id": "session-1",
            "invitation_id": "invitation-1",
            "role": "trader",
            "display_name": "Portfolio Trader",
            "identity_subject": "google-trader",
            "identity_email": "trader@example.com",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }
    )
    assert identity.role == "trader"
    assert identity.can_view_portfolio is True
    assert "portfolio:read" in identity.scope.split()


def test_viewer_session_does_not_get_portfolio_scope() -> None:
    manager = SessionManager(_settings())
    identity = manager.create_guest_identity(
        {
            "_id": "session-2",
            "invitation_id": "invitation-2",
            "role": "viewer",
            "display_name": "Backtest Viewer",
            "identity_subject": "google-viewer",
            "identity_email": "viewer@example.com",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }
    )
    assert identity.role == "viewer"
    assert identity.can_view_portfolio is False
    assert "portfolio:read" not in identity.scope.split()


def test_google_administrator_session_gets_full_scope() -> None:
    manager = SessionManager(_settings())
    identity = manager.create_access_identity(
        {
            "_id": "session-admin",
            "invitation_id": "invitation-admin",
            "role": "admin",
            "display_name": "Google Administrator",
            "identity_subject": "google-admin",
            "identity_email": "admin@example.com",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }
    )
    assert identity.role == "admin"
    assert identity.is_admin is True
    assert identity.can_view_portfolio is True
    assert "admin:manage" in identity.scope.split()
