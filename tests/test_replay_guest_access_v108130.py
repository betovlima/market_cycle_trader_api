"""Regression guards for professor access (no other MCT screen or endpoint)."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.auth.replay_guest import replay_guest_request_allowed
from market_cycle_trader_api.schemas.access_admin import InvitationCreateRequest
from market_cycle_trader_api.main import create_app


class ProfessorReplayAccessTests(unittest.TestCase):
    def test_replay_guest_has_only_annual_replay_capability(self):
        capabilities = capabilities_for_role("replay_guest")
        self.assertEqual({key for key, enabled in capabilities.items() if enabled}, {"annual_replay.view"})
        for role in ("viewer", "reviewer", "trader", "admin"):
            self.assertTrue(capabilities_for_role(role)["annual_replay.view"])

    def test_guest_invitation_requires_authorized_google_email(self):
        invite = InvitationCreateRequest(
            guest_name="Professor",
            authorized_email="professor@example.edu",
            role="replay_guest",
            duration_seconds=3600,
        )
        self.assertEqual(invite.role, "replay_guest")
        self.assertEqual(invite.max_active_sessions, 2)

    def test_allowed_methods_and_exact_path_guard(self):
        self.assertTrue(replay_guest_request_allowed("GET", "/api/auth/session"))
        self.assertTrue(replay_guest_request_allowed("POST", "/api/auth/logout"))
        self.assertTrue(replay_guest_request_allowed("GET", "/api/analytics/backtests"))
        self.assertTrue(replay_guest_request_allowed("GET", "/api/analytics/backtests/job-123"))
        for method, path in (
            ("GET", "/api/dashboard/summary"),
            ("GET", "/api/jobs"),
            ("POST", "/api/jobs"),
            ("GET", "/api/analytics/backtests/job-123/rotation-period"),
            ("GET", "/api/analytics/portfolio"),
            ("GET", "/api/analytics/processings"),
            ("GET", "/api/admin/invitations"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
            ("POST", "/api/analytics/backtests"),
            ("DELETE", "/api/analytics/backtests/job-123"),
        ):
            with self.subTest(path=path, method=method):
                self.assertFalse(replay_guest_request_allowed(method, path))

    def test_guest_cookie_is_rejected_by_non_replay_route(self):
        manager = Mock()
        manager.decode_session_token.return_value = Mock(role="replay_guest")
        app = create_app()
        with patch("market_cycle_trader_api.main.get_session_manager", return_value=manager):
            client = TestClient(app)
            for path in (
                "/api/dashboard/summary",
                "/api/jobs",
                "/api/analytics/portfolio",
                "/api/analytics/backtests/demo/rotation-period",
            ):
                with self.subTest(path=path):
                    response = client.get(
                        path, cookies={"market_cycle_trader_session": "signed-session"}
                    )
                    self.assertEqual(response.status_code, 403)
            response = client.post(
                "/api/jobs",
                cookies={"market_cycle_trader_session": "signed-session"},
            )
            self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
