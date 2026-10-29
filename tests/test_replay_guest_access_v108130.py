"""Regression guards for professor access (no other MCT screen or endpoint)."""
from __future__ import annotations

import unittest


from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.auth.replay_guest import replay_guest_denied, replay_guest_request_allowed
from market_cycle_trader_api.schemas.access_admin import InvitationCreateRequest


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
        self.assertTrue(replay_guest_request_allowed("GET", "/api/analytics/backtests/job-123/files"))
        for filename in ("equity_history.csv", "rotations.csv", "decision_scores.csv",
                         "metrics.json", "analytics.json", "manifest.json", "package.zip"):
            self.assertTrue(replay_guest_request_allowed(
                "GET", "/api/analytics/backtests/job-123/files/" + filename
            ))
        for method, path in (
            ("GET", "/api/dashboard/summary"),
            ("GET", "/api/jobs"),
            ("POST", "/api/jobs"),
            ("GET", "/api/analytics/backtests/job-123/rotation-period"),
            ("GET", "/api/analytics/backtests/job-123/files/secret.txt"),
            ("GET", "/api/analytics/backtests/job-123/files/../../secrets"),
            ("GET", "/api/analytics/backtests/job-123/files/package.zip/extra"),
            ("POST", "/api/analytics/backtests/job-123/files"),
            ("DELETE", "/api/analytics/backtests/job-123/files/rotations.csv"),
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

    def test_only_guest_is_rejected_outside_replay(self):
        self.assertFalse(replay_guest_denied("viewer", "GET", "/api/dashboard/summary"))
        self.assertFalse(replay_guest_denied("admin", "POST", "/api/jobs"))
        self.assertFalse(replay_guest_denied("replay_guest", "GET", "/api/analytics/backtests"))
        self.assertFalse(replay_guest_denied("replay_guest", "GET", "/api/analytics/backtests/job-123"))
        self.assertTrue(replay_guest_denied("replay_guest", "GET", "/api/dashboard/summary"))
        self.assertTrue(replay_guest_denied("replay_guest", "POST", "/api/jobs"))
        self.assertTrue(replay_guest_denied("replay_guest", "GET", "/docs"))

if __name__ == "__main__":
    unittest.main()
