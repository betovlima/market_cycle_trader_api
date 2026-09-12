from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_contextual_marginal_signature_v1 as base  # noqa: E402
import research_contextual_marginal_signature_v101 as research  # noqa: E402


class ContextualMarginalSignatureV101Tests(unittest.TestCase):
    def test_horizon_end_accepts_tz_naive_calendar(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=80, freq="B")
        decision = pd.Timestamp("2024-01-15", tz="UTC")
        end = research._horizon_end(sessions, decision, 10)
        normalized = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True))
        position = int(normalized.searchsorted(decision, side="left"))
        self.assertEqual(end, pd.Timestamp(normalized[position + 9]))
        self.assertIsNotNone(end.tzinfo)

    def test_horizon_end_accepts_tz_aware_calendar(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=80, freq="B", tz="UTC")
        decision = pd.Timestamp("2024-01-15", tz="UTC")
        end = research._horizon_end(sessions, decision, 10)
        position = int(sessions.searchsorted(decision, side="left"))
        self.assertEqual(end, pd.Timestamp(sessions[position + 9]))

    def test_install_v101_replaces_only_calendar_horizon_resolver_and_version(self) -> None:
        original = base._horizon_end
        original_version = base.SCRIPT_VERSION
        try:
            research.install_v101()
            self.assertIs(base._horizon_end, research._horizon_end)
            self.assertEqual(base.SCRIPT_VERSION, "contextual-marginal-signature-v1.0.1")
        finally:
            base._horizon_end = original
            base.SCRIPT_VERSION = original_version


if __name__ == "__main__":
    unittest.main()
