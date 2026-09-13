from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1141_snapshot as research  # noqa: E402


class ContextualMarginalSignatureV1141Tests(unittest.TestCase):
    def _table(self, symbol: str, close: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": pd.to_datetime(["2026-01-02"], utc=True),
                "open": [close],
                "high": [close],
                "low": [close],
                "close": [close],
                "volume": [100.0],
            }
        )

    def test_merge_preserves_existing_and_adds_missing_symbol(self) -> None:
        existing = self._table("A", 10.0)
        additions = self._table("B", 20.0)
        result = research._merge_snapshot(existing, additions, ["A", "B"])
        self.assertEqual(set(result["symbol"]), {"A", "B"})
        self.assertEqual(float(result.loc[result["symbol"] == "A", "close"].iloc[0]), 10.0)

    def test_merge_refuses_to_overwrite_frozen_symbol(self) -> None:
        existing = self._table("A", 10.0)
        additions = self._table("A", 11.0)
        with self.assertRaisesRegex(RuntimeError, "overwrite frozen symbols"):
            research._merge_snapshot(existing, additions, ["A"])

    def test_merge_rejects_missing_required_symbol(self) -> None:
        existing = self._table("A", 10.0)
        additions = existing.iloc[0:0].copy()
        with self.assertRaisesRegex(RuntimeError, "still does not provide"):
            research._merge_snapshot(existing, additions, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
