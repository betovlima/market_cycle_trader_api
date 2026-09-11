from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_sequential_exact_marginal_search_v101 as profiler  # noqa: E402


class SequentialExactMarginalSearchV101Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_replay = profiler._ORIGINAL_REPLAY
        self.original_dir = profiler._PROFILE_DIR
        profiler._PROFILE_ROWS = []
        profiler._PROFILE_COUNTER = 0

    def tearDown(self) -> None:
        profiler._ORIGINAL_REPLAY = self.original_replay
        profiler._PROFILE_DIR = self.original_dir

    def test_profiled_replay_preserves_result_and_writes_profile_artifacts(self) -> None:
        expected = ({"ending_capital": 123.0}, pd.DatetimeIndex([pd.Timestamp("2026-01-02")]))

        def fake_replay(*args, **kwargs):
            total = 0
            for value in range(500):
                total += value
            self.assertGreater(total, 0)
            return expected

        profiler._ORIGINAL_REPLAY = fake_replay
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir = Path(temporary)
            profiler._PROFILE_DIR = profile_dir
            frames = {
                "AAA": pd.DataFrame({"close": [1.0]}),
                "BBB": pd.DataFrame({"close": [2.0]}),
            }
            request = SimpleNamespace(strategy_mode="TEST_MODE")
            result = profiler._profiled_run_rotation_replay(frames, request)

            self.assertEqual(result, expected)
            index = pd.read_csv(profile_dir / "replay_profile_index.csv")
            self.assertEqual(len(index), 1)
            self.assertEqual(int(index.iloc[0]["asset_count"]), 2)
            self.assertEqual(index.iloc[0]["strategy_mode"], "TEST_MODE")
            self.assertTrue((profile_dir / index.iloc[0]["profile_file"]).exists())
            self.assertTrue((profile_dir / index.iloc[0]["top_functions_file"]).exists())

    def test_profiled_replay_records_error_and_reraises(self) -> None:
        def failing_replay(*args, **kwargs):
            raise RuntimeError("expected failure")

        profiler._ORIGINAL_REPLAY = failing_replay
        with tempfile.TemporaryDirectory() as temporary:
            profiler._PROFILE_DIR = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                profiler._profiled_run_rotation_replay({"AAA": pd.DataFrame({"close": [1.0]})}, SimpleNamespace())
            index = pd.read_csv(Path(temporary) / "replay_profile_index.csv")
            self.assertIn("expected failure", str(index.iloc[0]["error"]))


if __name__ == "__main__":
    unittest.main()
