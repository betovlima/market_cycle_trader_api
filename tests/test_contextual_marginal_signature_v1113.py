from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1113 as research  # noqa: E402


class ContextualMarginalSignatureV1113Tests(unittest.TestCase):
    def test_windows_completion_sound_uses_three_notes(self) -> None:
        calls: list[tuple[int, int]] = []
        mode = research._play_completion_sound(
            platform_name="nt",
            beep=lambda frequency, duration: calls.append((frequency, duration)),
        )
        self.assertEqual(mode, "windows_beep")
        self.assertEqual(calls, [(880, 160), (1175, 160), (1568, 280)])

    def test_non_windows_falls_back_to_terminal_bell(self) -> None:
        writes: list[str] = []
        mode = research._play_completion_sound(
            platform_name="posix",
            terminal_write=lambda value: writes.append(value),
        )
        self.assertEqual(mode, "terminal_bell")
        self.assertEqual(writes, ["\a"])


if __name__ == "__main__":
    unittest.main()
