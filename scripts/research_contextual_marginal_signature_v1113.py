from __future__ import annotations

import os
import sys
import time
from typing import Callable

import research_contextual_marginal_signature_v1112 as progress

base = progress.base
storage = progress.storage

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.11.3"
EXPERIMENT_NAME = base.EXPERIMENT_NAME


def _play_completion_sound(
    *,
    platform_name: str | None = None,
    beep: Callable[[int, int], None] | None = None,
    terminal_write: Callable[[str], object] | None = None,
) -> str:
    """Play a short completion signal without adding third-party dependencies.

    Windows uses winsound.Beep for a clearly audible three-note sequence. Other
    platforms fall back to the terminal bell. Sound failures never invalidate a
    completed scientific run.
    """
    platform_name = platform_name or os.name

    if platform_name == "nt":
        try:
            if beep is None:
                import winsound

                beep = winsound.Beep
            for frequency, duration_ms in ((880, 160), (1175, 160), (1568, 280)):
                beep(frequency, duration_ms)
                time.sleep(0.04)
            return "windows_beep"
        except Exception:
            # Audio availability must never turn a successful experiment into a
            # failed one. Fall through to the terminal bell.
            pass

    try:
        writer = terminal_write or sys.stdout.write
        writer("\a")
        sys.stdout.flush()
        return "terminal_bell"
    except Exception:
        return "unavailable"


def main() -> int:
    # Preserve the v1.0.11 scientific protocol, v1.0.11.1 compact storage and
    # v1.0.11.2 heartbeat. This patch changes completion observability only.
    base.SCRIPT_VERSION = SCRIPT_VERSION
    storage.SCRIPT_VERSION = SCRIPT_VERSION
    progress.SCRIPT_VERSION = SCRIPT_VERSION

    exit_code = int(progress.main())
    if exit_code == 0:
        base._log("[complete] Contextual path-attribution audit finished successfully.")
        mode = _play_completion_sound()
        base._log(f"[complete] Completion sound mode={mode}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
