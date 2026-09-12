from __future__ import annotations

import threading
import time
from typing import Any

import research_contextual_marginal_signature_v1111 as storage

base = storage.base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.11.2"
EXPERIMENT_NAME = base.EXPERIMENT_NAME
HEARTBEAT_SECONDS = 30.0

_ORIGINAL_LOG = base._log
_ORIGINAL_RUN_WITH_CAPTURE = base._run_with_capture
_CURRENT_REPLAY_LABEL = "replay"


def _progress_log(message: str) -> None:
    global _CURRENT_REPLAY_LABEL
    text = str(message)
    stripped = text.strip()
    if stripped.endswith("baseline trace") or stripped.endswith("challenger trace"):
        _CURRENT_REPLAY_LABEL = stripped
    _ORIGINAL_LOG(text)


def _run_with_heartbeat(frames: dict[str, Any], request: Any):
    label = _CURRENT_REPLAY_LABEL
    started = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - started
            _ORIGINAL_LOG(f"    [heartbeat] {label} still running; elapsed={elapsed:.0f}s")

    thread = threading.Thread(target=heartbeat, name="contextual-signature-heartbeat", daemon=True)
    thread.start()
    try:
        metrics, sessions, captured = _ORIGINAL_RUN_WITH_CAPTURE(frames, request)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    elapsed = time.monotonic() - started
    ending_capital = base.discovery._finite_number(metrics.get("ending_capital")) if isinstance(metrics, dict) else None
    capital_text = f" ending_capital={ending_capital:.6f}" if ending_capital is not None else ""
    session_count = len(sessions) if sessions is not None else 0
    _ORIGINAL_LOG(
        f"    [done] {label}; elapsed={elapsed:.1f}s; sessions={session_count}; "
        f"backends={len(captured)};{capital_text}"
    )
    return metrics, sessions, captured


def main() -> int:
    # Preserve the v1.0.11 scientific protocol and v1.0.11.1 compact storage.
    # This version changes observability only: replay inputs, outputs and ordering
    # remain unchanged.
    base.SCRIPT_VERSION = SCRIPT_VERSION
    storage.SCRIPT_VERSION = SCRIPT_VERSION
    base._save_capture = storage._compact_save_capture
    base._log = _progress_log
    base._run_with_capture = _run_with_heartbeat
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
