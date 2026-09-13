from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v1111 as storage
import research_contextual_marginal_signature_v1112 as progress
import research_contextual_marginal_signature_v1113 as sound

base = storage.base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.11.4"
EXPERIMENT_NAME = base.EXPERIMENT_NAME
HEARTBEAT_SECONDS = 30.0

_ORIGINAL_LOG = base._log
_CURRENT_REPLAY_LABEL = "startup"
_STATE_PATH: Path | None = None
_FATAL_LOG_PATH: Path | None = None
_FATAL_HANDLE: Any | None = None
_STARTED_MONOTONIC = time.monotonic()


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sidecar_paths(output_dir: Path) -> tuple[Path, Path]:
    resolved = output_dir.resolve()
    parent = resolved.parent
    stem = resolved.name
    return parent / f"{stem}.run_state.json", parent / f"{stem}.fatal.log"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_state(*, status: str, label: str | None = None, elapsed: float | None = None, event: str | None = None) -> None:
    if _STATE_PATH is None:
        return
    payload = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": status,
        "pid": os.getpid(),
        "updated_utc": _utc_now(),
        "process_elapsed_seconds": round(time.monotonic() - _STARTED_MONOTONIC, 3),
        "current_replay": label or _CURRENT_REPLAY_LABEL,
        "replay_elapsed_seconds": None if elapsed is None else round(float(elapsed), 3),
        "last_event": event,
        "fatal_log": str(_FATAL_LOG_PATH) if _FATAL_LOG_PATH is not None else None,
    }
    _atomic_json(_STATE_PATH, payload)


def _diagnostic_log(message: str) -> None:
    global _CURRENT_REPLAY_LABEL
    text = str(message)
    stripped = text.strip()
    if stripped.endswith("baseline trace") or stripped.endswith("challenger trace"):
        _CURRENT_REPLAY_LABEL = stripped
        _write_state(status="running", label=_CURRENT_REPLAY_LABEL, elapsed=0.0, event="replay_started")
    _ORIGINAL_LOG(text)


def _run_with_diagnostics(frames: dict[str, Any], request: Any):
    label = _CURRENT_REPLAY_LABEL
    started = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - started
            _write_state(status="running", label=label, elapsed=elapsed, event="heartbeat")
            _ORIGINAL_LOG(
                f"    [heartbeat] {label} still running; elapsed={elapsed:.0f}s; pid={os.getpid()}"
            )

    thread = threading.Thread(target=heartbeat, name="contextual-signature-heartbeat", daemon=True)
    thread.start()
    try:
        metrics, sessions, captured = progress._ORIGINAL_RUN_WITH_CAPTURE(frames, request)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    elapsed = time.monotonic() - started
    ending_capital = base.discovery._finite_number(metrics.get("ending_capital")) if isinstance(metrics, dict) else None
    capital_text = f" ending_capital={ending_capital:.6f}" if ending_capital is not None else ""
    session_count = len(sessions) if sessions is not None else 0
    _write_state(status="running", label=label, elapsed=elapsed, event="replay_completed")
    _ORIGINAL_LOG(
        f"    [done] {label}; elapsed={elapsed:.1f}s; sessions={session_count}; "
        f"backends={len(captured)};{capital_text}; pid={os.getpid()}"
    )
    return metrics, sessions, captured


def _install_exception_hook() -> None:
    original = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        _write_state(status="python_exception", event=f"{exc_type.__name__}: {exc}")
        if _FATAL_HANDLE is not None:
            try:
                _FATAL_HANDLE.write("\n=== Python exception ===\n")
                traceback.print_exception(exc_type, exc, tb, file=_FATAL_HANDLE)
                _FATAL_HANDLE.flush()
            except Exception:
                pass
        original(exc_type, exc, tb)

    sys.excepthook = hook


def _enable_fatal_diagnostics(output_dir: Path) -> None:
    global _STATE_PATH, _FATAL_LOG_PATH, _FATAL_HANDLE
    _STATE_PATH, _FATAL_LOG_PATH = _sidecar_paths(output_dir)
    _FATAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FATAL_HANDLE = _FATAL_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    _FATAL_HANDLE.write(
        f"\n=== {SCRIPT_VERSION} process start {_utc_now()} pid={os.getpid()} ===\n"
    )
    _FATAL_HANDLE.flush()
    faulthandler.enable(file=_FATAL_HANDLE, all_threads=True)
    _install_exception_hook()
    _write_state(status="starting", label="startup", elapsed=0.0, event="process_started")
    _ORIGINAL_LOG(
        f"[diagnostics] pid={os.getpid()} run_state={_STATE_PATH} fatal_log={_FATAL_LOG_PATH}"
    )


def main() -> int:
    args = base._parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    _enable_fatal_diagnostics(output_dir)

    base.SCRIPT_VERSION = SCRIPT_VERSION
    storage.SCRIPT_VERSION = SCRIPT_VERSION
    progress.SCRIPT_VERSION = SCRIPT_VERSION
    sound.SCRIPT_VERSION = SCRIPT_VERSION
    base._save_capture = storage._compact_save_capture
    base._log = _diagnostic_log
    base._run_with_capture = _run_with_diagnostics

    try:
        exit_code = int(base.main())
    except BaseException as exc:
        _write_state(status="python_exception", event=f"{type(exc).__name__}: {exc}")
        raise

    if exit_code == 0:
        _write_state(status="completed", event="process_completed")
        _ORIGINAL_LOG("[complete] Contextual path-attribution audit finished successfully.")
        mode = sound._play_completion_sound()
        _ORIGINAL_LOG(f"[complete] Completion sound mode={mode}")
    else:
        _write_state(status="failed_exit_code", event=f"exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
