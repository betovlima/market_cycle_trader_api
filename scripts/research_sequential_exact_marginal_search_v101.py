from __future__ import annotations

import cProfile
import hashlib
import io
import pstats
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_sequential_exact_marginal_search as base  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "sequential-exact-marginal-search-v1.0.1"
_ORIGINAL_REPLAY = discovery._run_rotation_replay
_PROFILE_LOCK = threading.Lock()
_PROFILE_ROWS: list[dict[str, Any]] = []
_PROFILE_COUNTER = 0
_PROFILE_DIR: Path | None = None
_PROFILE_TOP = 50


def _utc_log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _argv_value(flag: str) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return str(sys.argv[index + 1]).strip() or None


def _resolve_profile_dir() -> Path:
    global _PROFILE_DIR
    if _PROFILE_DIR is not None:
        return _PROFILE_DIR
    output_dir = _argv_value("--output-dir")
    if output_dir:
        root = Path(output_dir).resolve()
    else:
        root = (PROJECT_ROOT / "research_output" / "sequential_exact_marginal_search_profile_v101").resolve()
    _PROFILE_DIR = root / "replay_profiles"
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return _PROFILE_DIR


def _profile_label(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[str], str | None]:
    frames = args[0] if args else kwargs.get("frames") or kwargs.get("bars_by_symbol") or {}
    symbols = sorted(str(symbol).strip().upper() for symbol in dict(frames or {}) if str(symbol).strip())
    request = args[1] if len(args) > 1 else kwargs.get("request")
    strategy_mode = str(getattr(request, "strategy_mode", "") or "") or None
    return symbols, strategy_mode


def _write_profile_index(profile_dir: Path) -> None:
    frame = pd.DataFrame(_PROFILE_ROWS)
    temporary = profile_dir / "replay_profile_index.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(profile_dir / "replay_profile_index.csv")


def _profiled_run_rotation_replay(*args: Any, **kwargs: Any):
    global _PROFILE_COUNTER
    symbols, strategy_mode = _profile_label(args, kwargs)
    symbol_hash = hashlib.sha256("|".join(symbols).encode("utf-8")).hexdigest()[:10]
    with _PROFILE_LOCK:
        _PROFILE_COUNTER += 1
        call_id = _PROFILE_COUNTER
    profile_dir = _resolve_profile_dir()
    stem = f"replay_{call_id:03d}_{len(symbols)}assets_{symbol_hash}"

    profiler = cProfile.Profile()
    started = time.perf_counter()
    error: str | None = None
    try:
        profiler.enable()
        result = _ORIGINAL_REPLAY(*args, **kwargs)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        profiler.disable()
        elapsed = float(time.perf_counter() - started)
        profiler.dump_stats(str(profile_dir / f"{stem}.prof"))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(_PROFILE_TOP)
        (profile_dir / f"{stem}.txt").write_text(stream.getvalue(), encoding="utf-8")
        row = {
            "call_id": call_id,
            "asset_count": len(symbols),
            "symbols_sha256_10": symbol_hash,
            "strategy_mode": strategy_mode,
            "elapsed_seconds": elapsed,
            "profile_file": f"{stem}.prof",
            "top_functions_file": f"{stem}.txt",
            "error": error,
        }
        with _PROFILE_LOCK:
            _PROFILE_ROWS.append(row)
            _PROFILE_ROWS.sort(key=lambda item: int(item["call_id"]))
            _write_profile_index(profile_dir)
        _utc_log(
            f"Profiler: replay #{call_id} with {len(symbols)} assets completed in {elapsed:.1f}s; "
            f"profile={profile_dir / (stem + '.txt')}"
        )


def install_v101() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._log = _utc_log
    discovery._run_rotation_replay = _profiled_run_rotation_replay


if __name__ == "__main__":
    install_v101()
    _utc_log(
        "Sequential Exact Marginal Search v1.0.1 profiling enabled. The economic judge is unchanged; "
        "cProfile only measures where exact replay time is spent."
    )
    raise SystemExit(base.main())
