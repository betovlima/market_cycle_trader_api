from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_sequential_exact_marginal_search as base  # noqa: E402
from market_cycle_trader_api.engine import capital_rotation as rotation  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "sequential-exact-marginal-search-v1.1.0"
_ORIGINAL_MODEL_UTILITIES = rotation._model_utilities
_ORIGINAL_REPLAY = discovery._run_rotation_replay
_THREAD = threading.local()
_INDEX_LOCK = threading.Lock()
_INDEX_ROWS: list[dict[str, Any]] = []
_REPLAY_COUNTER = 0
_PARITY_MODE = False
_OUTPUT_DIR: Path | None = None
_PARITY_ATOL = 1e-12


def _utc_log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _consume_flag(flag: str) -> bool:
    found = False
    while flag in sys.argv:
        sys.argv.remove(flag)
        found = True
    return found


def _argv_value(flag: str) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    value = str(sys.argv[index + 1]).strip()
    return value or None


def _resolve_output_dir() -> Path:
    global _OUTPUT_DIR
    if _OUTPUT_DIR is not None:
        return _OUTPUT_DIR
    configured = _argv_value("--output-dir")
    if configured:
        _OUTPUT_DIR = Path(configured).resolve()
    else:
        _OUTPUT_DIR = (
            PROJECT_ROOT / "research_output" / "sequential_exact_marginal_search_accelerated_v110"
        ).resolve()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def _new_state() -> dict[str, Any]:
    return {
        "active": True,
        "cache": {},
        "utility_calls": 0,
        "cache_hits": 0,
        "cache_builds": 0,
        "batch_predict_calls": 0,
        "batch_predict_rows": 0,
        "batch_build_seconds": 0.0,
    }


def _model_signature(models: dict[str, Any], symbols: list[str]) -> tuple[tuple[str, int], ...]:
    return tuple((symbol, id(models.get(symbol))) for symbol in symbols)


def _build_prediction_cache(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, pd.Series | None]:
    state = getattr(_THREAD, "state", None)
    started = time.perf_counter()
    output: dict[str, pd.Series | None] = {}
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames.get(symbol)
        if model is None or frame is None or frame.empty:
            output[symbol] = None
            continue
        if not frame.index.is_unique:
            output[symbol] = None
            continue

        features = frame.loc[:, rotation.ROTATION_FEATURES]
        valid = ~features.isna().any(axis=1)
        next_open = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
        next_close = pd.to_numeric(frame["close"], errors="coerce").shift(-1)
        valid &= np.isfinite(next_open.to_numpy(dtype=float)) & (next_open.to_numpy(dtype=float) > 0.0)
        valid &= np.isfinite(next_close.to_numpy(dtype=float)) & (next_close.to_numpy(dtype=float) > 0.0)
        if not bool(valid.any()):
            output[symbol] = pd.Series(dtype=float)
            continue

        batch = features.loc[valid]
        predictions = np.asarray(model.predict(batch), dtype=float).reshape(-1)
        if len(predictions) != len(batch):
            raise RuntimeError(
                f"AcceleratedPredictionShapeMismatch: {symbol} returned "
                f"{len(predictions)} predictions for {len(batch)} rows."
            )
        output[symbol] = pd.Series(predictions, index=batch.index, dtype=float)
        if state is not None:
            state["batch_predict_calls"] += 1
            state["batch_predict_rows"] += int(len(batch))

    if state is not None:
        state["cache_builds"] += 1
        state["batch_build_seconds"] += float(time.perf_counter() - started)
    return output


def _accelerated_model_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
) -> np.ndarray:
    state = getattr(_THREAD, "state", None)
    if not state or not bool(state.get("active")):
        return _ORIGINAL_MODEL_UTILITIES(models, frames, symbols, timestamp, config)

    state["utility_calls"] += 1
    normalized_symbols = list(symbols)
    key = (id(frames), _model_signature(models, normalized_symbols))
    cached = state["cache"].get(key)
    if cached is None:
        cached = _build_prediction_cache(models, frames, normalized_symbols)
        state["cache"][key] = cached
    else:
        state["cache_hits"] += 1

    ts = pd.Timestamp(timestamp)
    values = [0.0]
    for symbol in normalized_symbols:
        series = cached.get(symbol)
        if series is None:
            model = models.get(symbol)
            frame = frames.get(symbol)
            if model is None or frame is None or ts not in frame.index or not frame.index.is_unique:
                values.append(float("-inf"))
                continue
            row = frame.loc[[ts], rotation.ROTATION_FEATURES]
            if row.empty or row.isna().any(axis=None):
                values.append(float("-inf"))
                continue
            location = frame.index.get_loc(ts)
            if not isinstance(location, (int, np.integer)) or int(location) + 1 >= len(frame.index):
                values.append(float("-inf"))
                continue
            next_row = frame.iloc[int(location) + 1]
            next_open = float(next_row.get("open", float("nan")))
            next_close = float(next_row.get("close", float("nan")))
            if not (
                np.isfinite(next_open)
                and next_open > 0
                and np.isfinite(next_close)
                and next_close > 0
            ):
                values.append(float("-inf"))
                continue
            values.append(float(model.predict(row)[0]))
            continue
        value = series.get(ts, np.nan)
        values.append(float(value) if pd.notna(value) and np.isfinite(float(value)) else float("-inf"))
    return np.asarray(values, dtype=np.float64)


def _compare_values(left: Any, right: Any, path: str = "root") -> tuple[bool, float, str | None]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf, f"{path}: key mismatch"
        max_diff = 0.0
        for key in sorted(left):
            ok, diff, error = _compare_values(left[key], right[key], f"{path}.{key}")
            max_diff = max(max_diff, diff)
            if not ok:
                return False, max_diff, error
        return True, max_diff, None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, math.inf, f"{path}: length mismatch {len(left)} != {len(right)}"
        max_diff = 0.0
        for index, (a, b) in enumerate(zip(left, right)):
            ok, diff, error = _compare_values(a, b, f"{path}[{index}]")
            max_diff = max(max_diff, diff)
            if not ok:
                return False, max_diff, error
        return True, max_diff, None
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        a = float(left)
        b = float(right)
        if math.isnan(a) and math.isnan(b):
            return True, 0.0, None
        diff = abs(a - b)
        if diff <= _PARITY_ATOL:
            return True, diff, None
        return False, diff, f"{path}: numeric mismatch {a!r} != {b!r}"
    if left == right:
        return True, 0.0, None
    return False, math.inf, f"{path}: value mismatch {left!r} != {right!r}"


def _compare_replay_results(reference: Any, accelerated: Any) -> tuple[bool, float, str | None]:
    reference_metrics, reference_sessions = reference
    accelerated_metrics, accelerated_sessions = accelerated
    if not pd.DatetimeIndex(reference_sessions).equals(pd.DatetimeIndex(accelerated_sessions)):
        return False, math.inf, "decision sessions differ"
    return _compare_values(reference_metrics, accelerated_metrics, "metrics")


def _write_index() -> None:
    output = _resolve_output_dir() / "accelerator_replay_index.csv"
    temp = output.with_suffix(".csv.tmp")
    pd.DataFrame(_INDEX_ROWS).to_csv(temp, index=False)
    temp.replace(output)


def _replay_label(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, str]:
    frames = args[0] if args else kwargs.get("frames") or kwargs.get("bars_by_symbol") or {}
    symbols = sorted(str(symbol).strip().upper() for symbol in dict(frames or {}) if str(symbol).strip())
    return len(symbols), "|".join(symbols)


def _accelerated_replay(*args: Any, **kwargs: Any):
    global _REPLAY_COUNTER
    asset_count, symbols_text = _replay_label(args, kwargs)
    with _INDEX_LOCK:
        _REPLAY_COUNTER += 1
        call_id = _REPLAY_COUNTER

    reference_seconds: float | None = None
    reference_result: Any | None = None
    if _PARITY_MODE:
        _THREAD.state = {"active": False}
        started = time.perf_counter()
        reference_result = _ORIGINAL_REPLAY(*args, **kwargs)
        reference_seconds = float(time.perf_counter() - started)

    _THREAD.state = _new_state()
    started = time.perf_counter()
    try:
        accelerated_result = _ORIGINAL_REPLAY(*args, **kwargs)
        accelerated_seconds = float(time.perf_counter() - started)
        state = dict(_THREAD.state)
    finally:
        _THREAD.state = None

    parity_ok: bool | None = None
    parity_max_abs_diff: float | None = None
    parity_error: str | None = None
    if _PARITY_MODE:
        parity_ok, parity_max_abs_diff, parity_error = _compare_replay_results(
            reference_result, accelerated_result
        )
        if not parity_ok:
            raise RuntimeError(
                "AcceleratedExactReplayParityFailed: "
                f"call={call_id}, assets={asset_count}, max_abs_diff={parity_max_abs_diff}, "
                f"detail={parity_error}"
            )

    row = {
        "call_id": call_id,
        "asset_count": asset_count,
        "symbols": symbols_text,
        "parity_mode": _PARITY_MODE,
        "reference_seconds": reference_seconds,
        "accelerated_seconds": accelerated_seconds,
        "speedup_vs_reference": (
            float(reference_seconds / accelerated_seconds)
            if reference_seconds is not None and accelerated_seconds > 0
            else None
        ),
        "parity_ok": parity_ok,
        "parity_max_abs_diff": parity_max_abs_diff,
        "parity_error": parity_error,
        "utility_calls": int(state.get("utility_calls") or 0),
        "utility_cache_hits": int(state.get("cache_hits") or 0),
        "prediction_cache_builds": int(state.get("cache_builds") or 0),
        "batch_predict_calls": int(state.get("batch_predict_calls") or 0),
        "batch_predict_rows": int(state.get("batch_predict_rows") or 0),
        "prediction_cache_build_seconds": float(state.get("batch_build_seconds") or 0.0),
    }
    with _INDEX_LOCK:
        _INDEX_ROWS.append(row)
        _INDEX_ROWS.sort(key=lambda item: int(item["call_id"]))
        _write_index()

    _utc_log(
        f"Accelerator replay #{call_id}: assets={asset_count}, accelerated={accelerated_seconds:.1f}s"
        + (
            f", reference={reference_seconds:.1f}s, speedup={reference_seconds / accelerated_seconds:.2f}x, parity=PASS"
            if reference_seconds is not None
            else ""
        )
    )
    return accelerated_result


def install_v110(*, parity_mode: bool = False) -> None:
    global _PARITY_MODE
    _PARITY_MODE = bool(parity_mode)
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._log = _utc_log
    rotation._model_utilities = _accelerated_model_utilities
    discovery._run_rotation_replay = _accelerated_replay


if __name__ == "__main__":
    parity = _consume_flag("--parity-check")
    workers = int(_argv_value("--workers") or "1")
    if parity and workers != 1:
        raise SystemExit("--parity-check requires --workers 1 so reference and accelerated replays cannot overlap.")
    install_v110(parity_mode=parity)
    _utc_log(
        "Sequential Exact Marginal Search v1.1.0 prediction-cache accelerator enabled. "
        "The economic judge, model fitting, policies, fees, slippage and DeltaCapital rule are unchanged."
    )
    if parity:
        _utc_log(
            "Parity mode enabled: every exact replay is executed once with the original utility path and "
            "once with the batched prediction cache; any metric/session mismatch aborts the run."
        )
    raise SystemExit(base.main())
