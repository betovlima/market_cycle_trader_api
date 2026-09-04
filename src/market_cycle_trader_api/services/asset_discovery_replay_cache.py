from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine import research_challengers
from ..engine.capital_rotation import ROTATION_FEATURES


_CACHE_CONDITION = threading.Condition(threading.RLock())
_CACHE: OrderedDict[str, Any | None] = OrderedDict()
_INFLIGHT: set[str] = set()
_ORIGINAL_FIT_MODELS: Callable[..., dict[str, Any]] | None = None
_INSTALLED = False


def _cache_capacity() -> int:
    raw = str(os.getenv("ASSET_DISCOVERY_BASE_MODEL_CACHE_SIZE") or "512").strip()
    try:
        return max(32, min(2048, int(raw)))
    except ValueError:
        return 512


def _normalized_assets(values: Any) -> set[str]:
    return {
        str(value or "").strip().upper()
        for value in (values or [])
        if str(value or "").strip()
    }


def _cacheable_reference_asset(config: Any, symbol: str) -> bool:
    reference_assets = _normalized_assets(getattr(config, "research_reference_assets", None))
    candidate_assets = _normalized_assets(getattr(config, "research_candidate_assets", None))
    return bool(reference_assets and candidate_assets and str(symbol).upper() in reference_assets)


def _series_signature(frame: pd.DataFrame, column: str) -> dict[str, Any] | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return {"count": int(len(values)), "finite": 0}
    clean = values[finite]
    return {
        "count": int(len(values)),
        "finite": int(len(clean)),
        "sum": float(np.sum(clean, dtype=np.float64)),
        "sum_sq": float(np.dot(clean, clean)),
        "first": float(clean[0]),
        "last": float(clean[-1]),
    }


def _model_cache_key(
    frame: pd.DataFrame,
    symbol: str,
    train_dates: pd.DatetimeIndex,
    config: Any,
    target_column: str,
) -> str:
    sample = frame.loc[train_dates]
    settings = research_challengers._lightgbm_settings(config)
    columns = [
        column
        for column in ("open", "high", "low", "close", "volume", target_column)
        if column in sample.columns
    ]
    payload = {
        "schema": 1,
        "symbol": str(symbol).upper(),
        "target_column": str(target_column),
        "train_start": pd.Timestamp(train_dates[0]).isoformat() if len(train_dates) else None,
        "train_end": pd.Timestamp(train_dates[-1]).isoformat() if len(train_dates) else None,
        "train_sessions": int(len(train_dates)),
        "frame_rows": int(len(sample)),
        "frame_signature": {
            column: _series_signature(sample, column)
            for column in columns
        },
        "random_state": int(config.random_state),
        "deterministic_execution": bool(config.deterministic_execution),
        "effective_n_jobs": int(
            research_challengers._effective_n_jobs(int(settings["n_jobs"]))
        ),
        "minimum_training_rows": int(config.rotation_minimum_training_rows),
        "settings": settings,
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cached_model(key: str) -> tuple[bool, Any | None]:
    with _CACHE_CONDITION:
        while key in _INFLIGHT and key not in _CACHE:
            _CACHE_CONDITION.wait()
        if key not in _CACHE:
            _INFLIGHT.add(key)
            return False, None
        model = _CACHE.pop(key)
        _CACHE[key] = model
        return True, model


def _publish_model(key: str, model: Any | None) -> None:
    with _CACHE_CONDITION:
        _INFLIGHT.discard(key)
        _CACHE.pop(key, None)
        _CACHE[key] = model
        while len(_CACHE) > _cache_capacity():
            _CACHE.popitem(last=False)
        _CACHE_CONDITION.notify_all()


def _release_failed_model(key: str) -> None:
    with _CACHE_CONDITION:
        _INFLIGHT.discard(key)
        _CACHE_CONDITION.notify_all()


def _cached_lightgbm_fit_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
    target_column: str = "forward_risk_adjusted_utility",
) -> dict[str, Any]:
    original = _ORIGINAL_FIT_MODELS
    if original is None:
        raise RuntimeError("Asset Discovery replay model cache is not installed.")

    cacheable = {
        symbol
        for symbol in symbols
        if _cacheable_reference_asset(config, symbol)
    }
    if not cacheable:
        return original(
            frames,
            symbols,
            train_dates,
            config,
            phase=phase,
            progress_callback=progress_callback,
            technical_log_callback=technical_log_callback,
            target_column=target_column,
        )

    started = time.perf_counter()
    fitted: dict[str, Any] = {}
    cache_hits = 0
    cache_misses = 0
    trained_uncached = 0

    if technical_log_callback is not None:
        technical_log_callback(
            f"model=lightgbm phase={phase} event=fit_start device=cpu "
            f"models={len(symbols)} train_sessions={len(train_dates)} "
            f"asset_discovery_base_cache=enabled cacheable_models={len(cacheable)}"
        )

    for position, symbol in enumerate(symbols, start=1):
        if symbol not in cacheable:
            result = original(
                frames,
                [symbol],
                train_dates,
                config,
                phase=phase,
                progress_callback=None,
                technical_log_callback=None,
                target_column=target_column,
            )
            if symbol in result:
                fitted[symbol] = result[symbol]
            trained_uncached += 1
            if progress_callback is not None:
                progress_callback(position, len(symbols), "cpu")
            continue

        key = _model_cache_key(
            frames[symbol],
            symbol,
            train_dates,
            config,
            target_column,
        )
        hit, model = _cached_model(key)
        if hit:
            cache_hits += 1
            if model is not None:
                fitted[symbol] = model
            if progress_callback is not None:
                progress_callback(position, len(symbols), "cpu")
            continue

        cache_misses += 1
        try:
            result = original(
                frames,
                [symbol],
                train_dates,
                config,
                phase=phase,
                progress_callback=None,
                technical_log_callback=None,
                target_column=target_column,
            )
            model = result.get(symbol)
            _publish_model(key, model)
            if model is not None:
                fitted[symbol] = model
        except Exception:
            _release_failed_model(key)
            raise
        if progress_callback is not None:
            progress_callback(position, len(symbols), "cpu")

    if technical_log_callback is not None:
        technical_log_callback(
            f"model=lightgbm phase={phase} event=fit_complete device=cpu "
            f"models={len(fitted)} cache_hits={cache_hits} cache_misses={cache_misses} "
            f"trained_uncached={trained_uncached} cache_entries={len(_CACHE)} "
            f"duration_seconds={time.perf_counter() - started:.3f}"
        )
    return fitted


def install_asset_discovery_replay_model_cache() -> None:
    global _INSTALLED, _ORIGINAL_FIT_MODELS
    if _INSTALLED:
        return
    original = research_challengers._lightgbm_fit_models
    if getattr(original, "_asset_discovery_replay_cache", False):
        _INSTALLED = True
        return
    _ORIGINAL_FIT_MODELS = original
    setattr(_cached_lightgbm_fit_models, "_asset_discovery_replay_cache", True)
    research_challengers._lightgbm_fit_models = _cached_lightgbm_fit_models
    _INSTALLED = True
