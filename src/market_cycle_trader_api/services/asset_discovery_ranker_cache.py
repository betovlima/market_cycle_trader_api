from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import os
import threading
import time
from typing import Any

import numpy as np
import pandas as pd


_CACHE_LOCK = threading.RLock()
_CACHE: OrderedDict[str, Any] = OrderedDict()
_INSTALLED = False
_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _cache_capacity() -> int:
    raw = str(os.getenv("ASSET_DISCOVERY_RANKER_CACHE_SIZE") or "4").strip()
    try:
        return max(1, min(16, int(raw)))
    except ValueError:
        return 4


def _frame_digest(symbol: str, frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(symbol or "").strip().upper().encode("utf-8"))
    if frame is None:
        digest.update(b"<none>")
        return digest.hexdigest()

    columns = [column for column in _REQUIRED_COLUMNS if column in frame.columns]
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(str(len(frame)).encode("ascii"))
    if not columns:
        return digest.hexdigest()

    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    hashed = pd.util.hash_pandas_object(normalized, index=True, categorize=False)
    digest.update(np.asarray(hashed.to_numpy(dtype=np.uint64), dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _ranker_cache_key(frames: dict[str, pd.DataFrame], random_state: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"asset-discovery-ranker-cache-v1")
    digest.update(str(int(random_state)).encode("ascii"))
    normalized_frames = {
        str(symbol or "").strip().upper(): frame
        for symbol, frame in frames.items()
        if str(symbol or "").strip()
    }
    for symbol in sorted(normalized_frames):
        digest.update(symbol.encode("utf-8"))
        digest.update(_frame_digest(symbol, normalized_frames[symbol]).encode("ascii"))
    return digest.hexdigest()


def _bundle_with_cache_metadata(bundle: Any, *, hit: bool, key: str, lookup_seconds: float) -> Any:
    diagnostics = copy.deepcopy(dict(getattr(bundle, "diagnostics", {}) or {}))
    diagnostics["ranker_cache"] = {
        "enabled": True,
        "hit": bool(hit),
        "key": str(key)[:16],
        "lookup_seconds": round(max(0.0, float(lookup_seconds)), 6),
        "scope": "process_local_exact_baseline_snapshot",
    }
    try:
        return type(bundle)(model=bundle.model, diagnostics=diagnostics)
    except Exception:
        try:
            bundle.diagnostics.update(diagnostics)
        except Exception:
            pass
        return bundle


def install_asset_discovery_ranker_cache() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_train_ranker = service.train_ranker
    if getattr(original_train_ranker, "_asset_discovery_ranker_cache", False):
        _INSTALLED = True
        return

    def cached_train_ranker(
        frames: dict[str, pd.DataFrame],
        *,
        random_state: int,
        stop_check: Any | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        if stop_check is not None and stop_check():
            raise service.AssetDiscoveryRankerCancelled(
                "Asset Discovery Learning-to-Rank cancellation requested."
            )

        started = time.perf_counter()
        key = _ranker_cache_key(frames, int(random_state))
        lookup_seconds = time.perf_counter() - started

        with _CACHE_LOCK:
            cached = _CACHE.pop(key, None)
            if cached is not None:
                _CACHE[key] = cached

        if cached is not None:
            if progress_callback is not None:
                progress_callback(100.0, "ranker_completed")
            return _bundle_with_cache_metadata(
                cached,
                hit=True,
                key=key,
                lookup_seconds=lookup_seconds,
            )

        bundle = original_train_ranker(
            frames,
            random_state=int(random_state),
            stop_check=stop_check,
            progress_callback=progress_callback,
        )

        with _CACHE_LOCK:
            _CACHE.pop(key, None)
            _CACHE[key] = bundle
            while len(_CACHE) > _cache_capacity():
                _CACHE.popitem(last=False)

        return _bundle_with_cache_metadata(
            bundle,
            hit=False,
            key=key,
            lookup_seconds=lookup_seconds,
        )

    setattr(cached_train_ranker, "_asset_discovery_ranker_cache", True)
    service.train_ranker = cached_train_ranker
    _INSTALLED = True
