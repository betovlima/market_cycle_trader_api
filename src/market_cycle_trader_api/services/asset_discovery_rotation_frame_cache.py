from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine import capital_rotation


_CACHE_CONDITION = threading.Condition(threading.RLock())
_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()
_INFLIGHT: set[str] = set()
_ORIGINAL_BUILD_ROTATION_FRAME: Callable[..., pd.DataFrame] | None = None
_INSTALLED = False


def _cache_capacity() -> int:
    raw = str(os.getenv("ASSET_DISCOVERY_ROTATION_FRAME_CACHE_SIZE") or "192").strip()
    try:
        return max(32, min(512, int(raw)))
    except ValueError:
        return 192


def _asset_discovery_request(config: Any) -> bool:
    reference_assets = [
        str(value or "").strip().upper()
        for value in list(getattr(config, "research_reference_assets", None) or [])
        if str(value or "").strip()
    ]
    market_mode = str(getattr(config, "research_market_data_mode", "") or "").strip().lower()
    return bool(reference_assets) and market_mode == "database_only"


def _frame_digest(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("open", "high", "low", "close", "volume")
        if column in frame.columns
    ]
    sample = frame.loc[:, columns].copy() if columns else frame.copy()
    hashed = pd.util.hash_pandas_object(sample, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(hashed.tobytes())
    digest.update(str(tuple(str(dtype) for dtype in sample.dtypes)).encode("utf-8"))
    digest.update(str(sample.shape).encode("utf-8"))
    return digest.hexdigest()


def _config_signature(config: Any) -> dict[str, Any]:
    return {
        "schema": 1,
        "strategy_mode": str(getattr(config, "strategy_mode", "") or ""),
        "rotation_target_horizons": [
            int(value) for value in list(getattr(config, "rotation_target_horizons", None) or [])
        ],
        "rotation_target_horizon_weights": [
            float(value) for value in list(getattr(config, "rotation_target_horizon_weights", None) or [])
        ],
        "slippage_bps": float(getattr(config, "slippage_bps", 0.0) or 0.0),
        "commission_rate": float(getattr(config, "commission_rate", 0.0) or 0.0),
        "rotation_downside_penalty": float(getattr(config, "rotation_downside_penalty", 0.0) or 0.0),
        "rotation_drawdown_penalty": float(getattr(config, "rotation_drawdown_penalty", 0.0) or 0.0),
        "rotation_movement_capture_weight": float(
            getattr(config, "rotation_movement_capture_weight", 0.0) or 0.0
        ),
        "rotation_trend_persistence_weight": float(
            getattr(config, "rotation_trend_persistence_weight", 0.0) or 0.0
        ),
    }


def _cache_key(frame: pd.DataFrame, config: Any) -> str:
    payload = {
        "frame": _frame_digest(frame),
        "config": _config_signature(config),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _acquire(key: str) -> tuple[bool, pd.DataFrame | None]:
    with _CACHE_CONDITION:
        while key in _INFLIGHT and key not in _CACHE:
            _CACHE_CONDITION.wait()
        if key not in _CACHE:
            _INFLIGHT.add(key)
            return False, None
        frame = _CACHE.pop(key)
        _CACHE[key] = frame
        return True, frame


def _publish(key: str, frame: pd.DataFrame) -> None:
    with _CACHE_CONDITION:
        _INFLIGHT.discard(key)
        _CACHE.pop(key, None)
        _CACHE[key] = frame.copy(deep=True)
        while len(_CACHE) > _cache_capacity():
            _CACHE.popitem(last=False)
        _CACHE_CONDITION.notify_all()


def _release_failed(key: str) -> None:
    with _CACHE_CONDITION:
        _INFLIGHT.discard(key)
        _CACHE_CONDITION.notify_all()


def _cached_build_rotation_frame(bars: pd.DataFrame, config: Any) -> pd.DataFrame:
    original = _ORIGINAL_BUILD_ROTATION_FRAME
    if original is None:
        raise RuntimeError("Asset Discovery rotation-frame cache is not installed.")
    if not _asset_discovery_request(config):
        return original(bars, config)

    key = _cache_key(bars, config)
    hit, cached = _acquire(key)
    if hit and cached is not None:
        # prepare_rotation_panel immediately reindexes/copies this frame, so a shallow
        # view is sufficient and avoids duplicating the large feature matrix on hits.
        return cached.copy(deep=False)

    try:
        built = original(bars, config)
        _publish(key, built)
        return built
    except Exception:
        _release_failed(key)
        raise


def rotation_frame_cache_stats() -> dict[str, int]:
    with _CACHE_CONDITION:
        return {
            "entries": len(_CACHE),
            "inflight": len(_INFLIGHT),
            "capacity": _cache_capacity(),
        }


def install_asset_discovery_rotation_frame_cache() -> None:
    global _INSTALLED, _ORIGINAL_BUILD_ROTATION_FRAME
    if _INSTALLED:
        return

    original = capital_rotation.build_rotation_frame
    if getattr(original, "_asset_discovery_rotation_frame_cache", False):
        _INSTALLED = True
        return

    _ORIGINAL_BUILD_ROTATION_FRAME = original
    setattr(_cached_build_rotation_frame, "_asset_discovery_rotation_frame_cache", True)
    capital_rotation.build_rotation_frame = _cached_build_rotation_frame
    _INSTALLED = True
