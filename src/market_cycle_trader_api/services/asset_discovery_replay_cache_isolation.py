from __future__ import annotations

import threading
from typing import Any, Callable

from . import asset_discovery_replay_cache as replay_cache


_INSTALLED = False
_ORIGINAL_CACHED_MODEL: Callable[[str], tuple[bool, Any | None]] | None = None
_ORIGINAL_PUBLISH_MODEL: Callable[[str, Any | None], None] | None = None
_MODEL_LOCKS_GUARD = threading.RLock()
_MODEL_LOCKS: dict[str, threading.RLock] = {}


def _lock_for_key(key: str) -> threading.RLock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[key] = lock
        return lock


class _LockedModelView:
    """Read-only view that serializes calls made against one cached fitted model.

    Asset Discovery candidate replays run concurrently.  The baseline models are
    immutable after fitting, so copying every LightGBM estimator for every cache
    hit is unnecessary and very expensive.  This view keeps one canonical cached
    model and only serializes method calls that can enter the native estimator.
    Attribute reads remain direct.
    """

    __slots__ = ("_model", "_lock")

    def __init__(self, model: Any, lock: threading.RLock) -> None:
        self._model = model
        self._lock = lock

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._model, name)
        if not callable(attribute):
            return attribute

        def locked_call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attribute(*args, **kwargs)

        return locked_call


def _isolated_cached_model(key: str) -> tuple[bool, Any | None]:
    original = _ORIGINAL_CACHED_MODEL
    if original is None:
        raise RuntimeError("Asset Discovery replay cache isolation is not installed.")
    hit, model = original(key)
    if not hit or model is None:
        return hit, model
    return True, _LockedModelView(model, _lock_for_key(key))


def _isolated_publish_model(key: str, model: Any | None) -> None:
    original = _ORIGINAL_PUBLISH_MODEL
    if original is None:
        raise RuntimeError("Asset Discovery replay cache isolation is not installed.")
    _lock_for_key(key)
    original(key, model)


def install_asset_discovery_replay_cache_isolation() -> None:
    global _INSTALLED, _ORIGINAL_CACHED_MODEL, _ORIGINAL_PUBLISH_MODEL
    if _INSTALLED:
        return

    cached_model = replay_cache._cached_model
    publish_model = replay_cache._publish_model
    if getattr(cached_model, "_asset_discovery_replay_cache_isolated", False):
        _INSTALLED = True
        return

    _ORIGINAL_CACHED_MODEL = cached_model
    _ORIGINAL_PUBLISH_MODEL = publish_model
    setattr(_isolated_cached_model, "_asset_discovery_replay_cache_isolated", True)
    setattr(_isolated_publish_model, "_asset_discovery_replay_cache_isolated", True)
    replay_cache._cached_model = _isolated_cached_model
    replay_cache._publish_model = _isolated_publish_model
    _INSTALLED = True
