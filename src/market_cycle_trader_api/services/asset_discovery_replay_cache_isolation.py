from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from . import asset_discovery_replay_cache as replay_cache


_INSTALLED = False
_ORIGINAL_CACHED_MODEL: Callable[[str], tuple[bool, Any | None]] | None = None
_ORIGINAL_PUBLISH_MODEL: Callable[[str, Any | None], None] | None = None


def _isolated_cached_model(key: str) -> tuple[bool, Any | None]:
    original = _ORIGINAL_CACHED_MODEL
    if original is None:
        raise RuntimeError("Asset Discovery replay cache isolation is not installed.")
    hit, model = original(key)
    if not hit or model is None:
        return hit, model
    return True, deepcopy(model)


def _isolated_publish_model(key: str, model: Any | None) -> None:
    original = _ORIGINAL_PUBLISH_MODEL
    if original is None:
        raise RuntimeError("Asset Discovery replay cache isolation is not installed.")
    cached_model = deepcopy(model) if model is not None else None
    original(key, cached_model)


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
