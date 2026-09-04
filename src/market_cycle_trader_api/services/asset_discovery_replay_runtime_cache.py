from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import threading
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine import capital_rotation, research_challengers
from ..engine.capital_rotation import ROTATION_FEATURES


_FEATURE_CACHE_LOCK = threading.RLock()
_FEATURE_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()
_PREDICTION_CACHE_CONDITION = threading.Condition(threading.RLock())
_PREDICTION_CACHE: OrderedDict[str, pd.Series] = OrderedDict()
_PREDICTION_INFLIGHT: set[str] = set()
_ORIGINAL_PREPARE_ROTATION_PANEL: Callable[..., tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]] | None = None
_ORIGINAL_MODEL_UTILITIES: Callable[..., np.ndarray] | None = None
_INSTALLED = False


def _feature_cache_capacity() -> int:
    raw = str(os.getenv("ASSET_DISCOVERY_FEATURE_CACHE_SIZE") or "128").strip()
    try:
        return max(32, min(512, int(raw)))
    except ValueError:
        return 128


def _prediction_cache_capacity() -> int:
    raw = str(os.getenv("ASSET_DISCOVERY_PREDICTION_CACHE_SIZE") or "1024").strip()
    try:
        return max(128, min(4096, int(raw)))
    except ValueError:
        return 1024


def _normalized_assets(values: Any) -> set[str]:
    return {
        str(value or "").strip().upper()
        for value in (values or [])
        if str(value or "").strip()
    }


def _replay_reference_assets(config: Any) -> set[str]:
    references = _normalized_assets(getattr(config, "research_reference_assets", None))
    candidates = _normalized_assets(getattr(config, "research_candidate_assets", None))
    return references if references and candidates else set()


def _numeric_signature(values: pd.Series) -> dict[str, Any]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(array)
    if not bool(finite.any()):
        return {"count": int(len(array)), "finite": 0}
    clean = array[finite]
    return {
        "count": int(len(array)),
        "finite": int(len(clean)),
        "sum": float(np.sum(clean, dtype=np.float64)),
        "sum_sq": float(np.dot(clean, clean)),
        "first": float(clean[0]),
        "last": float(clean[-1]),
    }


def _frame_signature(symbol: str, frame: pd.DataFrame, config: Any) -> str:
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    payload = {
        "schema": 1,
        "symbol": str(symbol).upper(),
        "rows": int(len(frame)),
        "index_start": index[0].isoformat() if len(index) else None,
        "index_end": index[-1].isoformat() if len(index) else None,
        "ohlcv": {
            column: _numeric_signature(frame[column])
            for column in ("open", "high", "low", "close", "volume")
            if column in frame.columns
        },
        "rotation_target_horizons": [int(value) for value in config.rotation_target_horizons],
        "rotation_target_horizon_weights": [float(value) for value in config.rotation_target_horizon_weights],
        "rotation_downside_penalty": float(config.rotation_downside_penalty),
        "rotation_drawdown_penalty": float(config.rotation_drawdown_penalty),
        "rotation_movement_capture_weight": float(config.rotation_movement_capture_weight),
        "rotation_trend_persistence_weight": float(config.rotation_trend_persistence_weight),
        "slippage_bps": float(config.slippage_bps),
        "commission_rate": float(config.commission_rate),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cached_rotation_frame(symbol: str, frame: pd.DataFrame, config: Any) -> tuple[pd.DataFrame, str]:
    key = _frame_signature(symbol, frame, config)
    with _FEATURE_CACHE_LOCK:
        cached = _FEATURE_CACHE.pop(key, None)
        if cached is not None:
            _FEATURE_CACHE[key] = cached
            return cached, key

    built = capital_rotation.build_rotation_frame(frame, config)
    built.attrs["_mct_replay_feature_key"] = key
    with _FEATURE_CACHE_LOCK:
        _FEATURE_CACHE[key] = built
        while len(_FEATURE_CACHE) > _feature_cache_capacity():
            _FEATURE_CACHE.popitem(last=False)
    return built, key


def _cached_prepare_rotation_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    references = _replay_reference_assets(config)
    if not references:
        original = _ORIGINAL_PREPARE_ROTATION_PANEL
        if original is None:
            raise RuntimeError("Asset Discovery replay runtime cache is not installed.")
        return original(bars_by_symbol, config)

    frames: dict[str, pd.DataFrame] = {}
    feature_keys: dict[str, str] = {}
    for symbol, frame in bars_by_symbol.items():
        if frame is None or frame.empty:
            continue
        normalized = str(symbol).upper()
        if normalized in references:
            built, feature_key = _cached_rotation_frame(normalized, frame, config)
            frames[normalized] = built
            feature_keys[normalized] = feature_key
        else:
            built = capital_rotation.build_rotation_frame(frame, config)
            built.attrs["_mct_replay_feature_key"] = f"candidate:{normalized}:{uuid.uuid4().hex}"
            frames[normalized] = built

    if len(frames) < 2:
        raise ValueError("Compound rotation needs at least two assets with valid aligned data.")

    configured_anchors = list(getattr(config, "calendar_anchor_assets", []) or [])
    anchor_symbols = [str(symbol).upper() for symbol in configured_anchors if str(symbol).upper() in frames]
    if len(anchor_symbols) < 2:
        anchor_symbols = sorted(frames)

    common: pd.DatetimeIndex | None = None
    for symbol in anchor_symbols:
        index = pd.DatetimeIndex(frames[symbol].index)
        common = index if common is None else common.intersection(index)
    if common is None or len(common) < 700:
        raise ValueError("The anchored aligned history is too short for train/calibration/test.")

    common = common.sort_values()
    common_key = hashlib.sha256(
        f"{common[0].isoformat()}|{common[-1].isoformat()}|{len(common)}".encode("utf-8")
    ).hexdigest()
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        current = frame.reindex(common).copy()
        if symbol in feature_keys:
            current.attrs["_mct_replay_prediction_frame_key"] = f"reference:{feature_keys[symbol]}:{common_key}"
        else:
            current.attrs["_mct_replay_prediction_frame_key"] = f"candidate:{symbol}:{common_key}:{uuid.uuid4().hex}"
        aligned[symbol] = current
    return aligned, common


def _model_token(model: Any) -> str:
    token = getattr(model, "_mct_replay_prediction_token", None)
    if token:
        return str(token)
    token = uuid.uuid4().hex
    try:
        setattr(model, "_mct_replay_prediction_token", token)
    except Exception:
        token = f"id:{id(model)}"
    return token


def _prediction_key(model: Any, symbol: str, frame: pd.DataFrame) -> str:
    frame_key = str(frame.attrs.get("_mct_replay_prediction_frame_key") or f"frame:{id(frame)}")
    return f"{_model_token(model)}|{str(symbol).upper()}|{frame_key}"


def _prediction_series(model: Any, symbol: str, frame: pd.DataFrame) -> pd.Series:
    key = _prediction_key(model, symbol, frame)
    with _PREDICTION_CACHE_CONDITION:
        while key in _PREDICTION_INFLIGHT and key not in _PREDICTION_CACHE:
            _PREDICTION_CACHE_CONDITION.wait()
        cached = _PREDICTION_CACHE.pop(key, None)
        if cached is not None:
            _PREDICTION_CACHE[key] = cached
            return cached
        _PREDICTION_INFLIGHT.add(key)

    try:
        features = frame.loc[:, ROTATION_FEATURES]
        valid = features.notna().all(axis=1)
        next_open = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
        next_close = pd.to_numeric(frame["close"], errors="coerce").shift(-1)
        valid &= next_open.notna() & next_close.notna() & (next_open > 0.0) & (next_close > 0.0)
        output = pd.Series(float("-inf"), index=frame.index, dtype=float)
        if bool(valid.any()):
            predicted = np.asarray(model.predict(features.loc[valid]), dtype=np.float64)
            output.loc[valid] = predicted
    except Exception:
        with _PREDICTION_CACHE_CONDITION:
            _PREDICTION_INFLIGHT.discard(key)
            _PREDICTION_CACHE_CONDITION.notify_all()
        raise

    with _PREDICTION_CACHE_CONDITION:
        _PREDICTION_INFLIGHT.discard(key)
        _PREDICTION_CACHE[key] = output
        while len(_PREDICTION_CACHE) > _prediction_cache_capacity():
            _PREDICTION_CACHE.popitem(last=False)
        _PREDICTION_CACHE_CONDITION.notify_all()
    return output


def _cached_model_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
) -> np.ndarray:
    if not _replay_reference_assets(config):
        original = _ORIGINAL_MODEL_UTILITIES
        if original is None:
            raise RuntimeError("Asset Discovery replay runtime cache is not installed.")
        return original(models, frames, symbols, timestamp, config)

    values = [0.0]
    key_time = pd.Timestamp(timestamp)
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames[symbol]
        if model is None or key_time not in frame.index:
            values.append(float("-inf"))
            continue
        series = _prediction_series(model, symbol, frame)
        value = series.get(key_time, float("-inf"))
        values.append(float(value) if value is not None and np.isfinite(float(value)) else float("-inf"))
    return np.asarray(values, dtype=np.float64)


def install_asset_discovery_replay_runtime_cache() -> None:
    global _INSTALLED, _ORIGINAL_PREPARE_ROTATION_PANEL, _ORIGINAL_MODEL_UTILITIES
    if _INSTALLED:
        return

    _ORIGINAL_PREPARE_ROTATION_PANEL = research_challengers.prepare_rotation_panel
    _ORIGINAL_MODEL_UTILITIES = capital_rotation._model_utilities

    capital_rotation.prepare_rotation_panel = _cached_prepare_rotation_panel
    research_challengers.prepare_rotation_panel = _cached_prepare_rotation_panel
    capital_rotation._model_utilities = _cached_model_utilities
    research_challengers._model_utilities = _cached_model_utilities
    _INSTALLED = True
