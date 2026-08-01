from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd



def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def locked_configuration_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("Unsupported configuration payload.")


def strategy_configuration_fingerprint(config: Any) -> str:
    return _sha256_json(locked_configuration_payload(config))


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "xgboost": _package_version("xgboost"),
        "scikit_learn": _package_version("scikit-learn"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "threadpoolctl": _package_version("threadpoolctl"),
    }


def _series_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


def market_data_manifest(
    bars_by_symbol: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, dict[str, Any]]]:
    manifests: dict[str, dict[str, Any]] = {}
    for symbol in sorted(bars_by_symbol):
        frame = bars_by_symbol[symbol].copy().sort_index()
        columns = [
            column
            for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count")
            if column in frame.columns
        ]
        canonical = frame[columns].copy()
        canonical.index = pd.to_datetime(canonical.index, utc=True, errors="coerce")
        canonical = canonical.loc[~canonical.index.isna()]
        for column in columns:
            canonical[column] = pd.to_numeric(canonical[column], errors="coerce")

        row_hashes = pd.util.hash_pandas_object(canonical, index=True).to_numpy(
            dtype=np.uint64,
            copy=False,
        )
        digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
        provenance = dict(frame.attrs.get("market_data_provenance", {}))
        manifests[symbol] = {
            "sha256": digest,
            "rows": int(len(canonical)),
            "first_timestamp": _series_timestamp(canonical.index.min()) if len(canonical) else None,
            "last_timestamp": _series_timestamp(canonical.index.max()) if len(canonical) else None,
            "columns": columns,
            "history_complete": bool(provenance.get("history_complete", True)),
            "provider": provenance.get("provider") or provenance.get("effective_provider"),
            "effective_provider": provenance.get("effective_provider"),
            "historical_feed": provenance.get("historical_feed"),
            "live_feed": provenance.get("live_feed"),
            "adjustment": provenance.get("adjustment"),
            "initial_rows": provenance.get("initial_rows"),
            "history_backfill_provider": provenance.get("history_backfill_provider"),
            "history_backfill_rows": provenance.get("history_backfill_rows"),
            "requested_start": provenance.get("requested_start"),
            "actual_start": provenance.get("actual_start"),
        }

    return _sha256_json(manifests), manifests


def build_reproducibility_manifest(
    config: Any,
    bars_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    data_hash, data_manifests = market_data_manifest(bars_by_symbol)
    versions = runtime_versions()
    incomplete_assets = [
        symbol
        for symbol, item in data_manifests.items()
        if not bool(item.get("history_complete", False))
    ]
    backfilled_assets = [
        symbol
        for symbol, item in data_manifests.items()
        if int(item.get("history_backfill_rows") or 0) > 0
    ]
    return {
        "strategy_configuration_sha256": strategy_configuration_fingerprint(config),
        "market_data_signature_sha256": data_hash,
        "market_data_signatures": data_manifests,
        "market_data_history_complete": not incomplete_assets,
        "market_data_incomplete_assets": incomplete_assets,
        "market_data_backfilled_assets": backfilled_assets,
        "runtime_versions": versions,
        "python_version": versions.get("python"),
        "xgboost_version": versions.get("xgboost"),
        "scikit_learn_version": versions.get("scikit_learn"),
        "numpy_version": versions.get("numpy"),
        "pandas_version": versions.get("pandas"),
        "scipy_version": versions.get("scipy"),
        "threadpoolctl_version": versions.get("threadpoolctl"),
        "deterministic_execution": bool(config.deterministic_execution),
        "numeric_thread_limit": int(config.numeric_thread_limit),
        "xgb_n_jobs": int(config.xgb_n_jobs),
        "alpaca_historical_feed": str(config.alpaca_historical_feed),
        "alpaca_live_feed": str(config.alpaca_live_feed),
        "alpaca_adjustment": str(config.alpaca_adjustment),
    }
