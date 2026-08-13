from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import API_VERSION, PACKAGE_DIR
from ..schemas.requests import BacktestRequest


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


def _sha256_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _git_commit() -> str | None:
    for key in (
        "RAILWAY_GIT_COMMIT_SHA",
        "GIT_COMMIT_SHA",
        "GITHUB_SHA",
        "SOURCE_COMMIT",
    ):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        value = completed.stdout.strip()
        if completed.returncode == 0 and value:
            return value
    except Exception:
        pass
    return None


def _xgboost_build_info() -> dict[str, Any] | None:
    try:
        import xgboost as xgb

        info = xgb.build_info()
        return _json_safe(info) if isinstance(info, dict) else {"value": str(info)}
    except Exception:
        return None


def _threadpool_runtime() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info

        return _json_safe(threadpool_info())
    except Exception:
        return []


def locked_configuration_payload(config: Any) -> dict[str, Any]:
    return {
        field_name: getattr(config, field_name)
        for field_name in BacktestRequest.model_fields
    }


def strategy_configuration_fingerprint(config: Any) -> str:
    return _sha256_json(locked_configuration_payload(config))


def runtime_versions() -> dict[str, Any]:
    package_files = list(PACKAGE_DIR.rglob("*.py"))
    engine_dir = PACKAGE_DIR / "engine"
    engine_files = list(engine_dir.rglob("*.py"))
    thread_environment = {
        key: os.getenv(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    image_digest = next(
        (
            str(os.getenv(key)).strip()
            for key in ("RAILWAY_IMAGE_DIGEST", "CONTAINER_IMAGE_DIGEST", "IMAGE_DIGEST")
            if str(os.getenv(key) or "").strip()
        ),
        None,
    )
    deployment = {
        "railway_deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID"),
        "railway_service_id": os.getenv("RAILWAY_SERVICE_ID"),
        "railway_environment_id": os.getenv("RAILWAY_ENVIRONMENT_ID"),
        "container_image_digest": image_digest,
    }
    xgb_build = _xgboost_build_info()
    threadpools = _threadpool_runtime()
    runtime = {
        "api_version": API_VERSION,
        "git_commit": _git_commit(),
        "engine_source_sha256": _sha256_files(engine_files, PACKAGE_DIR),
        "package_source_sha256": _sha256_files(package_files, PACKAGE_DIR),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_build": list(platform.python_build()),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "xgboost": _package_version("xgboost"),
        "lightgbm": _package_version("lightgbm"),
        "torch": _package_version("torch"),
        "xgboost_build_info": xgb_build,
        "scikit_learn": _package_version("scikit-learn"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "threadpoolctl": _package_version("threadpoolctl"),
        "numeric_thread_environment": thread_environment,
        "threadpool_runtime": threadpools,
        "deployment": deployment,
    }
    runtime["runtime_fingerprint_sha256"] = _sha256_json(runtime)
    return runtime


def _series_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


def market_data_research_signature_from_manifests(
    manifests: dict[str, dict[str, Any]],
) -> str:
    






    stable: dict[str, dict[str, Any]] = {}
    for symbol in sorted(manifests):
        item = manifests[symbol] if isinstance(manifests[symbol], dict) else {}
        stable[symbol] = {
            "sha256": item.get("sha256"),
            "rows": int(item.get("rows") or 0),
            "first_timestamp": item.get("first_timestamp"),
            "last_timestamp": item.get("last_timestamp"),
            "columns": list(item.get("columns") or []),
            "historical_feed": item.get("historical_feed"),
            "adjustment": item.get("adjustment"),
        }
    return _sha256_json(stable)


def market_data_manifest(
    bars_by_symbol: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, dict[str, Any]]]:
    





    manifests: dict[str, dict[str, Any]] = {}
    research_columns_order = ("open", "high", "low", "close", "volume")
    audit_columns_order = (*research_columns_order, "vwap", "trade_count")
    for symbol in sorted(bars_by_symbol):
        frame = bars_by_symbol[symbol].copy().sort_index()
        research_columns = [column for column in research_columns_order if column in frame.columns]
        audit_columns = [column for column in audit_columns_order if column in frame.columns]

        canonical = frame[research_columns].copy()
        canonical.index = pd.to_datetime(canonical.index, utc=True, errors="coerce")
        canonical = canonical.loc[~canonical.index.isna()]
        try:
            canonical.index = canonical.index.as_unit("ns")
        except AttributeError:  
            canonical.index = pd.DatetimeIndex(
                canonical.index.to_numpy(dtype="datetime64[ns]"),
                tz="UTC",
            )
        for column in research_columns:
            canonical[column] = pd.to_numeric(canonical[column], errors="coerce").astype(
                np.float64, copy=False
            )
        row_hashes = pd.util.hash_pandas_object(canonical, index=True).to_numpy(
            dtype=np.uint64, copy=False
        )
        digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()

        audit = frame[audit_columns].copy()
        audit.index = pd.to_datetime(audit.index, utc=True, errors="coerce")
        audit = audit.loc[~audit.index.isna()]
        try:
            audit.index = audit.index.as_unit("ns")
        except AttributeError:  
            audit.index = pd.DatetimeIndex(
                audit.index.to_numpy(dtype="datetime64[ns]"),
                tz="UTC",
            )
        for column in audit_columns:
            audit[column] = pd.to_numeric(audit[column], errors="coerce").astype(
                np.float64, copy=False
            )
        audit_hashes = pd.util.hash_pandas_object(audit, index=True).to_numpy(
            dtype=np.uint64, copy=False
        )
        audit_digest = hashlib.sha256(audit_hashes.tobytes()).hexdigest()

        provenance = dict(frame.attrs.get("market_data_provenance", {}))
        manifests[symbol] = {
            "sha256": digest,
            "audit_sha256": audit_digest,
            "rows": int(len(canonical)),
            "first_timestamp": _series_timestamp(canonical.index.min()) if len(canonical) else None,
            "last_timestamp": _series_timestamp(canonical.index.max()) if len(canonical) else None,
            "columns": research_columns,
            "audit_columns": audit_columns,
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

    return market_data_research_signature_from_manifests(manifests), manifests


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
    assets = list(getattr(config, "assets", []) or [])
    configured_reference_assets = list(
        getattr(config, "research_reference_assets", []) or []
    )
    reference_assets = [
        symbol for symbol in configured_reference_assets if symbol in set(assets)
    ]
    if len(reference_assets) < 2:
        reference_assets = [
            symbol
            for symbol in list(getattr(config, "calendar_anchor_assets", []) or [])
            if symbol in set(assets)
        ]
    reference_set = set(reference_assets)
    configured_candidate_assets = list(
        getattr(config, "research_candidate_assets", []) or []
    )
    candidate_assets = [
        symbol
        for symbol in configured_candidate_assets
        if symbol in set(assets) and symbol not in reference_set
    ]
    if not configured_candidate_assets:
        candidate_assets = [symbol for symbol in assets if symbol not in reference_set]
    return {
        "reproducibility_schema_version": 3,
        "api_version": API_VERSION,
        "strategy_configuration_sha256": strategy_configuration_fingerprint(config),
        "market_data_signature_schema_version": 4,
        "market_data_signature_sha256": data_hash,
        "market_data_audit_signature_sha256": _sha256_json(data_manifests),
        "market_data_signatures": data_manifests,
        "market_data_history_complete": not incomplete_assets,
        "market_data_incomplete_assets": incomplete_assets,
        "market_data_backfilled_assets": backfilled_assets,
        "research_reference_assets": reference_assets,
        "research_candidate_assets": candidate_assets,
        "runtime_versions": versions,
        "runtime_fingerprint_sha256": versions.get("runtime_fingerprint_sha256"),
        "git_commit": versions.get("git_commit"),
        "engine_source_sha256": versions.get("engine_source_sha256"),
        "package_source_sha256": versions.get("package_source_sha256"),
        "xgboost_build_info": versions.get("xgboost_build_info"),
        "numeric_thread_environment": versions.get("numeric_thread_environment"),
        "threadpool_runtime": versions.get("threadpool_runtime"),
        "deployment_runtime": versions.get("deployment"),
        "python_version": versions.get("python"),
        "xgboost_version": versions.get("xgboost"),
        "lightgbm_version": versions.get("lightgbm"),
        "torch_version": versions.get("torch"),
        "research_model_family": str(getattr(config, "research_model_family", "xgboost_utility")),
        "research_model_settings": getattr(config, "research_model_settings", {}) or {},
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
