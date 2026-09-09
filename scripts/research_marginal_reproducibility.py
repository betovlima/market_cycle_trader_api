from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def code_identity() -> dict:
    digest = hashlib.sha256()
    paths = sorted([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"), ROOT / "pyproject.toml"])
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for package in ("pandas", "numpy", "lightgbm", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"git_commit": commit, "source_sha256": digest.hexdigest(), "python": platform.python_version(), "packages": packages}


def market_data_hashes(raw_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    hashes = {}
    for symbol, raw in sorted(raw_frames.items()):
        frame = raw[["open", "high", "low", "close", "volume"]].astype(float).copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate market sessions: {symbol}")
        serialized = frame.sort_index().to_csv(index=True, float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S%z", lineterminator="\n")
        hashes[symbol] = hashlib.sha256(serialized.encode()).hexdigest()
    return hashes


def verify_validation_pair(baseline: dict, expanded: dict) -> None:
    for key in ("initial_capital", "test_start", "test_end", "validation_sessions", "selection_end",
                "training_last_session", "calibration_first_session", "calibration_last_session", "final_fit_last_labeled_session"):
        if key not in baseline or key not in expanded or baseline[key] != expanded[key]:
            raise ValueError(f"Independent comparison mismatch: {key}")
    for key in ("selection_used_validation_period", "training_used_validation_period", "calibration_used_validation_period"):
        if baseline.get(key) is not False or expanded.get(key) is not False:
            raise ValueError(f"Independent comparison needs a verified pre-validation split: {key}")
    base_assets, expanded_assets = set(baseline["assets"]), set(expanded["assets"])
    if not base_assets or not base_assets.issubset(expanded_assets):
        raise ValueError("Expanded universe must contain the entire immutable baseline.")
    for result in (baseline, expanded):
        if set(result.get("market_data_sha256_by_asset", {})) != set(result["assets"]):
            raise ValueError("Missing market snapshot hashes; rerun both validations with v2.")
    if any(baseline["market_data_sha256_by_asset"][s] != expanded["market_data_sha256_by_asset"][s] for s in base_assets):
        raise ValueError("Baseline and expanded validation loaded different market prices.")
    if not baseline.get("code_identity") or baseline["code_identity"] != expanded.get("code_identity"):
        raise ValueError("Baseline and expanded validation used different code/environments.")
