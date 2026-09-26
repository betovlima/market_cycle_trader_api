"""Strict optional frozen-TCC input for scientific parity of the MCT reference job.

The default operational MCT source remains the refreshed Alpaca/Mongo feed.
This module reads the published later TCC-main research snapshot *as is*:
no numerical rounding, ticker repair, network access, or database writes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .market_data import (
    BAR_COLUMNS, REQUIRED_BAR_COLUMNS, _history_frame_sha256,
)
from .research_market_data import (
    RAW_TOTAL_CAUSAL_PROTOCOL,
    StructuralResearchAssetExclusion,
    split_normalize,
    structural_identity_issue,
)

FROZEN_TCC_MAIN_SHA256 = (
    "4e2fd225cc0ea05da56dad8f0628ca989ad332812a5fa3a7dc796b2b8a6d5128"
)
FROZEN_TCC_MAIN_COMMIT = "f9cf29fdb736676d0d3e26780481be99813c602a"
FROZEN_TCC_START = "2016-01-01"
FROZEN_TCC_END = "2026-09-17"
DATA_SOURCE_ENV = "MCT_TCC_V106_INPUT_SOURCE"
FROZEN_ROOT_ENV = "MCT_TCC_V106_FROZEN_DATA_DIR"
DEFAULT_SOURCE = "mct_current"
FROZEN_SOURCE = "tcc_frozen_main"


def selected_tcc_reference_input_source(config: Any | None = None) -> str:
    """Use the immutable job's data-source choice before the server default."""
    pinned = getattr(config, "tcc_reference_input_source", None)
    source = str(pinned or os.getenv(DATA_SOURCE_ENV) or DEFAULT_SOURCE).strip().lower()
    if source not in {DEFAULT_SOURCE, FROZEN_SOURCE}:
        raise ValueError(
            f"Unsupported TCC reference input source {source!r}; "
            f"choose {DEFAULT_SOURCE!r} or {FROZEN_SOURCE!r}."
        )
    return source


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_tcc_main(
    root: Path,
    *,
    assets: tuple[str, ...],
    expected_sha: str = FROZEN_TCC_MAIN_SHA256,
) -> dict[str, Any]:
    """Validate manifest identity, all byte hashes, universe and data cutoff.

    A manifest-provided SHA is not enough by itself: recompute the canonical
    identity and every actual file hash, then verify the pinned SHA.
    """
    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported TCC frozen manifest schema.")
    identity = {
        key: value for key, value in manifest.items()
        if key not in {"snapshot_sha256", "created_for_experiment_version"}
    }
    recomputed = _sha256_bytes(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    )
    if (
        manifest.get("snapshot_sha256") != recomputed
        or recomputed != expected_sha
    ):
        raise ValueError("Frozen TCC snapshot identity is not the pinned SHA-256.")
    bars = manifest.get("bars") or {}
    if bars != {
        "feed": "sip",
        "timeframe": "1Day",
        "adjustment": "raw",
        "start": FROZEN_TCC_START,
        "bar_snapshot_as_of_end": FROZEN_TCC_END,
    }:
        raise ValueError("Frozen TCC RAW bar-source contract differs.")
    actions = manifest.get("corporate_actions") or {}
    if actions.get("query_end") != FROZEN_TCC_END:
        raise ValueError("Frozen TCC Corporate Actions cutoff differs.")
    if list(manifest.get("assets") or []) != list(assets):
        raise ValueError("Frozen TCC asset list or order differs from the engine.")
    expected_files = {
        f"raw_bars/{symbol}.csv" for symbol in assets
    } | {
        f"corporate_actions/{symbol}.csv" for symbol in assets
    }
    hashes = manifest.get("file_hashes") or {}
    if set(hashes) != expected_files:
        raise ValueError(
            "Frozen TCC snapshot must have exactly one RAW and one "
            "Corporate Actions CSV per configured asset."
        )
    for relative in sorted(expected_files):
        actual = _sha256_bytes((root / relative).read_bytes())
        if actual != hashes[relative]:
            raise ValueError(f"Frozen TCC file hash mismatch: {relative}")
    return manifest


def frozen_tcc_root_from_environment() -> Path:
    raw = (os.getenv(FROZEN_ROOT_ENV) or "").strip()
    if not raw:
        raise RuntimeError(
            f"{FROZEN_ROOT_ENV} is required when {DATA_SOURCE_ENV}={FROZEN_SOURCE}."
        )
    root = Path(raw).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def load_frozen_tcc_main_symbol(
    root: Path,
    symbol: str,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    if symbol not in manifest["assets"]:
        raise ValueError(f"{symbol}: not part of the pinned research universe.")
    raw_path = root / "raw_bars" / f"{symbol}.csv"
    data = pd.read_csv(raw_path)
    if "timestamp" not in data:
        raise ValueError(f"{symbol}: frozen CSV has no timestamp.")
    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in data]
    if missing:
        raise ValueError(f"{symbol}: frozen CSV lacks required columns {missing}.")
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], utc=True, errors="raise",
    )
    data = data.set_index("timestamp").sort_index()
    if data.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate frozen session timestamps.")
    if int(manifest["row_counts"][symbol]) != len(data):
        raise ValueError(f"{symbol}: frozen RAW CSV row count changed.")
    for column in BAR_COLUMNS:
        if column in data.columns:
            # Explicit float64 before split adjustment prevents fractional
            # volume assignments to int64 columns in pandas >= 2.
            data[column] = pd.to_numeric(
                data[column], errors="raise",
            ).astype(np.float64)
    with (root / "corporate_actions" / f"{symbol}.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        events = list(csv.DictReader(stream))
    if int(manifest["corporate_action_counts"][symbol]) != len(events):
        raise ValueError(f"{symbol}: frozen Corporate Actions count changed.")
    issue = structural_identity_issue(symbol, events)
    if issue is not None:
        raise StructuralResearchAssetExclusion(issue)
    normalized, applied = split_normalize(data, events)
    normalized.attrs["market_data_provenance"] = {
        "research_access_path": "verified_frozen_tcc_main_csv",
        "research_market_data_protocol": RAW_TOTAL_CAUSAL_PROTOCOL,
        "source_adjustment": "raw",
        "effective_adjustment": "raw_plus_causal_split_normalization",
        "research_source_snapshot_id": FROZEN_TCC_MAIN_SHA256,
        "research_source_commit": FROZEN_TCC_MAIN_COMMIT,
        "raw_csv_sha256": manifest["file_hashes"][f"raw_bars/{symbol}.csv"],
        "corporate_actions_csv_sha256":
            manifest["file_hashes"][f"corporate_actions/{symbol}.csv"],
        "raw_sha256": _history_frame_sha256(data),
        "raw_audit_sha256": _history_frame_sha256(
            data, columns=BAR_COLUMNS,
        ),
        "splits_applied": len(applied),
        "split_events": applied,
        "corporate_action_count": len(events),
        "dividend_adjustment_applied": False,
        "dividend_events_used_by_model": False,
        "structural_identity_verified": True,
        "history_complete": True,
    }
    return normalized
