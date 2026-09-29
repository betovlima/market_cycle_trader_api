"""Fresh, isolated MCT snapshot for a no-order Control shadow job.

Downloads Alpaca RAW/SIP daily OHLCV and complete corporate actions. Applies
the same causal split normalization as the MCT TCC reference, without MongoDB
market-data cache or frozen TCC CSV dependencies. The snapshot is stored
locally under API_ROOT/dados/control_shadow/snapshots/<job_id>.

No trading endpoints, paper plans, Winner selection or portfolio writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..core.environment import PROJECT_ROOT
from ..tcc_v106_reference.config import (
    ASSETS, REFERENCE_ASSETS, CONFIG, build_control_config,
)
from .market_data import (
    _download_alpaca_bars,
    _history_frame_sha256,
    latest_safe_completed_xnys_session,
    validate_and_clean_bars,
)
from .research_market_data import (
    _download_corporate_actions,
    split_normalize,
    structural_identity_issue,
)

DATA_DIRECTORY = PROJECT_ROOT / "dados" / "control_shadow" / "snapshots"
SOURCE_CONTRACT = "alpaca_raw_sip_corporate_actions_split_normalized_v1"
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class CurrentControlSnapshot:
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    directory: Path


def _write_bytes(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _bars_payload(frame: pd.DataFrame) -> bytes:
    stable = frame.copy()
    stable.index = pd.to_datetime(stable.index, utc=True)
    stable.index.name = "timestamp"
    # Round-trip binary64 precision; never use display rounding for a model.
    return stable.to_csv(index=True, float_format="%.17g").encode("utf-8")


def download_current_control_snapshot(
    *,
    job_id: str,
    completed_session: str | None = None,
    progress_callback: ProgressCallback | None = None,
    data_directory: Path | None = None,
    per_asset_pause_seconds: float = 0.25,
) -> CurrentControlSnapshot:
    """Always request all 56 assets anew; publish only a complete manifest.

    The caller supplies a server-generated job ID, not an HTTP filesystem
    path. A dated cutoff may be supplied by internal tests only and cannot
    exceed the latest safely completed XNYS session.
    """
    if not job_id.startswith("control-shadow-") or not all(
        character.isalnum() or character == "-" for character in job_id
    ):
        raise ValueError("A server-generated Control Shadow job ID is required.")

    safe = latest_safe_completed_xnys_session().date()
    cutoff = safe.isoformat()
    if completed_session is not None:
        requested = pd.Timestamp(completed_session)
        if (
            requested.tzinfo is not None
            or requested.strftime("%Y-%m-%d") != completed_session
            or requested.date() > safe
        ):
            raise ValueError("Control shadow cutoff must be a safely completed XNYS date.")
        cutoff = completed_session

    root = (data_directory or DATA_DIRECTORY).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / job_id
    temporary = root / (".partial-" + job_id)
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Control Shadow snapshot already exists: {job_id}")
    temporary.mkdir()

    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": cutoff,
        "end_date": cutoff,
        "alpaca_historical_feed": "sip",
        "alpaca_adjustment": "raw",
    })
    reference = set(REFERENCE_ASSETS)
    frames: dict[str, pd.DataFrame] = {}
    excluded: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    total = len(ASSETS)

    try:
        for index, symbol in enumerate(ASSETS, start=1):
            if progress_callback is not None:
                progress_callback("download", index - 1, total, symbol)
            raw = _download_alpaca_bars(
                symbol,
                config,
                str(config.start_date),
                cutoff,
                single_request_daily=True,
            )
            if raw.empty:
                raise RuntimeError(
                    f"{symbol}: Alpaca RAW/SIP returned no daily bars; "
                    "the Control universe must not be silently reduced."
                )
            # Ensure timestamps are offset-aware before causal split processing.
            # Never infer a local workstation timezone from provider timestamps.
            raw = raw.copy()
            raw.index = pd.to_datetime(raw.index, utc=True)
            raw.index.name = "timestamp"
            if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
                raise ValueError(f"{symbol}: duplicate or unsorted RAW daily sessions.")
            # Never mix RAW with all/dividend-adjusted candles.
            hashes[f"raw_bars/{symbol}.csv"] = _write_bytes(
                temporary, f"raw_bars/{symbol}.csv", _bars_payload(raw),
            )
            actions, query_start, query_end = _download_corporate_actions(symbol, config)
            hashes[f"corporate_actions/{symbol}.json"] = _write_bytes(
                temporary,
                f"corporate_actions/{symbol}.json",
                (json.dumps(
                    actions, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), default=str,
                ) + "\n").encode("utf-8"),
            )
            issue = structural_identity_issue(symbol, actions)
            record: dict[str, Any] = {
                "raw_rows": len(raw),
                "raw_history_sha256": _history_frame_sha256(raw),
                "corporate_actions": len(actions),
                "corporate_actions_query_start": query_start,
                "corporate_actions_query_end": query_end,
            }
            if issue is not None:
                record["status"] = "excluded_structural_identity"
                record["exclusion"] = issue
                excluded.append(dict(issue))
                records[symbol] = record
                if progress_callback is not None:
                    progress_callback("excluded", index, total, symbol)
                continue

            normalized, splits = split_normalize(raw, actions)
            asset_config = config.model_copy(update={
                "market_data_require_complete_history": symbol in reference,
            })
            cleaned = validate_and_clean_bars(normalized, asset_config)
            if symbol in reference and cleaned.index[-1].date().isoformat() != cutoff:
                raise RuntimeError(
                    f"{symbol}: anchor missing latest complete XNYS session {cutoff}."
                )
            if (cleaned.index.date > pd.Timestamp(cutoff).date()).any():
                raise RuntimeError(f"{symbol}: market feed included a future session.")
            hashes[f"normalized_bars/{symbol}.csv"] = _write_bytes(
                temporary, f"normalized_bars/{symbol}.csv", _bars_payload(cleaned),
            )
            record.update({
                "status": "eligible",
                "normalized_rows": len(cleaned),
                "normalized_history_sha256": _history_frame_sha256(cleaned),
                "first_session": cleaned.index[0].date().isoformat(),
                "last_session": cleaned.index[-1].date().isoformat(),
                "splits_applied": len(splits),
                "split_events": splits,
            })
            records[symbol] = record
            frames[symbol] = cleaned
            if progress_callback is not None:
                progress_callback("download", index, total, symbol)
            if per_asset_pause_seconds > 0 and index != total:
                time.sleep(per_asset_pause_seconds)

        missing_anchors = [symbol for symbol in REFERENCE_ASSETS if symbol not in frames]
        if missing_anchors:
            raise RuntimeError(
                "Control reference anchors structurally excluded or absent: "
                + ", ".join(missing_anchors)
            )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "source_contract": SOURCE_CONTRACT,
            "source": "alpaca",
            "feed": "sip",
            "download_adjustment": "raw",
            "effective_adjustment": "raw_plus_split_normalization",
            "dividend_adjustment_applied": False,
            "start_date": str(config.start_date),
            "completed_session": cutoff,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_assets": list(ASSETS),
            "eligible_assets": sorted(frames),
            "structural_exclusions": excluded,
            "per_asset": records,
            "file_hashes": hashes,
        }
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        manifest["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
        _write_bytes(
            temporary, "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, default=str)
             + "\n").encode("utf-8"),
        )
        os.replace(temporary, destination)
        return CurrentControlSnapshot(frames, manifest, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
