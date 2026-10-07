"""Fresh audited market-data snapshot for TCC U67 operational backtests."""
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
from ..tcc_u67_v1210_reference.contract import U67_REQUESTED_ASSETS
from .market_data import (
    _download_alpaca_bars,
    _history_frame_sha256,
    validate_and_clean_bars,
)
from .research_market_data import (
    _download_corporate_actions,
    split_normalize,
    structural_identity_issue,
)

DATA_DIRECTORY = PROJECT_ROOT / "dados" / "tcc_u67_operational" / "snapshots"
SOURCE_CONTRACT = "alpaca_raw_sip_ca_split_normalized_u67_v1"
ProgressCallback = Callable[[str, int, int, str], None]

# This identity break was explicitly adjudicated by the TCC pipeline.
KNOWN_U67_STRUCTURAL_EXCLUSIONS: dict[str, dict[str, Any]] = {
    "CLMT": {
        "symbol": "CLMT",
        "reason": "structural_identity_change",
        "action_type": "name_change",
        "process_date": "2024-07-11",
        "effective_date": None,
        "old_cusip": "131476103",
        "new_cusip": "131428104",
        "acquiree_symbol": "CLMT",
        "acquirer_symbol": "CLMT",
        "note": (
            "same ticker with legal/CUSIP identity transition; "
            "operational MCT excludes instead of bridging the series"
        ),
    }
}


@dataclass(frozen=True)
class U67OperationalSnapshot:
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    directory: Path


def _write_bytes(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _frame_payload(frame: pd.DataFrame) -> bytes:
    stable = frame.copy()
    stable.index = pd.to_datetime(stable.index, utc=True)
    stable.index.name = "timestamp"
    return stable.to_csv(index=True, float_format="%.17g").encode("utf-8")


def _structural_issue(
    symbol: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    known = KNOWN_U67_STRUCTURAL_EXCLUSIONS.get(
        str(symbol).strip().upper()
    )
    if known is not None:
        return dict(known)
    return structural_identity_issue(symbol, actions)


def download_u67_operational_snapshot(
    *,
    job_id: str,
    config: Any,
    progress_callback: ProgressCallback | None = None,
    data_directory: Path | None = None,
    per_asset_pause_seconds: float = 0.20,
) -> U67OperationalSnapshot:
    """Download all U67 identities fresh from Alpaca for one immutable job."""
    if not str(job_id).strip():
        raise ValueError("A job id is required for the U67 snapshot.")

    analysis_end = str(
        getattr(config, "analysis_end_date", None)
        or getattr(config, "end_date", None)
        or ""
    ).strip()
    if not analysis_end:
        raise ValueError(
            "U67 operational snapshot requires a locked analysis_end_date."
        )

    fresh_config = config.model_copy(
        update={
            "assets": list(U67_REQUESTED_ASSETS),
            "start_date": "2016-01-01",
            "end_date": None,
            "analysis_end_date": analysis_end,
            "timeframe": "1Day",
            "market_data_provider": "alpaca",
            "alpaca_historical_feed": "sip",
            "alpaca_adjustment": "raw",
            "research_market_data_protocol": "raw_total_causal_v1",
            "research_market_data_refresh_mode": "full",
            "research_market_data_mode": "backtest_bootstrap_missing",
        }
    )

    root = (data_directory or DATA_DIRECTORY).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(job_id)
    temporary = root / f".partial-{job_id}"
    if destination.exists() or temporary.exists():
        raise FileExistsError(
            f"U67 operational snapshot already exists: {job_id}"
        )
    temporary.mkdir(parents=True)

    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    total = len(U67_REQUESTED_ASSETS)

    try:
        for position, symbol in enumerate(U67_REQUESTED_ASSETS, start=1):
            if progress_callback is not None:
                progress_callback("download", position - 1, total, symbol)

            raw = _download_alpaca_bars(
                symbol,
                fresh_config,
                str(fresh_config.start_date),
                analysis_end,
                single_request_daily=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError(
                    f"{symbol}: fresh Alpaca RAW/SIP returned no daily bars."
                )
            raw = raw.copy()
            raw.index = pd.to_datetime(raw.index, utc=True)
            raw.index.name = "timestamp"
            if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
                raise ValueError(
                    f"{symbol}: duplicate or unsorted RAW sessions."
                )

            hashes[f"raw_bars/{symbol}.csv"] = _write_bytes(
                temporary,
                f"raw_bars/{symbol}.csv",
                _frame_payload(raw),
            )
            actions, query_start, query_end = _download_corporate_actions(
                symbol,
                fresh_config,
            )
            hashes[f"corporate_actions/{symbol}.json"] = _write_bytes(
                temporary,
                f"corporate_actions/{symbol}.json",
                (
                    json.dumps(
                        actions,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                ).encode("utf-8"),
            )

            record: dict[str, Any] = {
                "raw_rows": int(len(raw)),
                "raw_history_sha256": _history_frame_sha256(raw),
                "corporate_actions": int(len(actions)),
                "corporate_actions_query_start": query_start,
                "corporate_actions_query_end": query_end,
            }
            issue = _structural_issue(symbol, actions)
            if issue is not None:
                record["status"] = "excluded_structural_identity"
                record["exclusion"] = issue
                records[symbol] = record
                exclusions.append(dict(issue))
                if progress_callback is not None:
                    progress_callback("excluded", position, total, symbol)
                continue

            normalized, applied = split_normalize(raw, actions)
            asset_config = fresh_config.model_copy(
                update={"market_data_require_complete_history": False}
            )
            cleaned = validate_and_clean_bars(normalized, asset_config)
            if cleaned is None or cleaned.empty:
                raise RuntimeError(
                    f"{symbol}: no usable bars after U67 normalization."
                )
            if (
                pd.DatetimeIndex(cleaned.index).date
                > pd.Timestamp(analysis_end).date()
            ).any():
                raise RuntimeError(
                    f"{symbol}: snapshot contains a future session."
                )

            provenance = dict(getattr(cleaned, "attrs", {}) or {})
            provenance.update(
                {
                    "research_market_data_protocol": "raw_total_causal_v1",
                    "research_access_path": "fresh_u67_operational_snapshot",
                    "source_adjustment": "raw",
                    "effective_adjustment": (
                        "raw_plus_causal_split_normalization"
                    ),
                    "corporate_action_source": "alpaca",
                    "corporate_action_count": int(len(actions)),
                    "splits_applied": int(len(applied)),
                    "split_events": applied,
                    "split_normalization_direction": "pre_ex_date_history",
                    "split_normalization_uses_future_events": True,
                    "dividend_adjustment_applied": False,
                    "structural_identity_verified": True,
                    "fresh_snapshot_job_id": str(job_id),
                }
            )
            cleaned.attrs.update(provenance)

            hashes[f"normalized_bars/{symbol}.csv"] = _write_bytes(
                temporary,
                f"normalized_bars/{symbol}.csv",
                _frame_payload(cleaned),
            )
            record.update(
                {
                    "status": "eligible",
                    "normalized_rows": int(len(cleaned)),
                    "normalized_history_sha256": _history_frame_sha256(
                        cleaned
                    ),
                    "first_session": pd.Timestamp(
                        cleaned.index.min()
                    ).date().isoformat(),
                    "last_session": pd.Timestamp(
                        cleaned.index.max()
                    ).date().isoformat(),
                    "splits_applied": int(len(applied)),
                    "split_events": applied,
                }
            )
            records[symbol] = record
            frames[symbol] = cleaned

            if progress_callback is not None:
                progress_callback("download", position, total, symbol)
            if per_asset_pause_seconds > 0 and position != total:
                time.sleep(per_asset_pause_seconds)

        if len(frames) < 2:
            raise RuntimeError(
                "U67 operational snapshot has fewer than two eligible assets."
            )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "source_contract": SOURCE_CONTRACT,
            "source": "alpaca",
            "feed": "sip",
            "download_adjustment": "raw",
            "effective_adjustment": "raw_plus_split_normalization",
            "dividend_adjustment_applied": False,
            "start_date": "2016-01-01",
            "analysis_end_date": analysis_end,
            "downloaded_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "requested_assets": list(U67_REQUESTED_ASSETS),
            "eligible_assets": [
                symbol
                for symbol in U67_REQUESTED_ASSETS
                if symbol in frames
            ],
            "structural_exclusions": exclusions,
            "per_asset": records,
            "file_hashes": hashes,
        }
        canonical = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        manifest["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
        _write_bytes(
            temporary,
            "manifest.json",
            (
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            ).encode("utf-8"),
        )
        os.replace(temporary, destination)
        return U67OperationalSnapshot(frames, manifest, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
