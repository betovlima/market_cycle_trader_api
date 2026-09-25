"""Read-only Fold-1 data parity audit: TCC frozen CSV vs MCT Strategy #11/#12.

Run outside Model Tuning. No Alpaca calls, no Mongo writes, no TCC writes.
The MCT cache is mutable: historical comparisons are allowed only when the
cache's raw and normalized hashes match the archived Strategy #11 manifest.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from market_cycle_trader_api.core.environment import load_project_environment

# mongo_repository resolves MONGO_URI/MONGO_DATABASE at import time.
# Load the same local environment as the API BEFORE importing engine modules,
# which import mongo_repository transitively.
load_project_environment()

from market_cycle_trader_api.engine.market_data import (
    REQUIRED_BAR_COLUMNS,
    _history_frame_sha256,
    _read_frame,
    inclusive_end_exclusive_boundary,
    validate_and_clean_bars,
)
from market_cycle_trader_api.engine.research_market_data import (
    RAW_TOTAL_CAUSAL_PROTOCOL,
    split_normalize,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    ALPACA_CORPORATE_ACTIONS_COLLECTION,
    ALPACA_MARKET_BARS_COLLECTION,
    create_client,
    get_database,
)
from market_cycle_trader_api.services.reproducibility import market_data_manifest
from market_cycle_trader_api.tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES,
    build_rotation_frame,
)
from market_cycle_trader_api.tcc_v106_reference.config import (
    CONFIG as TCC_CONFIG,
    build_control_config,
)

TCC_FROZEN_MANIFEST_SHA = (
    "4e2fd225cc0ea05da56dad8f0628ca989ad332812a5fa3a7dc796b2b8a6d5128"
)
TCC_SOURCE_COMMIT = "f9cf29fdb736676d0d3e26780481be99813c602a"
TCC_SOURCE_ENGINE = "v1.0.6"
CUTOFF = "2026-09-17"
FOLD1_CALIBRATION_END = pd.Timestamp("2020-04-25", tz="UTC")
NUMERIC_ATOL = 1e-12
NUMERIC_RTOL = 1e-12
RAW_COLUMNS = ("open", "high", "low", "close", "volume", "vwap", "trade_count")


def _load_zip_manifest(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        return json.loads(archive.read("experiment_manifest.json"))


def _assert_reference_files(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_sha256") != TCC_FROZEN_MANIFEST_SHA:
        raise ValueError(
            "TCC frozen snapshot mismatch. Expected "
            f"{TCC_FROZEN_MANIFEST_SHA}; observed "
            f"{manifest.get('snapshot_sha256')!r}."
        )
    bars = manifest.get("bars") or {}
    if (
        bars.get("adjustment") != "raw"
        or bars.get("feed") != "sip"
        or bars.get("timeframe") != "1Day"
        or bars.get("bar_snapshot_as_of_end") != CUTOFF
    ):
        raise ValueError("TCC frozen data contract does not match this audit.")
    for relative, expected in sorted((manifest.get("file_hashes") or {}).items()):
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"TCC frozen file is missing: {target}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"TCC frozen CSV changed: {relative}")
    return manifest


def _frozen_frame(root: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "raw_bars" / f"{symbol}.csv")
    if "timestamp" not in frame:
        raise ValueError(f"TCC {symbol} raw CSV has no timestamp.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    for column in RAW_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _frozen_actions(root: Path, symbol: str) -> list[dict[str, Any]]:
    with (root / "corporate_actions" / f"{symbol}.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return [dict(item) for item in csv.DictReader(stream)]


def _canonical_ca_value(key: str, value: Any) -> str:
    """Compare numerical Corporate Action fields by value, not CSV formatting."""
    if value is None or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    # Keep identifiers, dates and ticker symbols as text. E.g. a CUSIP
    # may have significant leading zeroes and must never be parsed as float.
    numeric = (
        key in {"rate", "new_rate", "old_rate", "acquirer_rate", "acquiree_rate"}
        or key.endswith(("_amount", "_ratio", "_percent", "_percentage"))
    )
    if numeric:
        try:
            number = Decimal(raw)
            if number.is_finite():
                return format(number.normalize(), "f")
        except InvalidOperation:
            pass
    return raw


def _clean_actions(actions: list[dict[str, Any]], fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action in actions:
        row: dict[str, str] = {
            key: _canonical_ca_value(key, action.get(key))
            for key in fields
        }
        rows.append(row)
    return sorted(rows, key=lambda x: json.dumps(x, sort_keys=True))


def _split_safe_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent pandas >= 2 raising when fractional split volumes hit int64.

    This only casts audit-local copies; it neither edits the TCC files nor
    alters the frozen v1.0.6 split-normalization implementation.
    """
    converted = frame.copy()
    for column in REQUIRED_BAR_COLUMNS:
        if column in converted:
            converted[column] = pd.to_numeric(
                converted[column], errors="coerce"
            ).astype(np.float64)
    converted.attrs.update(getattr(frame, "attrs", {}))
    return converted


def _compare_numeric_frames(
    *,
    symbol: str,
    stage: str,
    frozen: pd.DataFrame,
    mct: pd.DataFrame,
    columns: list[str],
    details: list[dict[str, Any]],
    sample_per_column: int = 2,
) -> dict[str, Any]:
    left = frozen.copy().sort_index()
    right = mct.copy().sort_index()
    left.index = pd.to_datetime(left.index, utc=True)
    right.index = pd.to_datetime(right.index, utc=True)
    common = left.index.intersection(right.index)
    missing_tcc = right.index.difference(left.index)
    missing_mct = left.index.difference(right.index)
    raw_differences = 0
    substantial_differences = 0
    first_substantial = None
    for column in columns:
        if column not in left or column not in right:
            if column not in left and column not in right:
                continue
            raw_differences += len(common)
            substantial_differences += len(common)
            if first_substantial is None:
                first_substantial = {
                    "timestamp": None, "column": column,
                    "frozen": "missing" if column not in left else "present",
                    "mct": "missing" if column not in right else "present",
                }
            continue
        a = pd.to_numeric(left.loc[common, column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        b = pd.to_numeric(right.loc[common, column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        differing = np.flatnonzero(~equal)
        raw_differences += len(differing)
        close = np.isclose(
            a, b, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL, equal_nan=True
        )
        substantial = np.flatnonzero(~close)
        substantial_differences += len(substantial)
        if len(substantial) and (
            first_substantial is None
            or common[substantial[0]].isoformat()
            < str(first_substantial.get("timestamp") or "9999")
        ):
            j = int(substantial[0])
            first_substantial = {
                "timestamp": common[j].isoformat(),
                "column": column,
                "frozen": repr(float(a[j])),
                "mct": repr(float(b[j])),
            }
        for j in differing[:sample_per_column]:
            j = int(j)
            details.append(
                {
                    "symbol": symbol,
                    "stage": stage,
                    "timestamp": common[j].isoformat(),
                    "column": column,
                    "tcc_value": repr(float(a[j])),
                    "mct_value": repr(float(b[j])),
                    "substantial": bool(not close[j]),
                }
            )
    return {
        "symbol": symbol,
        "stage": stage,
        "frozen_rows": int(len(left)),
        "mct_rows": int(len(right)),
        "missing_in_tcc": int(len(missing_tcc)),
        "missing_in_mct": int(len(missing_mct)),
        "first_missing_in_tcc": (
            missing_tcc.min().isoformat() if len(missing_tcc) else None
        ),
        "first_missing_in_mct": (
            missing_mct.min().isoformat() if len(missing_mct) else None
        ),
        "exact_different_values": int(raw_differences),
        "substantial_different_values": int(substantial_differences),
        "first_substantial_difference": first_substantial,
        "status": (
            "DIFFERENT"
            if substantial_differences or len(missing_tcc) or len(missing_mct)
            else "PRECISION_ONLY" if raw_differences else "MATCH"
        ),
    }


def _write_outputs(
    output: Path,
    report: dict[str, Any],
    summaries: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    (output / "report.json").write_text(payload + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(details).to_csv(output / "first_differences.csv", index=False)
    with ZipFile(output / "tcc_v106_fold1_parity_audit.zip", "w", ZIP_DEFLATED) as archive:
        for name in ("report.json", "summary.csv", "first_differences.csv"):
            archive.write(output / name, arcname=name)


def audit(
    frozen_data: Path,
    strategy11_zip: Path,
    strategy12_zip: Path,
    output: Path,
) -> dict[str, Any]:
    frozen_manifest = _assert_reference_files(frozen_data)
    old = _load_zip_manifest(strategy11_zip)
    current = _load_zip_manifest(strategy12_zip)
    if old.get("strategy_profile_name") != "Strategy #11":
        raise ValueError("First MCT ZIP is not Strategy #11.")
    if current.get("strategy_profile_name") != "Strategy #12":
        raise ValueError("Second MCT ZIP is not Strategy #12.")
    if (
        not old.get("market_data_signature_sha256")
        or old.get("market_data_signature_sha256")
        != current.get("market_data_signature_sha256")
    ):
        raise ValueError("Strategy #11 and #12 used different market-data signatures.")
    signatures = old.get("market_data_signatures") or {}
    if not signatures:
        raise ValueError("Strategy #11 archive has no per-asset data signatures.")

    symbols = [
        symbol for symbol in TCC_CONFIG.assets
        if symbol != "DOC" and symbol in signatures
    ]
    if len(symbols) != 55:
        raise ValueError(f"Expected 55 eligible symbols; observed {len(symbols)}.")

    config = build_control_config(TCC_CONFIG)
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    stale_assets: list[str] = []
    errors: list[dict[str, str]] = []
    start = pd.Timestamp("2016-01-01", tz="UTC")
    end = inclusive_end_exclusive_boundary(CUTOFF)

    # Environment was initialized before mongo_repository was imported above.
    client = create_client()
    try:
        db = get_database(client)
        bars_collection = db[ALPACA_MARKET_BARS_COLLECTION]
        actions_collection = db[ALPACA_CORPORATE_ACTIONS_COLLECTION]
        for index, symbol in enumerate(symbols, 1):
            print(f"[{index:02d}/55] {symbol}", flush=True)
            try:
                raw = _read_frame(
                    bars_collection,
                    {
                        "symbol": symbol, "interval": "1Day",
                        "feed": "sip", "adjustment": "raw",
                    },
                    start, end,
                )
                expected = signatures[symbol]
                raw_hash = _history_frame_sha256(raw)
                if raw_hash != expected.get("raw_sha256"):
                    stale_assets.append(symbol)
                    summaries.append({
                        "symbol": symbol, "stage": "snapshot",
                        "status": "STALE_MCT_CACHE",
                        "expected_raw_sha256": expected.get("raw_sha256"),
                        "observed_raw_sha256": raw_hash,
                    })
                    continue

                cached = actions_collection.find_one(
                    {"symbol": symbol, "protocol": RAW_TOTAL_CAUSAL_PROTOCOL},
                    {"_id": 0},
                )
                if (
                    cached is None
                    or str(cached.get("query_end") or "") < CUTOFF
                ):
                    errors.append({
                        "symbol": symbol,
                        "error": "Corporate Actions snapshot unavailable through cutoff.",
                    })
                    continue
                mct_actions = [
                    dict(item) for item in (cached.get("actions") or [])
                    if isinstance(item, dict)
                ]
                tcc_actions = _frozen_actions(frozen_data, symbol)
                tcc_raw = _frozen_frame(frozen_data, symbol)
                summaries.append(_compare_numeric_frames(
                    symbol=symbol, stage="raw_bars", frozen=tcc_raw,
                    mct=raw, columns=list(RAW_COLUMNS), details=details,
                ))

                fields = []
                with (frozen_data / "corporate_actions" / f"{symbol}.csv").open(
                    encoding="utf-8", newline=""
                ) as source:
                    fields = list(csv.DictReader(source).fieldnames or [])
                frozen_ca = _clean_actions(tcc_actions, fields)
                mct_ca = _clean_actions(mct_actions, fields)
                summaries.append({
                    "symbol": symbol, "stage": "corporate_actions",
                    "status": "MATCH" if frozen_ca == mct_ca else "DIFFERENT",
                    "tcc_count": len(frozen_ca), "mct_count": len(mct_ca),
                    "first_substantial_difference": (
                        None if frozen_ca == mct_ca else {
                            "tcc": next(
                                (v for v in frozen_ca if v not in mct_ca),
                                None,
                            ),
                            "mct": next(
                                (v for v in mct_ca if v not in frozen_ca),
                                None,
                            ),
                        }
                    ),
                })

                frozen_normalized, frozen_splits = split_normalize(
                    _split_safe_float_frame(tcc_raw), tcc_actions,
                )
                mct_normalized, mct_splits = split_normalize(
                    _split_safe_float_frame(raw), mct_actions,
                )
                frozen_normalized = validate_and_clean_bars(
                    frozen_normalized, config,
                )
                mct_normalized = validate_and_clean_bars(
                    mct_normalized, config,
                )
                actual_normalized_hash = market_data_manifest(
                    {symbol: mct_normalized}
                )[1][symbol]["sha256"]
                if actual_normalized_hash != expected.get("sha256"):
                    stale_assets.append(symbol)
                    summaries.append({
                        "symbol": symbol, "stage": "snapshot",
                        "status": "STALE_MCT_ACTIONS_OR_PREPARATION",
                        "expected_normalized_sha256": expected.get("sha256"),
                        "observed_normalized_sha256": actual_normalized_hash,
                    })
                    continue
                summaries.append({
                    "symbol": symbol, "stage": "split_events",
                    "status": (
                        "MATCH" if frozen_splits == mct_splits else "DIFFERENT"
                    ),
                    "tcc_count": len(frozen_splits),
                    "mct_count": len(mct_splits),
                })
                summaries.append(_compare_numeric_frames(
                    symbol=symbol, stage="normalized_ohlcv",
                    frozen=frozen_normalized, mct=mct_normalized,
                    columns=list(REQUIRED_BAR_COLUMNS), details=details,
                ))

                tcc_features = build_rotation_frame(frozen_normalized, config)
                mct_features = build_rotation_frame(mct_normalized, config)
                tcc_features = tcc_features.loc[
                    tcc_features.index < FOLD1_CALIBRATION_END
                ]
                mct_features = mct_features.loc[
                    mct_features.index < FOLD1_CALIBRATION_END
                ]
                features = [
                    col for col in tcc_features.columns
                    if col in ROTATION_FEATURES
                    or col.startswith("forward_")
                    or col.startswith("target_")
                ]
                summaries.append(_compare_numeric_frames(
                    symbol=symbol, stage="fold1_features_targets",
                    frozen=tcc_features, mct=mct_features,
                    columns=features, details=details,
                ))
            except Exception as exc:
                # Pandas includes entire vectors in dtype errors; report a
                # bounded message so status JSON stays useful/readable.
                errors.append({
                    "symbol": symbol,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })

    finally:
        client.close()

    report = {
        "schema_version": 1,
        "audit": "tcc_v106_fold1_parity_read_only",
        "tcc_source_commit": TCC_SOURCE_COMMIT,
        "tcc_source_engine_tag": TCC_SOURCE_ENGINE,
        "tcc_snapshot_sha256": TCC_FROZEN_MANIFEST_SHA,
        "verified_tcc_files": len(frozen_manifest.get("file_hashes") or {}),
        "mct_job11_id": old.get("job_id"),
        "mct_job12_id": current.get("job_id"),
        "mct_archived_data_signature_sha256": old.get(
            "market_data_signature_sha256"
        ),
        "matched_mct_archived_signatures": True,
        "mct_cache_is_mutable": True,
        "mct_stale_assets": stale_assets,
        "errors": errors,
        "stage_status_counts": {
            stage: {
                status: sum(
                    1 for row in summaries
                    if row["stage"] == stage and row["status"] == status
                )
                for status in sorted(
                    set(
                        row["status"] for row in summaries
                        if row["stage"] == stage
                    )
                )
            }
            for stage in sorted(set(row["stage"] for row in summaries))
        },
        "method": (
            "Read-only Mongo RAW and Corporate Actions vs manifest-verified "
            "frozen TCC CSVs; exact and 1e-12 numerical comparisons; "
            "feature/target preparation uses the verbatim TCC v1.0.6 engine. "
            "Any cache-hash mismatch is reported as STALE, not TCC divergence."
        ),
        "next_if_data_match": (
            "Audit Fold-1 model input row ordering, LightGBM runtime, "
            "calibration predictions and candidate scores; never change TCC."
        ),
    }
    _write_outputs(output, report, summaries, details)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tcc-data", type=Path, required=True,
        help="Read-only path to official TCC main dados/pesquisa directory.",
    )
    parser.add_argument("--strategy11-zip", type=Path, required=True)
    parser.add_argument("--strategy12-zip", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output") / "tcc_v106_fold1_parity_audit",
    )
    args = parser.parse_args()
    result = audit(
        args.tcc_data, args.strategy11_zip,
        args.strategy12_zip, args.output,
    )
    print(json.dumps(
        {
            "output_zip": str(
                args.output / "tcc_v106_fold1_parity_audit.zip"
            ),
            "mct_stale_assets": result["mct_stale_assets"],
            "stage_status_counts": result["stage_status_counts"],
            "errors": result["errors"],
        }, ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
