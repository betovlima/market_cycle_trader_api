from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from pymongo import ASCENDING, UpdateOne


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.infrastructure.market_data.alpaca import download_stock_bars
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    JOBS_COLLECTION,
    create_client,
    get_alpaca_credentials,
    get_database,
    utc_now,
)


DEFAULT_TARGET_COLLECTION = "alpaca_market_bars_fresh_20260919"
MANIFEST_COLLECTION = "market_data_snapshot_manifests"
SCRIPT_VERSION = "fresh-alpaca-snapshot-v1.0.0"


def _latest_job(db: Any, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = db[JOBS_COLLECTION].find_one({"id": str(job_id)})
    else:
        job = db[JOBS_COLLECTION].find_one(
            {"internal_job": {"$ne": True}, "status": "completed"},
            sort=[("finished_at", -1), ("created_at", -1)],
        )
    if job is None or not isinstance(job.get("request"), dict):
        raise RuntimeError("No completed Backtest job with an immutable request snapshot was found.")
    return job


def _guard_target_collection(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Target collection cannot be empty.")
    if normalized == ALPACA_MARKET_BARS_COLLECTION:
        raise ValueError(
            "Refusing to write to alpaca_market_bars. "
            "The historical Alpaca cache must remain untouched."
        )
    if not normalized.startswith("alpaca_market_bars_fresh_"):
        raise ValueError(
            "Fresh Alpaca snapshots must use a collection beginning with "
            "'alpaca_market_bars_fresh_'."
        )
    return normalized


def _frame_documents(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    feed: str,
    adjustment: str,
    snapshot_id: str,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if frame is None or frame.empty:
        return docs

    for timestamp, row in frame.iterrows():
        doc: dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "timestamp": pd.Timestamp(timestamp).to_pydatetime(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "snapshot_id": snapshot_id,
            "source": "alpaca_api_fresh",
        }
        for column in ("vwap", "trade_count"):
            if column in row.index and pd.notna(row[column]):
                doc[column] = float(row[column])
        docs.append(doc)
    return docs


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    canonical = []
    for row in rows:
        canonical.append(
            {
                "symbol": row["symbol"],
                "interval": row["interval"],
                "feed": row["feed"],
                "adjustment": row["adjustment"],
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a fresh Alpaca historical snapshot directly from the Alpaca API "
            "into a new MongoDB collection without modifying alpaca_market_bars."
        )
    )
    parser.add_argument("--job-id", default=None, help="Backtest job whose immutable request defines the universe.")
    parser.add_argument(
        "--target-collection",
        default=DEFAULT_TARGET_COLLECTION,
        help="New MongoDB collection for the fresh snapshot.",
    )
    parser.add_argument(
        "--replace-target",
        action="store_true",
        help="Delete only the selected fresh target collection before downloading.",
    )
    args = parser.parse_args()

    target_name = _guard_target_collection(args.target_collection)
    snapshot_id = target_name.removeprefix("alpaca_market_bars_fresh_")

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = job["request"]

        assets = [str(item).strip().upper() for item in request.get("assets", []) if str(item).strip()]
        if not assets:
            raise RuntimeError("The selected job contains no assets.")

        timeframe = str(request.get("timeframe") or "1Day")
        feed = str(request.get("alpaca_historical_feed") or "sip").lower()
        adjustment = str(request.get("alpaca_adjustment") or "all").lower()
        start = pd.Timestamp(request["start_date"])
        end_text = request.get("analysis_end_date") or request.get("end_date")
        end = pd.Timestamp(end_text) if end_text else pd.Timestamp.utcnow()
        # Alpaca request end is treated as an upper bound. Add one calendar day
        # so the configured final market session is included.
        api_end = end + pd.Timedelta(days=1)

        target = db[target_name]
        existing = target.estimated_document_count()
        if existing:
            if not args.replace_target:
                raise RuntimeError(
                    f"Target collection {target_name!r} already contains {existing} rows. "
                    "Use a new collection name or pass --replace-target. "
                    "The legacy alpaca_market_bars collection is never modified."
                )
            target.drop()
            target = db[target_name]

        target.create_index(
            [
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
                ("feed", ASCENDING),
                ("adjustment", ASCENDING),
                ("timestamp", ASCENDING),
            ],
            unique=True,
            name="uq_fresh_alpaca_bar",
        )
        target.create_index(
            [("snapshot_id", ASCENDING), ("symbol", ASCENDING)],
            name="ix_fresh_alpaca_snapshot_symbol",
        )

        credentials = get_alpaca_credentials(db)
        started_at = utc_now()
        total_rows = 0
        symbol_summaries: list[dict[str, Any]] = []
        all_hash_rows: list[dict[str, Any]] = []

        print(
            f"[fresh-alpaca] job={job.get('id')} assets={len(assets)} "
            f"target={target_name} feed={feed} adjustment={adjustment} "
            f"start={start.date()} end={end.date()}",
            flush=True,
        )
        print(
            f"[fresh-alpaca] protected_collection={ALPACA_MARKET_BARS_COLLECTION} untouched=true",
            flush=True,
        )

        for position, symbol in enumerate(assets, start=1):
            print(f"[fresh-alpaca] {position}/{len(assets)} downloading {symbol}...", flush=True)
            frame = download_stock_bars(
                api_key_id=credentials["api_key_id"],
                secret_key=credentials["secret_key"],
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=api_end,
                feed=feed,
                adjustment=adjustment,
            )
            docs = _frame_documents(
                frame,
                symbol=symbol,
                timeframe=timeframe,
                feed=feed,
                adjustment=adjustment,
                snapshot_id=snapshot_id,
            )
            if not docs:
                raise RuntimeError(f"Fresh Alpaca API returned no rows for {symbol}.")

            operations = [
                UpdateOne(
                    {
                        "symbol": doc["symbol"],
                        "interval": doc["interval"],
                        "feed": doc["feed"],
                        "adjustment": doc["adjustment"],
                        "timestamp": doc["timestamp"],
                    },
                    {"$set": doc},
                    upsert=True,
                )
                for doc in docs
            ]
            target.bulk_write(operations, ordered=True)
            all_hash_rows.extend(docs)
            total_rows += len(docs)
            first = pd.Timestamp(frame.index.min()).date().isoformat()
            last = pd.Timestamp(frame.index.max()).date().isoformat()
            symbol_summaries.append(
                {
                    "symbol": symbol,
                    "rows": len(docs),
                    "first": first,
                    "last": last,
                }
            )
            print(
                f"[fresh-alpaca] {symbol} rows={len(docs)} first={first} last={last}",
                flush=True,
            )

        finished_at = utc_now()
        manifest = {
            "snapshot_id": snapshot_id,
            "script_version": SCRIPT_VERSION,
            "source": "alpaca_api",
            "source_job_id": job.get("id"),
            "source_job_api_version": job.get("api_version"),
            "target_collection": target_name,
            "protected_legacy_collection": ALPACA_MARKET_BARS_COLLECTION,
            "legacy_collection_modified": False,
            "feed": feed,
            "adjustment": adjustment,
            "timeframe": timeframe,
            "requested_start": start.date().isoformat(),
            "requested_end": end.date().isoformat(),
            "asset_count": len(assets),
            "assets": assets,
            "total_rows": total_rows,
            "sha256": _sha256_rows(all_hash_rows),
            "started_at": started_at,
            "finished_at": finished_at,
            "symbols": symbol_summaries,
        }
        db[MANIFEST_COLLECTION].replace_one(
            {"snapshot_id": snapshot_id},
            manifest,
            upsert=True,
        )

        print("", flush=True)
        print("[fresh-alpaca] completed", flush=True)
        print(f"[fresh-alpaca] collection={target_name}", flush=True)
        print(f"[fresh-alpaca] total_rows={total_rows}", flush=True)
        print(f"[fresh-alpaca] sha256={manifest['sha256']}", flush=True)
        print(f"[fresh-alpaca] legacy_collection_modified=false", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
