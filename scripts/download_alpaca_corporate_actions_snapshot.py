from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import requests
from pymongo import ASCENDING


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    create_client,
    get_alpaca_credentials,
    get_database,
    utc_now,
)


ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
DEFAULT_COLLECTION = "alpaca_corporate_actions_20260919"
MANIFEST_COLLECTION = "market_data_snapshot_manifests"
SCRIPT_VERSION = "alpaca-corporate-actions-snapshot-v1.0.0"
REQUEST_TYPES = (
    "forward_split",
    "reverse_split",
    "unit_split",
    "cash_dividend",
    "stock_dividend",
    "spin_off",
)
ARRAY_TO_TYPE = {
    "forward_splits": "forward_split",
    "reverse_splits": "reverse_split",
    "unit_splits": "unit_split",
    "cash_dividends": "cash_dividend",
    "stock_dividends": "stock_dividend",
    "spin_offs": "spin_off",
}


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


def _flatten_page(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return result
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = ARRAY_TO_TYPE.get(str(array_name), str(array_name))
        for value in values:
            if not isinstance(value, dict):
                continue
            document = dict(value)
            document["action_type"] = action_type
            document["source_array"] = str(array_name)
            result.append(document)
    return result


def _request_page(
    *,
    headers: dict[str, str],
    symbols: list[str],
    start: str,
    end: str,
    page_token: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "types": ",".join(REQUEST_TYPES),
        "start": start,
        "end": end,
        "region": "us",
        "data_quality": "complete",
        "limit": 1000,
        "sort": "asc",
    }
    if page_token:
        params["page_token"] = page_token

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(ENDPOINT, headers=headers, params=params, timeout=30)
            if response.status_code == 429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected Alpaca corporate-actions response.")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Alpaca corporate-actions request failed: {last_error}") from last_error


def _sha256_documents(documents: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted(
            (
                {
                    key: value
                    for key, value in document.items()
                    if key not in {"downloaded_at", "_id"}
                }
                for document in documents
            ),
            key=lambda item: (
                str(item.get("action_type") or ""),
                str(item.get("symbol") or item.get("source_symbol") or ""),
                str(item.get("process_date") or ""),
                str(item.get("id") or ""),
            ),
        ),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a frozen Alpaca corporate-actions snapshot for point-in-time "
            "research without modifying market bars."
        )
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--replace-target", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Symbols per Alpaca request.",
    )
    args = parser.parse_args()

    collection_name = str(args.collection or "").strip()
    if not collection_name.startswith("alpaca_corporate_actions_"):
        raise ValueError("Corporate-action snapshot collection must start with 'alpaca_corporate_actions_'.")

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = job["request"]
        symbols = sorted({str(item).strip().upper() for item in request.get("assets", []) if str(item).strip()})
        if not symbols:
            raise RuntimeError("Selected job contains no assets.")

        start = str(request.get("start_date"))
        end = str(request.get("analysis_end_date") or request.get("end_date"))
        if not start or not end:
            raise RuntimeError("Selected job must contain a closed research date range.")

        target = db[collection_name]
        existing = target.estimated_document_count()
        if existing:
            if not args.replace_target:
                raise RuntimeError(
                    f"Target collection {collection_name!r} already contains {existing} documents. "
                    "Use a new collection or --replace-target."
                )
            target.drop()
            target = db[collection_name]

        target.create_index([("action_type", ASCENDING), ("symbol", ASCENDING), ("process_date", ASCENDING)], name="ix_ca_type_symbol_process")
        target.create_index([("symbol", ASCENDING), ("ex_date", ASCENDING)], name="ix_ca_symbol_ex")
        target.create_index([("id", ASCENDING)], unique=True, sparse=True, name="uq_ca_id")

        credentials = get_alpaca_credentials(db)
        headers = {
            "APCA-API-KEY-ID": credentials["api_key_id"],
            "APCA-API-SECRET-KEY": credentials["secret_key"],
        }

        downloaded_at = utc_now()
        all_documents: list[dict[str, Any]] = []
        chunk_size = max(1, min(100, int(args.chunk_size)))
        for offset in range(0, len(symbols), chunk_size):
            chunk = symbols[offset:offset + chunk_size]
            page_token: str | None = None
            while True:
                payload = _request_page(
                    headers=headers,
                    symbols=chunk,
                    start=start,
                    end=end,
                    page_token=page_token,
                )
                documents = _flatten_page(payload)
                for document in documents:
                    document["downloaded_at"] = downloaded_at
                    document["snapshot_collection"] = collection_name
                    document["source"] = "alpaca_corporate_actions_api"
                if documents:
                    target.insert_many(documents, ordered=False)
                    all_documents.extend(documents)

                page_token = payload.get("next_page_token")
                if not page_token:
                    break

            print(
                f"[corporate-actions] symbols={chunk[0]}..{chunk[-1]} "
                f"progress={min(offset + len(chunk), len(symbols))}/{len(symbols)}",
                flush=True,
            )

        counts: dict[str, int] = {}
        for document in all_documents:
            key = str(document.get("action_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1

        sha256 = _sha256_documents(all_documents)
        manifest = {
            "snapshot_id": collection_name.removeprefix("alpaca_corporate_actions_"),
            "script_version": SCRIPT_VERSION,
            "source": "alpaca_corporate_actions_api",
            "source_job_id": job.get("id"),
            "target_collection": collection_name,
            "requested_start": start,
            "requested_end": end,
            "asset_count": len(symbols),
            "assets": symbols,
            "types": list(REQUEST_TYPES),
            "document_count": len(all_documents),
            "counts_by_type": counts,
            "sha256": sha256,
            "downloaded_at": downloaded_at,
        }
        db[MANIFEST_COLLECTION].replace_one(
            {"source": "alpaca_corporate_actions_api", "target_collection": collection_name},
            manifest,
            upsert=True,
        )

        print("")
        print("[corporate-actions] completed")
        print(f"[corporate-actions] collection={collection_name}")
        print(f"[corporate-actions] documents={len(all_documents)}")
        print(f"[corporate-actions] counts={json.dumps(counts, sort_keys=True)}")
        print(f"[corporate-actions] sha256={sha256}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
