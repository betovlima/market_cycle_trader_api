from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from pymongo import ReplaceOne
from pymongo.errors import AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError

T = TypeVar("T")
_TRANSIENT_MONGO_ERRORS = (AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mongo_value(value: Any) -> Any:
    """Convert pandas/numpy values into BSON-safe Python values.

    Research traces contain nullable timestamps.  pandas.NaT cannot be encoded by
    PyMongo because its utcoffset implementation raises ValueError, so missing
    temporal values must be normalized to None before any MongoDB write.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): mongo_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [mongo_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.to_pydatetime()
    if isinstance(value, np.generic):
        return mongo_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Catch scalar pandas missing values that are not pd.NaT/pd.NA while avoiding
    # array-like pd.isna() results with ambiguous truth values.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


def mongo_retry(
    operation: Callable[[], T],
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 0.25,
) -> T:
    """Retry only transient local-Mongo failures with bounded exponential backoff."""
    last_error: BaseException | None = None
    delay = float(initial_delay_seconds)
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return operation()
        except _TRANSIENT_MONGO_ERRORS as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 4.0)
    assert last_error is not None
    raise last_error


def runtime_mongo_settings(args: Any, mongo_repository: Any, base: Any) -> tuple[str, str]:
    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or ""
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not mongo_uri:
        raise RuntimeError("MONGO_URL/MONGO_URI is required in .env or --mongo-uri.")
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or --database.")
    base._assert_local_mongo(mongo_uri, allow_remote=False)
    mongo_repository.MONGO_URI = mongo_uri
    mongo_repository.MONGO_DATABASE = database_name
    return mongo_uri, database_name


def persist_capture(
    *,
    run_id: str,
    decision_date: str,
    universe_name: str,
    candidate: str | None,
    arm: str,
    captured: list[Any],
    trace_runs: Any,
    trace_rows: Any,
) -> None:
    """Persist one replay capture idempotently.

    Every document has a deterministic id. Re-running the same arm after a crash
    replaces/upserts the same records instead of creating duplicates, making a
    long campaign safe to resume from the last completed observation.
    """
    for result_index, result in enumerate(captured, start=1):
        trace_id = f"{run_id}:{decision_date}:{universe_name}:{candidate or 'BASELINE'}:{arm}:{result_index}"
        metrics = dict(getattr(result, "metrics", {}) or {})
        trace_document = mongo_value(
            {
                "_id": trace_id,
                "run_id": run_id,
                "decision_date": decision_date,
                "universe_name": universe_name,
                "candidate": candidate,
                "arm": arm,
                "result_index": result_index,
                "backend": str(getattr(result, "backend", "")),
                "metrics": metrics,
                "summary": str(getattr(result, "summary", "") or ""),
            }
        )
        mongo_retry(lambda: trace_runs.replace_one({"_id": trace_id}, trace_document, upsert=True))

        rows: list[dict[str, Any]] = []
        predictions = getattr(result, "predictions", None)
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            frame = predictions.reset_index()
            for row_index, record in enumerate(frame.to_dict(orient="records")):
                row_id = f"{trace_id}:prediction:{row_index}"
                rows.append(
                    mongo_value(
                        {
                            "_id": row_id,
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "row_type": "prediction",
                            "row_index": row_index,
                            "payload": record,
                        }
                    )
                )
        trades = getattr(result, "trades", None)
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            for row_index, record in enumerate(trades.to_dict(orient="records")):
                row_id = f"{trace_id}:trade:{row_index}"
                rows.append(
                    mongo_value(
                        {
                            "_id": row_id,
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "row_type": "trade",
                            "row_index": row_index,
                            "payload": record,
                        }
                    )
                )
        if rows:
            operations = [ReplaceOne({"_id": row["_id"]}, row, upsert=True) for row in rows]
            # Keep batches modest so one large trace cannot exceed MongoDB message limits.
            for start in range(0, len(operations), 200):
                batch = operations[start:start + 200]
                mongo_retry(lambda batch=batch: trace_rows.bulk_write(batch, ordered=False))


def upsert_observation(collection: Any, document: dict[str, Any]) -> None:
    payload = mongo_value(document)
    identity = {
        "run_id": payload["run_id"],
        "decision_date": payload["decision_date"],
        "universe_name": payload["universe_name"],
        "candidate": payload["candidate"],
    }
    mongo_retry(lambda: collection.replace_one(identity, payload, upsert=True))


def export_result(
    *,
    project_root: Path,
    run_id: str,
    script_version: str,
    dataset: pd.DataFrame,
    summary: dict[str, Any],
    export_folder_name: str,
    export_zip_name: str,
    collections: tuple[str, ...],
) -> tuple[Path, Path]:
    research_root = (project_root / "research_output").resolve()
    research_root.mkdir(parents=True, exist_ok=True)
    export_dir = research_root / export_folder_name
    zip_path = research_root / export_zip_name
    temp_dir = research_root / f".{export_folder_name}.tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(temp_dir / "dataset.csv", index=False)
    (temp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (temp_dir / "README.txt").write_text(
        "\n".join(
            [
                "Market Cycle Trader - Contextual Marginal Signature",
                "",
                f"Mongo run id: {run_id}",
                f"Internal script version: {script_version}",
                "",
                "MongoDB is the source of truth for analysis.",
                "This folder and ZIP are export-only and are never required as input.",
                "The stable runner is scripts/research_contextual_marginal_signature.py.",
                "",
                f"Collections: {', '.join(collections)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if export_dir.exists():
        shutil.rmtree(export_dir)
    temp_dir.replace(export_dir)
    temporary_zip = research_root / f".{export_zip_name}.tmp"
    if temporary_zip.exists():
        temporary_zip.unlink()
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(research_root))
    if zip_path.exists():
        zip_path.unlink()
    temporary_zip.replace(zip_path)
    return export_dir, zip_path
