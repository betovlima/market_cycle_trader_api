from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION = 3


def encode_market_frame(frame: pd.DataFrame) -> tuple[bytes, list[str]]:
    """Serialize a canonical market frame without changing float/timestamp bits."""
    canonical = frame.copy().sort_index()
    canonical.index = pd.DatetimeIndex(pd.to_datetime(canonical.index, utc=True, errors="raise"))
    columns = list(canonical.columns)
    values = canonical[columns].to_numpy(dtype=np.float64, copy=True)

    # PyMongo-backed frames may arrive as datetime64[us, UTC] while pandas-created
    # frames are commonly datetime64[ns, UTC]. DatetimeIndex.asi8 preserves the
    # index unit, so persisting asi8 directly makes microsecond timestamps look
    # like nanoseconds during restore (e.g. 2026 becomes a date near 1970).
    # Normalize the index explicitly to nanoseconds before serializing it.
    try:
        timestamp_index = canonical.index.as_unit("ns")
    except AttributeError:  # pandas < 2.0 compatibility
        timestamp_index = pd.DatetimeIndex(
            canonical.index.to_numpy(dtype="datetime64[ns]"),
            tz="UTC",
        )
    timestamps = timestamp_index.asi8.astype(np.int64, copy=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, timestamps=timestamps, values=values)
    return buffer.getvalue(), columns


def decode_market_frame(payload: bytes, columns: list[str]) -> pd.DataFrame:
    """Restore a frame produced by :func:`encode_market_frame` bit-for-bit."""
    with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as archive:
        timestamps = archive["timestamps"].astype(np.int64, copy=False)
        values = archive["values"].astype(np.float64, copy=False)
    index = pd.to_datetime(timestamps, unit="ns", utc=True)
    return pd.DataFrame(values, index=index, columns=list(columns)).sort_index()


def json_safe_provenance(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe_provenance(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_provenance(item) for item in value]
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return stamp.isoformat()
    return str(value)
