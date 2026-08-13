from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION = 3


def encode_market_frame(frame: pd.DataFrame) -> tuple[bytes, list[str]]:
    
    canonical = frame.copy().sort_index()
    canonical.index = pd.DatetimeIndex(pd.to_datetime(canonical.index, utc=True, errors="raise"))
    columns = list(canonical.columns)
    values = canonical[columns].to_numpy(dtype=np.float64, copy=True)

    
    
    
    
    
    try:
        timestamp_index = canonical.index.as_unit("ns")
    except AttributeError:  
        timestamp_index = pd.DatetimeIndex(
            canonical.index.to_numpy(dtype="datetime64[ns]"),
            tz="UTC",
        )
    timestamps = timestamp_index.asi8.astype(np.int64, copy=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, timestamps=timestamps, values=values)
    return buffer.getvalue(), columns


def decode_market_frame(payload: bytes, columns: list[str]) -> pd.DataFrame:
    
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
