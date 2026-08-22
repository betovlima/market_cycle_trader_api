from __future__ import annotations

import json
import zlib
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
)
from .utils import as_datetime


def decode_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


def observation_rows(db: Any, run_id: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)},
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        timestamp = as_datetime(document.get("timestamp"))
        if timestamp is not None:
            grouped[timestamp.isoformat()] = decode_rows(document)
    return grouped


def artifact_rows(db: Any, run_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    economics: dict[str, dict[str, Any]] = {}
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": "decision_diagnostics"},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for document in cursor:
        for row in decode_rows(document):
            kind = str(row.get("artifact_kind") or "")
            if kind == "multi_horizon_decision_diagnostics":
                diagnostics.append(row)
            elif kind == "multi_horizon_equity_curve":
                timestamp = as_datetime(row.get("decision_timestamp"))
                if timestamp is not None:
                    economics[timestamp.isoformat()] = row
    diagnostics.sort(key=lambda item: str(item.get("timestamp") or ""))
    return diagnostics, economics
