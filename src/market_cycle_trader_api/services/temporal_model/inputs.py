from __future__ import annotations

from copy import deepcopy
import json
import zlib
from typing import Any, Callable

import pandas as pd

from ...engine.market_data import load_market_bars, validate_and_clean_bars
from ...infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
)
from ...schemas.requests import BacktestExecutionRequest
from ..model_research import execution_settings_from_values, model_values_from_snapshot
from .constants import TEMPORAL_MODEL_FAMILY
from .errors import TemporalModelTuningCancelled


def source_run(db: Any, strategy: dict[str, Any]) -> dict[str, Any]:
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    run_id = str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or "").strip()
    if not run_id:
        raise ValueError("TEMPORAL Strategy does not reference its source Temporal Intelligence run.")
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id})
    if run is None or str(run.get("status") or "") != "completed":
        raise ValueError("The source Temporal Intelligence run is unavailable or not completed.")
    return run


def artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        current = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            current = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        rows.extend(dict(row) for row in current if isinstance(row, dict))
    return rows


def winner_override(db: Any, run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    summary = deepcopy(result.get("winner_reference") or {})
    run_id = str(run.get("id") or "")
    daily_rows = artifact_rows(db, run_id, "winner_reference_daily")
    trade_rows = artifact_rows(db, run_id, "winner_reference_trades")
    if not summary or not daily_rows:
        raise ValueError("The source Temporal run does not contain the immutable Winner replay required by Temporal Model Tuning.")
    return {"summary": summary, "daily_rows": daily_rows, "trade_rows": trade_rows}


def candidate_request(
    run: dict[str, Any],
    model_snapshot: dict[str, Any],
    settings: dict[str, Any],
    *,
    fold_count: int | None = None,
) -> tuple[BacktestExecutionRequest, dict[str, Any]]:
    request_payload = deepcopy(run.get("request") or {})
    values = model_values_from_snapshot(model_snapshot)
    values.update(deepcopy(settings))
    revision = max(1, int(model_snapshot.get("settings_revision") or 1))
    settings_snapshot = execution_settings_from_values(
        TEMPORAL_MODEL_FAMILY,
        values,
        settings_revision=revision,
        profile_id="temporal-tuning",
    )
    snapshot_id = str(run.get("market_data_snapshot_id") or request_payload.get("research_market_data_snapshot_id") or "").strip().lower()
    if not snapshot_id:
        raise ValueError("The source Temporal run does not contain a frozen market-data snapshot id.")
    request_payload.update({
        "research_model_family": TEMPORAL_MODEL_FAMILY,
        "research_model_settings": settings_snapshot,
        "research_market_data_mode": "database_only",
        "research_market_data_snapshot_id": snapshot_id,
        "expected_market_data_signature_sha256": snapshot_id,
        "deterministic_execution": True,
        "numeric_thread_limit": 1,
        "xgb_n_jobs": 1,
        "walk_forward_fold_count_override": (int(fold_count) if fold_count is not None else None),
    })
    return BacktestExecutionRequest.model_validate(request_payload), settings_snapshot


def load_frozen_bars(
    request: BacktestExecutionRequest,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, pd.DataFrame]:
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    anchors = set(request.calendar_anchor_assets)
    assets = list(request.assets)
    for position, symbol in enumerate(assets, start=1):
        if cancel_check is not None and bool(cancel_check()):
            raise TemporalModelTuningCancelled("Temporal Model Tuning cancelled by user.")
        if progress_callback:
            progress_callback(
                2.0 + 8.0 * ((position - 1) / max(1, len(assets))),
                f"Loading frozen market data {position}/{len(assets)}",
            )
        asset_request = request if symbol in anchors else request.model_copy(update={"market_data_require_complete_history": False})
        raw = load_market_bars(symbol, asset_request)
        bars_by_symbol[symbol] = validate_and_clean_bars(raw, asset_request)
    return bars_by_symbol
