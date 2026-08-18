from __future__ import annotations

from copy import deepcopy
import json
import math
import zlib
from typing import Any, Callable

import pandas as pd

from ..engine.market_data import load_market_bars, validate_and_clean_bars
from ..engine.temporal_intelligence import run_temporal_intelligence
from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestExecutionRequest
from .model_research import execution_settings_from_values, model_execution_snapshot, model_values_from_snapshot
from .temporal_policy_tuning import temporal_policy_baseline

TEMPORAL_MODEL_TUNING_SCOPE = "temporal_model"
TEMPORAL_MODEL_FAMILY = "lightgbm_utility"


class TemporalModelTuningCancelled(RuntimeError):
    pass


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise TemporalModelTuningCancelled("Temporal Model Tuning cancelled by user.")


TEMPORAL_MODEL_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"name": "n_estimators", "type": "integer", "min": 220, "max": 380},
    {"name": "learning_rate", "type": "number", "min": 0.020, "max": 0.050, "precision": 6},
    {"name": "max_depth", "type": "integer", "min": 2, "max": 4},
    {"name": "num_leaves", "type": "integer", "min": 4, "max": 12},
    {"name": "min_child_samples", "type": "integer", "min": 15, "max": 30},
    {"name": "colsample_bytree", "type": "number", "min": 0.75, "max": 0.95, "precision": 6},
    {"name": "reg_alpha", "type": "number", "min": 0.0, "max": 0.50, "precision": 6},
    {"name": "reg_lambda", "type": "number", "min": 1.0, "max": 4.0, "precision": 6},
)


def is_temporal_strategy(strategy: dict[str, Any]) -> bool:
    return str(strategy.get("strategy_kind") or "") == "temporal_intelligence"


def temporal_model_plan(strategy: dict[str, Any], model_snapshot: dict[str, Any]) -> dict[str, Any]:
    if not is_temporal_strategy(strategy):
        raise ValueError("Temporal Model Tuning requires a materialized TEMPORAL Strategy.")
    if str(model_snapshot.get("family") or "") != TEMPORAL_MODEL_FAMILY:
        raise ValueError("Temporal Model Tuning requires the TEMPORAL Strategy to use LightGBM.")
    search_space = [dict(item) for item in TEMPORAL_MODEL_SEARCH_SPACE]
    frozen = model_values_from_snapshot(model_snapshot)
    if not frozen:
        raise ValueError("The TEMPORAL Strategy does not contain a frozen LightGBM model snapshot.")
    tuned_names = [item["name"] for item in search_space]
    return {
        "scope": TEMPORAL_MODEL_TUNING_SCOPE,
        "scope_label": "Temporal LightGBM Model",
        "description": "Retrain the Temporal Intelligence LightGBM classifiers/regressors on the immutable frozen market snapshot and walk-forward protocol. Winner allocation and Temporal policy thresholds remain frozen.",
        "search_space": search_space,
        "tuned_parameters": tuned_names,
        "tuned_model_parameters": tuned_names,
        "tuned_strategy_parameters": [],
        "base_values": deepcopy(frozen),
        "base_model_values": deepcopy(frozen),
        "frozen_model_values": deepcopy(frozen),
        "fixed_model_values": {name: value for name, value in frozen.items() if name not in set(tuned_names)},
        "strategy_mode": "TEMPORAL_WINNER_ANCHORED_TIMING",
    }


def temporal_model_baseline(strategy: dict[str, Any], model_snapshot: dict[str, Any]) -> dict[str, Any]:
    baseline = temporal_policy_baseline(strategy)
    baseline["model_family"] = TEMPORAL_MODEL_FAMILY
    baseline["model_label"] = "LightGBM Temporal Intelligence"
    baseline["model_settings_hash"] = model_snapshot.get("settings_hash")
    baseline["model_settings_revision"] = model_snapshot.get("settings_revision")
    return baseline


def _source_run(db: Any, strategy: dict[str, Any]) -> dict[str, Any]:
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    run_id = str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or "").strip()
    if not run_id:
        raise ValueError("TEMPORAL Strategy does not reference its source Temporal Intelligence run.")
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id})
    if run is None or str(run.get("status") or "") != "completed":
        raise ValueError("The source Temporal Intelligence run is unavailable or not completed.")
    return run


def _artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        artifact_rows = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            artifact_rows = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        rows.extend(dict(row) for row in artifact_rows if isinstance(row, dict))
    return rows


def _winner_override(db: Any, source_run: dict[str, Any]) -> dict[str, Any]:
    result = source_run.get("result") if isinstance(source_run.get("result"), dict) else {}
    summary = deepcopy(result.get("winner_reference") or {})
    run_id = str(source_run.get("id") or "")
    daily_rows = _artifact_rows(db, run_id, "winner_reference_daily")
    trade_rows = _artifact_rows(db, run_id, "winner_reference_trades")
    if not summary or not daily_rows:
        raise ValueError("The source Temporal run does not contain the immutable Winner replay required by Temporal Model Tuning.")
    return {"summary": summary, "daily_rows": daily_rows, "trade_rows": trade_rows}


def _candidate_request(
    source_run: dict[str, Any],
    model_snapshot: dict[str, Any],
    settings: dict[str, Any],
    *,
    fold_count: int | None = None,
) -> tuple[BacktestExecutionRequest, dict[str, Any]]:
    request_payload = deepcopy(source_run.get("request") or {})
    base_values = model_values_from_snapshot(model_snapshot)
    values = deepcopy(base_values)
    values.update(deepcopy(settings))
    revision = max(1, int(model_snapshot.get("settings_revision") or 1))
    settings_snapshot = execution_settings_from_values(
        TEMPORAL_MODEL_FAMILY,
        values,
        settings_revision=revision,
        profile_id="temporal-tuning",
    )
    snapshot_id = str(source_run.get("market_data_snapshot_id") or request_payload.get("research_market_data_snapshot_id") or "").strip().lower()
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
    request = BacktestExecutionRequest.model_validate(request_payload)
    return request, settings_snapshot


def _metrics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    winner = result.get("winner_reference") if isinstance(result.get("winner_reference"), dict) else {}
    folds: list[dict[str, Any]] = []
    fold_returns: list[float] = []
    for item in result.get("multi_horizon_fold_metrics") or []:
        if not isinstance(item, dict):
            continue
        fold_capital = item.get("shadow_capital") if isinstance(item.get("shadow_capital"), dict) else {}
        strategy_return = float(fold_capital.get("total_return") or 0.0)
        fold_returns.append(strategy_return)
        folds.append({
            "fold_id": int(item.get("fold_id") or len(folds) + 1),
            "strategy_return": strategy_return,
            "maximum_drawdown": float(fold_capital.get("max_drawdown") or 0.0),
            "benchmark_return": float(item.get("winner_reference_return") or 0.0),
        })
    return {
        "initial_capital": float(capital.get("initial_capital") or 0.0),
        "ending_capital": float(capital.get("ending_capital") or 0.0),
        "strategy_return": float(capital.get("total_return") or 0.0),
        "cagr": float(capital.get("cagr") or 0.0),
        "sharpe": float(capital.get("sharpe") or 0.0),
        "maximum_drawdown": float(capital.get("max_drawdown") or 0.0),
        "risk_adjusted_compound_score": 0.0,
        "turnover_ratio": 0.0,
        "capital_rotations": int(capital.get("switch_count") or 0),
        "average_holding_days": float(capital.get("median_holding_days") or 0.0),
        "market_exposure": float(capital.get("exposure") or 0.0),
        "cash_days": int(capital.get("cash_days") or 0),
        "benchmark_ending_capital": float(winner.get("ending_capital") or 0.0),
        "market_data_signature_sha256": None,
        "market_data_last_timestamp": result.get("oos_end"),
        "folds": folds,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": bool(fold_returns) and all(value > 0 for value in fold_returns),
        "timing_override_count": int(capital.get("timing_override_count") or 0),
    }


def _equity_preview(result: dict[str, Any], limit: int = 180) -> list[dict[str, Any]]:
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    rows = [dict(item) for item in (capital.get("economic_curve") or []) if isinstance(item, dict)]
    if len(rows) <= limit:
        return rows
    step = max(1, math.ceil(len(rows) / max(2, int(limit))))
    sampled = rows[::step]
    if sampled and rows and sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled[: limit + 1]


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    multi = deepcopy(result.get("multi_horizon_metrics") or {})
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    if capital:
        capital.pop("decision_diagnostics", None)
        capital.pop("economic_curve", None)
    return {
        "experiment": result.get("experiment"),
        "model_family": result.get("model_family"),
        "model_label": result.get("model_label"),
        "lightgbm_version": result.get("lightgbm_version"),
        "horizons": deepcopy(result.get("horizons") or []),
        "asset_count": result.get("asset_count"),
        "walk_forward_fold_count": result.get("walk_forward_fold_count"),
        "purge_sessions": result.get("purge_sessions"),
        "oos_start": result.get("oos_start"),
        "oos_end": result.get("oos_end"),
        "latest_as_of": result.get("latest_as_of"),
        "decision_policy": deepcopy(result.get("decision_policy") or {}),
        "multi_horizon_metrics": multi,
        "multi_horizon_fold_metrics": deepcopy(result.get("multi_horizon_fold_metrics") or []),
        "winner_reference": deepcopy(result.get("winner_reference") or {}),
        "duration_seconds": result.get("duration_seconds"),
        "shadow_only": True,
        "affects_strategy_decisions": False,
        "affects_winner": False,
        "affects_paper_trading": False,
    }


def evaluate_temporal_model_candidate(
    db: Any,
    strategy: dict[str, Any],
    model_snapshot: dict[str, Any],
    settings: dict[str, Any],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    fold_count: int | None = None,
) -> dict[str, Any]:
    source_run = _source_run(db, strategy)
    request, settings_snapshot = _candidate_request(
        source_run, model_snapshot, settings, fold_count=fold_count
    )
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    _raise_if_cancelled(cancel_check)
    for position, symbol in enumerate(request.assets, start=1):
        _raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(2.0 + 12.0 * ((position - 1) / max(1, len(request.assets))), f"Loading frozen market data {position}/{len(request.assets)}")
        asset_request = request if symbol in set(request.calendar_anchor_assets) else request.model_copy(update={"market_data_require_complete_history": False})
        raw = load_market_bars(symbol, asset_request)
        bars_by_symbol[symbol] = validate_and_clean_bars(raw, asset_request)
        _raise_if_cancelled(cancel_check)

    winner_override = _winner_override(db, source_run)
    _raise_if_cancelled(cancel_check)
    result = run_temporal_intelligence(
        bars_by_symbol,
        request,
        progress_callback=progress_callback,
        cancel_callback=(lambda: _raise_if_cancelled(cancel_check)) if cancel_check is not None else None,
        winner_reference_override=winner_override,
        candidate_evaluation_only=True,
    )
    observation_rows = list(result.pop("_multi_horizon_observations", []) or [])
    winner_daily_rows = list(result.pop("_winner_reference_daily_rows", []) or winner_override["daily_rows"])
    winner_trade_rows = list(result.pop("_winner_reference_trade_rows", []) or winner_override["trade_rows"])
    model_execution = model_execution_snapshot(TEMPORAL_MODEL_FAMILY, settings_snapshot)
    metrics = _metrics_from_result(result)
    metrics["market_data_signature_sha256"] = str(source_run.get("market_data_snapshot_id") or "") or None
    return {
        "metrics": metrics,
        "equity_preview": _equity_preview(result),
        "temporal_result": _compact_result(result),
        "observation_rows": observation_rows,
        "winner_reference_daily_rows": winner_daily_rows,
        "winner_reference_trade_rows": winner_trade_rows,
        "execution_request": request.model_dump(mode="python"),
        "model_snapshot": model_execution,
        "source_run_id": str(source_run.get("id") or ""),
        "source_run": source_run,
    }


def _compressed_artifact_documents(run_id: str, kind: str, rows: list[dict[str, Any]], *, chunk_size: int = 250) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    normalized = [bson_value(dict(row)) for row in rows if isinstance(row, dict)]
    for sequence, start in enumerate(range(0, len(normalized), max(1, int(chunk_size)))):
        chunk = normalized[start:start + max(1, int(chunk_size))]
        encoded = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        documents.append({
            "run_id": str(run_id),
            "kind": str(kind),
            "sequence": int(sequence),
            "encoding": "zlib-json-v1",
            "row_count": int(len(chunk)),
            "payload": zlib.compress(encoded, level=9),
            "created_at": utc_now(),
        })
    return documents


def persist_temporal_model_champion_cache(
    db: Any,
    *,
    tuning_run_id: str,
    candidate_id: int,
    strategy: dict[str, Any],
    evaluation: dict[str, Any],
) -> str:
    cache_run_id = f"{str(tuning_run_id)}-temporal-model-champion"
    observations = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION]
    artifacts = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION]
    observations.delete_many({"run_id": cache_run_id})
    artifacts.delete_many({"run_id": cache_run_id})

    grouped: dict[str, dict[str, Any]] = {}
    for row in evaluation.get("observation_rows") or []:
        if not isinstance(row, dict) or row.get("timestamp") is None:
            continue
        timestamp = pd.Timestamp(row.get("timestamp"))
        key = timestamp.isoformat()
        document = grouped.setdefault(key, {"run_id": cache_run_id, "timestamp": timestamp, "rows": []})
        payload = dict(row)
        payload.pop("timestamp", None)
        document["rows"].append(bson_value(payload))
    observation_documents: list[dict[str, Any]] = []
    for _, document in sorted(grouped.items()):
        rows_payload = bson_value(document.get("rows") or [])
        encoded = json.dumps(rows_payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        observation_documents.append({
            "run_id": cache_run_id,
            "timestamp": bson_value(document.get("timestamp")),
            "encoding": "zlib-json-v1",
            "row_count": int(len(rows_payload)),
            "payload": zlib.compress(encoded, level=9),
        })
    for start in range(0, len(observation_documents), 200):
        observations.insert_many(observation_documents[start:start + 200], ordered=False)

    artifact_documents = _compressed_artifact_documents(cache_run_id, "winner_reference_daily", evaluation.get("winner_reference_daily_rows") or [], chunk_size=250)
    artifact_documents.extend(_compressed_artifact_documents(cache_run_id, "winner_reference_trades", evaluation.get("winner_reference_trade_rows") or [], chunk_size=250))
    for start in range(0, len(artifact_documents), 100):
        artifacts.insert_many(artifact_documents[start:start + 100], ordered=False)

    source_run = evaluation.get("source_run") if isinstance(evaluation.get("source_run"), dict) else {}
    model_snapshot = evaluation.get("model_snapshot") if isinstance(evaluation.get("model_snapshot"), dict) else {}
    request = evaluation.get("execution_request") if isinstance(evaluation.get("execution_request"), dict) else {}
    now = utc_now()
    cache_document = {
        "id": cache_run_id,
        "status": "completed",
        "stage": "Temporal Model Tuning champion cache",
        "progress": 100.0,
        "created_at": now,
        "updated_at": now,
        "started_at": now,
        "finished_at": now,
        "actor_email": None,
        "strategy_profile_id": strategy.get("id"),
        "strategy_profile_name": strategy.get("name"),
        "strategy_profile_revision": strategy.get("revision"),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": TEMPORAL_MODEL_FAMILY,
        "model_label": "LightGBM Temporal Intelligence",
        "model_settings_hash": model_snapshot.get("settings_hash"),
        "model_settings_revision": model_snapshot.get("settings_revision"),
        "market_data_snapshot_id": source_run.get("market_data_snapshot_id"),
        "market_data_snapshot_source": "temporal_model_tuning_champion",
        "market_data_snapshot_source_run_id": source_run.get("id"),
        "analysis_end_date": source_run.get("analysis_end_date") or request.get("analysis_end_date"),
        "certified_backtest_cutoff": source_run.get("certified_backtest_cutoff"),
        "live_market_cutoff": source_run.get("live_market_cutoff"),
        "research_snapshot_cutoff": source_run.get("research_snapshot_cutoff") or source_run.get("analysis_end_date") or request.get("analysis_end_date"),
        "horizons": list(request.get("rotation_target_horizons") or []),
        "request": bson_value(request),
        "experiment": (evaluation.get("temporal_result") or {}).get("experiment") or source_run.get("experiment"),
        "result": bson_value(evaluation.get("temporal_result") or {}),
        "failure_message": None,
        "technical_error": None,
        "shadow_only": True,
        "derived_from_model_tuning_run_id": str(tuning_run_id),
        "derived_from_model_tuning_candidate_id": int(candidate_id),
    }
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].replace_one({"id": cache_run_id}, cache_document, upsert=True)
    return cache_run_id
