from __future__ import annotations

from copy import deepcopy
import json
import zlib
from typing import Any


from .temporal_policy_replay import _finite, _replay_rows, _timestamp_key

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
)

TEMPORAL_POLICY_TUNING_SCOPE = "temporal_policy"
TEMPORAL_POLICY_MODEL_FAMILY = "temporal_policy"
TEMPORAL_POLICY_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"name": "timing_base_weak_threshold", "type": "number", "min": 0.30, "max": 0.70, "precision": 6},
    {"name": "timing_challenger_minimum", "type": "number", "min": 0.40, "max": 0.80, "precision": 6},
    {"name": "timing_minimum_advantage", "type": "number", "min": 0.05, "max": 0.40, "precision": 6},
    {"name": "timing_maximum_advantage", "type": "number", "min": 0.45, "max": 1.00, "precision": 6},
)


def is_temporal_policy_strategy(strategy: dict[str, Any]) -> bool:
    return (
        str(strategy.get("strategy_kind") or "") == "temporal_intelligence"
        and str(strategy.get("tuning_target") or "") == TEMPORAL_POLICY_TUNING_SCOPE
    )


def _load_artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
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


def _load_observations(db: Any, run_id: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {"run_id": str(run_id)},
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        key = _timestamp_key(document.get("timestamp"))
        if not key:
            continue
        observation_rows = document.get("rows") or []
        if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
            observation_rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
        rows_by_symbol: dict[str, dict[str, Any]] = {}
        fold_id: int | None = None
        for row in observation_rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            rows_by_symbol[symbol] = dict(row)
            if fold_id is None and row.get("fold_id") is not None:
                fold_id = int(row["fold_id"])
        grouped[key] = {"fold_id": fold_id, "rows_by_symbol": rows_by_symbol}
    return grouped


def observations_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _timestamp_key(row.get("timestamp"))
        symbol = str(row.get("symbol") or "").strip()
        if not key or not symbol:
            continue
        payload = grouped.setdefault(key, {"fold_id": None, "rows_by_symbol": {}})
        row_payload = dict(row)
        row_payload.pop("timestamp", None)
        payload["rows_by_symbol"][symbol] = row_payload
        if payload.get("fold_id") is None and row.get("fold_id") is not None:
            payload["fold_id"] = int(row["fold_id"])
    return grouped


def _source_run(db: Any, strategy: dict[str, Any]) -> dict[str, Any]:
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    run_id = str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or "").strip()
    if not run_id:
        raise ValueError("Temporal Policy Strategy does not reference its source Temporal Intelligence run.")
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id})
    if run is None or str(run.get("status") or "") != "completed":
        raise ValueError("The source Temporal Intelligence run is not available as a completed frozen replay.")
    return run


def _base_parameters(strategy: dict[str, Any]) -> dict[str, float]:
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    parameters = policy.get("parameters") if isinstance(policy.get("parameters"), dict) else {}
    defaults = {
        "timing_base_weak_threshold": 0.50,
        "timing_challenger_minimum": 0.60,
        "timing_minimum_advantage": 0.25,
        "timing_maximum_advantage": 1.00,
    }
    return {
        name: float(parameters.get(name) if _finite(parameters.get(name)) is not None else default)
        for name, default in defaults.items()
    }


def temporal_policy_plan(strategy: dict[str, Any]) -> dict[str, Any]:
    base_values = _base_parameters(strategy)
    return {
        "scope": TEMPORAL_POLICY_TUNING_SCOPE,
        "scope_label": "Temporal Policy — Winner-Anchored Timing",
        "description": "Tune the causal Top-1/Top-2 Temporal timing thresholds and the overconfidence ceiling against the immutable frozen Temporal replay without retraining the Winner model or downloading market data.",
        "search_space": [dict(item) for item in TEMPORAL_POLICY_SEARCH_SPACE],
        "tuned_parameters": [item["name"] for item in TEMPORAL_POLICY_SEARCH_SPACE],
        "tuned_model_parameters": [],
        "tuned_strategy_parameters": [item["name"] for item in TEMPORAL_POLICY_SEARCH_SPACE],
        "base_values": deepcopy(base_values),
        "base_model_values": {},
        "frozen_model_values": {},
        "fixed_model_values": {},
        "strategy_mode": "TEMPORAL_WINNER_ANCHORED_TIMING",
    }


def _fold_metrics_from_validation(validation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(validation.get("folds") or []):
        if not isinstance(item, dict):
            continue
        capital = item.get("shadow_capital") if isinstance(item.get("shadow_capital"), dict) else item
        strategy_return = _finite(capital.get("total_return"))
        maximum_drawdown = _finite(capital.get("max_drawdown"))
        benchmark_return = _finite(item.get("winner_reference_return"))
        rows.append({
            "fold_id": int(item.get("fold_id") or index + 1),
            "strategy_return": float(strategy_return or 0.0),
            "maximum_drawdown": float(maximum_drawdown or 0.0),
            "benchmark_return": float(benchmark_return or 0.0),
        })
    return rows


def temporal_policy_baseline(strategy: dict[str, Any]) -> dict[str, Any]:
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    validation = policy.get("validation") if isinstance(policy.get("validation"), dict) else {}
    folds = _fold_metrics_from_validation(validation)
    fold_returns = [float(item["strategy_return"]) for item in folds]
    metrics = {
        "initial_capital": float(validation.get("initial_capital") or 0.0),
        "ending_capital": float(validation.get("ending_capital") or 0.0),
        "strategy_return": float(validation.get("total_return") or 0.0),
        "cagr": float(validation.get("cagr") or 0.0),
        "sharpe": float(validation.get("sharpe") or 0.0),
        "maximum_drawdown": float(validation.get("max_drawdown") or 0.0),
        "risk_adjusted_compound_score": 0.0,
        "turnover_ratio": 0.0,
        "capital_rotations": int(validation.get("switch_count") or 0),
        "average_holding_days": 0.0,
        "market_exposure": float(validation.get("exposure") or 0.0),
        "cash_days": 0,
        "benchmark_ending_capital": float(validation.get("winner_ending_capital") or 0.0),
        "market_data_signature_sha256": policy.get("market_data_snapshot_id"),
        "market_data_last_timestamp": policy.get("analysis_end_date"),
        "folds": folds,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": bool(fold_returns) and all(value > 0 for value in fold_returns),
        "timing_override_count": int(validation.get("timing_override_count") or 0),
    }
    return {
        "job_id": str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or ""),
        "source_temporal_run_id": str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or ""),
        "strategy_profile_id": strategy.get("id"),
        "strategy_profile_name": strategy.get("name"),
        "strategy_profile_revision": strategy.get("revision"),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": TEMPORAL_POLICY_MODEL_FAMILY,
        "model_label": "Temporal Policy",
        "model_settings_hash": None,
        "model_settings_revision": strategy.get("temporal_policy_revision"),
        "metrics": metrics,
    }


def evaluate_temporal_policy_candidate(
    db: Any,
    strategy: dict[str, Any],
    settings: dict[str, Any],
    *,
    source_run_id_override: str | None = None,
) -> dict[str, Any]:
    if source_run_id_override:
        run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(source_run_id_override)})
        if run is None or str(run.get("status") or "") != "completed":
            raise ValueError("The fold-specific Temporal research cache is unavailable.")
    else:
        run = _source_run(db, strategy)
    run_id = str(run["id"])
    observations = _load_observations(db, run_id)
    winner_rows = _load_artifact_rows(db, run_id, "winner_reference_daily")
    if not observations or not winner_rows:
        raise ValueError("Frozen Temporal replay artifacts are incomplete for policy tuning.")
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    initial_capital = float(request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(0.0, float(request.get("commission_rate") or 0.0))
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    fold_rows = result.get("multi_horizon_fold_metrics") if isinstance(result.get("multi_horizon_fold_metrics"), list) else []
    winner_fold_returns = {
        int(item.get("fold_id")): float(item.get("winner_reference_return") or 0.0)
        for item in fold_rows
        if isinstance(item, dict) and item.get("fold_id") is not None
    }
    metrics, preview = _replay_rows(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=settings,
        winner_fold_returns=winner_fold_returns,
    )
    metrics["market_data_signature_sha256"] = str(run.get("market_data_snapshot_id") or "") or None
    metrics["benchmark_ending_capital"] = float(
        ((result.get("winner_reference") or {}).get("ending_capital") if isinstance(result.get("winner_reference"), dict) else 0.0) or 0.0
    )
    return {"metrics": metrics, "equity_preview": preview}


def derived_temporal_policy_snapshot(
    strategy: dict[str, Any],
    *,
    tuning_run_id: str,
    candidate_id: int,
    settings: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    policy = deepcopy(strategy.get("temporal_policy") or {})
    parameters = deepcopy(policy.get("parameters") or {})
    parameters.update({name: settings[name] for name in (item["name"] for item in TEMPORAL_POLICY_SEARCH_SPACE) if name in settings})
    validation = deepcopy(policy.get("validation") or {})
    validation.update({
        "initial_capital": metrics.get("initial_capital"),
        "ending_capital": metrics.get("ending_capital"),
        "total_return": metrics.get("strategy_return"),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("maximum_drawdown"),
        "exposure": metrics.get("market_exposure"),
        "switch_count": metrics.get("capital_rotations"),
        "timing_override_count": metrics.get("timing_override_count"),
        "folds": bson_value(metrics.get("folds") or []),
    })
    policy["parameters"] = bson_value(parameters)
    policy["validation"] = bson_value(validation)
    policy["tuning_source_run_id"] = str(tuning_run_id)
    policy["tuning_source_candidate_id"] = int(candidate_id)
    policy["tuning_execution_mode"] = "frozen_temporal_replay"
    return policy
