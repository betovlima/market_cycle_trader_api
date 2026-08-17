from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    MODEL_TUNING_RUNS_COLLECTION,
    MODEL_TUNING_VALIDATIONS_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from .analytics import analytics_from_equity_rotations
from .strategy_lab import create_tuned_temporal_strategy, get_strategy
from .temporal_policy_replay import replay_temporal_policy_details
from .temporal_policy_tuning import (
    TEMPORAL_POLICY_TUNING_SCOPE,
    _load_artifact_rows,
    _load_observations,
    _source_run,
    derived_temporal_policy_snapshot,
)


class ModelTuningValidationConflict(RuntimeError):
    pass


class ModelTuningValidationNotFound(RuntimeError):
    pass


def _candidate(document: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    candidate = next(
        (
            item
            for item in document.get("candidates") or []
            if int(item.get("candidate_id") if item.get("candidate_id") is not None else -1) == int(candidate_id)
        ),
        None,
    )
    if candidate is None:
        raise ModelTuningValidationNotFound("Model tuning candidate not found.")
    if str(candidate.get("status") or "") != "completed":
        raise ModelTuningValidationConflict("Only a completed candidate can be validated.")
    return candidate


def _find_observation(document: dict[str, Any], candidate_id: int) -> dict[str, Any] | None:
    for item in list(document.get("candidates") or []) + list(document.get("prior_observations") or []):
        source_id = item.get("source_candidate_id")
        item_id = item.get("candidate_id")
        if item_id is not None and int(item_id) == int(candidate_id):
            return item
        if source_id is not None and int(source_id) == int(candidate_id):
            return item
    return None


def _anchor_for_candidate(document: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    starting = deepcopy(document.get("starting_probability_anchor") or document.get("probability_anchor") or {})
    history = [
        item
        for item in document.get("probability_champion_history") or []
        if item.get("candidate_id") is not None and int(item.get("candidate_id")) < int(candidate_id)
    ]
    history.sort(key=lambda item: int(item.get("candidate_id") or 0))
    if history:
        prior_id = int(history[-1]["candidate_id"])
        observation = _find_observation(document, prior_id)
        if observation is not None and isinstance(observation.get("settings"), dict):
            return {
                "source": "prior_campaign_champion",
                "candidate_id": prior_id,
                "settings": deepcopy(observation["settings"]),
                "metrics": deepcopy(observation.get("metrics") or history[-1].get("metrics") or {}),
            }
    if isinstance(starting.get("settings"), dict) and isinstance(starting.get("metrics"), dict):
        return starting
    control = next(
        (
            item for item in document.get("candidates") or []
            if bool(item.get("is_control")) and isinstance(item.get("settings"), dict) and isinstance(item.get("metrics"), dict)
        ),
        None,
    )
    if control is None:
        raise ModelTuningValidationConflict("The CARO Champion anchor snapshot is unavailable.")
    return {
        "source": "campaign_control",
        "candidate_id": int(control.get("candidate_id") or 0),
        "settings": deepcopy(control["settings"]),
        "metrics": deepcopy(control["metrics"]),
    }


def _source_context(db: Any, strategy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], float, float, dict[int, float]]:
    source = _source_run(db, strategy)
    run_id = str(source["id"])
    observations = _load_observations(db, run_id)
    winner_rows = _load_artifact_rows(db, run_id, "winner_reference_daily")
    if not observations or not winner_rows:
        raise ModelTuningValidationConflict("Frozen Temporal replay artifacts are incomplete for Champion validation.")
    request = source.get("request") if isinstance(source.get("request"), dict) else {}
    initial_capital = float(request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(
        0.0, float(request.get("commission_rate") or 0.0)
    )
    result = source.get("result") if isinstance(source.get("result"), dict) else {}
    fold_rows = result.get("multi_horizon_fold_metrics") if isinstance(result.get("multi_horizon_fold_metrics"), list) else []
    winner_fold_returns = {
        int(item.get("fold_id")): float(item.get("winner_reference_return") or 0.0)
        for item in fold_rows
        if isinstance(item, dict) and item.get("fold_id") is not None
    }
    return source, observations, winner_rows, initial_capital, one_side_cost, winner_fold_returns


def _reference_map(details: dict[str, Any]) -> dict[str, float]:
    return {
        str(row.get("timestamp")): float(row["simulation_equity"])
        for row in details.get("equity") or []
        if row.get("timestamp") and row.get("simulation_equity") is not None
    }


def _analytics_metrics(candidate_metrics: dict[str, Any], anchor_metrics: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(candidate_metrics)
    payload.update(
        {
            "reference_return": anchor_metrics.get("strategy_return"),
            "reference_cagr": anchor_metrics.get("cagr"),
            "reference_sharpe": anchor_metrics.get("sharpe"),
            "reference_maximum_drawdown": anchor_metrics.get("maximum_drawdown"),
            "reference_ending_capital": anchor_metrics.get("ending_capital"),
        }
    )
    return payload


def _materialize_validated_strategy(
    db: Any,
    *,
    document: dict[str, Any],
    candidate: dict[str, Any],
    source_strategy: dict[str, Any],
    actor_email: str | None,
) -> dict[str, Any]:
    candidate_id = int(candidate.get("candidate_id") or 0)
    adopted_id = str(document.get("adopted_strategy_id") or "").strip()
    adopted_candidate = document.get("adopted_candidate_id")
    if adopted_id and adopted_candidate is not None and int(adopted_candidate) == candidate_id:
        return get_strategy(db, adopted_id)

    policy_snapshot = derived_temporal_policy_snapshot(
        source_strategy,
        tuning_run_id=str(document["id"]),
        candidate_id=candidate_id,
        settings=deepcopy(candidate.get("settings") or {}),
        metrics=deepcopy(candidate.get("metrics") or {}),
    )
    created = create_tuned_temporal_strategy(
        db,
        str(source_strategy["id"]),
        name=f"{source_strategy.get('name') or 'TEMPORAL'} — CARO Champion #{candidate_id}",
        description=(
            f"Validated TEMPORAL CARO Champion #{candidate_id} from tuning campaign {document['id']}. "
            "Research validation only; Trader promotion remains blocked until the Temporal live execution engine is available."
        ),
        policy_snapshot=policy_snapshot,
        tuning_run_id=str(document["id"]),
        tuning_candidate_id=candidate_id,
        tuning_metrics=deepcopy(candidate.get("metrics") or {}),
        actor_email=actor_email,
    )
    return created


def validate_temporal_policy_champion(
    db: Any,
    run_id: str,
    candidate_id: int,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise ModelTuningValidationNotFound("Model tuning run not found.")
    if str(document.get("status") or "") != "completed":
        raise ModelTuningValidationConflict("Champion validation is available after the research campaign completes.")
    if str(document.get("tuning_scope") or "") != TEMPORAL_POLICY_TUNING_SCOPE:
        raise ModelTuningValidationConflict(
            "Validate Champion is the final step of Temporal Policy Tuning. "
            "A Temporal Model Champion must first be materialized and continued into Policy Tuning."
        )

    candidate = _candidate(document, candidate_id)
    best_id = document.get("best_candidate_id")
    if best_id is None or int(best_id) != int(candidate_id):
        raise ModelTuningValidationConflict("Only the final ranked campaign Champion can be validated.")
    if bool(candidate.get("is_control")):
        raise ModelTuningValidationConflict("The unchanged Control does not require Champion validation.")

    existing = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
        {"_id": 0},
    )
    if existing is not None:
        now = utc_now()
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": str(run_id)},
            {"$set": {
                "validated_candidate_id": int(candidate_id),
                "validation_processing_id": existing.get("id"),
                "validation_strategy_id": existing.get("strategy_profile_id"),
                "validated_at": existing.get("finished_at") or existing.get("created_at") or now,
                "updated_at": now,
            }},
        )
        return bson_value(existing)

    source_strategy = get_strategy(db, str(document.get("strategy_profile_id") or ""))
    source_run, observations, winner_rows, initial_capital, one_side_cost, winner_fold_returns = _source_context(
        db, source_strategy
    )
    candidate_settings = deepcopy(candidate.get("settings") or {})
    candidate_details = replay_temporal_policy_details(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=candidate_settings,
        winner_fold_returns=winner_fold_returns,
    )

    anchor = _anchor_for_candidate(document, int(candidate_id))
    anchor_settings = deepcopy(anchor.get("settings") or {})
    anchor_details = replay_temporal_policy_details(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=anchor_settings,
        winner_fold_returns=winner_fold_returns,
    )
    reference_by_timestamp = _reference_map(anchor_details)
    candidate_equity = deepcopy(candidate_details["equity"])
    for row in candidate_equity:
        row["reference_equity"] = reference_by_timestamp.get(str(row.get("timestamp")))

    derived_strategy = _materialize_validated_strategy(
        db,
        document=document,
        candidate=candidate,
        source_strategy=source_strategy,
        actor_email=actor_email,
    )

    now = utc_now()
    validation_id = f"caro-{str(run_id)}-c{int(candidate_id)}"
    analytics = analytics_from_equity_rotations(
        processing_id=validation_id,
        equity=candidate_equity,
        rotations=deepcopy(candidate_details["rotations"]),
        metrics=_analytics_metrics(candidate_details["metrics"], anchor_details["metrics"]),
        created_at=now,
        finished_at=now,
        processing_kind="caro_champion",
        processing_label=f"CARO Champion #{int(candidate_id)} · TEMPORAL",
        reference_label="Champion Anchor",
    )
    validation = {
        "_id": validation_id,
        "id": validation_id,
        "schema_version": 1,
        "status": "completed",
        "kind": "caro_champion",
        "tuning_scope": TEMPORAL_POLICY_TUNING_SCOPE,
        "tuning_run_id": str(run_id),
        "candidate_id": int(candidate_id),
        "candidate_settings_hash": candidate.get("settings_hash"),
        "candidate_settings": bson_value(candidate_settings),
        "candidate_metrics": bson_value(candidate_details["metrics"]),
        "anchor_source": anchor.get("source"),
        "anchor_candidate_id": anchor.get("candidate_id"),
        "anchor_settings": bson_value(anchor_settings),
        "anchor_metrics": bson_value(anchor_details["metrics"]),
        "reference_label": "Champion Anchor",
        "strategy_profile_id": derived_strategy.get("id"),
        "strategy_profile_name": derived_strategy.get("name"),
        "strategy_profile_revision": derived_strategy.get("revision"),
        "source_strategy_profile_id": source_strategy.get("id"),
        "source_temporal_run_id": str(source_run.get("id") or ""),
        "market_data_snapshot_id": source_run.get("market_data_snapshot_id"),
        "analytics": bson_value(analytics),
        "created_at": now,
        "finished_at": now,
        "created_by": (actor_email or "").strip().lower() or None,
        "trader_winner_eligible": False,
        "trader_winner_block_reason": (
            "TEMPORAL live execution is not installed in the Paper/Trader runtime. "
            "This Champion is validated for research and analytics but cannot be promoted to Trader Winner yet."
        ),
    }
    try:
        db[MODEL_TUNING_VALIDATIONS_COLLECTION].insert_one(bson_value(validation))
    except Exception:
        existing = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
            {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
            {"_id": 0},
        )
        if existing is not None:
            db[MODEL_TUNING_RUNS_COLLECTION].update_one(
                {"id": str(run_id)},
                {"$set": {
                    "validated_candidate_id": int(candidate_id),
                    "validation_processing_id": existing.get("id"),
                    "validation_strategy_id": existing.get("strategy_profile_id"),
                    "validated_at": existing.get("finished_at") or existing.get("created_at") or utc_now(),
                    "updated_at": utc_now(),
                }},
            )
            return bson_value(existing)
        raise

    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {
            "validated_candidate_id": int(candidate_id),
            "validation_processing_id": validation_id,
            "validation_strategy_id": derived_strategy.get("id"),
            "validated_at": now,
            "updated_at": now,
        }},
    )
    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": str(derived_strategy.get("id") or "")},
        {"$set": {
            "temporal_validation_status": "validated_candidate",
            "temporal_validation_id": validation_id,
            "temporal_validation_at": now,
            "temporal_validation_by": (actor_email or "").strip().lower() or None,
            "temporal_trader_eligible": False,
            "temporal_trader_block_reason": validation["trader_winner_block_reason"],
            "updated_at": now,
        }},
    )
    return bson_value({key: value for key, value in validation.items() if key != "_id"})


def get_tuning_validation(db: Any, run_id: str, candidate_id: int) -> dict[str, Any] | None:
    document = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
        {"_id": 0, "analytics": 0},
    )
    return bson_value(document) if document is not None else None
