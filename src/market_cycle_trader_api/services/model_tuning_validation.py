from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    MODEL_TUNING_RUNS_COLLECTION,
    MODEL_TUNING_VALIDATIONS_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    bson_value,
    utc_now,
)
from .analytics import analytics_from_equity_rotations
from .strategy_lab import create_tuned_temporal_strategy, get_strategy, get_strategy_model_snapshot
from .temporal_model_tuning import evaluate_temporal_model_candidate
from .temporal_policy_replay import replay_temporal_policy_details
from .temporal_policy_tuning import (
    TEMPORAL_POLICY_TUNING_SCOPE,
    derived_temporal_policy_snapshot,
    observations_from_rows,
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
    if bool(candidate.get("is_control")):
        raise ModelTuningValidationConflict("The unchanged Control does not require finalist validation.")
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
    if isinstance(starting.get("settings"), dict):
        return starting
    control = next(
        (
            item for item in document.get("candidates") or []
            if bool(item.get("is_control")) and isinstance(item.get("settings"), dict)
        ),
        None,
    )
    if control is None:
        raise ModelTuningValidationConflict("The CARO Champion anchor snapshot is unavailable.")
    return {
        "source": "campaign_control",
        "candidate_id": int(control.get("candidate_id") or 0),
        "settings": deepcopy(control["settings"]),
        "metrics": deepcopy(control.get("metrics") or {}),
    }


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


def _fold_protocol(document: dict[str, Any]) -> dict[str, int]:
    payload = dict(document.get("fold_protocol") or {})
    return {
        "research_folds": int(payload.get("research_folds") or 3),
        "validation_folds": int(payload.get("validation_folds") or 5),
        "certification_folds": int(payload.get("certification_folds") or 7),
    }


def _gate(document: dict[str, Any], candidate_metrics: dict[str, Any], anchor_metrics: dict[str, Any]) -> dict[str, Any]:
    config = dict(document.get("probability_config") or {})
    min_capital_improvement = float(config.get("min_capital_improvement") or 0.03)
    sharpe_tolerance = float(config.get("sharpe_tolerance") or 0.05)
    drawdown_tolerance = float(config.get("drawdown_tolerance") or 0.03)
    min_worst_fold_return = float(config.get("min_worst_fold_return") or 0.0)

    candidate_capital = float(candidate_metrics.get("ending_capital") or 0.0)
    anchor_capital = float(anchor_metrics.get("ending_capital") or 0.0)
    candidate_sharpe = float(candidate_metrics.get("sharpe") or 0.0)
    anchor_sharpe = float(anchor_metrics.get("sharpe") or 0.0)
    candidate_dd = float(candidate_metrics.get("maximum_drawdown") or 0.0)
    anchor_dd = float(anchor_metrics.get("maximum_drawdown") or 0.0)
    candidate_worst = float(candidate_metrics.get("worst_fold_return") or 0.0)

    capital_threshold = anchor_capital * (1.0 + min_capital_improvement)
    sharpe_threshold = anchor_sharpe - sharpe_tolerance
    drawdown_threshold = anchor_dd - drawdown_tolerance
    checks = {
        "capital": candidate_capital >= capital_threshold,
        "sharpe": candidate_sharpe >= sharpe_threshold,
        "drawdown": candidate_dd >= drawdown_threshold,
        "worst_fold": candidate_worst > min_worst_fold_return,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate": {
            "ending_capital": candidate_capital,
            "sharpe": candidate_sharpe,
            "maximum_drawdown": candidate_dd,
            "worst_fold_return": candidate_worst,
        },
        "anchor": {
            "ending_capital": anchor_capital,
            "sharpe": anchor_sharpe,
            "maximum_drawdown": anchor_dd,
            "worst_fold_return": anchor_metrics.get("worst_fold_return"),
        },
        "thresholds": {
            "ending_capital": capital_threshold,
            "sharpe": sharpe_threshold,
            "maximum_drawdown": drawdown_threshold,
            "worst_fold_return": min_worst_fold_return,
        },
    }


def _full_fold_replay(
    db: Any,
    source_strategy: dict[str, Any],
    *,
    fold_count: int,
    candidate_settings: dict[str, Any],
    anchor_settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_snapshot = get_strategy_model_snapshot(db, str(source_strategy.get("id") or ""))
    evaluation = evaluate_temporal_model_candidate(
        db,
        source_strategy,
        model_snapshot,
        {},
        fold_count=int(fold_count),
    )
    observations = observations_from_rows(list(evaluation.get("observation_rows") or []))
    winner_rows = list(evaluation.get("winner_reference_daily_rows") or [])
    request = evaluation.get("execution_request") if isinstance(evaluation.get("execution_request"), dict) else {}
    initial_capital = float(request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(
        0.0, float(request.get("commission_rate") or 0.0)
    )
    candidate_details = replay_temporal_policy_details(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=candidate_settings,
        winner_fold_returns={},
    )
    anchor_details = replay_temporal_policy_details(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=anchor_settings,
        winner_fold_returns={},
    )
    return candidate_details, anchor_details, evaluation


def _materialize_validated_strategy(
    db: Any,
    *,
    document: dict[str, Any],
    candidate: dict[str, Any],
    source_strategy: dict[str, Any],
    validation_metrics: dict[str, Any],
    actor_email: str | None,
) -> dict[str, Any]:
    candidate_id = int(candidate.get("candidate_id") or 0)
    existing_validation = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"tuning_run_id": str(document["id"]), "candidate_id": candidate_id},
        {"_id": 0, "strategy_profile_id": 1},
    )
    existing_strategy_id = str((existing_validation or {}).get("strategy_profile_id") or "").strip()
    if existing_strategy_id:
        return get_strategy(db, existing_strategy_id)

    policy_snapshot = derived_temporal_policy_snapshot(
        source_strategy,
        tuning_run_id=str(document["id"]),
        candidate_id=candidate_id,
        settings=deepcopy(candidate.get("settings") or {}),
        metrics=deepcopy(validation_metrics),
    )
    return create_tuned_temporal_strategy(
        db,
        str(source_strategy["id"]),
        name=f"{source_strategy.get('name') or 'TEMPORAL'} — CARO Finalist #{candidate_id}",
        description=(
            f"TEMPORAL CARO finalist #{candidate_id} from tuning campaign {document['id']}. "
            "Validated with a full walk-forward Temporal LightGBM rerun. Trader promotion remains blocked until the Temporal live execution engine is available."
        ),
        policy_snapshot=policy_snapshot,
        tuning_run_id=str(document["id"]),
        tuning_candidate_id=candidate_id,
        tuning_metrics=deepcopy(validation_metrics),
        actor_email=actor_email,
    )


def _validation_document(db: Any, run_id: str, candidate_id: int) -> dict[str, Any] | None:
    return db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
        {"_id": 0},
    )


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
        raise ModelTuningValidationConflict("Finalist validation is available after the research campaign completes.")
    if str(document.get("tuning_scope") or "") != TEMPORAL_POLICY_TUNING_SCOPE:
        raise ModelTuningValidationConflict(
            "Finalist validation is the robustness stage of Temporal Policy Tuning. "
            "A Temporal Model Champion must first be materialized and continued into Policy Tuning."
        )

    candidate = _candidate(document, candidate_id)
    protocol = _fold_protocol(document)
    fold_count = int(protocol["validation_folds"])
    existing = _validation_document(db, run_id, candidate_id)
    if existing is not None and int(existing.get("validation_fold_count") or 0) == fold_count and existing.get("validation_completed_at"):
        return bson_value(existing)

    source_strategy = get_strategy(db, str(document.get("strategy_profile_id") or ""))
    anchor = _anchor_for_candidate(document, int(candidate_id))
    candidate_settings = deepcopy(candidate.get("settings") or {})
    anchor_settings = deepcopy(anchor.get("settings") or {})
    candidate_details, anchor_details, evaluation = _full_fold_replay(
        db,
        source_strategy,
        fold_count=fold_count,
        candidate_settings=candidate_settings,
        anchor_settings=anchor_settings,
    )
    gate = _gate(document, candidate_details["metrics"], anchor_details["metrics"])

    reference_by_timestamp = _reference_map(anchor_details)
    candidate_equity = deepcopy(candidate_details["equity"])
    for row in candidate_equity:
        row["reference_equity"] = reference_by_timestamp.get(str(row.get("timestamp")))

    derived_strategy = None
    if bool(gate.get("passed")):
        derived_strategy = _materialize_validated_strategy(
            db,
            document=document,
            candidate=candidate,
            source_strategy=source_strategy,
            validation_metrics=candidate_details["metrics"],
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
        processing_kind="caro_validation",
        processing_label=f"CARO Finalist #{int(candidate_id)} · {fold_count} folds",
        reference_label="Champion Anchor",
    )
    validation = {
        "id": validation_id,
        "schema_version": 2,
        "status": "completed",
        "kind": "caro_validation",
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
        "strategy_profile_id": derived_strategy.get("id") if derived_strategy else None,
        "strategy_profile_name": derived_strategy.get("name") if derived_strategy else None,
        "strategy_profile_revision": derived_strategy.get("revision") if derived_strategy else None,
        "source_strategy_profile_id": source_strategy.get("id"),
        "source_temporal_run_id": str((evaluation.get("source_run") or {}).get("id") or ""),
        "market_data_snapshot_id": (evaluation.get("source_run") or {}).get("market_data_snapshot_id"),
        "fold_protocol": bson_value(protocol),
        "validation_fold_count": fold_count,
        "validation_gate": bson_value(gate),
        "validation_passed": bool(gate.get("passed")),
        "validation_completed_at": now,
        "analytics": bson_value(analytics),
        "created_at": (existing or {}).get("created_at") or now,
        "finished_at": now,
        "created_by": (actor_email or "").strip().lower() or None,
        "trader_winner_eligible": False,
        "trader_winner_block_reason": (
            "TEMPORAL live execution is not installed in the Paper/Trader runtime. "
            "This finalist can be validated and certified for research but cannot be promoted to Trader Winner yet."
        ),
    }
    if existing is None:
        db[MODEL_TUNING_VALIDATIONS_COLLECTION].insert_one({"_id": validation_id, **bson_value(validation)})
    else:
        db[MODEL_TUNING_VALIDATIONS_COLLECTION].update_one(
            {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
            {"$set": bson_value(validation)},
        )

    if bool(gate.get("passed")) and derived_strategy is not None:
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
                "temporal_validation_folds": fold_count,
                "temporal_validation_passed": True,
                "temporal_validation_at": now,
                "temporal_validation_by": (actor_email or "").strip().lower() or None,
                "temporal_trader_eligible": False,
                "temporal_trader_block_reason": validation["trader_winner_block_reason"],
                "updated_at": now,
            }},
        )
    else:
        db[MODEL_TUNING_RUNS_COLLECTION].update_one(
            {"id": str(run_id)},
            {"$set": {"updated_at": now}},
        )
    return bson_value(validation)


def certify_temporal_policy_candidate(
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
        raise ModelTuningValidationConflict("Certification is available after the research campaign completes.")
    if str(document.get("tuning_scope") or "") != TEMPORAL_POLICY_TUNING_SCOPE:
        raise ModelTuningValidationConflict("Certification is available for Temporal Policy Tuning finalists.")
    candidate = _candidate(document, candidate_id)
    validation = _validation_document(db, run_id, candidate_id)
    if validation is None or not bool(validation.get("validation_passed")):
        raise ModelTuningValidationConflict("The candidate must pass the validation-fold gate before certification.")

    protocol = _fold_protocol(document)
    fold_count = int(protocol["certification_folds"])
    if int(validation.get("certification_fold_count") or 0) == fold_count and validation.get("certification_completed_at"):
        return bson_value(validation)

    source_strategy = get_strategy(db, str(document.get("strategy_profile_id") or ""))
    anchor = _anchor_for_candidate(document, int(candidate_id))
    candidate_settings = deepcopy(candidate.get("settings") or {})
    anchor_settings = deepcopy(anchor.get("settings") or {})
    candidate_details, anchor_details, _evaluation = _full_fold_replay(
        db,
        source_strategy,
        fold_count=fold_count,
        candidate_settings=candidate_settings,
        anchor_settings=anchor_settings,
    )
    gate = _gate(document, candidate_details["metrics"], anchor_details["metrics"])

    reference_by_timestamp = _reference_map(anchor_details)
    candidate_equity = deepcopy(candidate_details["equity"])
    for row in candidate_equity:
        row["reference_equity"] = reference_by_timestamp.get(str(row.get("timestamp")))

    now = utc_now()
    certification_processing_id = f"{validation['id']}-certification"
    certification_analytics = analytics_from_equity_rotations(
        processing_id=certification_processing_id,
        equity=candidate_equity,
        rotations=deepcopy(candidate_details["rotations"]),
        metrics=_analytics_metrics(candidate_details["metrics"], anchor_details["metrics"]),
        created_at=now,
        finished_at=now,
        processing_kind="caro_certification",
        processing_label=f"CARO Candidate #{int(candidate_id)} · Certification · {fold_count} folds",
        reference_label="Champion Anchor",
    )
    update = {
        "certification_fold_count": fold_count,
        "certification_metrics": bson_value(candidate_details["metrics"]),
        "certification_anchor_metrics": bson_value(anchor_details["metrics"]),
        "certification_gate": bson_value(gate),
        "certification_passed": bool(gate.get("passed")),
        "certification_processing_id": certification_processing_id,
        "certification_analytics": bson_value(certification_analytics),
        "certification_completed_at": now,
        "finished_at": now,
    }
    db[MODEL_TUNING_VALIDATIONS_COLLECTION].update_one(
        {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
        {"$set": update},
    )
    strategy_id = str(validation.get("strategy_profile_id") or "")
    if strategy_id:
        db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": strategy_id},
            {"$set": {
                "temporal_certification_status": "certified_candidate" if gate.get("passed") else "certification_failed",
                "temporal_certification_folds": fold_count,
                "temporal_certification_passed": bool(gate.get("passed")),
                "temporal_certification_at": now,
                "temporal_certification_by": (actor_email or "").strip().lower() or None,
                "temporal_trader_eligible": False,
                "updated_at": now,
            }},
        )
    db[MODEL_TUNING_RUNS_COLLECTION].update_one(
        {"id": str(run_id)},
        {"$set": {
            "certified_candidate_id": int(candidate_id) if gate.get("passed") else None,
            "certification_processing_id": certification_processing_id,
            "certified_at": now if gate.get("passed") else None,
            "updated_at": now,
        }},
    )
    refreshed = _validation_document(db, run_id, candidate_id)
    return bson_value(refreshed or {**validation, **update})


def get_tuning_validation(db: Any, run_id: str, candidate_id: int) -> dict[str, Any] | None:
    document = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"tuning_run_id": str(run_id), "candidate_id": int(candidate_id)},
        {"_id": 0, "analytics": 0, "certification_analytics": 0},
    )
    return bson_value(document) if document is not None else None
