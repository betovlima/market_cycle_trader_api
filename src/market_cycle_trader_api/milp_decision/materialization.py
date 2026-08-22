from __future__ import annotations

from typing import Any

from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import (
    STRATEGY_PROFILES_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from ..services.strategy_lab import StrategyLabConflict, StrategyLabNotFound, create_strategy, get_strategy
from .config import COLLECTION
from .errors import MilpDecisionError
from .persistence import get_completed


def materialize(db: Any, run_id: str, optimization_id: str, *, actor_email: str | None) -> dict[str, Any]:
    optimization = get_completed(db, run_id, optimization_id)
    if optimization is None:
        raise MilpDecisionError("Completed MILP Decision Optimization result not found.")
    control_parity = optimization.get("control_parity") if isinstance(optimization.get("control_parity"), dict) else {}
    if str(control_parity.get("status") or "") != "passed":
        raise MilpDecisionError("MILP Strategy materialization requires passed exact Control replay parity.")
    existing_id = str(optimization.get("materialized_strategy_id") or "").strip()
    if existing_id:
        try:
            return {"created": False, "strategy": get_strategy(db, existing_id)}
        except StrategyLabNotFound:
            pass

    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if run is None:
        raise MilpDecisionError("Temporal Intelligence run not found.")
    source_strategy_id = str(run.get("strategy_profile_id") or "").strip()
    source = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": source_strategy_id})
    if source is None:
        raise MilpDecisionError("The source Strategy is no longer available in the catalog.")
    expected_revision = int(run.get("strategy_profile_revision") or 0)
    expected_hash = str(run.get("strategy_configuration_hash") or "").strip()
    if expected_revision and int(source.get("revision") or 1) != expected_revision:
        raise MilpDecisionError("The source Strategy revision no longer matches the research run.")
    if expected_hash and str(source.get("configuration_hash") or "") != expected_hash:
        raise MilpDecisionError("The source Strategy configuration no longer matches the research run.")

    try:
        created = create_strategy(
            db,
            name="",
            description="",
            clone_from_strategy_id=source_strategy_id,
            actor_email=actor_email,
        )
    except (StrategyLabConflict, StrategyLabNotFound, ValueError) as exc:
        raise MilpDecisionError(str(exc)) from exc

    policy_snapshot = {
        "schema_version": 1,
        "family": "milp_decision_optimization",
        "source_run_id": str(run_id),
        "source_decision_optimization_id": str(optimization_id),
        "source_strategy_id": source_strategy_id,
        "source_strategy_revision": expected_revision or int(source.get("revision") or 1),
        "source_strategy_configuration_hash": expected_hash or str(source.get("configuration_hash") or ""),
        "configuration": bson_value(optimization.get("configuration") or {}),
        "solver": bson_value(optimization.get("solver") or {}),
        "validation": {
            "metrics": bson_value(optimization.get("metrics") or {}),
            "folds": bson_value(optimization.get("folds") or []),
            "cost_stress": bson_value(optimization.get("cost_stress") or []),
            "attribution": bson_value(optimization.get("attribution") or {}),
        },
    }
    replay_snapshot = bson_value(optimization.get("analytics") or {})
    now = utc_now()
    updated = db[STRATEGY_PROFILES_COLLECTION].find_one_and_update(
        {"_id": created["id"], "revision": int(created["revision"])},
        {
            "$set": {
                "strategy_kind": "temporal_intelligence",
                "tuning_target": "decision_optimization",
                "temporal_strategy_variant": "milp_decision_overlay",
                "source_temporal_run_id": str(run_id),
                "source_temporal_experiment": str(run.get("experiment") or ""),
                "source_decision_optimization_id": str(optimization_id),
                "source_decision_processing_id": str(optimization.get("processing_id") or ""),
                "temporal_policy_revision": 1,
                "temporal_policy_snapshot": bson_value(policy_snapshot),
                "decision_replay_snapshot": replay_snapshot,
                "research_only": True,
                "research_only_reason": "milp_live_runtime_not_available",
                "updated_at": now,
                "updated_by": (actor_email or "").strip().lower() or None,
            },
            "$unset": {
                "source_stateful_replay_id": "",
                "source_stateful_processing_id": "",
                "stateful_candidate_key": "",
                "stateful_candidate_label": "",
            },
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise MilpDecisionError("Unable to materialize the MILP Decision Optimization Strategy.")
    db[COLLECTION].update_one(
        {"id": str(optimization_id)},
        {"$set": {"materialized_strategy_id": created["id"], "materialized_at": now, "updated_at": now}},
    )
    return {"created": True, "strategy": get_strategy(db, created["id"])}
