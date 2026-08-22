from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import STRATEGY_PROFILES_COLLECTION, bson_value
from .config import PROCESSING_PREFIX


def processing_id(strategy_id: str) -> str:
    normalized = str(strategy_id or "").strip()
    if not normalized:
        raise ValueError("MILP Strategy id is required.")
    return f"{PROCESSING_PREFIX}{normalized}"


def strategy_id_from_processing(value: str) -> str | None:
    normalized = str(value or "").strip()
    if not normalized.startswith(PROCESSING_PREFIX):
        return None
    strategy_id = normalized[len(PROCESSING_PREFIX):].strip()
    return strategy_id or None


def is_milp_strategy(strategy: dict[str, Any] | None) -> bool:
    if not isinstance(strategy, dict):
        return False
    return (
        str(strategy.get("strategy_kind") or "") == "temporal_intelligence"
        and str(strategy.get("temporal_strategy_variant") or "") == "milp_decision_overlay"
        and str(strategy.get("tuning_target") or "") == "decision_optimization"
    )


def resolve_research_processing_context(strategy: dict[str, Any]) -> tuple[str, str, str, None] | None:
    if not is_milp_strategy(strategy):
        return None
    strategy_id = str(strategy.get("id") or strategy.get("_id") or "").strip()
    if not strategy_id:
        raise ValueError("MILP Strategy id is missing.")
    policy = strategy.get("temporal_policy") if isinstance(strategy.get("temporal_policy"), dict) else {}
    binding = str(strategy.get("source_decision_optimization_id") or policy.get("source_decision_optimization_id") or "").strip()
    if not binding:
        raise ValueError("The selected MILP Strategy Research snapshot is missing its Decision Optimization binding.")
    return processing_id(strategy_id), "strategy_research_decision_optimization", "Strategy Research · MILP", None


def processing_analytics(db: Any, value: str) -> dict[str, Any] | None:
    strategy_id = strategy_id_from_processing(value)
    if not strategy_id:
        return None
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one(
        {"_id": strategy_id},
        {
            "_id": 1,
            "name": 1,
            "revision": 1,
            "configuration_hash": 1,
            "strategy_kind": 1,
            "temporal_strategy_variant": 1,
            "tuning_target": 1,
            "source_decision_optimization_id": 1,
            "decision_replay_snapshot": 1,
        },
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Strategy Research MILP processing source was not found.")
    if not is_milp_strategy(profile):
        raise HTTPException(status_code=409, detail="The selected Strategy Research profile is not a MILP Decision Strategy.")
    replay = profile.get("decision_replay_snapshot") if isinstance(profile.get("decision_replay_snapshot"), dict) else None
    if replay is None:
        raise HTTPException(status_code=409, detail="The selected MILP Strategy does not contain a reproducible Decision Optimization replay snapshot.")
    payload = deepcopy(bson_value(replay))
    payload.update({
        "job_id": str(value),
        "processing_id": str(value),
        "processing_kind": "strategy_research_decision_optimization",
        "processing_label": "Strategy Research · MILP",
        "reference_label": "Source MILP Strategy",
        "strategy_profile_id": strategy_id,
        "strategy_profile_name": str(profile.get("name") or "MILP Strategy"),
        "strategy_profile_revision": int(profile.get("revision") or 1),
        "strategy_configuration_hash": str(profile.get("configuration_hash") or ""),
        "source_decision_optimization_id": str(profile.get("source_decision_optimization_id") or ""),
    })
    return bson_value(payload)
