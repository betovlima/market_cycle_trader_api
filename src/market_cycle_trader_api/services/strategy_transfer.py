from __future__ import annotations

from copy import deepcopy
from typing import Any
import os

from bson import json_util
from pymongo import ReturnDocument

from ..core.config import API_VERSION
from ..core.environment import PROJECT_ROOT
from ..infrastructure.persistence.mongo_repository import (
    MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
    utc_now,
)
from .strategy_lab import create_strategy, get_strategy

TRANSFER_SCHEMA_VERSION = 1
TRANSFER_FILE = PROJECT_ROOT / "scripts" / "strategy_import.json"
SUPPORTED_STRATEGY_KIND = "temporal_intelligence"
SUPPORTED_TEMPORAL_VARIANT = "winner_anchored_timing"
DEFAULT_LOCAL_TRANSFER_STRATEGY_ID = "strategy-bc88335cfe854dad97d0055cc4089d12"


class StrategyTransferError(RuntimeError):
    pass


class StrategyTransferNotFound(StrategyTransferError):
    pass


class StrategyTransferConflict(StrategyTransferError):
    pass


def _normalized_id(value: Any) -> str:
    return str(value or "").strip()


def _ending_capital(run: dict[str, Any]) -> float | None:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    value = capital.get("ending_capital")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _validate_source_strategy(strategy: dict[str, Any], run: dict[str, Any]) -> None:
    if _normalized_id(strategy.get("strategy_kind")) != SUPPORTED_STRATEGY_KIND:
        raise StrategyTransferConflict("Only materialized Temporal Intelligence Strategies can be transferred by this endpoint.")
    if _normalized_id(strategy.get("temporal_strategy_variant")) != SUPPORTED_TEMPORAL_VARIANT:
        raise StrategyTransferConflict(
            "Only winner_anchored_timing Temporal Strategies are supported by this transfer endpoint. "
            "Stateful and MILP materializations require additional dependencies and are intentionally rejected."
        )
    if _normalized_id(strategy.get("tuning_target")) != "temporal_policy":
        raise StrategyTransferConflict("The selected Temporal Strategy is not a temporal_policy materialization.")
    run_id = _normalized_id(strategy.get("source_temporal_run_id"))
    if not run_id or run_id != _normalized_id(run.get("id")):
        raise StrategyTransferConflict("The selected Strategy does not match its source Temporal Intelligence run.")
    if _normalized_id(run.get("status")) != "completed" or not isinstance(run.get("result"), dict):
        raise StrategyTransferConflict("The source Temporal Intelligence run is not completed.")
    strategy_hash = _normalized_id(strategy.get("configuration_hash"))
    run_hash = _normalized_id(run.get("strategy_configuration_hash"))
    if not strategy_hash or strategy_hash != run_hash:
        raise StrategyTransferConflict("Strategy and source Temporal run configuration hashes do not match.")


def export_strategy_transfer_package(
    db: Any,
    *,
    strategy_id: str | None,
    strategy_sequence: int | None,
    include_market_snapshot: bool,
    actor_email: str | None,
) -> dict[str, Any]:
    strategy_key = _normalized_id(strategy_id)
    if not strategy_key and strategy_sequence is None:
        strategy_key = DEFAULT_LOCAL_TRANSFER_STRATEGY_ID
    if strategy_key:
        strategy = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_key})
    elif strategy_sequence is not None:
        strategy = db[STRATEGY_PROFILES_COLLECTION].find_one({"strategy_sequence": int(strategy_sequence)})
        strategy_key = _normalized_id((strategy or {}).get("_id"))
    else:
        strategy = None
    if strategy is None:
        raise StrategyTransferNotFound("Strategy not found in the local Strategy Catalog.")

    run_id = _normalized_id(strategy.get("source_temporal_run_id"))
    if not run_id:
        raise StrategyTransferConflict("The selected Strategy does not contain source_temporal_run_id.")
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id})
    if run is None:
        raise StrategyTransferNotFound("The source Temporal Intelligence run is unavailable in the local MongoDB.")
    _validate_source_strategy(strategy, run)

    observations = list(db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find({"run_id": run_id}))
    artifacts = list(db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find({"run_id": run_id}))
    if not artifacts:
        raise StrategyTransferConflict("The source Temporal run does not contain artifacts required to reproduce the materialized Strategy.")

    snapshot_id = _normalized_id(run.get("market_data_snapshot_id")).lower()
    snapshot_documents: list[dict[str, Any]] = []
    if include_market_snapshot and snapshot_id:
        snapshot_documents = list(
            db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].find({"snapshot_id": snapshot_id})
        )
        manifest = next(
            (
                item
                for item in snapshot_documents
                if _normalized_id(item.get("kind")) == "manifest" and bool(item.get("ready"))
            ),
            None,
        )
        if manifest is None:
            raise StrategyTransferConflict(
                "The source Temporal run references a frozen market snapshot, but its ready manifest is unavailable locally."
            )

    source_strategy_id = _normalized_id(run.get("strategy_profile_id") or strategy.get("source_strategy_id"))
    source_strategy = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": source_strategy_id}) if source_strategy_id else None
    source_hash = _normalized_id(run.get("strategy_configuration_hash") or strategy.get("configuration_hash"))

    package = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "package_kind": "market_cycle_trader_temporal_strategy_transfer",
        "created_at": utc_now(),
        "created_by": (actor_email or "").strip().lower() or None,
        "source_api_version": API_VERSION,
        "manifest": {
            "strategy_id": strategy_key,
            "strategy_sequence": int(strategy.get("strategy_sequence") or 0),
            "strategy_kind": _normalized_id(strategy.get("strategy_kind")),
            "temporal_strategy_variant": _normalized_id(strategy.get("temporal_strategy_variant")),
            "source_temporal_run_id": run_id,
            "configuration_hash": source_hash,
            "source_strategy_id": source_strategy_id or None,
            "source_strategy_revision": int(run.get("strategy_profile_revision") or (source_strategy or {}).get("revision") or 1),
            "source_strategy_catalog_status": _normalized_id((source_strategy or {}).get("catalog_status")) or None,
            "source_ending_capital": _ending_capital(run),
            "market_data_snapshot_id": snapshot_id or None,
            "observation_count": len(observations),
            "artifact_count": len(artifacts),
            "market_snapshot_document_count": len(snapshot_documents),
        },
        "documents": {
            "strategy": strategy,
            "temporal_run": run,
            "observations": observations,
            "artifacts": artifacts,
            "market_snapshot": snapshot_documents,
        },
    }

    TRANSFER_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRANSFER_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json_util.dumps(package, indent=2, json_options=json_util.RELAXED_JSON_OPTIONS),
        encoding="utf-8",
    )
    os.replace(temporary, TRANSFER_FILE)
    size = TRANSFER_FILE.stat().st_size
    return {
        "status": "exported",
        "file": str(TRANSFER_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "size_bytes": int(size),
        "strategy_id": strategy_key,
        "strategy_sequence": int(strategy.get("strategy_sequence") or 0),
        "source_temporal_run_id": run_id,
        "ending_capital": _ending_capital(run),
        "configuration_hash": source_hash,
        "market_data_snapshot_id": snapshot_id or None,
        "observation_count": len(observations),
        "artifact_count": len(artifacts),
        "market_snapshot_document_count": len(snapshot_documents),
        "next_step": "Commit scripts/strategy_import.json with API v6.6.2, deploy it to production, then call POST /api/admin/strategy-transfer/import with confirm=IMPORT.",
    }


def _load_transfer_package() -> dict[str, Any]:
    if not TRANSFER_FILE.is_file():
        raise StrategyTransferNotFound("scripts/strategy_import.json was not found in the deployed API package.")
    if TRANSFER_FILE.stat().st_size <= 2:
        raise StrategyTransferConflict("scripts/strategy_import.json is empty.")
    try:
        payload = json_util.loads(TRANSFER_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StrategyTransferConflict(f"Unable to parse scripts/strategy_import.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise StrategyTransferConflict("The Strategy transfer file does not contain a JSON object.")
    if int(payload.get("schema_version") or 0) != TRANSFER_SCHEMA_VERSION:
        raise StrategyTransferConflict("Unsupported Strategy transfer schema_version.")
    if _normalized_id(payload.get("package_kind")) != "market_cycle_trader_temporal_strategy_transfer":
        raise StrategyTransferConflict("Invalid Strategy transfer package_kind.")
    return payload


def _select_production_base(db: Any, configuration_hash: str) -> dict[str, Any]:
    candidates = list(db[STRATEGY_PROFILES_COLLECTION].find({"configuration_hash": configuration_hash}))
    candidates = [
        item
        for item in candidates
        if _normalized_id(item.get("strategy_kind") or "standard") == "standard"
    ]
    if not candidates:
        raise StrategyTransferConflict(
            "Production does not contain a standard Strategy with the same configuration_hash as the imported Temporal Strategy."
        )
    status_order = {"winner": 0, "research": 1, "saved": 2}
    candidates.sort(
        key=lambda item: (
            status_order.get(_normalized_id(item.get("catalog_status")), 9),
            -int(item.get("strategy_sequence") or 0),
        )
    )
    return candidates[0]


def _without_id(document: dict[str, Any], **updates: Any) -> dict[str, Any]:
    payload = deepcopy(document)
    payload.pop("_id", None)
    payload.update(updates)
    return payload


def import_strategy_transfer_package(db: Any, *, actor_email: str | None) -> dict[str, Any]:
    package = _load_transfer_package()
    manifest = package.get("manifest") if isinstance(package.get("manifest"), dict) else {}
    documents = package.get("documents") if isinstance(package.get("documents"), dict) else {}
    source_strategy = documents.get("strategy") if isinstance(documents.get("strategy"), dict) else None
    source_run = documents.get("temporal_run") if isinstance(documents.get("temporal_run"), dict) else None
    observations = documents.get("observations") if isinstance(documents.get("observations"), list) else []
    artifacts = documents.get("artifacts") if isinstance(documents.get("artifacts"), list) else []
    snapshot_documents = documents.get("market_snapshot") if isinstance(documents.get("market_snapshot"), list) else []
    if source_strategy is None or source_run is None:
        raise StrategyTransferConflict("The transfer package does not contain Strategy and Temporal run documents.")
    _validate_source_strategy(source_strategy, source_run)

    run_id = _normalized_id(source_run.get("id"))
    configuration_hash = _normalized_id(source_strategy.get("configuration_hash"))
    if run_id != _normalized_id(manifest.get("source_temporal_run_id")):
        raise StrategyTransferConflict("Transfer manifest source_temporal_run_id does not match the packaged run.")
    if configuration_hash != _normalized_id(manifest.get("configuration_hash")):
        raise StrategyTransferConflict("Transfer manifest configuration_hash does not match the packaged Strategy.")
    if len(observations) != int(manifest.get("observation_count") or 0):
        raise StrategyTransferConflict("Transfer observation count does not match its manifest.")
    if len(artifacts) != int(manifest.get("artifact_count") or 0) or not artifacts:
        raise StrategyTransferConflict("Transfer artifact count does not match its manifest.")
    if len(snapshot_documents) != int(manifest.get("market_snapshot_document_count") or 0):
        raise StrategyTransferConflict("Transfer market snapshot count does not match its manifest.")

    existing_strategy = db[STRATEGY_PROFILES_COLLECTION].find_one({
        "source_temporal_run_id": run_id,
        "strategy_kind": SUPPORTED_STRATEGY_KIND,
        "temporal_strategy_variant": SUPPORTED_TEMPORAL_VARIANT,
    })
    if existing_strategy is not None:
        return {
            "status": "already_imported",
            "strategy": get_strategy(db, _normalized_id(existing_strategy.get("_id"))),
            "source_temporal_run_id": run_id,
        }
    existing_run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 1})
    if existing_run is not None:
        raise StrategyTransferConflict(
            "Production already contains the packaged Temporal run without its imported Strategy. Refusing to continue from a partial state."
        )

    production_base = _select_production_base(db, configuration_hash)
    base_id = _normalized_id(production_base.get("_id"))
    base_revision = int(production_base.get("revision") or 1)
    base_name = _normalized_id(production_base.get("name"))

    snapshot_id = _normalized_id(source_run.get("market_data_snapshot_id")).lower()
    existing_snapshot_count = 0
    if snapshot_id and snapshot_documents:
        existing_snapshot_count = int(db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].count_documents({"snapshot_id": snapshot_id}) or 0)
        if existing_snapshot_count not in {0, len(snapshot_documents)}:
            raise StrategyTransferConflict(
                "Production contains a partial frozen market snapshot with the same snapshot_id. Refusing to merge partial snapshot data."
            )
        if existing_snapshot_count == len(snapshot_documents):
            manifest_document = db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].find_one({
                "snapshot_id": snapshot_id,
                "kind": "manifest",
                "ready": True,
            })
            if manifest_document is None:
                raise StrategyTransferConflict("Production snapshot documents exist but the ready manifest is missing.")

    inserted_snapshot = False
    inserted_run = False
    inserted_observations = False
    inserted_artifacts = False
    created_strategy_id: str | None = None
    now = utc_now()
    actor = (actor_email or "").strip().lower() or None
    try:
        if snapshot_documents and existing_snapshot_count == 0:
            db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].insert_many(deepcopy(snapshot_documents), ordered=True)
            inserted_snapshot = True

        run_document = _without_id(
            source_run,
            strategy_profile_id=base_id,
            strategy_profile_name=base_name,
            strategy_profile_revision=base_revision,
            strategy_configuration_hash=configuration_hash,
            strategy_kind="standard",
            temporal_strategy_variant=None,
            research_processing_id=None,
            research_processing_kind=None,
            research_processing_label=None,
            stateful_reference_bundle=None,
            materialized_strategy_id=None,
            materialized_strategy_name=None,
            materialized_strategy_at=None,
            imported_at=now,
            imported_by=actor,
            imported_from_strategy_id=_normalized_id(source_strategy.get("_id")) or None,
            imported_from_api_version=_normalized_id(package.get("source_api_version")) or None,
        )
        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].insert_one(run_document)
        inserted_run = True

        if observations:
            db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].insert_many(
                [_without_id(item, run_id=run_id) for item in observations if isinstance(item, dict)],
                ordered=True,
            )
            inserted_observations = True
        if artifacts:
            db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].insert_many(
                [_without_id(item, run_id=run_id) for item in artifacts if isinstance(item, dict)],
                ordered=True,
            )
            inserted_artifacts = True

        created = create_strategy(
            db,
            name="Imported Temporal Strategy",
            description="Imported Temporal Strategy",
            clone_from_strategy_id=base_id,
            actor_email=actor,
        )
        created_strategy_id = _normalized_id(created.get("id"))
        if not created_strategy_id:
            raise StrategyTransferConflict("Unable to reserve a Strategy Catalog identity for the imported Strategy.")
        created_document = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": created_strategy_id})
        if created_document is None or _normalized_id(created_document.get("configuration_hash")) != configuration_hash:
            raise StrategyTransferConflict("The production base clone does not reproduce the packaged Strategy configuration_hash.")

        temporal_policy = deepcopy(source_strategy.get("temporal_policy_snapshot")) if isinstance(source_strategy.get("temporal_policy_snapshot"), dict) else None
        if temporal_policy is not None:
            temporal_policy["source_strategy_id"] = base_id
            temporal_policy["source_strategy_revision"] = base_revision
            temporal_policy["source_strategy_configuration_hash"] = configuration_hash
        research_model = source_strategy.get("research_model_snapshot") if isinstance(source_strategy.get("research_model_snapshot"), dict) else None
        updated = db[STRATEGY_PROFILES_COLLECTION].find_one_and_update(
            {"_id": created_strategy_id},
            {
                "$set": {
                    "status": "draft",
                    "catalog_status": "saved",
                    "locked": False,
                    "strategy_kind": SUPPORTED_STRATEGY_KIND,
                    "tuning_target": "temporal_policy",
                    "source_temporal_run_id": run_id,
                    "source_temporal_experiment": _normalized_id(source_strategy.get("source_temporal_experiment") or source_run.get("experiment")),
                    "temporal_strategy_variant": SUPPORTED_TEMPORAL_VARIANT,
                    "temporal_policy_revision": int(source_strategy.get("temporal_policy_revision") or 1),
                    "temporal_policy_snapshot": bson_value(temporal_policy) if temporal_policy is not None else None,
                    "research_model_snapshot": bson_value(deepcopy(research_model)) if research_model is not None else created_document.get("research_model_snapshot"),
                    "research_model_revision": int(source_strategy.get("research_model_revision") or 1),
                    "research_reference_assets": list(source_strategy.get("research_reference_assets") or created_document.get("research_reference_assets") or []),
                    "source_strategy_id": base_id,
                    "source_strategy_revision": base_revision,
                    "updated_at": now,
                    "updated_by": actor,
                    "imported_at": now,
                    "imported_by": actor,
                    "imported_from_strategy_id": _normalized_id(source_strategy.get("_id")) or None,
                    "imported_from_strategy_sequence": int(source_strategy.get("strategy_sequence") or 0) or None,
                    "imported_from_api_version": _normalized_id(package.get("source_api_version")) or None,
                },
                "$unset": {
                    "source_stateful_replay_id": "",
                    "source_stateful_processing_id": "",
                    "stateful_candidate_key": "",
                    "stateful_candidate_label": "",
                    "last_backtest_id": "",
                    "last_backtest_status": "",
                    "last_backtest_revision": "",
                    "candidate_at": "",
                    "candidate_by": "",
                    "candidate_note": "",
                    "candidate_revision": "",
                    "candidate_backtest_id": "",
                    "promoted_at": "",
                    "promoted_by": "",
                    "superseded_at": "",
                    "superseded_by_strategy_id": "",
                },
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            raise StrategyTransferConflict("Unable to finalize the imported Temporal Strategy.")

        db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].update_one(
            {"id": run_id},
            {
                "$set": {
                    "materialized_strategy_id": created_strategy_id,
                    "materialized_strategy_name": _normalized_id(updated.get("name")),
                    "materialized_strategy_at": now,
                    "updated_at": now,
                }
            },
        )

        verified_observations = int(db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].count_documents({"run_id": run_id}) or 0)
        verified_artifacts = int(db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].count_documents({"run_id": run_id}) or 0)
        if verified_observations != len(observations) or verified_artifacts != len(artifacts):
            raise StrategyTransferConflict("Imported Temporal run dependency counts do not match the transfer package.")

        return {
            "status": "imported",
            "strategy": get_strategy(db, created_strategy_id),
            "source_temporal_run_id": run_id,
            "production_base_strategy_id": base_id,
            "production_base_strategy_name": base_name,
            "ending_capital": _ending_capital(source_run),
            "observation_count": verified_observations,
            "artifact_count": verified_artifacts,
            "market_snapshot_document_count": int(
                db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].count_documents({"snapshot_id": snapshot_id}) or 0
            ) if snapshot_id else 0,
            "winner_changed": False,
        }
    except Exception as exc:
        if created_strategy_id:
            db[STRATEGY_PROFILES_COLLECTION].delete_one({"_id": created_strategy_id})
        if inserted_artifacts:
            db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].delete_many({"run_id": run_id})
        if inserted_observations:
            db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].delete_many({"run_id": run_id})
        if inserted_run:
            db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].delete_many({"id": run_id})
        if inserted_snapshot and snapshot_id:
            db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].delete_many({"snapshot_id": snapshot_id})
        if isinstance(exc, StrategyTransferError):
            raise
        raise StrategyTransferError(f"Strategy transfer import failed: {exc}") from exc
