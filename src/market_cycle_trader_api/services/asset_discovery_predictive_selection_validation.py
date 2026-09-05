from __future__ import annotations

from typing import Any
from uuid import uuid4


_INSTALLED = False
_PREDICTIVE_VALIDATION_METHOD = "predictive_selection_integrity"
_ECONOMIC_REPLAY_NOT_RUN = frozenset({"", "pending", "not_run"})


def _uses_predictive_selection_validation(document: dict[str, Any]) -> bool:
    if str(document.get("discovery_mode") or "").strip().lower() != "predictive_only":
        return False
    marginal = document.get("marginal_replay") if isinstance(document.get("marginal_replay"), dict) else {}
    status = str(marginal.get("status") or "").strip().lower()
    return status in _ECONOMIC_REPLAY_NOT_RUN


def install_asset_discovery_predictive_selection_validation() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original = service.start_full_strategy_validation
    if getattr(original, "_asset_discovery_predictive_selection_validation", False):
        _INSTALLED = True
        return

    def predictive_selection_validation(
        db: Any,
        *,
        run_id: str | None,
        symbols: list[str],
    ) -> dict[str, Any]:
        document = service._campaign(db) or {}
        if not _uses_predictive_selection_validation(document):
            return original(db, run_id=run_id, symbols=symbols)

        requested_symbols = service._selection_symbols(symbols)
        if not requested_symbols:
            raise service.AssetDiscoveryConflict("Select at least one discovered asset.")

        current_run_id = str(document.get("run_id") or "").strip()
        if not current_run_id:
            raise service.AssetDiscoveryConflict("Complete an Asset Discovery search before validating a selection.")
        normalized_run_id = str(run_id or "").strip()
        if normalized_run_id and normalized_run_id != current_run_id:
            raise service.AssetDiscoveryConflict("The selected Asset Discovery run is no longer the current campaign.")
        if str(document.get("status") or "").strip().lower() != "completed":
            raise service.AssetDiscoveryConflict("Complete Predictive Asset Discovery before validating the selected assets.")

        metadata = service._discovery_metadata_for_symbols(db, document, requested_symbols)
        service._require_persistent_candidate_selection(metadata, requested_symbols)

        source_raw, source_config = service._current_research_source(db)
        source_id = str(source_raw.get("_id") or "").strip()
        source_assets = [str(item).strip().upper() for item in source_config.assets if str(item).strip()]
        source_asset_set = set(source_assets)
        added_symbols = [symbol for symbol in requested_symbols if symbol not in source_asset_set]
        if not added_symbols:
            raise service.AssetDiscoveryConflict("The selected assets are already present in the current Strategy Research source.")

        baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}
        campaign_source_id = str(baseline.get("strategy_id") or "").strip()
        campaign_source_hash = str(baseline.get("configuration_hash") or "").strip()
        if campaign_source_id and campaign_source_id != source_id:
            raise service.AssetDiscoveryConflict(
                "The Strategy source changed after Predictive Asset Discovery. Run a new Discovery campaign before validating the selection."
            )
        if campaign_source_hash and campaign_source_hash != str(source_raw.get("configuration_hash") or ""):
            raise service.AssetDiscoveryConflict(
                "The Strategy configuration changed after Predictive Asset Discovery. Run a new Discovery campaign before validating the selection."
            )

        model_snapshot = service.get_strategy_model_snapshot(db, source_id)
        snapshot_end = str(baseline.get("market_snapshot_end") or "").strip()
        if not snapshot_end:
            snapshot_end = str((document.get("discovery_selection_model") or {}).get("snapshot_end") or "").strip()

        identity_ok = all(
            str((((metadata.get(symbol) or {}).get("identity_integrity") or {}).get("status") or "")).lower() == "passed"
            for symbol in requested_symbols
        )
        predictive_ok = all(service._item_is_persistent_candidate(metadata.get(symbol)) for symbol in requested_symbols)
        source_ok = bool(source_id) and (not campaign_source_id or campaign_source_id == source_id)
        gates = {
            "predictive_candidate_selection": bool(predictive_ok),
            "identity_integrity": bool(identity_ok),
            "source_snapshot_integrity": bool(source_ok),
        }
        decision = "PASS" if all(gates.values()) else "FAIL"
        now = service.utc_now()
        validation = {
            "validation_id": f"asset-predictive-{uuid4().hex[:12]}",
            "status": "completed",
            "selected_assets": requested_symbols,
            "added_assets": added_symbols,
            "source_strategy_id": source_id,
            "source_strategy_sequence": source_raw.get("strategy_sequence"),
            "source_strategy_revision": int(source_raw.get("revision") or 1),
            "source_strategy_hash": str(source_raw.get("configuration_hash") or ""),
            "source_model_family": str(model_snapshot.get("family") or ""),
            "source_model_settings_hash": str(model_snapshot.get("settings_hash") or ""),
            "source_model_settings_revision": int(model_snapshot.get("settings_revision") or 0),
            "source_asset_count": len(source_assets),
            "snapshot_end": snapshot_end or None,
            "research_window": document.get("research_window") if isinstance(document.get("research_window"), dict) else {},
            "validation_method": _PREDICTIVE_VALIDATION_METHOD,
            "economic_validation_status": "not_run",
            "current_stage": "Predictive selection integrity validated",
            "progress_percent": 100.0,
            "context": {
                "predictive_campaign": True,
                "exact_campaign_selection": True,
                "economic_replay_run": False,
                "economic_replay_required_for_research_append": False,
            },
            "deltas": {},
            "gates": gates,
            "decision": decision,
            "created_at": now,
            "completed_at": now,
        }
        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": current_run_id},
            {"$set": {
                "status": "completed",
                "phase": "completed",
                "cancel_requested": False,
                "completed_at": document.get("completed_at") or now,
                "updated_at": now,
                "message": (
                    f"Predictive selection integrity {decision} for {', '.join(added_symbols)}. "
                    "No full-history economic replay was executed."
                ),
                "full_strategy_validation": service.bson_value(validation),
            }},
        )
        service._update_catalog_full_validation(db, requested_symbols, validation)
        return service.get_asset_discovery_status(db)

    setattr(predictive_selection_validation, "_asset_discovery_predictive_selection_validation", True)
    service.start_full_strategy_validation = predictive_selection_validation
    _INSTALLED = True
