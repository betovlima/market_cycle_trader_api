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


def _coverage_from_metadata(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("history_window_complete") is not True:
        return None
    coverage = {
        key: value
        for key, value in item.items()
        if key == "history_window_complete" or str(key).startswith("history_")
    }
    coverage["history_window_complete"] = True
    return coverage


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
        if not snapshot_end:
            raise service.AssetDiscoveryConflict("The Predictive Asset Discovery snapshot is unavailable. Run a new Discovery campaign.")

        identity_failed = [
            symbol for symbol in requested_symbols
            if str((((metadata.get(symbol) or {}).get("identity_integrity") or {}).get("status") or "")).lower() != "passed"
        ]
        adherence_failed = [
            symbol for symbol in requested_symbols
            if str((((metadata.get(symbol) or {}).get("market_adherence") or {}).get("status") or "")).lower() != "passed"
        ]
        predictive_failed = [
            symbol for symbol in requested_symbols
            if not service._item_is_persistent_candidate(metadata.get(symbol))
        ]
        source_ok = bool(source_id) and (not campaign_source_id or campaign_source_id == source_id)

        coverage_by_symbol: dict[str, Any] = {}
        needs_history_download: list[str] = []
        for symbol in added_symbols:
            cached_coverage = _coverage_from_metadata(metadata.get(symbol) or {})
            if cached_coverage is not None:
                coverage_by_symbol[symbol] = cached_coverage
            else:
                needs_history_download.append(symbol)

        if needs_history_download:
            baseline_frames = service._baseline_frames(source_config, snapshot_end)
            required_sessions = service._baseline_required_sessions(baseline_frames, source_config, snapshot_end)
            for symbol in needs_history_download:
                try:
                    _frame, coverage = service._candidate_history_coverage(
                        db,
                        symbol,
                        source_config,
                        snapshot_end,
                        required_sessions,
                    )
                    coverage_by_symbol[symbol] = coverage
                except Exception as exc:
                    coverage_by_symbol[symbol] = {
                        "history_window_complete": False,
                        "reason": str(exc)[:300],
                    }

        history_failed = [
            symbol for symbol in added_symbols
            if not bool((coverage_by_symbol.get(symbol) or {}).get("history_window_complete"))
        ]

        gates = {
            "predictive_candidate_selection": not predictive_failed,
            "market_adherence": not adherence_failed,
            "identity_integrity": not identity_failed,
            "full_history_integrity": not history_failed,
            "source_snapshot_integrity": bool(source_ok),
        }
        decision = "PASS" if all(gates.values()) else "FAIL"
        failed_assets = {
            "predictive_candidate_selection": predictive_failed,
            "market_adherence": adherence_failed,
            "identity_integrity": identity_failed,
            "full_history_integrity": history_failed,
        }
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
            "snapshot_end": snapshot_end,
            "research_window": document.get("research_window") if isinstance(document.get("research_window"), dict) else {},
            "validation_method": _PREDICTIVE_VALIDATION_METHOD,
            "economic_validation_status": "not_run",
            "current_stage": "Predictive selection and historical integrity validated",
            "progress_percent": 100.0,
            "context": {
                "predictive_campaign": True,
                "exact_campaign_selection": True,
                "market_adherence_checked": True,
                "full_history_integrity_checked": True,
                "economic_replay_run": False,
                "economic_replay_required_for_research_append": False,
            },
            "history_coverage": coverage_by_symbol,
            "failed_assets": failed_assets,
            "deltas": {},
            "gates": gates,
            "decision": decision,
            "created_at": now,
            "completed_at": now,
        }
        failed_labels = [name for name, passed in gates.items() if not passed]
        message = (
            f"Predictive selection integrity {decision} for {', '.join(added_symbols)}. "
            "Market adherence and complete historical coverage were checked; no economic replay was executed."
        )
        if failed_labels:
            message += " Failed gates: " + ", ".join(failed_labels) + "."
        if history_failed:
            message += " Incomplete Strategy history: " + ", ".join(history_failed) + "."

        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": current_run_id},
            {"$set": {
                "status": "completed",
                "phase": "completed",
                "cancel_requested": False,
                "completed_at": document.get("completed_at") or now,
                "updated_at": now,
                "message": message,
                "full_strategy_validation": service.bson_value(validation),
            }},
        )
        service._update_catalog_full_validation(db, requested_symbols, validation)
        return service.get_asset_discovery_status(db)

    setattr(predictive_selection_validation, "_asset_discovery_predictive_selection_validation", True)
    service.start_full_strategy_validation = predictive_selection_validation
    _INSTALLED = True
