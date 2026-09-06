from __future__ import annotations

from typing import Any


_INSTALLED = False


def install_asset_discovery_predictive_selection_validation() -> None:
    """Keep Predictive Discovery fast, but restore economic validation before append.

    Predictive Discovery remains the cheap pre-filter.  When the user explicitly
    validates the selected universe we delegate to the original Full Strategy
    validation worker, which replays the current Strategy and the exact selected
    universe over the same complete historical window and requires capital lift.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original = service.start_full_strategy_validation
    if getattr(original, "_asset_discovery_predictive_economic_validation", False):
        _INSTALLED = True
        return

    def predictive_economic_validation(
        db: Any,
        *,
        run_id: str | None,
        symbols: list[str],
    ) -> dict[str, Any]:
        document = service._campaign(db) or {}
        if str(document.get("discovery_mode") or "").strip().lower() != "predictive_only":
            return original(db, run_id=run_id, symbols=symbols)

        requested_symbols = service._selection_symbols(symbols)
        if not requested_symbols:
            raise service.AssetDiscoveryConflict("Select at least one discovered asset.")

        current_run_id = str(document.get("run_id") or "").strip()
        normalized_run_id = str(run_id or "").strip()
        if not current_run_id:
            raise service.AssetDiscoveryConflict("Complete an Asset Discovery search before validating a selection.")
        if normalized_run_id and normalized_run_id != current_run_id:
            raise service.AssetDiscoveryConflict("The selected Asset Discovery run is no longer the current campaign.")
        if str(document.get("status") or "").strip().lower() != "completed":
            raise service.AssetDiscoveryConflict("Complete Predictive Asset Discovery before validating the selected assets.")

        metadata = service._discovery_metadata_for_symbols(db, document, requested_symbols)
        service._require_persistent_candidate_selection(metadata, requested_symbols)

        # The base Full Strategy validation performs the authoritative checks:
        # exact source revision/hash, complete historical coverage, same decision
        # window, baseline replay, combined selected-universe replay and
        # capital_improves > 0 before PASS.
        return original(db, run_id=current_run_id, symbols=requested_symbols)

    setattr(predictive_economic_validation, "_asset_discovery_predictive_economic_validation", True)
    service.start_full_strategy_validation = predictive_economic_validation
    _INSTALLED = True
