from __future__ import annotations

from typing import Any, Callable


_INSTALLED = False
_ORIGINAL_VALIDATED_CREATION_SOURCE: Callable[..., Any] | None = None
_ORIGINAL_GET_STATUS: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_GET_CATALOG: Callable[..., dict[str, Any]] | None = None


def _predictive_campaign(document: dict[str, Any]) -> bool:
    return str(document.get("discovery_mode") or "").strip().lower() == "predictive_only"


def _economic_validation_passed(validation: dict[str, Any]) -> bool:
    context = validation.get("context") if isinstance(validation.get("context"), dict) else {}
    gates = validation.get("gates") if isinstance(validation.get("gates"), dict) else {}
    method = str(context.get("validation_method") or validation.get("validation_method") or "").strip()
    return (
        str(validation.get("status") or "").strip().lower() == "completed"
        and str(validation.get("decision") or "").strip().upper() == "PASS"
        and method == "full_strategy_history_replay"
        and gates.get("capital_improves") is True
        and gates.get("full_history_integrity") is True
        and gates.get("research_context") is True
    )


def install_asset_discovery_economic_append_guard() -> None:
    global _INSTALLED, _ORIGINAL_VALIDATED_CREATION_SOURCE, _ORIGINAL_GET_STATUS, _ORIGINAL_GET_CATALOG
    if _INSTALLED:
        return

    from . import asset_discovery as service

    current = service._validated_creation_source
    if getattr(current, "_asset_discovery_economic_append_guard", False):
        _INSTALLED = True
        return

    _ORIGINAL_VALIDATED_CREATION_SOURCE = current
    _ORIGINAL_GET_STATUS = service.get_asset_discovery_status
    _ORIGINAL_GET_CATALOG = service.get_discovery_catalog

    def guarded_validated_creation_source(
        db: Any,
        document: dict[str, Any],
        requested_symbols: list[str],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        original = _ORIGINAL_VALIDATED_CREATION_SOURCE
        if original is None:
            raise RuntimeError("Asset Discovery economic append guard is not installed.")

        source, config, validation = original(db, document, requested_symbols)
        if _predictive_campaign(document) and not _economic_validation_passed(validation):
            raise service.AssetDiscoveryConflict(
                "Run selected-universe Full Strategy economic validation and obtain PASS with positive final-capital delta before adding assets to the Strategy."
            )
        return source, config, validation

    def get_status_with_economic_policy(db: Any) -> dict[str, Any]:
        original = _ORIGINAL_GET_STATUS
        payload = dict(original(db)) if original is not None else {}
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "selection_policy": "predictive_discovery_then_selected_universe_full_strategy_economic_validation",
            "full_history_capital_lift_required_before_append": True,
            "selected_universe_validation_runs": 2,
            "selected_universe_validation_scope": "baseline_and_exact_combined_selection",
            "per_candidate_marginal_replay_required_before_append": False,
        })
        payload["persistence_policy"] = policy
        return payload

    def get_catalog_with_economic_policy(db: Any) -> dict[str, Any]:
        original = _ORIGINAL_GET_CATALOG
        payload = dict(original(db)) if original is not None else {"count": 0, "assets": []}
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "full_history_capital_lift_required_before_append": True,
            "economic_append_gate": "exact_selected_universe_must_increase_final_strategy_capital",
        })
        payload["persistence_policy"] = policy
        return payload

    setattr(guarded_validated_creation_source, "_asset_discovery_economic_append_guard", True)
    service._validated_creation_source = guarded_validated_creation_source
    service.get_asset_discovery_status = get_status_with_economic_policy
    service.get_discovery_catalog = get_catalog_with_economic_policy
    _INSTALLED = True
